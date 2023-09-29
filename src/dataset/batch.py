import torch
from einops import rearrange

from util.utility import recursive_operation

class Batch:
    def __init__(self, samples, target_info=None, with_transformer_targets=False, with_image_targets=True):
        self.samples = samples
        self.training = target_info is not None
        self.with_transformer_targets = with_transformer_targets
        self.with_image_targets = with_image_targets

        self.images = self._generate_image_tensor() if with_image_targets else None
        self.transform_matrices = self._generate_transform_matrices()
        self.targets = self._generate_targets(target_info)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def to(self, device):
        self.images = self.images.to(device)
        self.transform_matrices = recursive_operation(self.transform_matrices, lambda obj: obj.to(device))
        if self.training:
            self.targets = recursive_operation(self.targets, lambda obj: obj.to(device))
        return self

    def _extract_box_parameters(self, selector):
        n_frames = len(self.samples[0].additional_targets) + 1
        parameter_tensor = [[
            torch.tensor([selector(i, box) for i, box in enumerate(frame_targets)], dtype=torch.float32)
            for frame_targets in (sample.additional_targets + [sample.targets])
        ] for sample in self.samples]  # (samples, time_steps, boxes)
        parameter_tensor = [
            torch.cat([sample_vis[i] for sample_vis in parameter_tensor], dim=0)
            for i in range(n_frames)
        ] # (time_steps, cat_of_batch_box_visibilities)
        return parameter_tensor

    def _generate_image_tensor(self):
        images = rearrange([sample.image for sample in self.samples], 'b t h w c -> (t b) c h w')
        images = torch.from_numpy(images.copy()).float() # for some reason .copy() is needed here, otherwise .to is super slow (threading related?)
        return images

    def _generate_transform_matrices(self):
        transform_matrices = [sample.transform.compute(inverse=True) for sample in self.samples]
        transform_matrices = recursive_operation(transform_matrices, lambda obj: torch.tensor(obj, dtype=torch.float32))
        return transform_matrices

    def _generate_targets(self, target_info):
        targets = {}
        targets['box_visibilities'] = self._extract_box_parameters(lambda i, b: b.visibility)
        targets['label_uncertainty'] = self._extract_box_parameters(lambda i, b: b.label_uncertainty)

        if self.training:
            from model.detection_head import DetectionHead
            pose_modifier = -target_info['n_classes'] + 1
            action_modifier = -len(DetectionHead.static_spec['action_cls']) + 1
            target_channels = target_info['n_channels'] + pose_modifier + action_modifier # target pose and action is represented as int instead of one-hot
            target_shapes = [(len(self), *shape[1:], target_channels) for shape in target_info['tensor_shapes']] # b, d, y, x, c
            targets['detection_heads'] = [DetectionHead.build_targets(self, target_shape) for target_shape in target_shapes]
            for head in targets['detection_heads']:
                head['box_visibilities'] = targets['box_visibilities'][-1]

            from model.orientation import OrientationModule
            targets['orientation_module'] = OrientationModule.build_targets(self)
        if self.with_transformer_targets:
            from model.tracking_model import TrackModel
            transformer_targets = TrackModel.build_targets(self)
            targets['transformer'] = [{
                'targets': transformer_targets[i],
                'box_visibilities': targets['box_visibilities'][i],
                'label_uncertainty': targets['label_uncertainty'][i]
            } for i in range(len(transformer_targets))]

        return targets

    def postprocess(self, raw_output, nms_kwargs=None):
        for raw_predictions, transform_matrices, sample in zip(raw_output, self.transform_matrices, self.samples):
            sample.postprocess(raw_predictions, transform_matrices=transform_matrices, nms_kwargs=nms_kwargs)

def get_collate_fn(model, phase, **kwargs):
    head_shape_info = model.head_shape_info
    def collate_fn(samples):
        batch = Batch(samples, target_info=(head_shape_info if phase == 'train' else None), **kwargs)
        return batch
    return collate_fn