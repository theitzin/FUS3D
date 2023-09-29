import numpy as np
from scipy.stats.mstats import gmean
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from einops import rearrange, reduce

from model.util import tanh_nd, dot_product_loss
from dataset.box import Box
from util.utility import DataTable

def target_distance(loc1, loc2, threshold=0.5):
    dist = torch.norm(loc1 - loc2, dim=1)
    ious = torch.zeros(dist.shape, device=dist.device)
    mask = dist < threshold
    ious[mask] = -dist[mask]/threshold + 1
    return ious

class DetectionHead(nn.Module):
    static_spec = {
        'x': [0],
        'y': [1],
        'd': [2],
        'direction': [3, 4],
        'dimensions': [5, 6, 7],
        'centerness': [8],
        'conf': [9],
        'action_cls': [10, 11, 12] # no action, friendly, aggressive
    } # does not include cls as it depends on __init__ parameters
    target_spec = {
        'action_cls': [10],
        'cls': [11]
    }
    static_n_features = sum([len(cols) for key, cols in static_spec.items()])

    def __init__(self, grid_shape, n_classes, n_tracking_frames):
        super().__init__()
        self.n_tracking_frames = n_tracking_frames
        self.n_channels = DetectionHead.get_n_channels(n_classes)
        self.spec = DetectionHead.static_spec.copy()
        self.spec['cls'] = [DetectionHead.static_n_features + i for i in range(n_classes)]

        _grid_offsets, _grid_normalization = DetectionHead.compute_anchor_grid(grid_shape)
        self.grid_shape = grid_shape
        self.grid_offsets = Parameter(torch.tensor(_grid_offsets[..., [2, 1, 0]]).unsqueeze(0), requires_grad=False) # offsets as (x, y, d) format
        self.grid_normalization = Parameter(torch.tensor(_grid_normalization[..., [2, 1, 0]], dtype=torch.float32).unsqueeze(0), requires_grad=False)

        self.metrics = {}
        self.loss_weights = {'x': 1, 'y': 1, 'd': 2, 'dir': 20, 'dimensions': 3, 'centerness': 1, 'conf': 1, 'cls': 1, 'action_cls': 10}

    @staticmethod
    def get_n_channels(n_classes):
        return DetectionHead.static_n_features + n_classes

    def transform_data(self, _data):
        data = torch.empty_like(_data) # new tensor so torch won't complain about backprop through inplace operations
        spatials = [self.spec[c] for c in ['x', 'y', 'd']]

        data[..., spatials] = _data[..., spatials]
        data[..., self.spec['direction']] = tanh_nd(_data[..., self.spec['direction']], dim=-1) # normalized with 'euclidean tanh' (euclidean length capped at 1)
        data[..., self.spec['dimensions']] = torch.exp(torch.clamp(_data[..., self.spec['dimensions']], -2, 2))  # clip values to avoid very large / small gradients
        data[..., self.spec['centerness']] = torch.sigmoid(_data[..., self.spec['centerness']])
        data[..., self.spec['conf']] = torch.sigmoid(_data[..., self.spec['conf']])
        data[..., self.spec['action_cls']] = _data[..., self.spec['action_cls']]
        data[..., self.spec['cls']] = _data[..., self.spec['cls']] # no normalization for classification channels as it is not needed with CrossEntropyLoss
        return data

    def forward(self, data, targets=None, out_transform=True):
        features = self.transform_data(rearrange(data, 'tb c d h w -> tb d h w c'))

        loss = 0.
        if self.training: # compute loss if targets are available
            assert targets is not None
            trained_features = rearrange(features, '(t b) c d h w -> t b c d h w', t=self.n_tracking_frames)[-1]
            loss = self.compute_loss(trained_features, targets)

        if out_transform:
            # add grid offset to x, y, depth and normalized them
            features[..., :3] += self.grid_offsets
            features[..., :3] /= self.grid_normalization
            features[..., self.spec['action_cls']] = torch.softmax(features[..., self.spec['action_cls']], dim=-1)
            features[..., self.spec['cls']] = torch.softmax(features[..., self.spec['cls']], dim=-1) # must be done after `compute_loss` which operates on logits

        return {
            'features': features,
            'loss': loss
        }

    def compute_loss(self, predictions, targets_dict, obj_scale=1, noobj_scale=100):
        targets, obj_mask, noobj_mask, box_visibilities, box_affiliation, box_target_scale = \
            [targets_dict[key] for key in ['grid_targets', 'obj_mask', 'noobj_mask', 'box_visibilities', 'box_affiliation', 'box_target_scale']]
        box_weights = 0.5 + 0.5 * (1 - box_visibilities[box_affiliation[torch.where(obj_mask)]])
        box_weights = box_weights / torch.sum(box_weights)

        mse_loss = nn.MSELoss(reduction='none')
        bce_loss = nn.BCELoss(reduction='none')
        cls_loss = nn.CrossEntropyLoss(reduction='none')

        def apply_loss(measure, mask, cols, no_box_weights=False, classification=False, scaling=None, *args, **kwargs):
            if classification:
                target_tensor = targets[mask][:, self.target_spec[cols][0]].long()
            else:
                target_tensor = targets[mask][:, self.spec[cols]]

            loss_tensor = measure(predictions[mask][:, self.spec[cols]], target_tensor, *args, **kwargs)
            if len(loss_tensor.shape) > 1:
                loss_tensor = reduce(loss_tensor, 'b ... -> b', reduction='mean')
            if scaling is not None:
                loss_tensor *= torch.clamp(scaling[mask] ** 2, min=0.1, max=10)

            if no_box_weights:
                return loss_tensor.mean()
            else:
                loss_scalar = torch.dot(loss_tensor, box_weights)
                return loss_scalar

        # computing losses
        box_scale_factor = 5. / (box_target_scale + 1e-10)
        box_scaling = box_scale_factor[box_affiliation]

        loss_dict = {}
        loss_dict['x'] = apply_loss(mse_loss, obj_mask, 'x', scaling=box_scaling)
        loss_dict['y'] = apply_loss(mse_loss, obj_mask, 'y', scaling=box_scaling)
        loss_dict['d'] = apply_loss(mse_loss, obj_mask, 'd', scaling=box_scaling)
        loss_dict['dir'] = apply_loss(dot_product_loss, obj_mask, 'direction', with_acos=True, reduce=False)
        loss_dict['dimensions'] = apply_loss(mse_loss, obj_mask, 'dimensions')
        loss_dict['centerness'] = apply_loss(mse_loss, obj_mask, 'centerness')
        loss_dict['conf_obj'] = apply_loss(bce_loss, obj_mask, 'conf')
        loss_dict['conf_noobj'] = apply_loss(bce_loss, noobj_mask, 'conf', no_box_weights=True)
        loss_dict['action_cls'] = apply_loss(cls_loss, obj_mask, 'action_cls', classification=True)
        loss_dict['cls'] = apply_loss(cls_loss, obj_mask, 'cls', classification=True)

        loss_dict['conf'] = obj_scale * loss_dict['conf_obj'] + noobj_scale * loss_dict['conf_noobj']
        loss_dict['total'] = sum(weight * loss_dict[key] for key, weight in self.loss_weights.items())

        self.metrics = {key: value.item() for key, value in loss_dict.items()}
        self.add_metrics(predictions, targets, obj_mask, noobj_mask)

        return loss_dict['total']

    def add_metrics(self, predictions, targets, obj_mask, noobj_mask):
        # compute metrics
        iou_scores = torch.zeros(obj_mask.shape, device=obj_mask.device)
        iou_scores[obj_mask] = target_distance(predictions[obj_mask][:, :3], targets[obj_mask][:, :3])
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()

        class_mask = torch.zeros(obj_mask.shape, device=obj_mask.device)
        action_mask = torch.zeros(obj_mask.shape, device=obj_mask.device)
        if obj_mask.max().item() == True: # invalid operation if there are no objects
            prediction_classes = predictions[obj_mask][:, self.spec['cls']].argmax(-1)
            target_classes = targets[obj_mask][:, self.target_spec['cls'][0]].long()
            class_mask[obj_mask] = (prediction_classes == target_classes).float()

            prediction_classes = predictions[obj_mask][:, self.spec['action_cls']].argmax(-1)
            target_classes = targets[obj_mask][:, self.target_spec['action_cls'][0]].long()
            action_mask[obj_mask] = (prediction_classes == target_classes).float()

        pred_conf = predictions[..., self.spec['conf'][0]]
        conf50 = (pred_conf > 0.5).float()
        detected_mask = conf50 * class_mask * obj_mask.float()

        cls_acc = 100 * class_mask[obj_mask].mean()
        action_acc = 100 * action_mask[obj_mask].mean()
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        conf_obj = pred_conf[obj_mask].mean()
        conf_noobj = pred_conf[noobj_mask].mean()

        self.metrics.update({
            'cls_acc': cls_acc.item(),
            'action_acc': action_acc.item(),
            'recall50': recall50.item(),
            'recall75': recall75.item(),
            'precision': precision.item(),
            'conf_obj': conf_obj.item(),
            'conf_noobj': conf_noobj.item(),
            'grid_shape': self.grid_shape,
        })

    @staticmethod
    def compute_anchor_grid(anchor_grid_shape, as_list=False):
        grid = np.array(np.meshgrid(*[np.arange(n) + 0.5 for n in anchor_grid_shape], indexing='ij'))
        grid = rearrange(grid, 'c d y x -> d y x c').astype(np.float32)
        grid_normalization = np.asarray(anchor_grid_shape).reshape((1, 3))

        if as_list:
            grid = grid.reshape(-1, 3)
        return grid, grid_normalization

    @staticmethod
    def build_targets(batch, target_shape, assign_threshold=0.8, ignore_threshold=1.2):
        def generate_box_tensor(batch):
            center = lambda sample, target: sample.transform.apply(target.center().reshape(3, 1)).reshape(3)
            line = lambda sample, target, top: sample.transform.apply(np.array([
                target.center() + (1. if top else -1.) * np.array([0., 0., target.dimensions[2] / 2])
            ]).T).T[0]

            targets = [target for sample in batch for target in sample.targets]
            data = {
                'sample_idx': [i for i, sample in enumerate(batch) for _ in sample.targets],
                'box_idx': [i for i, _ in enumerate([target for sample in batch for target in sample.targets])],
                'label': [target.label for target in targets],
                'action': [target.action for target in targets],
                'center': [center(sample, target) for sample in batch for target in sample.targets],
                'direction': [target.direction for target in targets],
                'dimensions': [np.asarray(target.dimensions) / Box.mean_dimensions for target in targets],
                'point0': [line(sample, target, top=False) for sample in batch for target in sample.targets],
                'point1': [line(sample, target, top=True) for sample in batch for target in sample.targets],
            }
            return DataTable.from_dict(data)

        # compute centerness according to FCOS: Fully Convolutional One-Stage Object Detection
        def compute_centerness(box_mask, box_coordinates):
            centerness = np.zeros(box_mask.shape)
            center_min_coordinates = 1 - abs(box_coordinates[box_mask]) # min = 1-abs, max = 2 - min = 1 + abs
            if center_min_coordinates.size > 0: # avoids error in gmean
                centerness[box_mask] = gmean(np.clip(center_min_coordinates / (2 - center_min_coordinates), a_min=1e-5, a_max=1.), axis=-1)
            return centerness

        def create_indexer(mask, anchor_grid, sample_idxs): # restructure data from list to grid
            anchors_x, anchors_y, anchors_d = anchor_grid[np.where(mask)[1]].astype(int).T
            mask_indexer = np.where(mask)
            grid_indexer = (sample_idxs[mask_indexer[0]], anchors_d, anchors_y, anchors_x)
            return mask_indexer, grid_indexer

        # target shape is (batch, depth, y, x, n_features) -> grid shape in (x, y, depth) format
        anchor_grid_shape = (target_shape[3], target_shape[2], target_shape[1])
        anchor_grid, grid_normalization = DetectionHead.compute_anchor_grid(anchor_grid_shape, as_list=True) # shape (m, 3)
        normalized_anchor_grid = anchor_grid / grid_normalization

        # compute anchor information with respect to boxes
        boxes = [target for sample in batch for target in sample.targets]

        obj_mask = np.zeros(target_shape[:-1], dtype=bool)
        noobj_mask = np.ones(target_shape[:-1], dtype=bool)
        grid_targets = np.zeros(target_shape)
        grid_box_affiliation = -np.ones(target_shape[:-1], dtype=int)

        if len(boxes) > 0:
            box_coordinates = np.array([box.box_coordinates(normalized_anchor_grid.T, from_img_space=True).T for box in boxes]) # shape (n, m, 3)
            box_coordinates_norm = np.max(abs(box_coordinates), axis=-1) # max norm
            box_tensor = generate_box_tensor(batch)
            sample_idxs = box_tensor['sample_idx'][:, 0].astype(int)

            # create object mask
            assign_mask = box_coordinates_norm <= assign_threshold  # points that get targets assigned
            obj_mask_indexer, obj_grid_indexer = create_indexer(assign_mask, anchor_grid, sample_idxs)
            obj_mask[obj_grid_indexer] = True

            # create noobject mask
            not_ignore_mask = box_coordinates_norm <= ignore_threshold  # points that are close enough so not to be ignored
            noobj_mask_indexer, noobj_grid_indexer = create_indexer(not_ignore_mask, anchor_grid, sample_idxs)
            noobj_mask[noobj_grid_indexer] = False

            # create box mask (anchors within boxes)
            box_mask = box_coordinates_norm <= 1.
            box_mask_indexer, box_grid_indexer = create_indexer(box_mask, anchor_grid, sample_idxs)
            grid_box_affiliation[box_grid_indexer] = box_tensor['box_idx'][np.where(box_mask)[0], 0].astype(int)

            # create grid targets
            centers = box_tensor['center'] * np.array(anchor_grid_shape).reshape((1, 3))
            targets = np.expand_dims(centers, axis=1) - np.expand_dims(anchor_grid, axis=0)  # shape (n, m, 3)
            targets_extra = box_tensor[['direction', 'dimensions']][obj_mask_indexer[0]]
            targets_centerness_dummy = np.zeros((len(obj_mask_indexer[0]), 1)) # everything else is assigned with obj_grid instead of box_grid, so a placeholder is needed here
            targets_conf = np.ones((len(obj_mask_indexer[0]), 1))
            targets_action = box_tensor['action'][obj_mask_indexer[0]]
            targets_cls = box_tensor['label'][obj_mask_indexer[0]]

            grid_targets[obj_grid_indexer] = np.concatenate([targets[assign_mask], targets_extra, targets_centerness_dummy, targets_conf, targets_action, targets_cls], axis=1)
            grid_targets[box_grid_indexer][:, DetectionHead.static_spec['centerness'][0]] = compute_centerness(box_mask, box_coordinates)[box_mask] # now assign centerness with box_grid_indexer

        box_target_scale = [np.linalg.norm(grid_targets[grid_box_affiliation == i], axis=-1) for i in range(len(boxes))]
        return {
            'grid_targets': torch.tensor(grid_targets, dtype=torch.float32),
            'obj_mask': torch.tensor(obj_mask),
            'noobj_mask': torch.tensor(noobj_mask),
            'box_affiliation': torch.tensor(grid_box_affiliation),
            'box_target_scale': torch.tensor([0.1 if len(scale) == 0 else scale.max() for scale in box_target_scale], dtype=torch.float32)
        }
