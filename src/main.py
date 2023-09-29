import torch
import torch.nn as nn
from einops import rearrange

from model.detection_model import DetectionModel, DetectionOutputType
from model.tracking_model import TrackModel
from dataset.box import Box

class CellSampler(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.max_features = 128

    def forward(self, features):
        device = features.device

        conf_mask = (features[..., 9] > 0.7)
        selected_features = torch.zeros((features.shape[0], self.max_features, features.shape[2]), device=device, dtype=features.dtype)
        selected_mask = torch.zeros((features.shape[0], self.max_features), dtype=torch.bool, device=device)

        for idx, (step_features, step_conf_mask) in enumerate(zip(features, conf_mask)):
            n_features = torch.sum(step_conf_mask)
            if n_features > self.max_features:
                print('warning: max features exceeded (%d)' % n_features.item())
                n_features = torch.clamp(torch.sum(step_conf_mask), max=self.max_features)
            selected_features[idx, :n_features] = step_features[step_conf_mask][:n_features]
            selected_mask[idx, :n_features] = True

        if self.training:
            mask = torch.rand(selected_features.shape[:2], device=device) >= self.dropout
        else:
            mask = torch.ones(selected_features.shape[:2], dtype=torch.bool, device=device)

        return {
            'features': selected_features,
            'cell_mask': mask & selected_mask
        }

    @staticmethod
    def compose_sampled_batch(samples):
        return {
            'features': torch.stack([s['features'] for s in samples], dim=1),
            'cell_mask': torch.stack([s['cell_mask'] for s in samples], dim=1)
        }


class TokenGenerator:
    def __init__(self, dataloader, detection_model, sampler, device):
        self.dataloader = dataloader
        self.detection_model = detection_model
        self.sampler = sampler
        self.device = device

        self.world_normalization = torch.tensor(Box.world_normalization, device=self.device, dtype=torch.float32)

    def __iter__(self):
        for batch in self.dataloader:
            batch = batch.to(self.device)

            if self.detection_model.training:
                outputs, loss_dict = self.detection_model(batch.images, batch, output_type=DetectionOutputType.Features, single_head=False)
                self.loss = sum([value for _, value in loss_dict.items()])
            else:
                outputs = self.detection_model(batch.images, output_type=DetectionOutputType.Features, single_head=True)
                self.loss = None
            feature_outputs = rearrange(outputs['features'][-1]['features'], 't b dhw c -> b (t dhw) c')

            features = []
            sampled_features = []
            for i, (output, sample) in enumerate(zip(feature_outputs, batch)):
                # transformation from image space to world space
                world_points = sample.transform.apply(output[:, :3].T, matrices=batch.transform_matrices[i]).T / self.world_normalization
                output = torch.cat([world_points, output[:, 3:]], dim=-1)
                output = rearrange(output, '(t dhw) c -> t dhw c', t=self.detection_model.n_tracking_frames)

                sample.activations = {'features': output}
                features.append(sample.activations['features'])
                sampled_features.append(self.sampler(sample.activations['features']))

            batch_features = torch.stack(features, dim=1)
            batch_sampled_features = CellSampler.compose_sampled_batch(sampled_features)
            batch_sampled_features['context'] = outputs['features'][-1]['context']

            sample_ids = ['%s_%d' % (s.recording_id, s.idx) for s in batch.samples]
            data = (batch_features, batch_sampled_features, batch.targets, sample_ids)
            yield data

def get_detection_model(img_size=320):
    return DetectionModel(
        backbone='mobilenet_v2',
        in_shape=(3, img_size, img_size),
        depth_channels=24,
        n_tracking_frames=6,
        head_feature_channels=64,
        head_kwargs={'n_classes': 3},
        backbone_kwargs=None
    )

def get_tracking_model(token_length=57):
    return TrackModel(token_length=token_length)

def main():
    pass