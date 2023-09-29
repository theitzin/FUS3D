from enum import Enum

import torch
import torch.nn as nn
from einops import rearrange

from model.backbones import get_backbone
from model.detection_head import DetectionHead
from model.orientation import OrientationModule
from util.utility import DataTable

class DetectionOutputType(Enum):
    Raw = 0
    Features = 1
    Detections = 2

class DetectionModel(nn.Module):
    def __init__(self, backbone, in_shape, depth_channels, head_kwargs, n_tracking_frames, head_feature_channels=None, backbone_kwargs=None):
        super().__init__()
        self.in_shape = in_shape # (n_ch, height, width)
        self.n_tracking_frames = n_tracking_frames
        self.head_channels = DetectionHead.get_n_channels(head_kwargs['n_classes'])
        if head_feature_channels is None:
            head_feature_channels = self.head_channels * 4

        self.features = get_backbone(backbone_name=backbone, in_channels=in_shape[0], out_channels=head_feature_channels,
                                     depth_channels=depth_channels, with_fpn=True, **backbone_kwargs)
        n_heads = len(self.features.return_layers)

        self.head_projections = nn.ModuleList([
            nn.Conv3d(head_feature_channels, self.head_channels, kernel_size=1, padding=0, bias=False)
            for _ in range(n_heads)
        ])

        self.head_shape_info = self.compute_head_shape_info(head_kwargs['n_classes'])

        self.heads = nn.ModuleList([
            DetectionHead(grid_shape=self.head_shape_info['tensor_shapes'][i][1:], n_tracking_frames=n_tracking_frames, **head_kwargs) for i in range(n_heads)
        ])
        self.orientation_module = OrientationModule(head_feature_channels, n_tracking_frames=n_tracking_frames)

    def compute_head_shape_info(self, n_classes):
        # this has to be determined to ensure static collate function can be defined (-> fast data loading)
        device = next(self.parameters()).device
        dummy_feature_output = self.features(torch.zeros((self.n_tracking_frames, *self.in_shape), device=device))
        dummy_head_inputs = [
            self.head_projections[i](dummy_feature_output[key]['features'])
            for i, key in enumerate(dummy_feature_output)
        ]

        shape_info = {
            'tensor_shapes': [tuple(tensor.shape[1:]) for tensor in dummy_head_inputs],
            'n_channels': self.head_channels,
            'n_classes': n_classes,
        }
        return shape_info

    def forward(self, input, batch=None, output_type=DetectionOutputType.Detections, single_head=False):
        # backbone
        x = self.features(input)
        x = [x[key] for key in x] # from OrderedDict to List

        # detection heads
        head_data_list = []
        losses = {}
        for i, (y, head_projection, head) in enumerate(zip(x, self.head_projections, self.heads)):
            if single_head and i < len(x) - 1: # in inference only
                continue

            z = head_projection(y['features'])

            head_targets = batch.targets['detection_heads'][i] if batch is not None else None
            head_result = head(z, head_targets, out_transform=(output_type != DetectionOutputType.Raw))
            losses['head%d' % i] = head_result['loss']

            if output_type == DetectionOutputType.Raw: # this is used as teacher targets for knowledge distillation
                head_features = head_result['features']
            else:
                head_features = rearrange(head_result['features'], '(t b) d h w c -> t b (d h w) c', t=self.n_tracking_frames)

            if output_type == DetectionOutputType.Features:
                head_raw_features = rearrange(y['features'], '(t b) c d h w -> t b (d h w) c', t=self.n_tracking_frames)
                head_features = {
                    'features': torch.cat([head_features, head_raw_features], dim=-1),
                    'context': rearrange(y['context'], '(t b) c -> t b c', t=self.n_tracking_frames)
                }

            head_data_list.append(head_features)

        # orientation module
        orientation_targets = batch.targets['orientation_module'] if batch is not None else None
        orientation_data, orientation_loss = self.orientation_module(x[-1]['features'], orientation_targets)
        losses.update({'orientation_module.%s' % key: orientation_loss[key] for key in orientation_loss})

        output = {
            'orientation': orientation_data
        }

        if output_type == DetectionOutputType.Features:
            output['features'] = head_data_list
        elif output_type == DetectionOutputType.Raw:
            output['raw_features'] = head_data_list
        else:
            features = torch.cat([data[-1] for data in head_data_list], dim=1)
            output['detections'] = [DataTable(f, spec=self.heads[0].spec) for f in features] # to DataTable

        if self.training:
            return output, losses
        else:
            return output