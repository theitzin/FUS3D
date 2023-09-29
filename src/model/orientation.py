from enum import IntEnum

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Reduce

from model.util import tanh_nd, dot_product_loss

def rotation_from_vector_target(origin, target):
    # returns a rotation matrix rotating unit vector 'origin' to target vector 'target'
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677#897677
    origin, target = np.array(origin) / np.linalg.norm(origin), np.array(target) / np.linalg.norm(target) # normalize
    cross_product_matrix = lambda v: np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) # cross product matrix
    v_cross = cross_product_matrix(np.cross(origin, target))
    rotation = np.eye(3) + v_cross + v_cross @ v_cross * 1 / (1 + np.dot(origin, target))
    return rotation

def build_extrinsic(orientation, height):
    inv_extrinsic = np.eye(4)
    inv_extrinsic[:3, :3] = rotation_from_vector_target(orientation, [0., 0., 1.])
    inv_extrinsic[:3, 3] = [0., 0., height]
    extrinsic = np.linalg.inv(inv_extrinsic)
    return extrinsic

class OrientationOutputType(IntEnum):
    Raw = 0
    HeightDirection = 1
    Extrinsic = 2

class OrientationModule(nn.Module):
    def __init__(self, in_channels, n_tracking_frames):
        super().__init__()

        self.layers = nn.Sequential(
            Reduce('(b t) c d h w -> b c', reduction='mean', t=n_tracking_frames),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )

    @staticmethod
    def build_targets(batch):
        # goal: predict upwards axis in image coordinates, length represents sensor height in image space units
        orientations, heights = np.zeros((len(batch), 3)), np.zeros(len(batch))
        for i, sample in enumerate(batch):
            world_points = sample.transform.apply(np.array([
                [0.5, 0.5, 0.5], # center (should be better conditioned than origin where projection to image plane with x / depth probably causes issues)
                [0.5, 0.5, 0.0] # sensor location
            ]).T, inverse=True).T
            sensor_height = world_points[1, 2]
            image_space_endpoints = sample.transform.apply(np.array([
                world_points[0], # image space center
                world_points[0] + [0., 0., sensor_height], # image space center offset with sensor height
            ]).T).T
            offset = image_space_endpoints[1] - image_space_endpoints[0]
            orientations[i] = offset / (np.linalg.norm(offset) + 1e-10)
            heights[i] = np.linalg.norm(offset)

        targets = {
            'orientation': torch.tensor(orientations, dtype=torch.float32),
            'height': torch.tensor(heights, dtype=torch.float32)
        }
        return targets

    @staticmethod
    def postprocess(orientation, height, sample):
        # mainly converts orientation and height prediction in image space to world space
        offset = (orientation * height).detach().cpu().numpy()
        # transform to world space but without extrinsic
        world_points = sample.transform.apply(np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5] + offset,
        ]).T, inverse=True, override_dict={'extrinsic': np.eye(4)}).T
        orientation = world_points[1] - world_points[0]
        height = np.linalg.norm(orientation)
        return orientation, height

    @staticmethod
    def as_format(output, batch, format):
        if format >= OrientationOutputType.HeightDirection:
            output = [OrientationModule.postprocess(o, h, s) for o, h, s in
                      zip(output['orientation'], output['height'], batch)]
        if format == OrientationOutputType.Extrinsic:
            output = [build_extrinsic(*out) for out in output]
        return output

    def forward(self, x, targets=None):
        x = self.layers(x)
        output = {
            'orientation': tanh_nd(x[:, :3]),
            'height': x[:, 3]
        }

        if self.training:
            assert targets is not None
            loss = {
                'orientation': dot_product_loss(output['orientation'], targets['orientation']),
                'height': torch.mean((output['height'] - targets['height'])**2)
            }
            return output, loss
        else:
            return output, {}

