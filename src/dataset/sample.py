import copy
import pickle

import numpy as np
import torch

from dataset.box import Box, location_iou
from util.utility import DataTable

class CameraTransform:
    def __init__(self, matrices, img_size):
        self.matrices = matrices
        self.img_size = img_size

    def compute(self, inverse=False, normalize=True, augment=True, override_dict=None):
        matrices = self.matrices
        if override_dict is not None:
            matrices = copy.deepcopy(matrices)
            matrices.update(override_dict)

        camera_matrix = matrices['intrinsic'] @ matrices['extrinsic']
        image_matrix = np.eye(4)
        if augment:
            image_matrix = matrices['augmentation'] @ image_matrix
        if normalize:
            depth_min_max = (matrices['max_depth'] - matrices['min_depth'])
            normalization = np.diag([1. / self.img_size[1], 1. / self.img_size[0], 1. / depth_min_max, 1.])
            normalization[2, 3] = -matrices['min_depth'] / depth_min_max
            image_matrix = normalization @ image_matrix

        transform = {
            'is_inverse': inverse,
            'camera_matrix': camera_matrix if not inverse else np.linalg.inv(camera_matrix),
            'image_matrix': image_matrix if not inverse else np.linalg.inv(image_matrix)
        }
        return transform

    def apply(self, points, matrices=None, **kwargs):
        if matrices is None:
            matrices = self.compute(**kwargs)

        return self.static_apply(points, matrices)

    @staticmethod
    def static_apply(points, matrices): # differentiable and traceable
        if torch.is_tensor(points):
            points = torch.cat([points, torch.ones((1, points.shape[1]), device=points.device, dtype=points.dtype)], dim=0)
        else:
            points = np.concatenate([points, np.ones((1, points.shape[1]), dtype=points.dtype)], axis=0) # homogeneous coordinates

        if not matrices['is_inverse']:
            points = matrices['camera_matrix'] @ points
            points[[0, 1]] /= (points[[2]] + 1e-10)
            points = matrices['image_matrix'] @ points
        else:
            points = matrices['image_matrix'] @ points
            points[[0, 1]] *= points[[2]]
            points = matrices['camera_matrix'] @ points
        return points[:3]

class Sample:
    def __init__(self, idx, transform, recording_id=None, path=None):
        self.idx = idx
        self.transform = transform
        self.recording_id = recording_id
        self.path = path

        # image data
        self.image = None
        self.original_image = None

        # boxes
        self.targets = []
        self.pre_nms_predictions = None # set in postprocessing
        self.predictions = [] # set in non_maximum_suppression
        self.interactions = {} # set in interaction module after non_maximum_suppression

        # for development or debugging. store intermediate activations
        self.activations = None

    def serialize(self, path):
        data = copy.deepcopy(self)
        data.image = None
        data.original_image = None

        pickle.dump(data, open(path, 'wb'))

    @staticmethod
    def _fuse_detections(detections):
        weights = detections['conf'] * detections['centerness']
        weights /= (np.sum(weights) + 1e-10)
        fused_detection = DataTable(np.sum(detections.data * weights, axis=0)[np.newaxis, :], detections.spec)

        box = Box(
            location=fused_detection[['x', 'y', 'd']][0],
            direction=fused_detection['direction'][0],
            dimensions=fused_detection['dimensions'][0],
            label=np.bincount(detections['label'][:, 0].astype(int)).argmax(),
            action=np.bincount(detections['action'][:, 0].astype(int)).argmax(),
            confidence=detections['conf'][0].item() # non averaged confidence
        )
        return box

    @staticmethod
    def non_max_suppression(predictions, conf_thres, iou_thres, n_max_predictions=16):
        # Scale boxes to their full un-normalized size
        predictions.data[:, predictions.spec['dimensions']] *= Box.mean_dimensions

        # Filter out confidence scores below threshold
        predictions = predictions[(predictions['conf'] >= conf_thres)[:, 0]]
        if predictions.data.shape[0] == 0:
            return []

        # score predictions using confidence times class confidence and sort by it
        score = predictions['conf'][:, 0] * predictions['cls'].max(axis=1)
        predictions = predictions[(-score).argsort()]

        # get class with highest confidence
        class_preds = np.argmax(predictions['cls'], axis=1)
        if predictions['cls'].shape[1] > 1:
            class_confs = predictions['cls'][(np.arange(len(class_preds)), class_preds)]
        else:
            # in case of single class 'cls' is meaningless (always =1 after softmax)
            class_confs = predictions['conf'][:, 0]
        action_preds = np.argmax(predictions['action_cls'], axis=1)

        detections = predictions.select(['x', 'y', 'd', 'direction', 'dimensions', 'centerness'])
        detections.append(DataTable.from_dict({'label': class_preds, 'action': action_preds, 'conf': class_confs}))

        nms_predictions = []
        # perform non-maximum suppression
        while detections.data.shape[0] and len(nms_predictions) < n_max_predictions:
            large_overlap = location_iou(detections[['x', 'y']][0], detections[['x', 'y']]) > iou_thres
            invalid = large_overlap

            nms_predictions.append(Sample._fuse_detections(detections[invalid]))
            detections = detections[~invalid]

        return nms_predictions

    def postprocess(self, prediction, transform_matrices, nms_kwargs=None):
        self.predictions = [] # just in case this is called multiple times
        world_points = self.transform.apply(prediction.data[:, :3].T, matrices=transform_matrices).T
        prediction.data[:, :3] = world_points
        self.pre_nms_predictions = prediction
        if nms_kwargs is not None:
            nms_predictions = self.non_max_suppression(self.pre_nms_predictions.to_numpy(), **nms_kwargs)
            [self.add_prediction(pred) for pred in nms_predictions]

    def add_target(self, box):
        box.transform = self.transform
        self.targets.append(box)

    def add_prediction(self, box):
        box.transform = self.transform
        self.predictions.append(box)

    def pointcloud(self, apply_extrinsic=True, in_image_space=False, subsample=1):
        # get pointcloud without extrinsic applied
        depth_image = self.original_image['depth']
        img_shape = depth_image.shape
        depth_image = depth_image[::subsample, ::subsample]

        xy_img_points = np.meshgrid(np.arange(img_shape[1], step=subsample), np.arange(img_shape[0], step=subsample))
        points = np.concatenate([*xy_img_points, depth_image]).reshape(3, -1)

        if not in_image_space:
            override_dict = {} if apply_extrinsic else {'extrinsic': np.eye(4)}
            points = self.transform.apply(points, inverse=True, normalize=False, augment=False, override_dict=override_dict)
        return points
