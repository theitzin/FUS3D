import cv2
import numpy as np
try:
    from shapely.geometry import Polygon, Point
except ImportError:
    print('shapely not installed, some Box IoU functions not available')

def rot_mat_2d(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

def location_iou(target, locs, radius=20):
    distances = np.linalg.norm(locs - np.expand_dims(target, axis=0), axis=1)
    ious = np.maximum(0, -distances / (2 * radius) + 1)
    return ious

class Box:
    mean_dimensions = np.array([76., 47., 80.])  # divide by mean bbox size
    world_normalization = np.array([200., 200., 100.])

    def __init__(self, location, direction, dimensions, label, label_uncertainty=None, action=-100, action_uncertainty=None, sequence_id=None, confidence=1., track_id=0, transform=None):
        self.location = np.asarray(location)
        self.velocity = np.zeros(2) # only used in tracking
        self.direction = np.asarray(direction) / (np.linalg.norm(direction) + 1e-10)
        self.dimensions = np.asarray(dimensions) # length, width, height
        self.label = label
        self.label_uncertainty = label_uncertainty # only set for targets
        self.action = action
        self.action_uncertainty = action_uncertainty
        self.sequence_id = sequence_id
        self.confidence = confidence
        self.transform = transform
        self.track_id = track_id # only used in inference for tracking
        self.visibility = 1.

    def center(self):
        return self.location

    def region(self):
        return self.dimensions[:2]

    def height(self):
        return self.dimensions[2]

    def area(self): # birds eye view area
        return self.dimensions[0] * self.dimensions[1]

    def radius(self):
        return np.max(self.dimensions[:2]) / 2

    def volume(self):
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

    def _base_polygon(self):
        angle = np.arctan2(self.direction[1], self.direction[0])
        region = self.dimensions[:2]
        corners = [(region[1] / 2 * dir_x, region[0] / 2 * dir_y) for dir_x, dir_y in [(1, 1), (-1, 1), (-1, -1), (1, -1)]]

        polygon = (rot_mat_2d(angle) @ np.array(corners).T).T + self.location[np.newaxis, :2]
        return Polygon(polygon)

    def _intersection_area(self, other):
        return self._base_polygon().intersection(other._base_polygon()).area

    def _intersection_circle(self, other):
        c_self, c_other = Point(self.location[:2]).buffer(self.radius()), Point(other.location[:2]).buffer(other.radius())
        return c_self.intersection(c_other).area

    def _intersection_height(self, other):
        inter_top = min(self.center()[2] + self.dimensions[2]/2, other.center()[2] + other.dimensions[2]/2)
        inter_bottom = max(self.center()[2] - self.dimensions[2]/2, other.center()[2] - other.dimensions[2]/2)
        return max(0, inter_top - inter_bottom)

    def _intersection_volume(self, other):
        return self._intersection_area(other) * self._intersection_height(other)

    def iou_loc(self, other):
        return location_iou(self.location[:2], other.location[np.newaxis, :2])[0]

    def iou_loc3d(self, other):
        return self.iou_loc(other) * self._intersection_height(other)

    def iou_adaloc(self, other, min_radius=None): # other = target
        radius = max(min_radius, self.radius()) if min_radius is not None else self.radius()
        other_radius = max(min_radius, other.radius()) if min_radius is not None else other.radius()
        area_self, area_other = np.pi * radius**2, np.pi * other_radius**2
        area_inter = self._intersection_circle(other)
        iou_circle = area_inter / (area_self + area_other - area_inter)
        return iou_circle

    def iou_adaloc3d(self, other): # other = target
        return self.iou_adaloc(other) * self._intersection_height(other)

    def iou_bev(self, other):
        inter_area = self._intersection_area(other)
        return inter_area / (self.area() + other.area() - inter_area)

    def iou_3d(self, other):
        inter_volume = self._intersection_volume(other)
        return inter_volume / (self.volume() + other.volume() - inter_volume + 1e-10)

    def heading_similarity(self, other):
        return (1 + np.dot(self.direction, other.direction)) / 2

    def box_coordinates(self, points, from_img_space=True, to_img_space=False, **transform_kwargs):
        # transforms points to "box coordinates". box center is (0, 0, 0), box hull has coordinates +- 1 (|p|_\infty = 1)
        if from_img_space:
            transform_kwargs['inverse'] = True
            transform_kwargs['augment'] = True
            # if (not from_img_space) and transform_kwargs.get('override_dict') is None:
            #     transform_kwargs['override_dict'] = {'intrinsic': np.eye(4)}

            # to points in world space
            points = self.transform.apply(points, **transform_kwargs)

        # box basis
        orthogonal_box_basis = np.array([
            [self.direction[0], self.direction[1], 0.],
            [-self.direction[1], self.direction[0], 0.],
            [0., 0., 1.]
        ]).T
        box_center = self.center().reshape((3, 1))

        # compute inverse scaled box basis
        ## scaled_box_basis = orthogonal_box_basis * (self.dimensions[np.newaxis, :] / 2)
        ## inv_scaled_box_basis = np.linalg.inv(scaled_box_basis)
        transform = (1 / (self.dimensions[:, np.newaxis] / 2 + 1e-10)) * orthogonal_box_basis.T # numerically stable
        if to_img_space:
            transform = orthogonal_box_basis @ transform # transforms img space to img space but coordinates are scaled by bbox dimensions
        coordinates = transform @ (points - box_center)
        return coordinates

    def compute_points_3d(self, as_world_coordinates=False):
        corners = np.stack([
            np.array([1, 1, -1, -1, 1, 1, -1, -1]) * self.dimensions[0] / 2,
            np.array([1, -1, -1, 1, 1, -1, -1, 1]) * self.dimensions[1] / 2,
            np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * self.dimensions[2] / 2
        ], axis=0)

        orientation = np.array([
            [0., self.dimensions[0]],
            [0., 0.],
            [-self.dimensions[2] * 0.45] * 2 # a bit off the ground
        ])

        R = np.eye(3)
        R[:2, 0] = self.direction
        R[:2, 1] = [-self.direction[1], self.direction[0]]

        location = self.location[:, np.newaxis]
        corners_3d = R @ corners + location
        orientation_3d = R @ orientation + location
        visible = not any(corners_3d[2, :] < 0.) # not behind camera

        if as_world_coordinates:
            return {
                'corners': corners_3d,
                'orientation': orientation_3d,
                'visible': visible
            }
        else:
            return {
                'corners': self.transform.apply(corners_3d, normalize=False, augment=False),
                'orientation': self.transform.apply(orientation_3d, normalize=False, augment=False),
                'visible': visible
            }


    def compute_points_2d(self):
        points_3d = self.compute_points_3d(as_world_coordinates=False)['corners']
        hull = {
            'xmin': np.min(points_3d[0]),
            'xmax': np.max(points_3d[0]),
            'ymin': np.min(points_3d[1]),
            'ymax': np.max(points_3d[1])
        }
        return hull

    def draw_3d(self, img_data, color=(255, 127, 0), orientation_color=(255, 0, 0), orientation_only=False):
        # img_data has to be tuple of (depth_img, color_img)
        box_3d = self.compute_points_3d()

        face_idxs = [
            [0, 1, 5, 4],  # front face
            [1, 2, 6, 5],  # left face
            [2, 3, 7, 6],  # back face
            [3, 0, 4, 7]  # right
        ]

        corners, orientation = box_3d['corners'].T, box_3d['orientation'].T
        img = img_data[1]
        # color = tuple(map(lambda c: int(c * 255), color))
        corners, orientation = corners.astype(np.int32), orientation.astype(np.int32)
        if not orientation_only:
            for idxs in face_idxs:
                for i, j in zip(idxs[:-1], idxs[1:]):
                    img = cv2.line(img, tuple(corners[i, :2]), tuple(corners[j, :2]), color=color, thickness=2)
        img = cv2.line(img, tuple(orientation[0, :2]), tuple(orientation[1, :2]), color=orientation_color, thickness=2)

    def draw_box_2d(self, ax):
        box_2d = self.compute_points_2d()
        pts_x = [box_2d['xmin'], box_2d['xmax'], box_2d['xmax'], box_2d['xmin'], box_2d['xmin']]
        pts_y = [box_2d['ymin'], box_2d['ymin'], box_2d['ymax'], box_2d['ymax'], box_2d['ymin']]
        ax.plot(pts_x, pts_y, c='b')

