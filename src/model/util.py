import math

import torch

from dataset.box import Box
from util.utility import static_vars
try:
    from iou_loss.oriented_iou_loss import cal_giou_3d, cal_diou_3d
except ImportError:
    print('oriented_iou_loss not installed, only dummy box loss available')
    cal_giou_3d = lambda predictions, targets: torch.zeros(len(predictions), device=predictions.data.device)
    cal_diou_3d = lambda predictions, targets: torch.zeros(len(predictions), device=predictions.data.device)

def tanh_nd(data, dim=-1):
    # direction - normalized with 'euclidean tanh' (euclidean length capped at 1)
    dir_norm = torch.norm(data, dim=dim)
    dir_modifier = torch.tanh(dir_norm) / (dir_norm + 1e-10) # constant for numerical stability
    scaled_data = data * dir_modifier.unsqueeze(dim=dim)
    return scaled_data

def dot_product_loss(a, b, with_acos=False, reduce=True):
    if with_acos:
        # enclosing_angle = torch.acos(dot_product) # gradient of acos not numerically stable
        enclosing_angle = (torch.atan2(b[:, 1], b[:, 0]) - torch.atan2(a[:, 1], a[:, 0])) % (2 * math.pi)
        enclosing_angle = torch.min(enclosing_angle, 2 * math.pi - enclosing_angle) # % to convert to [-pi, pi]
        loss = 2 * (enclosing_angle / math.pi) ** 2 # 2 * (angle / pi) ** 2 this very closely matches (1 - cos(x)) / 2 for small x
    else:
        dot_product = torch.bmm(a.unsqueeze(dim=1), b.unsqueeze(dim=2))
        loss = (1 - dot_product) / 2

    if reduce:
        loss = torch.mean(loss)
    return loss

@static_vars(scaling_factor=None)
def box_loss(predictions, targets, type='giou', with_direction=True):
    device = predictions.data.device
    if box_loss.scaling_factor is None:
        box_loss.scaling_factor = torch.tensor(Box.mean_dimensions / Box.world_normalization, dtype=torch.float32, device=device)

    def to_boxes(tensor):
        if with_direction:
            angles = torch.atan2(tensor['direction'][:, 1], tensor['direction'][:, 0]).view(-1, 1)
        else:
            angles = torch.zeros_like(tensor['direction'][:, [1]])
        center = tensor['center']
        dimensions = tensor['dimensions'] * box_loss.scaling_factor
        boxes = torch.cat([center, dimensions, angles], dim=1).unsqueeze(0)
        return boxes

    metric = cal_giou_3d if type == 'giou' else cal_diou_3d
    if predictions.data.shape[0] == 0:
        return torch.zeros(len(predictions), device=device)
    else:
        return metric(to_boxes(predictions), to_boxes(targets))[0][0]