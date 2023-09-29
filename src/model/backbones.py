from collections import OrderedDict

from einops import reduce
from einops.layers.torch import Rearrange
from torch import nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork

def adapt_first_conv(conv, in_channels, kernel_size=3, padding=1, stride=1, bias=False):
    out_channels = conv.out_channels
    new_first_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
    new_first_layer.weight.data[:, :3] = conv.weight.data
    return new_first_layer

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=(1, 3, 3), stride=1, groups=1, norm_layer=None):
        padding = tuple([(k-1) // 2 for k in kernel_size])

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        super(ConvBNReLU, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class To3DBlock(nn.Module):
    def __init__(self, block, out_channels, depth_channels):
        super().__init__()
        self.block = nn.Sequential(
            block,
            Rearrange('tb (c d) h w -> tb c d h w', d=depth_channels),
            ConvBNReLU(out_channels, out_channels)
        )

    def forward(self, x):
        return {
            'features': self.block(x),
            'context': reduce(x, 'tb cd h w -> tb cd', reduction='mean')
        }


class BackboneWithNeck(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, depth_channels, with_fpn=True):
        super().__init__()

        if with_fpn:
            self.return_layers = return_layers
        else:
            self.return_layers = OrderedDict([list(return_layers.items())[-1]]) # only last entry
            in_channels_list = [in_channels_list[-1]]

        self.body = create_feature_extractor(backbone, return_nodes=self.return_layers)
        self.neck = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=(out_channels * depth_channels))
        self.neck.layer_blocks = nn.ModuleList([
            To3DBlock(block, out_channels=out_channels, depth_channels=depth_channels)
            for block in self.neck.layer_blocks
        ])

        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.neck(x)
        return x

def get_backbone(backbone_name, in_channels, out_channels, depth_channels, with_fpn=True, **kwargs):
    if backbone_name == 'mobilenet_v2':
        from torchvision.models import mobilenet_v2
        kwargs['width_mult'] = kwargs.get('width_mult') or 1.
        pretrained = (True if kwargs['width_mult'] == 1. else False) and ('inverted_residual_setting' not in kwargs)
        backbone = mobilenet_v2(pretrained=pretrained, **kwargs).features
        if in_channels != 3:
            backbone[0][0] = adapt_first_conv(backbone[0][0], in_channels, stride=2)

        return_layers = OrderedDict([('14.conv.0', 'out1'), ('18', 'out0')])
        in_channels_list = [int(c * kwargs['width_mult']) for c in [576, 1280]]
    elif backbone_name == 'mobilenet_v3':
        from torchvision.models import mobilenet_v3_large
        backbone = mobilenet_v3_large(reduced_tail=False, **kwargs).features
        if in_channels != 3:
            first_layer_out_channels = backbone[0][0].out_channels
            backbone[0][0] = nn.Conv2d(in_channels, first_layer_out_channels, kernel_size=3, padding=1, stride=2, bias=False)

        return_layers = OrderedDict([('13.block.0', 'out1'), ('16', 'out0')])
        in_channels_list = [672, 960]
    elif backbone_name == 'mobilenet_v3_small':
        from torchvision.models import mobilenet_v3_small
        backbone = mobilenet_v3_small(**kwargs).features
        if in_channels != 3:
            backbone[0][0] = adapt_first_conv(backbone[0][0], in_channels, stride=2)

        return_layers = OrderedDict([('9.block.0', 'out1'), ('12', 'out0')])
        in_channels_list = [288, 576]
    elif backbone_name == 'mnasnet':
        from torchvision.models import mnasnet1_0
        backbone = mnasnet1_0(**kwargs).layers
        if in_channels != 3:
            backbone.layers[0] = adapt_first_conv(backbone.layers[0], in_channels, stride=2)

        return_layers = OrderedDict([('12.0.layers.2', 'out1'), ('16', 'out0')])
        in_channels_list = [576, 1280]
    elif backbone_name.startswith('resnet'):
        from torchvision.models import resnet
        backbone = resnet.__dict__[backbone_name](pretrained=True)
        if in_channels != 3:
            backbone.conv1 = adapt_first_conv(backbone.conv1, in_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        return_layers = {'layer2': 'out2', 'layer3': 'out1', 'layer4': 'out0'}
        in_channels_list = [512, 1024, 2048]
    elif backbone_name == 'efficientnet_b7':
        from torchvision.models import efficientnet_b7
        backbone = efficientnet_b7(pretrained=True).features
        if in_channels != 3:
            backbone.conv1 = adapt_first_conv(backbone.conv1, in_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        return_layers = {'4.0.block.0': 'out2', '6.0.block.0': 'out1', '8': 'out0'}
        in_channels_list = [480, 1344, 2560]
    else:
        raise ValueError('backbone %s is unknown or not implemented' % backbone_name)

    backbone_with_neck = BackboneWithNeck(backbone, return_layers, in_channels_list, out_channels, depth_channels, with_fpn=with_fpn)
    return backbone_with_neck