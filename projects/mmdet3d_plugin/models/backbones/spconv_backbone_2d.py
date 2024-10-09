from functools import partial

import torch.nn as nn
import numpy as np

import spconv
if float(spconv.__version__[2:]) >= 2.2:
    spconv.constants.SPCONV_USE_DIRECT_TABLE = False
try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

from projects.mmdet3d_plugin.models.utils import replace_feature
from mmdet.models import BACKBONES

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_fn=None):
    m = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

class SparseBasicBlockV(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlockV, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv0 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn0 = norm_fn(planes)
        self.conv1 = spconv.SubMConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.conv0(x)
        x = replace_feature(x, self.bn0(x.features))
        x = replace_feature(x, self.relu(x.features))
        
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None):
        super(BasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


@BACKBONES.register_module()
class PillarRes18BackBone8x(nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = np.array(grid_size)[[1,0]]
        
        block = post_act_block
        dense_block = post_act_block_dense
        
        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )
        
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            BasicBlock(256, 256, norm_fn=norm_fn),
            BasicBlock(256, 256, norm_fn=norm_fn),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }

    def forward(self, pillar_features, pillar_coords, batch_size):
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        spconv_backbone_outs = dict()
        
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        spconv_backbone_outs.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        spconv_backbone_outs.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })
        
        return spconv_backbone_outs


@BACKBONES.register_module()
class PillarRes18BackBone8x2(nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = np.array(grid_size)[[1,0]]
        
        block = post_act_block
        dense_block = post_act_block_dense
        
        self.conv1 = spconv.SparseSequential(
            SparseBasicBlockV(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlockV(32, 32, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlockV(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlockV(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlockV(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlockV(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlockV(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlockV(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )
        
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            BasicBlock(256, 256, norm_fn=norm_fn),
            BasicBlock(256, 256, norm_fn=norm_fn),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }

    def forward(self, pillar_features, pillar_coords, batch_size):
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        spconv_backbone_outs = dict()
        
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        spconv_backbone_outs.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        spconv_backbone_outs.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })
        
        return spconv_backbone_outs