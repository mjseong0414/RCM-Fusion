import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet.models import NECKS


@NECKS.register_module()
class SECONDFPN_v2(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 fused_channels_in=None,
                 fused_channels_out=None,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False,
                 init_cfg=None):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN_v2, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                stride = np.round(stride).astype(np.int64)
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU())
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)
        
        fuse_blocks = []
        for i, fused_channel_in in enumerate(fused_channels_in):
            upsample_layer = build_conv_layer(
                conv_cfg,
                in_channels=fused_channel_in,
                out_channels=fused_channels_out[i],
                kernel_size=1,
                stride=1)

            fuse_block = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, fused_channels_out[i])[1],
                                    nn.ReLU())
            fuse_blocks.append(fuse_block)
        self.fuse_blocks = nn.ModuleList(fuse_blocks)

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
        if len(ups) > 1:
            mid = torch.cat(ups, dim=1)
        else:
            mid = ups[0]
        out = self.fuse_blocks[0](mid)
        
        return out.flatten(2).permute(2, 0, 1)

# @NECKS.register_module()
# class BaseBEVBackboneV1(nn.Module):
#     def __init__(self, layer_nums,  num_filters, upsample_strides, num_upsample_filters):
#         super().__init__()
#         assert len(layer_nums) == len(num_filters) == 4
#         assert len(num_upsample_filters) == len(upsample_strides)
        
#         num_levels = len(layer_nums)
#         self.blocks = nn.ModuleList()
#         self.deblocks = nn.ModuleList()
#         for idx in range(num_levels):
#             if idx == 0:
#                 cur_layers = [
#                     nn.ZeroPad2d(1),
#                     nn.Conv2d(
#                         256, 256, kernel_size=3,
#                         stride=1, padding=0, bias=False
#                     ),
#                     nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
#                     nn.ReLU()
#                 ]
#                 for k in range(layer_nums[idx]):
#                     cur_layers.extend([
#                         nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
#                         nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
#                         nn.ReLU()
#                     ])
#             else:
#                 cur_layers = [
#                     nn.ZeroPad2d(1),
#                     nn.Conv2d(
#                         num_filters[idx], num_filters[idx], kernel_size=3,
#                         stride=1, padding=0, bias=False
#                     ),
#                     nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
#                     nn.ReLU()
#                 ]
#                 for k in range(layer_nums[idx]):
#                     cur_layers.extend([
#                         nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
#                         nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
#                         nn.ReLU()
#                     ])
#             self.blocks.append(nn.Sequential(*cur_layers))
#             if len(upsample_strides) > 0:
#                 stride = upsample_strides[idx]
#                 if stride >= 1:
#                     self.deblocks.append(nn.Sequential(
#                         nn.ConvTranspose2d(
#                             num_filters[idx], num_upsample_filters[idx],
#                             upsample_strides[idx],
#                             stride=upsample_strides[idx], bias=False
#                         ),
#                         nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
#                         nn.ReLU()
#                     ))
#                 else:
#                     stride = np.round(1 / stride).astype(np.int)
#                     self.deblocks.append(nn.Sequential(
#                         nn.Conv2d(
#                             num_filters[idx], num_upsample_filters[idx],
#                             stride,
#                             stride=stride, bias=False
#                         ),
#                         nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
#                         nn.ReLU()
#                     ))

#         c_in = sum(num_upsample_filters)
#         if len(upsample_strides) > num_levels:
#             self.deblocks.append(nn.Sequential(
#                 nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
#                 nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
#                 nn.ReLU(),
#             ))

#         self.num_bev_features = c_in

#     def forward(self, data_dict, bs):
#         """
#         Args:
#             data_dict:
#                 spatial_features
#         Returns:
#         """
#         spatial_features = data_dict['multi_scale_2d_features']

#         x_conv2 = spatial_features['x_conv2']
#         x_conv3 = spatial_features['x_conv3']
#         x_conv4 = spatial_features['x_conv4']
#         x_conv5 = spatial_features['x_conv5']
        
#         ups = [self.deblocks[0](x_conv2.dense())]

#         x = self.blocks[1](x_conv3.dense())
#         ups.append(self.deblocks[1](x))
        
#         x = self.blocks[2](x_conv4)
#         ups.append(self.deblocks[2](x))
        
#         x = self.blocks[3](x_conv5)
#         ups.append(self.deblocks[3](x))

#         x = torch.cat(ups, dim=1)
#         x = self.blocks[0](x)
        
#         return x.view(bs, 256, -1).permute(2, 0, 1).contiguous()

@NECKS.register_module()
class BaseBEVBackboneV1(nn.Module):
    def __init__(self, layer_nums,  num_filters, upsample_strides, num_upsample_filters):
        super().__init__()
        assert len(layer_nums) == len(num_filters) == 2
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict, bs):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)

        return x.view(bs, 256, -1).permute(2, 0, 1).contiguous()


@NECKS.register_module()
class BaseBEVBackboneV2(nn.Module):
    def __init__(self, layer_nums,  num_filters, upsample_strides, num_upsample_filters):
        super().__init__()
        assert len(layer_nums) == len(num_filters) == 2
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            if idx == 0:
                cur_layers = [
                    nn.ZeroPad2d(1),
                    nn.Conv2d(num_filters[idx]*2, num_filters[idx], kernel_size=3,
                              stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ]
            else:
                cur_layers = [
                    nn.ZeroPad2d(1),
                    nn.Conv2d(
                        num_filters[idx], num_filters[idx], kernel_size=3,
                        stride=1, padding=0, bias=False
                    ),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx]*2,
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx]*2, eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        self.deblocks = self.deblocks[1:]

    def forward(self, data_dict, bs):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [x_conv4]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[0](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)

        return x.view(bs, 256, -1).permute(2, 0, 1).contiguous()