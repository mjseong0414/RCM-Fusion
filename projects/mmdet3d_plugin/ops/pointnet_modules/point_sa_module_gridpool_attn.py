from builtins import breakpoint
import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.ops import (GroupAll, PAConv, QueryAndGroup)
from mmdet3d.ops.pointnet_modules.builder import SA_MODULES


class BasePointSAModuleAttn(nn.Module):
    """Base module for point set abstraction module used in PointNets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        dilated_group (bool): Whether to use dilated ball query.
            Default: False.
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
        grouper_return_grouped_xyz (bool): Whether to return grouped xyz in
            `QueryAndGroup`. Defaults to False.
        grouper_return_grouped_idx (bool): Whether to return grouped idx in
            `QueryAndGroup`. Defaults to False.
    """

    def __init__(self,
                 num_point,
                 radii,
                 sample_nums,
                 mlp_channels,
                 dilated_group=False,
                 use_xyz=True,
                 pool_mod='max',
                 normalize_xyz=False,
                 grouper_return_grouped_xyz=False,
                 grouper_return_grouped_idx=False):
        super(BasePointSAModuleAttn, self).__init__()

        assert len(radii) == len(sample_nums) == len(mlp_channels)
        assert pool_mod in ['max', 'avg']

        if isinstance(mlp_channels, tuple):
            mlp_channels = list(map(list, mlp_channels))
        self.mlp_channels = mlp_channels

        if isinstance(num_point, int):
            self.num_point = [num_point]
        elif isinstance(num_point, list) or isinstance(num_point, tuple):
            self.num_point = num_point
        else:
            raise NotImplementedError('Error type of num_point!')

        self.pool_mod = pool_mod
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.mlp_attns = nn.ModuleList()

        for i in range(len(radii)):
            radius = radii[i]
            sample_num = sample_nums[i]
            if num_point is not None:
                if dilated_group and i != 0:
                    min_radius = radii[i - 1]
                else:
                    min_radius = 0
                grouper = QueryAndGroup(
                    radius,
                    sample_num,
                    min_radius=min_radius,
                    use_xyz=use_xyz,
                    normalize_xyz=normalize_xyz,
                    return_grouped_xyz=grouper_return_grouped_xyz,
                    return_grouped_idx=grouper_return_grouped_idx)
            else:
                grouper = GroupAll(use_xyz)
            self.groupers.append(grouper)

    def _pool_features(self, features):
        """Perform feature aggregation using pooling operation.

        Args:
            features (torch.Tensor): (B, C, N, K)
                Features of locally grouped points before pooling.

        Returns:
            torch.Tensor: (B, C, N)
                Pooled features aggregating local information.
        """
        if self.pool_mod == 'max':
            # (B, C, N, 1)
            new_features = F.max_pool2d(
                features, kernel_size=[1, features.size(3)])
        elif self.pool_mod == 'avg':
            # (B, C, N, 1)
            new_features = F.avg_pool2d(
                features, kernel_size=[1, features.size(3)])
        else:
            raise NotImplementedError

        return new_features.squeeze(-1).contiguous()

    def forward(
        self,
        points_xyz,
        features=None,
        new_xyz=None,
    ):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
                Default: None.
            new_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tensor: (B, M, 3) where M is the number of points.
                New features xyz.
            Tensor: (B, M, sum_k(mlps[k][-1])) where M is the number
                of points. New feature descriptors.
            Tensor: (B, M) where M is the number of points.
                Index of the features.
        """
        new_features_list = []

        for i in range(len(self.groupers)):
            # grouped_results : (b, 128+3, grid_points per roi, nsample)
            grouped_results = self.groupers[i](points_xyz, new_xyz, features)
            # (B, mlp[-1], num_point, nsample)
            new_features = self.mlps[i](grouped_results)
            
            # feature 강화
            weight = self.mlp_attns[i](new_features).sigmoid()
            new_features = new_features * weight

            # this is a bit hack because PAConv outputs two values
            # we take the first one as feature
            if isinstance(self.mlps[i][0], PAConv):
                assert isinstance(new_features, tuple)
                new_features = new_features[0]

            # (B, mlp[-1], num_point)
            new_features = self._pool_features(new_features)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


@SA_MODULES.register_module()
class PointSAModuleMSGAttn(BasePointSAModuleAttn):
    """Point set abstraction module with multi-scale grouping (MSG) used in
    PointNets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        dilated_group (bool): Whether to use dilated ball query.
            Default: False.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
    """

    def __init__(self,
                 num_point,
                 radii,
                 sample_nums,
                 mlp_channels,
                 dilated_group=False,
                 norm_cfg=dict(type='BN2d'),
                 use_xyz=True,
                 pool_mod='max',
                 normalize_xyz=False,
                 bias='auto'):
        super(PointSAModuleMSGAttn, self).__init__(
            num_point=num_point,
            radii=radii,
            sample_nums=sample_nums,
            mlp_channels=mlp_channels,
            dilated_group=dilated_group,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            normalize_xyz=normalize_xyz)

        for i in range(len(self.mlp_channels)):
            mlp_channel = self.mlp_channels[i]
            if use_xyz:
                mlp_channel[0] += 3

            mlp = nn.Sequential()
            mlp_attn = nn.Sequential()
            for i in range(len(mlp_channel) - 1):
                mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        mlp_channel[i],
                        mlp_channel[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias=bias))
                mlp_attn.add_module(
                    f'layer{i}',
                    ConvModule(
                        mlp_channel[i + 1],
                        mlp_channel[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias=bias))
            self.mlps.append(mlp)
            self.mlp_attns.append(mlp_attn)


@SA_MODULES.register_module()
class PointSAModuleAttn(BasePointSAModuleAttn):
    """Point set abstraction module with single-scale grouping (SSG) used in
    PointNets.

    Args:
        mlp_channels (list[int]): Specify of the pointnet before
            the global pooling for each scale.
        num_point (int): Number of points.
            Default: None.
        radius (float): Radius to group with.
            Default: None.
        num_sample (int): Number of samples in each ball query.
            Default: None.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
    """

    def __init__(self,
                 mlp_channels,
                 num_point=None,
                 radius=None,
                 num_sample=None,
                 norm_cfg=dict(type='BN2d'),
                 use_xyz=True,
                 pool_mod='max',
                 normalize_xyz=False):
        super(PointSAModuleAttn, self).__init__(
            mlp_channels=[mlp_channels],
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            norm_cfg=norm_cfg,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            normalize_xyz=normalize_xyz)
