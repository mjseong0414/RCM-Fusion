
import torch
import torch.nn as nn
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean, build_assigner, build_sampler)
from mmcv.runner import force_fp32
import numpy as np
import cv2 as cv
from mmdet3d.models.builder import build_loss
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox_custom
from mmdet3d.models.builder import FUSION_LAYERS
from mmdet3d.ops import build_sa_module
from torch.nn import functional as F
import copy
import time
from mmdet3d.ops.furthest_point_sample import (furthest_point_sample)
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.rcm_fusion.modules.detr3d_cross_attention import Detr3DCrossAtten

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

def SoftPolarAssociation(polar_pts, polar_corners_):
    # proposal-wise grouping range
    point_num = int(polar_pts.shape[0])
    num_box = int(polar_corners_.shape[0])
    range_center = (polar_corners_[:,0] + polar_corners_[:,1])/2
    
    box_length = torch.abs(polar_corners_[:,0] - polar_corners_[:,1])
    range_off = torch.clamp(box_length*1.05, max=5.0)
    r_low = polar_corners_[:,1] - range_off
    r_upper = polar_corners_[:,0] + range_off
    
    mask1 = polar_corners_[:, 2] < 0
    mask2 = polar_corners_[:, 3] < 0
    polar_corners_[:, 2][mask1] = polar_corners_[:, 2][mask1] + 2*np.pi
    polar_corners_[:, 3][mask2] = polar_corners_[:, 3][mask2] + 2*np.pi
    abs_azi = torch.abs(polar_corners_[:,2] - polar_corners_[:,3])
    azi_mask = abs_azi > (np.pi/2)
    abs_azi[azi_mask] = 2*np.pi - abs_azi[azi_mask]
    box_width =  abs_azi * range_center
    width_off = torch.clamp(box_width*1.05, max=5.0)
    azi_off = torch.clamp(width_off / range_center, min=0, max=0.05)
    angle_max = polar_corners_[:, 2] + azi_off
    angle_min = polar_corners_[:, 3] - azi_off
    mask3 = angle_max > 2*np.pi
    mask4 = angle_min < 0
    angle_max[mask3] = angle_max[mask3] - 2*np.pi
    angle_min[mask4] = angle_min[mask4] + 2*np.pi
    
    polar_ranges = polar_pts[:,0].expand(num_box,point_num).transpose(0,1)
    mask0 = polar_pts[:,1] < 0
    polar_pts[:,1][mask0] = polar_pts[:,1][mask0] + 2*np.pi
    polar_angles = polar_pts[:,1].expand(num_box,point_num).transpose(0,1)
    
    point_masks_ = (polar_ranges < r_upper) * (polar_ranges > r_low)\
        * (polar_angles > angle_min) *(polar_angles < angle_max)

    return point_masks_


@FUSION_LAYERS.register_module()
class InstanceLevelFusion(nn.Module):
# Proposal-Aware, Point_Gating
    def __init__(self, 
                radii, 
                num_samples, 
                sa_mlps, 
                dilated_group, 
                norm_cfg, 
                sa_cfg, 
                grid_size, 
                grid_fps, 
                code_size,
                num_classes,
                input_channels,
                train_cfg=None,
                test_cfg=None,
                pc_range=None,
                loss_bbox=None,
                loss_cls=None,
                bbox_coder=None):
        super(InstanceLevelFusion, self).__init__()
        self.fp16_enabled = False
        self.grid_size = grid_size
        self.grid_fps = grid_fps
        self.code_size = code_size
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.pc_range = pc_range
        self.code_weights = [1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 
                            1.0, 1.0, 0.2, 0.2]
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.roi_grid_pool_layer = build_sa_module(
                                    num_point=self.grid_fps,
                                    radii=radii,
                                    sample_nums=num_samples,
                                    mlp_channels=sa_mlps,
                                    dilated_group=dilated_group,
                                    norm_cfg=norm_cfg,
                                    cfg=sa_cfg,
                                    bias=True)
        self.shared_fc_layer = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.reg_layer = nn.Conv1d(512, self.code_size, kernel_size=1, bias=True)
        
        if train_cfg is not None:
            self.assigner_refine = build_assigner(train_cfg['assigner'])

        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = True
        
        self.Detr3DCrossAtten = Detr3DCrossAtten()
        self.channelMixMLPs01 = nn.Sequential(
            nn.Conv1d(6, self.grid_fps//2, kernel_size=1),
            nn.BatchNorm1d(self.grid_fps//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.grid_fps//2, self.grid_fps//2, kernel_size=1))
        
        self.linear_p = nn.Sequential(
            nn.Conv1d(3, 3, kernel_size=1),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True), 
            nn.Conv1d(3, self.grid_fps//2, kernel_size=1))

        self.channelMixMLPs02 = nn.Sequential(
            nn.Conv1d(self.grid_fps, self.grid_fps//2, kernel_size=1),
            nn.BatchNorm1d(self.grid_fps//2), 
            nn.ReLU(inplace=True),
            nn.Conv1d(self.grid_fps//2, self.grid_fps//2, kernel_size=1))

        self.channelMixMLPs03 = nn.Conv1d(6, self.grid_fps//2, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, outs, refine_point, refine_box, img_feats, img_metas):
        batch_polar_corner = refine_box['polar_corner']
        batch_radar_point = refine_point['original']
        batch_polar_radar_point = refine_point['polar']
        bs = len(batch_polar_radar_point)
        num_box = int(batch_polar_corner.shape[1])
        device = batch_polar_corner.device
        bbox_preds = []
        e_box_indexes = []
        ne_box_indexes = []
        
        for i in range(bs):
            polar_corners = batch_polar_corner[i]
            all_radar_points = batch_radar_point[i]
            polar_radar_points = batch_polar_radar_point[i]
            head_feature = refine_box['boxes'][i][1].detach().clone()
            proposals = refine_box['boxes'][i][0].tensor.detach().clone()
            
            point_masks = SoftPolarAssociation(polar_radar_points, polar_corners)
            e_box_index = (torch.sum(point_masks, axis=0) == 0).nonzero().squeeze()
            ne_box_index = torch.nonzero(torch.sum(point_masks, axis=0), as_tuple=True)
            ne_box_num = ne_box_index[0].shape[0]
            
            if ne_box_num == 0:
                bbox_preds.append([])
                e_box_indexes.append(e_box_index)
                ne_box_indexes.append(ne_box_index)
                continue
            elif ne_box_num == 1:
                e_box_index = torch.cat((e_box_index, ne_box_index[0]))
                ne_box_index = (torch.tensor([]),)
                bbox_preds.append([])
                e_box_indexes.append(e_box_index)
                ne_box_indexes.append(ne_box_index)
                continue
            else:
                e_box_indexes.append(e_box_index)
                ne_box_indexes.append(ne_box_index)
            
            ne_proposals = proposals[ne_box_index[0]]
            non_empty_pred_box_centers = ne_proposals[:, :3]
            
            ############ Radar Grid Point Generation ############
            box_2d = ne_proposals[:,:2]
            box_angle = torch.atan2(box_2d[:,1],-box_2d[:,0])
            box_true_vel = ne_proposals[:,-2:]
            box_vel_value = torch.clamp(torch.sqrt((box_true_vel**2).sum(axis=1))/2,min=1,max=5)
            grid_interval = int((self.grid_size-1)/2)
            bin_weight = torch.arange(self.grid_size).to(device)-grid_interval
            grid_weight = box_vel_value/grid_interval
            roi_box_num = box_2d.shape[0]
            total_weights_2 = bin_weight.expand(roi_box_num,self.grid_size).transpose(0,1)*grid_weight
            vel_x = (torch.cos(box_angle)*total_weights_2).T
            vel_y = (torch.sin(box_angle)*total_weights_2).T
            roi_per_pts_num = point_masks.sum(axis=0)[ne_box_index]
            roi_x_all_pts = point_masks.T[ne_box_index]
            
            max_pts = int(roi_per_pts_num.max())
            fps_need_box = torch.nonzero(roi_per_pts_num>self.grid_fps//4).T[0]
            fps_num_box = fps_need_box.shape[0]
            fps_need_grid_box = torch.nonzero(roi_per_pts_num*self.grid_size>self.grid_fps).T[0]
            fps_num_grid_box = fps_need_grid_box.shape[0]

            radar_points_per_roi = torch.zeros((ne_box_num,self.grid_fps//4,6)).to(device)
            grid_points_per_roi = torch.zeros((ne_box_num,self.grid_fps,3)).to(device)
            if fps_num_box > 0 :
                batch_cur_pts = torch.zeros((fps_num_box,max_pts,6)).to(device)
            if fps_num_grid_box > 0 :
                batch_cur_grid_pts = torch.zeros((fps_num_grid_box,max_pts*self.grid_size,6)).to(device)
            
            for box_idx in range(ne_box_num):
                pts_index = (roi_x_all_pts[box_idx]==True).nonzero(as_tuple=True)[0]
                if box_idx not in fps_need_box:
                    random_idx = torch.randint(pts_index.shape[0], (self.grid_fps//4-pts_index.shape[0],))
                    random_idx = torch.cat((torch.arange(pts_index.shape[0]), random_idx))
                    cur_roi_point_xyz = all_radar_points[pts_index[random_idx]]
                    radar_points_per_roi[box_idx] = cur_roi_point_xyz
                else:
                    new_idx = torch.nonzero(fps_need_box == box_idx).item()
                    random_idx = torch.randint(pts_index.shape[0], (max_pts-pts_index.shape[0],))
                    random_idx = torch.cat((torch.arange(pts_index.shape[0]), random_idx))
                    cur_roi_point_xyz = all_radar_points[pts_index[random_idx]]
                    batch_cur_pts[new_idx] = cur_roi_point_xyz
                
                grid_pts = all_radar_points[pts_index][None, :, :].expand(self.grid_size, pts_index.shape[0], 6).reshape(-1, 6).clone() # 레이더 포인트 개수 * grid_size
                grid_pts[:, 0:1] = grid_pts[:, 0:1] + vel_x[box_idx][None,:, None].expand(pts_index.shape[0],self.grid_size,1).reshape(-1, 1)
                grid_pts[:, 1:2] = grid_pts[:, 1:2] + vel_y[box_idx][None,:, None].expand(pts_index.shape[0],self.grid_size,1).reshape(-1, 1)
                
                if grid_pts.shape[0] < self.grid_fps:
                    grid_random_idx = torch.randint(grid_pts.shape[0], (self.grid_fps-grid_pts.shape[0],))
                    grid_random_idx = torch.cat((torch.arange(grid_pts.shape[0]), grid_random_idx))
                    cur_roi_grid_point_xyz = grid_pts[grid_random_idx][...,:3]
                    grid_points_per_roi[box_idx] = cur_roi_grid_point_xyz
                else:
                    new_idx_g = torch.nonzero(fps_need_grid_box == box_idx).item()
                    random_idx_g = torch.randint(grid_pts.shape[0], (max_pts*self.grid_size - grid_pts.shape[0],))
                    random_idx_g = torch.cat((torch.arange(grid_pts.shape[0]), random_idx_g))
                    cur_roi_point_xyz_g = grid_pts[random_idx_g]
                    batch_cur_grid_pts[new_idx_g] = cur_roi_point_xyz_g
            if fps_num_box > 0 :
                fps_pts_idx = furthest_point_sample(batch_cur_pts[..., :3].contiguous().float(),self.grid_fps//4).long()
                for fps_box in range(fps_num_box):
                    fps_cur_pts = batch_cur_pts[fps_box][fps_pts_idx[fps_box]]
                    radar_points_per_roi[fps_need_box[fps_box]] = fps_cur_pts
            if fps_num_grid_box > 0 :
                fps_grid_idx = furthest_point_sample(batch_cur_grid_pts[..., :3].contiguous().float(),self.grid_fps).long()
                for fps_grid_box in range(fps_num_grid_box):
                    fps_cur_grid_pts = batch_cur_grid_pts[fps_grid_box][fps_grid_idx[fps_grid_box]][..., :3]
                    grid_points_per_roi[fps_need_grid_box[fps_grid_box]] = fps_cur_grid_pts
            
            ############ proposal-aware radar attention ############
            relative_pos = torch.abs(non_empty_pred_box_centers[:,None,:].expand(ne_box_num,self.grid_fps//4,3) - radar_points_per_roi[..., :3])
            energy = self.channelMixMLPs01(radar_points_per_roi.permute(0,2,1).contiguous())
            p_embed = self.linear_p(relative_pos.permute(0,2,1).contiguous())
            energy = torch.cat([energy, p_embed], dim=1)
            energy = self.channelMixMLPs02(energy)
            w = self.softmax(energy)
            x_v = self.channelMixMLPs03(radar_points_per_roi.permute(0,2,1).contiguous())
            radar_attn_feat = ((x_v + p_embed) * w).permute(0,2,1).contiguous()
            ############ proposal-aware radar attention ############
            
            
            ############ Radar Grid Point Pooling ############
            ### Set Abstraction ###
            # pooled_point : (1, M, 3)
            # pooled_feature : (1, C, M)
            grid_points_per_roi_ = grid_points_per_roi.view(-1, 3)
            pooled_point, pooled_feature = self.roi_grid_pool_layer(
                points_xyz = radar_points_per_roi[..., :3].view(-1, 3).unsqueeze(0).contiguous(), # (1, N, 3)
                features = radar_attn_feat.view(-1, 32).unsqueeze(0).permute(0,2,1).contiguous(), # (1, C, N)
                new_xyz = grid_points_per_roi_.unsqueeze(0) # (1, M, 3)
            )
            
            ### Detr3D style image feature pooling ###
            img_feats_ = img_feats.copy()
            value=[]
            for mlvl_feat_num in range(len(img_feats_)):
                value.append(img_feats_[mlvl_feat_num][i].unsqueeze(0))
            query = pooled_feature.permute(2, 0, 1).contiguous()
            key = None
            reference_points = pooled_point.clone()
            sampled_feat = self.Detr3DCrossAtten(query, key, value, reference_points=reference_points, img_metas=img_metas[i])
            sampled_feats = sampled_feat.permute(1, 2, 0).contiguous().view(1, -1, ne_box_num, self.grid_fps)
            
            # sampled_feats : (1, C, ne_box_num, 64)
            shared_features = self.shared_fc_layer(sampled_feats).max(-1)[0]
            head_feature_ = head_feature[ne_box_index[0]].unsqueeze(0).transpose(1, 2).contiguous()
            refine_feature = torch.cat((head_feature_, shared_features), dim=1)
            bbox_pred = self.reg_layer(refine_feature).transpose(1, 2).contiguous().squeeze() # x y z vx vy
            bbox_preds.append(bbox_pred)
        
        box_reg = torch.zeros((bs, num_box, 10)).to(device)
        for i in range(bs):
            if len(e_box_indexes[i]) == 0:
                pass
            else:
                e_boxes_ori = proposals[e_box_indexes[i]]
                box_reg[i][e_box_indexes[i]] = e_boxes_ori
            if len(ne_box_indexes[i][0]) == 0:
                pass
            else:
                ne_boxes_ori = proposals[ne_box_indexes[i]]
                ne_boxes_ori[..., 0:2] += bbox_preds[i][..., 0:2] # x, y offset
                ne_boxes_ori[..., 3:6] += bbox_preds[i][..., 2:5] # w, l, h offset
                ne_boxes_ori[..., 8:10] += bbox_preds[i][..., 5:7] # vx, vy offset
                box_reg[i][ne_box_indexes[i]] = ne_boxes_ori
        
        box_reg = box_reg.view(1, bs, num_box, 10)
        outs['all_bbox_preds_refine'] = box_reg
        return outs


    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        assign_result = self.assigner_refine.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(bbox_preds_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        
        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = torch.tensor([num_total_pos], dtype=torch.float).to(bbox_preds.device)
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox_custom(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        
        # center loss
        loss_center = self.loss_bbox(bbox_preds[isnotnan, :2], normalized_bbox_targets[isnotnan, :2], 
                                    bbox_weights[isnotnan, :2],avg_factor=num_total_pos)
        # dim loss
        loss_dim = self.loss_bbox(bbox_preds[isnotnan, 3:6], normalized_bbox_targets[isnotnan, 3:6], 
                                    bbox_weights[isnotnan, 3:6],avg_factor=num_total_pos)
        # velocity loss
        loss_vel = self.loss_bbox(bbox_preds[isnotnan, 8:10], normalized_bbox_targets[isnotnan, 8:10], 
                                    bbox_weights[isnotnan, 8:10],avg_factor=num_total_pos)
        loss_bbox = loss_center + loss_dim + loss_vel
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_bbox = torch.nan_to_num(loss_bbox)
        return num_imgs, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             losses=None,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        all_cls_scores = [preds_dicts['all_cls_scores'][-1].float()]
        all_bbox_preds = preds_dicts['all_bbox_preds_refine']
        # there is one decoder layers on refine module
        num_dec_layers = len(all_bbox_preds)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        _, loss_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)
        
        losses['refine_loss_bbox'] = loss_bbox[0]
        return losses

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

        return ret_list
