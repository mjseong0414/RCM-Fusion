
_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.064, 0.064, 8]
grid_size = [(point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i] for i in range(3)]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
with_info=True
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=False)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 1 # each sequence contains `queue_length` frames.

model = dict(
    type='RCMFusion',
    freeze_img=False,
    freeze_pts=False,
    use_grid_mask=True,
    video_test_mode=False, # Decide whether to use temporal information during inference.
    video_train_mode=False, # Decide whether to use temporal information during train.
    pts_voxel_encoder=dict( 
        type='DynamicPillarVFESimple2D',
        num_point_features=6,
        voxel_size=voxel_size,
        grid_size=grid_size[:2], # voxel grid = 0.064
        point_cloud_range=point_cloud_range,
        num_filters=[32],
        with_distance=False,
        use_absolute_xyz=True,
        with_cluster_center=True,
        legacy=False),
    pts_backbone=dict(type='PillarRes18BackBone8x2', grid_size=grid_size[:2]),
    pts_neck=dict(
        type='BaseBEVBackboneV2',
        layer_nums=[5,5],
        num_filters=[256,256],
        upsample_strides=[1,2],
        num_upsample_filters=[128,128]),
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp=True, # using checkpoint to save GPU memory
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='FeatureLevelFusion',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformerRadar',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='RadarGuidedBEVEncoder',
                num_layers=2,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='RadarGuidedBEVEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='RadarGuidedBEVAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 
                                     'ffn', 'norm','norm')),),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    pts_fusion_layer=dict(
        type= 'InstanceLevelFusion',
        radii=(1.6, 3.2),
        num_samples=(16, 16),
        dilated_group=False,
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
        sa_mlps=((32, 32, 32), (32, 32, 32)),
        sa_cfg=dict(
            type='PointSAModuleMSGAttn',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False),
        grid_size=7,
        grid_fps=64,
        code_size=7, # x,y,w,l,h,vx,vy
        num_classes=10,
        input_channels=67, # 64 + 3
        pc_range = point_cloud_range,
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        bbox_coder=dict(
            type='NMSFreeCoderILFusion',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            score_threshold=0.0,
            voxel_size=voxel_size,
            num_classes=10),
        ),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=grid_size,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=8,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDatasetNew'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadRadarPointsFromMultiSweepsV3',
        sweeps_num=6,
        invalid_states = [0,4,8,9,10,11,12,15,16],
        ambig_states=[1,2,3,4],
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    # dict(
    #     type='ObjectSample_V2',
    #     with_info=with_info,
    #     sample_2d=True,
    #     db_sampler=dict(
    #         data_root=data_root,
    #         info_path=data_root + 'nuscenes_dbinfos_train_radar_polar_loadB_0911.pkl',
    #         rate=1.0,
    #         prepare=dict(
    #             filter_by_difficulty=[-1],
    #             filter_by_min_points=dict(
    #                 car=2,
    #                 truck=2,
    #                 bus=2,
    #                 trailer=2,
    #                 construction_vehicle=2,
    #                 traffic_cone=2,
    #                 barrier=2,
    #                 motorcycle=2,
    #                 bicycle=2,
    #                 pedestrian=2)),
    #         classes=class_names,
    #         sample_groups=dict(
    #             car=2,
    #             truck=3,
    #             construction_vehicle=7,
    #             bus=4,
    #             trailer=6,
    #             barrier=2,
    #             motorcycle=6,
    #             bicycle=6,
    #             pedestrian=2,
    #             traffic_cone=2),
    #         points_loader=dict(
    #             type='LoadPointsFromFile',
    #             coord_type='LIDAR',
    #             load_dim=6,
    #             use_dim=[0, 1, 2, 3, 4, 5],
    #         ))),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type = 'MultCamImageAugmentation', ida_aug_conf=dict(
    resize= (-0.06, 0.11),
    rot= (-5.4, 5.4),
    flip= True,
    crop_h= (0.0, 0.0),
    resize_test= 0.00 )),
    dict(type='MultiModalBEVAugmentation',bda_aug_conf=dict(
    rot_lim=(-0.3925, 0.3925),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'points'])
]

test_pipeline = [
    dict(
        type='LoadRadarPointsFromMultiSweepsV3',
        sweeps_num=6,
        invalid_states = [0,4,8,9,10,11,12,15,16],
        ambig_states=[1,2,3,4],
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['points', 'img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(type=dataset_type,
                    data_root=data_root,
                    ann_file=data_root + 'nuscenes_infos_train_rcmfusion.pkl',
                    pipeline=train_pipeline,
                    classes=class_names,
                    modality=input_modality,
                    test_mode=False,
                    use_valid_flag=True,
                    bev_size=(bev_h_, bev_w_),
                    queue_length=queue_length,
                    box_type_3d='LiDAR')),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_val_rcmfusion.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_val_rcmfusion.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/bevformer_small_static_full_epoch_23.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)