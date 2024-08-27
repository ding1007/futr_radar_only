_base_ = [
    '../../../../configs/_base_/datasets/nus-3d.py',
    '../../../../configs/_base_/default_runtime.py'
]

plugin=True
#plugin_dir='plugin/fudet'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
radar_voxel_size = [0.8, 0.8, 8]
# x y z rcs vx_comp vy_comp x_rms y_rms vx_rms vy_rms
radar_use_dims = [0, 1, 2, 8, 9, 18]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=False,
    use_radar=True,
    use_map=False,
    use_external=False)

model = dict(
    type='FUTR3D',
    use_lidar=False,
    use_camera=False,
    use_radar=True,
    use_grid_mask=True,
    radar_voxel_encoder=dict(
        type='RadarFeatureNet',
        in_channels=6,
        feat_channels=[32, 64],
        with_distance=False,
        point_cloud_range=point_cloud_range,
        voxel_size=radar_voxel_size,
        norm_cfg=dict(
            type='BN1d',
            eps=1.0e-3,
            momentum=0.01)
    ),
    radar_voxel_layer=dict(
    max_num_points=10, 
    voxel_size=radar_voxel_size, 
    max_voxels=(90000, 120000),
    point_cloud_range=point_cloud_range),
    radar_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[128, 128],
    ),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=10,
        in_channels=256,
        feat_channels=64,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            scales=[1],
            sizes=[
                [2.5981, 0.8660, 1.],  # 1.5 / sqrt(3)
                [1.7321, 0.5774, 1.],  # 1 / sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500)))

dataset_type = 'NuScenesDataset'
data_root = 'data/nuScenes/'

file_client_args = dict(backend='disk')


train_pipeline = [
    # dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=4,
        use_dim=radar_use_dims,
        max_num=1200, ),
    # dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    # dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'radar'])
]
test_pipeline = [
    # dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=4,
        use_dim=radar_use_dims,
        max_num=1200, ),
    # dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=[ 'radar'])
        ])
]


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data/nuscenes_futr_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline, 
        classes=class_names, 
        modality=input_modality, 
        ann_file='data/nuscenes_futr_infos_val.pkl'),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline, 
        classes=class_names, 
        modality=input_modality, 
        ann_file='data/nuscenes_futr_infos_val.pkl'))

optimizer = dict(
    type='AdamW', 
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer
_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=24, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from=None
