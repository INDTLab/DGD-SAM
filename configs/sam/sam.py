from get_mIoU import image


default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5, save_best='coco/segm_mAP', rule='greater',
                    save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='DetVisualizationHook', draw=True, interval=1, test_out_dir='vis_data')
)
crop_size = (512, 512)

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend', init_kwargs=dict(project='rsprompter-nwpu', group='rsprompter-anchor',
                                                              name="rsprompter_anchor-nwpu-peft-512"))
                ]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

num_classes = 10
prompt_shape = (70, 5)  # (per img pointset, per pointset point)

#### should be changed when using different pretrain model

# sam base model
hf_sam_pretrain_name = "/data2/yihan/MyProject/RSPrompter-release/sam-vit-base"
# hf_sam_pretrain_name = "/data2/yihan/MyProject/RSPrompter-release/sam-vit-large"
# huggingface model name, e.g. facebook/sam-vit-base
# or local repo path, e.g. work_dirs/sam_cache/sam_vit_base
hf_sam_pretrain_ckpt_path = "/data2/yihan/MyProject/RSPrompter-release/sam-vit-base/pytorch_model.bin"
# hf_sam_pretrain_ckpt_path = "/data2/yihan/MyProject/RSPrompter-release/sam-vit-large/pytorch_model.bin"
# # sam large model
# hf_sam_pretrain_name = "facebook/sam-vit-large"
# hf_sam_pretrain_ckpt_path = "~/.cache//huggingface/hub/models--facebook--sam-vit-large/snapshots/70009d56dac23ebb3265377257158b1d6ed4c802/pytorch_model.bin"
# # sam huge model
# hf_sam_pretrain_name = "facebook/sam-vit-huge"
# hf_sam_pretrain_ckpt_path = "~/.cache/huggingface/hub/models--facebook--sam-vit-huge/snapshots/89080d6dcd9a900ebd712b13ff83ecf6f072e798/pytorch_model.bin"


batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=crop_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32,
    batch_augments=batch_augments
)

model = dict(
    type='MySam',
    data_preprocessor=data_preprocessor,
    # decoder_freeze=False,
    shared_image_embedding=dict(
        type='MyPositionalEmbedding',
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
    ),
    hf_pretrain_name=hf_sam_pretrain_name,
    backbone=dict(
        type='MyVisionEncoder',
        hf_pretrain_name=hf_sam_pretrain_name,
        extra_config=dict(output_hidden_states=False, image_size=crop_size[0]),
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)
    ),
    neck=dict(
        type='RSFPN',
        feature_aggregator=dict(
            type='PseudoFeatureAggregator',
            in_channels=256,
            hidden_channels=512,
            out_channels=256,
        ),
        feature_spliter=dict(
            type='RSSimpleFPN',
            backbone_channel=256,
            in_channels=[64, 128, 256, 256],
            out_channels=256,
            num_outs=5,
            norm_cfg=dict(type='LN2d', requires_grad=True)),
    ),
    prompt_encoder=dict(
        type='MyPromptEncoder',
        hf_pretrain_name=hf_sam_pretrain_name,
        extra_config=dict(image_size=crop_size[0], image_embedding_size=crop_size[0] // 16),
        # 不必用这个，用原版sam带的 SamPositionalEmbedding即可
        # shared_patch_embedding=dict(
        #     type='MyPositionalEmbedding',
        #     hf_pretrain_name=hf_sam_pretrain_name,
        #     init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
        # ),
    ),
    mask_decoder=dict(
        type='MyMaskDecoder',
        hf_pretrain_name=hf_sam_pretrain_name,
    ),
    # model training and testing settings
    train_cfg=None,
    # 测试配置（test_cfg）
    test_cfg=None,
)

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=(0.406 * 255, 0.456 * 255, 0.485 * 255), masks=0)),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor'))
]

dataset_type = 'LIACiInsSegDataset'
#### should be changed align with your code root and data root
code_root = "/data2/yihan/MyProject/RSPrompter-release"
data_root = "/data2/yihan/MyDataset/LIACi/LIACi_dataset_pretty/"

batch_size_per_gpu = 2
num_workers = 8
persistent_workers = True
train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        indices=None,
        data_root=data_root,
        ann_file="/data2/yihan/MyDataset/LIACi/LIACi_dataset_pretty/train_ann.json",
        data_prefix=dict(img='images'),
        pipeline=train_pipeline,
        backend_args=None,
    )
)

val_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        indices=None,
        data_root=data_root,
        ann_file="/data2/yihan/MyDataset/LIACi/LIACi_dataset_pretty/val_ann.json",
        data_prefix=dict(img='images'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None,
    )
)

val_evaluator = dict(
    type='CocoMetric',
    # metric=['bbox', 'segm'],
    metric=['segm'],
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator


test_dataloader = val_dataloader
resume = False
load_from = None

# base_lr = 0.0002
base_lr = 0.00002
max_epochs = 200

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
#     dict(
#         type='CosineAnnealingLR',
#         eta_min=base_lr * 0.001,
#         begin=1,
#         end=max_epochs,
#         T_max=max_epochs,
#         by_epoch=True
#     )
# ]
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[15, 21],
        gamma=0.1
    )
]

#### AMP training config
runner_type = 'Runner'
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05),
    accumulative_counts=4
)
# val_evaluator = dict(
#     type='CocoMetric',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args=backend_args,
# )
# evaluation = dict(
#     interval=1,  # 每隔多少个 epoch 进行一次评估
#     metric='bbox',  # 评估任务类型，比如 'bbox'（目标检测），'segm'（分割）
#     metric_options=dict(
#         maxDets=[1, 10, 100]  # 设置 maxDets 的不同取值
#     )
# )

# mmdet.apis.inference_detector 函数是用于推理目标检测模型的主要函数。这个函数本身没有直接的 maxDets 参数，
# 但是你可以通过模型的配置文件中的 测试配置（test_cfg） 部分来控制推理阶段每张图像保留的最大检测框数量
# test_cfg = dict(
#     rpn=dict(
#         nms_pre=1000,  # NMS 前保留的候选框数量
#         max_per_img=100,  # RPN 阶段保留的最大候选框数量
#     ),
#     rcnn=dict(
#         score_thr=0.05,  # 置信度分数阈值，只有超过此阈值的框才会保留
#         nms=dict(type='nms', iou_threshold=0.5),  # NMS 的 IOU 阈值
#         max_per_img=100  # 最终每张图像保留的最大检测框数量
#     )
# )




# #### DeepSpeed Configs
# runner_type = 'FlexibleRunner'
# strategy = dict(
#     type='DeepSpeedStrategy',
#     fp16=dict(
#         enabled=True,
#         auto_cast=False,
#         fp16_master_weights_and_grads=False,
#         loss_scale=0,
#         loss_scale_window=500,
#         hysteresis=2,
#         min_loss_scale=1,
#         initial_scale_power=15,
#     ),
#     gradient_clipping=0.1,
#     inputs_to_half=['inputs'],
#     zero_optimization=dict(
#         stage=2,
#         allgather_partitions=True,
#         allgather_bucket_size=2e8,
#         reduce_scatter=True,
#         reduce_bucket_size='auto',
#         overlap_comm=True,
#         contiguous_gradients=True,
#     ),
# )
# optim_wrapper = dict(
#     type='DeepSpeedOptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=base_lr,
#         weight_decay=0.05
#     )
# )
