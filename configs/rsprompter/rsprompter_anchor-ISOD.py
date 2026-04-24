_base_ = ['_base_/rsprompter_anchor.py']

work_dir = './work_dirs/rsprompter/rsprompter_anchor-nwpu-peft-512'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2, save_best=['coco/bbox_mAP', 'coco/segm_mAP'], rule='greater',
                    save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='DetVisualizationHook', draw=True, interval=1, test_out_dir='vis_data')
)
# 添加自定义 Hook的配置
custom_hooks = [
    dict(type='EpochInfoHook', priority='NORMAL'),
    dict(type='CustomFileLoggerHook', file_path="/data2/yihan/MyProject/RSPrompter-release/mmdet/rsprompter/__init__.py", priority='NORMAL'),
    # dict(type='CustomFileLoggerHook', file_path="/data3/yihan/MyProject/RSPrompter-release/mmdet/rsprompter/__init__.py", priority='NORMAL'),
    # dict(type='IgnoreOOMHook', priority='NORMAL'),
    # dict(type='EvalTestHook',test_dataloader_cfg=test_dataloader,test_evaluator_cfg=test_evaluator) # 如果训练集和测试集不是同一个，就使用EvalTestHook
]
crop_size = (512, 512)

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend', init_kwargs=dict(project='rsprompter-nwpu', group='rsprompter-anchor',
                                                              name="rsprompter_anchor-nwpu-peft-512"))
                ]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

num_classes = 1
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
        type='BatchFixedSizePad', # 指定数据增强类型为 批量固定尺寸填充，即对每个图像进行填充，确保所有图像的尺寸统一为固定值。
        size=crop_size, # 指定目标填充后的尺寸，所有图像将被填充或裁剪为这个大小。
        img_pad_value=0, # 指定填充区域的值，默认值为 0
        pad_mask=True, # 是否对掩膜进行填充
        mask_pad_value=0, # 指定填充掩膜的值
        pad_seg=False # 是否填充分割标注,不对分割标注进行填充操作，是因为已经有了pad_mask
    )
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
    decoder_freeze=False,
    data_preprocessor=data_preprocessor,
    shared_image_embedding=dict(
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
    ),
    backbone=dict(
        _delete_=True,
        img_size=crop_size[0],
        type='MMPretrainSamVisionEncoder',
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
        peft_config=None,
        # peft_config={},
        # peft_config=dict(
        #     peft_type="LORA",
        #     r=16,
        #     target_modules=["qkv"],
        #     lora_alpha=32,
        #     lora_dropout=0.05,
        #     bias="none",
        # ),
    ),
    neck=dict(
        feature_aggregator=dict(
            _delete_=True,
            type='PseudoFeatureAggregator',
            in_channels=256,
            hidden_channels=512,
            out_channels=256,
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=num_classes,
        ),
        mask_head=dict(
            mask_decoder=dict(
                hf_pretrain_name=hf_sam_pretrain_name,
                init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)
            ),
            per_pointset_point=prompt_shape[1],
            with_sincos=True,
        ),
    ),
    train_cfg=dict(
        rcnn=dict(
            mask_size=crop_size,
        )
    ),
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

# dataset_type = 'LIACiInsSegDataset'
dataset_type = 'ISODDataset'
#### should be changed align with your code root and data root
code_root = "/data2/yihan/MyProject/RSPrompter-release"

data_root = "/data2/yihan/MyDataset/isod/"

batch_size_per_gpu = 6
num_workers = 8
persistent_workers = True
train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="/data2/yihan/MyDataset/isod/train.json",
        data_prefix=dict(img='imgs'),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="/data2/yihan/MyDataset/isod/test.json",
        data_prefix=dict(img='imgs'),
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader
# test_dataloader = dict(
#     batch_size=batch_size_per_gpu,
#     num_workers=num_workers,
#     persistent_workers=persistent_workers,
#     # num_workers=0, # 2025-02-24为了调试，设为 0
#     # persistent_workers=False, # 2025-02-24为了调试，设为 False，注意如果num_workers=0，那么persistent_workers必须是False
#     dataset=dict(
#         type=dataset_type,
#         # data_root="/data2/yihan/MyDataset/COME15K/",
#         # ann_file="/data2/yihan/MyDataset/COME15K/annotations/COME15K-Test-E.json",
#         # data_prefix=dict(img="COME-E/RGB"),
#         data_root="/data2/yihan/MyDataset/SIP/",
#         ann_file="/data2/yihan/MyDataset/SIP/SIP.json",
#         data_prefix=dict(img="RGB"),
#         pipeline=test_pipeline,
#     )
# )
resume = False
load_from = None

base_lr = 0.0002
max_epochs = 100

train_cfg = dict(max_epochs=max_epochs)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.001,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True
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
    # accumulative_counts=4
)
val_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args,
    classwise=True,
)
test_evaluator=val_evaluator
# 如果训练集和测试集不是同一个，就使用EvalTestHook
# 通过钩子实现：每个epoch跑完后，既在验证集上计算指标（验证），也在测试集上计算指标（测试）
# custom_hooks.append(
#     dict(
#         type='EvalTestHook',
#         dataloader=test_dataloader # 或你实际配置的 test dataloader
#     )
# )

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
