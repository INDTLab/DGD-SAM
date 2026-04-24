_base_ = ['./_base_/rsprompter_anchor.py']

work_dir = './work_dirs/rsprompter/rsprompter_anchor-nwpu-peft-512'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5, save_best='coco/segm_mAP', rule='greater',
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
    decoder_freeze=False,
    data_preprocessor=data_preprocessor,
    shared_image_embedding=dict(
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
    ),
    backbone=dict(
        _delete_=True,
        type="ImageEncoderViT_EfficientSAM_pretrained",
        img_size=crop_size[0],
        patch_size=16,
        in_chans=3,
        patch_embed_dim=384, # 只能是384，不然预训练权重就对不上了
        normalization_type="layer_norm",
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        neck_dims=[256, 256],
        peft_config=None,
        # peft={},

        # peft_config=dict(
        #     bias='none',
        #     lora_alpha=32,
        #     lora_dropout=0.05,
        #     peft_type='LORA',
        #     r=16,
        #     target_modules=[
        #         'qkv',
        #     ]
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

dataset_type = 'LIACiInsSegDataset'
#### should be changed align with your code root and data root
code_root = "/data1/yihan/mmdetection-main/"
data_root = "/data2/yihan/MyDataset/LIACi/LIACi_dataset_pretty/"

batch_size_per_gpu = 4
num_workers = 8
persistent_workers = True
train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="/data2/yihan/MyDataset/LIACi/LIACi_dataset_pretty/train_ann.json",
        data_prefix=dict(img='images'),
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
        ann_file="/data2/yihan/MyDataset/LIACi/LIACi_dataset_pretty/val_ann.json",
        data_prefix=dict(img='images'),
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader
resume = True
load_from = "/data2/yihan/MyProject/RSPrompter-release/work_dirs/efficient_sam_froze_backbone/epoch_153.pth"

base_lr = 0.0002
max_epochs = 400

train_cfg = dict(max_epochs=max_epochs, val_interval=3)
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
