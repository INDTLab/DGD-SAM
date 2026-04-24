_base_ = ['_base_/rsprompter_query.py']

work_dir = './work_dirs/rsprompter/rsprompter_query-nwpu-peft-512'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=4, save_best='coco/segm_mAP', rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='DetVisualizationHook', draw=True, interval=1, test_out_dir='vis_data')
)

vis_backends = [dict(type='LocalVisBackend'),
                # dict(type='WandbVisBackend', init_kwargs=dict(project='rsprompter-nwpu', group='rsprompter-query', name="rsprompter_query-nwpu-peft-512"))
                ]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

num_classes = 10
prompt_shape = (70, 5)  # (per img pointset, per pointset point) (产生多少个query即每张图像生成的实例数量，每个实例需要几个提示信息)

#### should be changed when using different pretrain model

# sam base model
hf_sam_pretrain_name = "/data2/yihan/MyProject/RSPrompter-release/sam-vit-base"
# huggingface model name, e.g. facebook/sam-vit-base
# or local repo path, e.g. work_dirs/sam_cache/sam_vit_base
hf_sam_pretrain_ckpt_path = "/data2/yihan/MyProject/RSPrompter-release/sam-vit-base/pytorch_model.bin"
# # sam large model
# hf_sam_pretrain_name = "facebook/sam-vit-large"
# hf_sam_pretrain_ckpt_path = "~/.cache//huggingface/hub/models--facebook--sam-vit-large/snapshots/70009d56dac23ebb3265377257158b1d6ed4c802/pytorch_model.bin"
# # sam huge model
# hf_sam_pretrain_name = "facebook/sam-vit-huge"
# hf_sam_pretrain_ckpt_path = "~/.cache/huggingface/hub/models--facebook--sam-vit-huge/snapshots/89080d6dcd9a900ebd712b13ff83ecf6f072e798/pytorch_model.bin"

crop_size = (256, 256)

batch_augments = [
    dict(
        type='BatchFixedSizePad', # 增强类型，这里是固定大小的填充
        size=crop_size, # 填充后的目标大小，通常是一个元组，如(高度, 宽度)
        img_pad_value=0, # 填充时图像使用的值，通常为0表示黑色填充
        pad_mask=True, # 是否对掩码进行填充，常用于实例分割任务
        mask_pad_value=0, # 填充掩码时使用的值
        pad_seg=False, # 是否对分割掩码进行填充，通常用于语义分割任务
    )
]

data_preprocessor = dict(
    # 数据预处理的类型，这里指定为目标检测数据预处理器，
    # 具体实现见RSPrompter-release\mmdet\models\data_preprocessors\data_preprocessor.py
    type='DetDataPreprocessor',
    # # 均值，用于图像归一化，这些值是针对每个颜色通道（R、G、B）的均值，乘以255是为了将范围从[0, 1]转换到[0, 255]
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    # 标准差，用于图像归一化，这些值是针对每个颜色通道（R、G、B）的标准差，乘以255是为了将范围从[0, 1]转换到[0, 255]
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    bgr_to_rgb=True, # 是否将图像从BGR格式转换为RGB格式
    pad_mask=True, # 是否对掩码进行填充
    pad_size_divisor=32, # 该参数用于指定填充后的图像尺寸必须是该值的倍数，通常用于保证尺寸能被32整除，方便网络计算
    batch_augments=batch_augments, # 额外的批次增强配置
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
        peft_config=dict(
            peft_type="LORA",
            r=16,
            target_modules=["qkv"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
        ),
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
    panoptic_head=dict(
        decoder_plus=True,
        mask_decoder=dict(
            hf_pretrain_name=hf_sam_pretrain_name,
            init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)
        ),
        per_pointset_point=prompt_shape[1],
        with_sincos=True,
        num_things_classes=num_classes,
        num_queries=prompt_shape[0],
        loss_cls=dict(
            class_weight=[1.0] * num_classes + [0.1])
    ),
    panoptic_fusion_head=dict(
        num_things_classes=num_classes
    ),
    test_cfg=dict(
        max_per_image=prompt_shape[0],
    )
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
resume = False
load_from = None

base_lr = 0.0001
max_epochs = 200

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