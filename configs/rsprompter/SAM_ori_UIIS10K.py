_base_ = ['_base_/rsprompter_anchor.py']
# # ------ 固定所有随机种子 ------
# seed = 42  # 可自定义
# deterministic = True
#
# # Python内置随机
# import random
# random.seed(seed)
#
# # Numpy随机
# import numpy as np
# np.random.seed(seed)
#
# # PyTorch随机
# import torch
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
#
# # CuDNN确定性模式
# torch.backends.cudnn.deterministic = deterministic
# torch.backends.cudnn.benchmark = not deterministic  # 关闭自动优化
#
# # 设置环境变量(重要!)
# import os
# os.environ['PYTHONHASHSEED'] = str(seed)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2, save_best='coco/segm_mAP', rule='greater',
                    save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='DetVisualizationHook', draw=True, interval=1, test_out_dir='vis_data')
)
# 添加自定义 Hook的配置
custom_hooks = [
    dict(type='EpochInfoHook', priority='NORMAL'),
]
crop_size = (512, 512)
# crop_size = (1024, 1024)
# crop_size = (480, 480)
# crop_size = (384, 384) # vit_l用
decoder_layers = 2

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend', init_kwargs=dict(project='rsprompter-nwpu', group='rsprompter-anchor',
                                                              name="rsprompter_anchor-nwpu-peft-512"))
                ]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

num_classes = 10
prompt_shape = (100, 5)  # (per img pointset, per pointset point)

#### should be changed when using different pretrain model

# sam base model
# hf_sam_pretrain_name = "/data2/yihan/MyProject/RSPrompter-release/sam-vit-base"
hf_sam_pretrain_name = "/data2/yihan/MyProject/RSPrompter-release/sam-vit-huge"
# hf_sam_pretrain_name = "/data3/yihan/MyProject/RSPrompter-release/sam-vit-base"
# hf_sam_pretrain_name = "/data3/yihan/MyProject/RSPrompter-release/sam-vit-large"
# hf_sam_pretrain_name = "/data3/yihan/MyProject/RSPrompter-release/sam-vit-huge"
# hf_sam_pretrain_name = "/root/shared-nvme/MyProject/RSPrompter-release/sam-vit-base"

# hf_sam_pretrain_ckpt_path = "/data2/yihan/MyProject/RSPrompter-release/sam-vit-base/pytorch_model.bin"
hf_sam_pretrain_ckpt_path = "/data2/yihan/MyProject/RSPrompter-release/sam-vit-huge/pytorch_model.bin"
# hf_sam_pretrain_ckpt_path = "/data3/yihan/MyProject/RSPrompter-release/sam-vit-base/pytorch_model.bin"
# hf_sam_pretrain_ckpt_path = "/data3/yihan/MyProject/RSPrompter-release/sam-vit-large/pytorch_model.bin"
# hf_sam_pretrain_ckpt_path = "/data3/yihan/MyProject/RSPrompter-release/sam-vit-huge/pytorch_model.bin"
# hf_sam_pretrain_ckpt_path = "/root/shared-nvme/MyProject/RSPrompter-release/sam-vit-base/pytorch_model.bin"


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
    type='DetDataPreprocessor', # 为目标检测任务设计的预处理模块，能够处理图像的归一化、颜色转换、图像尺寸调整等
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], # 指定图像的均值，用于模型输入的数据 x的归一化
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255], # 指定图像的标准差，用于模型输入的数据 x的归一化
    bgr_to_rgb=True, # 是否将图像的通道顺序从 BGR 转换为 RGB
    pad_mask=True, # 是否对掩膜（mask）进行填充，则掩膜会被填充至图像的目标尺寸（在 train_pipeline 或 batch_augments 中定义）
    pad_size_divisor=32, # 指定填充后的图像尺寸应当是 32 的倍数。
    batch_augments=batch_augments # 在数据批处理阶段应用的增强操作
)

model = dict(
    type='SAM_Anchor_Prompt',
    data_preprocessor=data_preprocessor,
    decoder_freeze=False,
    num_classes=num_classes,
    shared_image_embedding=dict(
        type='RSSamPositionalEmbedding',
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
    ),
    loss_proposal=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=20.0,
        reduction='mean',
        class_weight=[1.0] * num_classes + [0.1]
    ),
    num_queries=prompt_shape[0],
    backbone=dict(
        _delete_=True,
        img_size=crop_size[0],
        type='MMPretrainSamVisionEncoder', # 不使用DVT
        # type='MyPretrainEncoder_DVT', # 使用DVT
        # type='MyPretrainEncoder_Adapter', # 使用Adapter
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
        # peft_config=None, # 冻结backbone
        # peft_config={}, # 全量训练backbone
        peft_config=dict( # LORA微调backbone
            peft_type="LORA",
            r=16,
            target_modules=["qkv"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
        ),
    ),

    neck=dict(
        type='RSFPN',
        # feature_aggregator=dict(
        #     type='RSFeatureAggregator',
        #     in_channels=hf_sam_pretrain_name,
        #     out_channels=256,
        #     # select_layers=range(1, 13, 2),
        #     hidden_channels=32,
        #     select_layers=range(1, 24+1, 2),# vit-base: range(1, 13, 2), large: range(1, 25, 2), huge: range(1, 33, 2)
        # ), # 不用LORA微调，neck就用RSFeatureAggregator
        feature_aggregator=dict(
            _delete_=True,
            type='PseudoFeatureAggregator',
            in_channels=256,
            hidden_channels=512,
            out_channels=256,
        ), # 用LORA微调，neck就用PseudoFeatureAggregator
        feature_spliter=dict(
            type='RSSimpleFPN',
            backbone_channel=256,
            in_channels=[64, 128, 256, 256],
            out_channels=256,
            num_outs=5,
            norm_cfg=dict(type='LN2d', requires_grad=True)
        ),
    ),
    # prompt_encoder=dict(
    #     type='SAM_Prompt_Encoder',
    #     hf_pretrain_name=hf_sam_pretrain_name,
    #     init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)
    # ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='MyPrompterAnchorRoIPromptHead',
        # 叠加几层 mask decoder
        decoder_layers = decoder_layers,
        with_extra_pe=True,
        # with_extra_pe=False, # 2025-02-18如果没改成8通道，就试试不加 extra_pe
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor', # RoI Align操作，根据RoI区域在原图中的坐标，从特征图中框出7x7大小的特征矩阵
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        # 计算cls损失、bbox损失等
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            # type='RSPrompterAnchorMaskHead',
            type='MyMaskHead',
            # queries_num=prompt_shape[0],
            decoder_layers=decoder_layers,
            prompt_encoder=dict(
                type='SAM_Prompt_Encoder',
                hf_pretrain_name=hf_sam_pretrain_name,
                init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
                extra_config=dict(image_size=crop_size[0], image_embedding_size=crop_size[0]/16),
            ),
            mask_decoder=dict(
                type='SAM_Mask_Decoder',
                # type='SAM_Mask_Decoder_tmp', # 使用余弦相似度
                # type='New_Mask_Decoder', # 去掉了mask_tokens和iou_tokens
                hf_pretrain_name=hf_sam_pretrain_name,
                init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)),
            in_channels=256,
            roi_feat_size=14,
            per_pointset_point=prompt_shape[1],
            with_sincos=True,
            multimask_output=False,
            class_agnostic=True,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0
            ),
            loss_boundary=dict(
                type='LaplacianCrossEntropyLoss',
                kernel_size=7,
            ),
        )
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=crop_size,
            pos_weight=-1,
            debug=False)),
    # 测试配置（test_cfg）
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000, # NMS 前保留的候选框数量
            max_per_img=1000, # RPN 阶段保留的最大候选框数量
            nms=dict(type='nms', iou_threshold=0.7), # NMS 的 IOU 阈值
            min_bbox_size=0), # 用于过滤掉尺寸过小的候选框，min_bbox_size=0 表示不过滤任何候选框。
        rcnn=dict(
            # score_thr=0.05, # 置信度分数阈值，只有超过此阈值的框才会保留
            score_thr=0.05, # 置信度分数阈值，只有超过此阈值的框才会保留
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100, # 最终每张图像保留的最大检测框数量
            mask_thr_binary=0.5) # 这是分割任务中使用的掩码阈值，当掩码得分大于 0.5 时，将该像素分类为前景，否则为背景。
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

code_root = "/data2/yihan/MyProject/RSPrompter-release"

''' 换数据集，修改这里 '''
# dataset_type = 'LIACiInsSegDataset'
# dataset_type = 'UIISInsSegDataset'
# dataset_type = 'USISInsSegDataset'
dataset_type = 'UIIS10KDataset'

# data_root = "/data2/yihan/MyDataset/LIACi/LIACi_dataset_pretty/"
# data_root = "/data1/yihan/MyDataset/UIIS/UDW/"
# data_root = "/data2/yihan/MyDataset/USIS10K/"
data_root = "/data2/yihan/MyDataset/UIIS10K/"
# data_root = "/root/shared-nvme/MyDataset/LIACi_dataset_pretty/"


batch_size_per_gpu = 4
num_workers = 8
persistent_workers = True
# num_workers = 0 # 2025-02-24为了调试，设为 0
# persistent_workers = False # 2025-02-24为了调试，设为 False，注意如果num_workers=0，那么persistent_workers必须是False

# seed = 42  # 设置你想要的随机数种子
# deterministic = True  # 如果需要强制使用确定性算法（例如 CUDA 操作）

train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file="/data2/yihan/MyDataset/LIACi/LIACi_dataset_pretty/train_ann.json",
        # data_prefix=dict(img='images'),
        # ann_file="/data1/yihan/MyDataset/UIIS/UDW/annotations/train.json",
        # data_prefix=dict(img='train'),
        ann_file="/data2/yihan/MyDataset/UIIS10K/annotations/multiclass_train.json",
        data_prefix=dict(img='img'),
        # ann_file="/root/shared-nvme/MyDataset/LIACi_dataset_pretty/train_ann.json",
        # data_prefix=dict(img='image_gamma'),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    # num_workers=0, # 2025-02-24为了调试，设为 0
    # persistent_workers=False, # 2025-02-24为了调试，设为 False，注意如果num_workers=0，那么persistent_workers必须是False
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file="/data2/yihan/MyDataset/LIACi/LIACi_dataset_pretty/val_ann.json",
        # data_prefix=dict(img='images'),
        # data_prefix=dict(img='RGBD'),
        # ann_file="/data1/yihan/MyDataset/UIIS/UDW/annotations/val.json",
        # data_prefix=dict(img='val'),
        ann_file="/data2/yihan/MyDataset/UIIS10K/annotations/multiclass_test.json",
        data_prefix=dict(img='img'),
        # ann_file="/root/shared-nvme/MyDataset/LIACi_dataset_pretty/val_ann.json",
        # data_prefix=dict(img='image_gamma'),
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader
#test_dataloader = dict(
#    batch_size=batch_size_per_gpu,
#    num_workers=num_workers,
#    persistent_workers=persistent_workers,
#    dataset=dict(
#        type=dataset_type,
#        data_root=data_root,
#        ann_file="/data2/yihan/MyDataset/USIS10K/multi_class_annotations/multi_class_test_annotations.json",
#        data_prefix=dict(img='test'),
#        pipeline=test_pipeline,
#    )
#)

# resume = True
# load_from = "/root/shared-nvme/MyProject/RSPrompter-release/work_dirs/self-prompt-1024/epoch_87.pth"
resume = False
load_from = None

base_lr = 0.0002
# max_epochs = 200
max_epochs = 50

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
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
    # accumulative_counts=8
)
val_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args,
)
test_evaluator=val_evaluator

# 通过钩子实现：每个epoch跑完后，既在验证集上计算指标（验证），也在测试集上计算指标（测试）
#custom_hooks.append(
#    dict(
#        type='EvalTestHook',
#        dataloader=test_dataloader # 或你实际配置的 test dataloader
#    )
#)