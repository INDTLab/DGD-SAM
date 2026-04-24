import copy
import warnings
import einops
import numpy as np
import torch
from mmcv.cnn import build_norm_layer, ConvModule
from mmcv.ops import point_sample
from mmengine import ConfigDict
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from mmengine.model import BaseModel
from mmengine.structures import InstanceData
from peft import get_peft_config, get_peft_model
from torch import nn, Tensor
from transformers import SamConfig
from transformers.models.sam.modeling_sam import SamVisionEncoder, SamMaskDecoder, SamPositionalEmbedding, \
    SamPromptEncoder, SamModel, SamVisionEncoderOutput
from typing import List, TypeVar, Tuple, Optional, Dict, Union

from triton.language import tensor

from mmdet.models import MaskRCNN, StandardRoIHead, FCNMaskHead, SinePositionalEncoding, Mask2Former, Mask2FormerHead, \
    MaskFormerFusionHead, BaseDetector
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import unpack_gt_instances, empty_instances, multi_apply, \
    get_uncertain_point_coords_with_randomness
from mmdet.registry import MODELS
from mmdet.structures import SampleList, DetDataSample, OptSampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import OptConfigType, MultiConfig, ConfigType, InstanceList, reduce_mean
import torch.nn.functional as F
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.detectors.single_stage import SingleStageDetector
from typing import List, Tuple, Type

from mmpretrain.models import LayerNorm2d
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample

from .utils import *
from .twowaytransformer import *
import math
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

T = TypeVar('T')

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: number of masks
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: number of masks
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

@MODELS.register_module(force=True)
class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

@MODELS.register_module()
class RSFPN(BaseModule):
    def __init__(
            self,
            feature_aggregator=None,
            feature_spliter=None,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if feature_aggregator is not None:
            self.feature_aggregator = MODELS.build(feature_aggregator)
        if feature_spliter is not None:
            self.feature_spliter = MODELS.build(feature_spliter)

    def forward(self, inputs):
        if hasattr(self, 'feature_aggregator'):
            x = self.feature_aggregator(inputs)
        else:
            x = inputs
        if hasattr(self, 'feature_spliter'):
            x = self.feature_spliter(x)
        else:
            x = (x,)
        return x

@MODELS.register_module()
class PseudoFeatureAggregator(BaseModule):
    def __init__(
            self,
            in_channels,  # rsprompter_anchor_LIACi，256，固定的，不受图像分辨率影响
            hidden_channels=64,  # 512，固定
            out_channels=256,  # 256，固定
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        '''
        实际上就是直接经过三层卷积，对SAM的image encoder输出的特征图做融合操作，连残差链接都没有
        '''
        self.channel_fusion = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(hidden_channels, eps=1e-6),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(hidden_channels, eps=1e-6),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels, eps=1e-6),
        )

    def forward(self, inputs):
        assert len(inputs) == 1 # 要求inputs是个tuple
        x = inputs[0]
        x = self.channel_fusion(x)
        return x

@MODELS.register_module()
class RSSimpleFPN(BaseModule):
    #
    def __init__(self,
                 backbone_channel: int,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel  # 256
        self.in_channels = in_channels  # [64, 128, 256, 256]
        self.out_channels = out_channels  # 256
        self.num_ins = len(in_channels)  # 4
        self.num_outs = num_outs  # 5

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(self.backbone_channel // 2,
                               self.backbone_channel // 4, 2, 2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())  # 直接返回输入，不对输入进行任何改变
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):  # self.num_ins=4
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, input: Tensor) -> tuple:
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build FPN
        inputs = []
        inputs.append(self.fpn1(input))
        inputs.append(self.fpn2(input))
        inputs.append(self.fpn3(input))
        inputs.append(self.fpn4(input))

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)

@MODELS.register_module()
class MySam(SingleStageDetector):
    def __init__(
            self,
            backbone: ConfigType,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
            shared_image_embedding=None,
            neck: OptConfigType = None,
            prompt_encoder: OptConfigType = None,
            prompt_generater: OptConfigType = None,
            mask_decoder: OptConfigType = None,
            mask_head: OptConfigType = None,
            cls_head: OptConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        self.shared_image_embedding = MODELS.build(shared_image_embedding)
        self.data_preprocessor=MODELS.build(data_preprocessor)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)

        # prompt_encoder.update({'shared_patch_embedding':shared_image_embedding})
        self.prompt_encoder = MODELS.build(prompt_encoder)

        # fastinst生成 Na的方式
        # self.prompt_generater = MODELS.build(prompt_generater)
        self.mask_decoder = MODELS.build(mask_decoder)
        # self.mask_decoder = MaskDecoder(
        #     num_multimask_outputs=3,
        #     transformer=TwoWayTransformer(
        #         depth=2,
        #         embedding_dim=256,
        #         mlp_dim=2048,
        #         num_heads=8,
        #     ),
        #     transformer_dim=256,
        #     iou_head_depth=3,
        #     iou_head_hidden_dim=256,
        # )

        # self.mask_head = MODELS.build(mask_head)
        # self.cls_head = MODELS.build(cls_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.no_mask_embed = nn.Embedding(1, 256)
        torch.nn.init.kaiming_normal_(self.no_mask_embed.weight, nonlinearity='relu')
        self.Cls_Head = MLP(256, 256, 11, 3)

        self.num_classes = 10 # 针对self.loss_proposals的特定设置
        self.losses = ['labels', 'masks']
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = 0.1
        self.register_buffer("empty_weight", empty_weight)
        self.num_points = 12544
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        self.matcher = HungarianMatcher(
            cost_class=2.0,
            cost_mask=5.0,
            cost_dice=5.0,
            cost_location=1000.0,
            # num_points=12544, # 采样 12544个点，减少计算量，并加速匹配过程
            num_points=12544, # 采样 12544个点，减少计算量，并加速匹配过程
        )
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        image_embeddings = x.last_hidden_state
        image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_embeddings.shape[-1])

        x = self.neck(x)
        return x, image_embeddings, image_positional_embeddings

    def init_weights(self):
        pass

    # def forward(self, *args, **kwargs):
    #     return self.sam_model(*args, **kwargs)
    def get_image_wide_positional_embeddings(self, size):
        target_device = self.shared_image_embedding.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        ''' 删，下面两行二选一 '''
        # tmp = self.backbone(batch_inputs)
        x, image_embeddings, image_pe = self.extract_feat(batch_inputs)
        # sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #     input_points=None, # Optional[Tuple[torch.Tensor, torch.Tensor]],
        #     input_labels=None, # Optional[torch.Tensor],
        #     input_boxes=None, # Optional[torch.Tensor],
        #     input_masks=None, # Optional[torch.Tensor],
        # )
        query_features, query_pos_embeds, query_locations, proposal_cls_logits = self.prompt_encoder.get_query(x[2])
        check_nan_inf(query_features, 'query_features')
        check_nan_inf(query_pos_embeds, 'query_pos_embeds')
        check_nan_inf(query_locations, 'query_locations')
        check_nan_inf(proposal_cls_logits, 'proposal_cls_logits')
        '''
        query_features.shape = (b, 256, 100)
        query_locations.shape = (b, 100, 2)
        query_pos_embeds.shape = (b, 256, 100)
        proposal_cls_logits.shape = (b, 11, 512//16, 512//16)
        '''
        num_query_per_image = torch.full((x[2].shape[0],), query_features.shape[-1]).to(x[2].device) # tensor([100, 100])

        # image_embeddings = x[2]
        # image_pe = self.get_image_wide_positional_embeddings(size=x[2].shape[-1])

        image_embeddings = image_embeddings.repeat_interleave(num_query_per_image, dim=0)  # (16, 256, 32, 32)
        image_pe = image_pe.expand(image_embeddings.shape[0], -1, -1, -1)

        # 构造 dense_embeddings
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            image_embeddings.shape[0], -1, image_embeddings.shape[-2], image_embeddings.shape[-1]
        )
        check_nan_inf(dense_embeddings, 'dense_embeddings')
        # 构造 sparse_embeddings
        sparse_embeddings = query_features.permute(0, 2, 1) # (2, 100, 256)
        sparse_embeddings = sparse_embeddings.reshape(-1, 256) # (200, 256)
        ''' 对齐mask decoder的输入要求 '''
        sparse_embeddings = sparse_embeddings.unsqueeze(1).unsqueeze(1)  # (200, 1, 1, 256)
        ''' 可选：query加上位置信息 '''
        sparse_embeddings = torch.cat((sparse_embeddings, query_pos_embeds.permute(0, 2, 1).reshape(-1, 256).unsqueeze(1).unsqueeze(1)), dim=-2)

        if torch.isnan(sparse_embeddings).any() or torch.isinf(sparse_embeddings).any():
            print("sparse_embeddings has NaN or Inf values")
        # assert torch.isnan(sparse_embeddings).sum() == 0, 'sparse_embeddings has NaN values'

        query, mask_predictions, iou_predictions, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # if torch.isnan(mask_predictions).any() or torch.isinf(mask_predictions).any():
        #     print("mask_predictions has NaN or Inf values")
        # if torch.isnan(iou_predictions).any() or torch.isinf(iou_predictions).any():
        #     print("iou_predictions has NaN or Inf values")
        check_nan_inf(mask_predictions, 'mask_predictions')
        check_nan_inf(iou_predictions, 'iou_predictions')
        # assert torch.isnan(mask_predictions).sum() == 0, 'mask_predictions has NaN values'
        # assert torch.isnan(iou_predictions).sum() == 0, 'iou_predictions has NaN values'

        # 计算分类头损失
        # 1、组装targets
        targets = []
        batches = batch_inputs.shape[0]
        for b in range(batches):
            batch_data_sample = batch_data_samples[b]
            labels = batch_data_sample.gt_instances.labels
            masks = torch.from_numpy(batch_data_sample.gt_instances.masks.masks).to(labels.device)
            targets.append({
                "labels": labels,
                "masks": masks,
            })
        # 2、组装 outputs
        # query_features, query_pos_embeds, query_locations, proposal_cls_logits
        pred_logits = self.Cls_Head(query.squeeze(1))

        outputs={}
        outputs['proposal_cls_logits'] = proposal_cls_logits
        outputs['query_locations'] = query_locations
        mh, mw = mask_predictions.shape[-2:]
        outputs['pred_masks'] = mask_predictions.squeeze(1).squeeze(1).reshape(batches, -1, mh, mw)
        outputs['pred_logits'] = pred_logits.reshape(2, 100, 11)
        outputs['pred_matching_indices'] = None


        # 计算粗粒度语义分组头的损失
        proposal_loss_dict = {}
        if outputs.get("proposal_cls_logits") is not None:
            output_proposals = {"proposal_cls_logits": outputs.pop("proposal_cls_logits")}
            indices = self.matcher(output_proposals, targets)
            proposal_loss_dict = self.loss_proposals(output_proposals, targets, indices)

        # 计算cls、mask损失
        indices = self.matcher(outputs, targets)

        losses = {}
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # losses.update(self.prompt_generater.loss(batch_data_samples))
        # losses.update(self.mask_head.loss(batch_data_samples))
        # losses.update(self.cls_head.loss(batch_data_samples))
        losses.update(proposal_loss_dict)

        return losses

    def loss_proposals(self, output_proposals, targets, indices):
        assert "proposal_cls_logits" in output_proposals

        proposal_size = output_proposals["proposal_cls_logits"].shape[-2:]
        proposal_cls_logits = output_proposals["proposal_cls_logits"].flatten(2).float()  # b, c, hw

        target_classes = self.num_classes * torch.ones([proposal_cls_logits.shape[0],
                                                        proposal_size[0] * proposal_size[1]],
                                                       device=proposal_cls_logits.device)
        target_classes = target_classes.to(torch.int64)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        idx = self._get_src_permutation_idx(indices)
        target_classes[idx] = target_classes_o

        loss_proposal = F.cross_entropy(proposal_cls_logits, target_classes, ignore_index=-1)
        losses = {"loss_proposal": loss_proposal}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        '''
        根据 indices重排预测结果
        '''
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )

        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            # (self.backbone.vision_encoder.img_size, self.backbone.vision_encoder.img_size), # 1024x1024
            input_size,
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]] # 722x1024
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False) # 767x1088

        # 后处理必须加上这个，不然 masks不是正常的二值掩码，而是一堆数值
        masks = masks > 0
        return masks

    def predict(self,
                batch_inputs: Tensor, # (b, c, h, w) = (2, 3, 512, 512)
                batch_data_samples: SampleList, # list列表，长度 = batch，列表中每个元素都是DetDataSample类的实例，该实例其中一个属性就是gt_instances
                rescale: bool = True) -> SampleList:
        x, image_embeddings, image_pe = self.extract_feat(batch_inputs)

        query_features, query_pos_embeds, query_locations, proposal_cls_logits = self.prompt_encoder.get_query(x[2])

        num_query_per_image = torch.full((x[2].shape[0],), query_features.shape[-1]).to(
            x[2].device)  # tensor([100, 100])

        image_embeddings = image_embeddings.repeat_interleave(num_query_per_image, dim=0)  # (16, 256, 32, 32)
        image_pe = image_pe.expand(image_embeddings.shape[0], -1, -1, -1)

        # 构造 dense_embeddings
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            image_embeddings.shape[0], -1, image_embeddings.shape[-2], image_embeddings.shape[-1]
        )

        sparse_embeddings = query_features.permute(0, 2, 1)  # (2, 100, 256)
        sparse_embeddings = sparse_embeddings.reshape(-1, 256)  # (200, 256)
        sparse_embeddings = sparse_embeddings.unsqueeze(1).unsqueeze(1)  # (200, 1, 1, 256)
        sparse_embeddings = torch.cat(
            (sparse_embeddings, query_pos_embeds.permute(0, 2, 1).reshape(-1, 256).unsqueeze(1).unsqueeze(1)), dim=-2)

        query, mask_predictions, iou_predictions, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )


        iou_predictions = iou_predictions.reshape(2,-1, 1, 1)
        iou_predictions = iou_predictions.squeeze(2).squeeze(2)

        # 1、组装targets
        targets = []
        batches = batch_inputs.shape[0]
        for b in range(batches):
            batch_data_sample = batch_data_samples[b]
            labels = batch_data_sample.gt_instances.labels
            masks = torch.from_numpy(batch_data_sample.gt_instances.masks.masks).to(labels.device)
            targets.append({
                "labels": labels,
                "masks": masks,
            })
        # 2、组装 outputs
        # query_features, query_pos_embeds, query_locations, proposal_cls_logits
        pred_logits = self.Cls_Head(query.squeeze(1)) # (b, 100, 11)
        # pred_labels = torch.argmax(pred_logits, dim=-1) # (b, 100, 1)

        outputs = {}
        # outputs['proposal_cls_logits'] = proposal_cls_logits
        outputs['query_locations'] = query_locations
        mh, mw = mask_predictions.shape[-2:]
        outputs['pred_masks'] = mask_predictions.squeeze(1).squeeze(1).reshape(batches, -1, mh, mw)
        outputs['pred_logits'] = pred_logits.reshape(2, 100, 11)
        outputs['pred_matching_indices'] = None

        # 计算cls、mask损失
        indices = self.matcher(outputs, targets)

        # num_masks = sum(len(t["labels"]) for t in targets)
        # num_masks = torch.as_tensor(
        #     [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        # )

        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )

        target_classes[idx] = target_classes_o

        # 根据 indices，选出对应的 pred_masks、pred_logits、gt_masks、gt_labels
        pred_masks = []
        pred_logits = []
        pred_scores = []
        gt_masks = []
        gt_labels = []
        for i in range(batches):
            pred_masks.append(outputs["pred_masks"][i, indices[i][0], :, :])
            pred_logits.append(outputs["pred_logits"][i, indices[i][0], :])
            pred_scores.append(iou_predictions[i, indices[i][0]])
            gt_masks.append(targets[i]['masks'][indices[i][1], :, :])
            gt_labels.append(targets[i]['labels'][indices[i][1]])


        # 构造 batch_data_samples
        for i in range(batches):
            p_instances = InstanceData()
            p_instances.masks = pred_masks[i]
            # 对 masks上采样
            p_instances.masks = self.postprocess_masks(p_instances.masks.unsqueeze(1),batch_data_samples[i].img_shape, batch_data_samples[i].ori_shape)
            p_instances.masks = p_instances.masks.squeeze(1)

            p_instances.labels = torch.argmax(pred_logits[i][:, 0:10], dim=-1)
            p_instances.scores = pred_scores[i]
            p_instances.bboxes = self.get_bboxes_from_masks(p_instances.masks)
            batch_data_samples[i].pred_instances = p_instances


        return batch_data_samples

    def get_bboxes_from_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        根据给定的二值掩码，返回每个实例的边界框。
        参数: masks (torch.Tensor): 形状为 (n, H, W) 的二值掩码张量，其中 n 是实例数，H 和 W 是掩码的高和宽。
        返回: bboxes (torch.Tensor): 形状为 (n, 4) 的边界框张量，每一行表示一个实例的 [x_min, y_min, x_max, y_max]。
        """
        n, H, W = masks.shape
        bboxes = []

        # 遍历每个实例掩码
        for i in range(n):
            # 获取第 i 个实例掩码
            mask = masks[i]

            # 获取所有非零像素的坐标 (y, x)
            non_zero_coords = torch.nonzero(mask)

            if non_zero_coords.size(0) == 0:
                # 如果没有找到任何非零像素，表示该实例为空掩码，返回默认边界框
                bboxes.append(torch.tensor([0, 0, 0, 0]))
            else:
                # 获取非零像素的最小和最大坐标
                y_min = non_zero_coords[:, 0].min().item()
                x_min = non_zero_coords[:, 1].min().item()
                y_max = non_zero_coords[:, 0].max().item()
                x_max = non_zero_coords[:, 1].max().item()
                bboxes.append(torch.tensor([x_min, y_min, x_max, y_max]))
        if len(bboxes) > 0:
            return torch.stack(bboxes)
        else:
            return torch.tensor([])



@MODELS.register_module()
class MyPositionalEmbedding(SamPositionalEmbedding, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.shared_image_embedding = SamPositionalEmbedding(sam_config)

    def forward(self, *args, **kwargs):
        return self.shared_image_embedding(*args, **kwargs)


@MODELS.register_module()
class MyVisionEncoder(BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            peft_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        vision_encoder = SamVisionEncoder(sam_config)

        pretrained_dict = torch.load("/data2/yihan/MyProject/RSPrompter-release/sam-vit-base/pytorch_model.bin")
        model_dict = vision_encoder.state_dict()

        pretrained_dict = {k.replace('vision_encoder.',''):v for k,v in pretrained_dict.items()
                           if 'vision_encoder.' in k}
        sum = 0
        dif=[]
        for k,v in pretrained_dict.items():
            if k in model_dict and v.shape != model_dict[k].shape:
                if k=='pos_embed':
                    pos_embed_resized = F.interpolate(v.permute(0, 3, 1, 2),size=(model_dict[k].shape[1], model_dict[k].shape[2]),mode='bilinear', align_corners=False)
                    pretrained_dict[k] = pos_embed_resized.permute(0, 2, 3, 1)
                else:
                    rel_pos_resized = F.interpolate(v.unsqueeze(0).unsqueeze(0),size=(model_dict[k].shape[0],model_dict[k].shape[1]),mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    pretrained_dict[k] = rel_pos_resized

            if k in model_dict and v.shape == model_dict[k].shape:
                sum+=1

        # assert sum==model_dict.__len__(), f'vision encoder预训练参数未完全加载成功，进度{sum}/{model_dict.__len__()}'
        print(f'vision encoder预训练参数加载进度{sum}/{model_dict.__len__()}')
        model_dict.update(pretrained_dict)
        vision_encoder.load_state_dict(model_dict, strict=False)
        self.vision_encoder = vision_encoder
        self.vision_encoder.is_init = True

    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')

    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)

@MODELS.register_module()
class MyPromptEncoder(SamPromptEncoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            shared_patch_embedding=None,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)

        # 下面这种方式得到的是Config类的实例，而不是一个dict，因此不能用 MODELS.build()方法构造实例，
        # 而是需要用hugging face的transformers 中的sam模块导入的 SamVisionEncoder, SamMaskDecoder,
        # SamPositionalEmbedding等这些类，来根据Config类的实例，构造SamVisionEncoder自己的实例
        # prompt_encoder_config = SamConfig.from_pretrained(hf_pretrain_name).prompt_encoder_config
        # positionEmbedding_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        # if extra_config is not None:
        #     prompt_encoder_config.update(extra_config)
        # # self.shared_patch_embedding = MODELS.build(shared_patch_embedding)
        # self.shared_patch_embedding = SamPositionalEmbedding(positionEmbedding_config)
        # self.prompt_encoder = SamPromptEncoder(prompt_encoder_config, shared_patch_embedding=self.shared_patch_embedding)



        self.hidden_dim, self.num_queries, self.num_classes = 256, 100, 10
        self.query_proposal = QueryProposal(self.hidden_dim, self.num_queries, self.num_classes)

        meta_pos_size = int(round(math.sqrt(self.num_queries)))
        self.meta_pos_embed = nn.Parameter(torch.empty(1, self.hidden_dim, meta_pos_size, meta_pos_size))



    def forward(self, *args, **kwargs):
        # return self.prompt_encoder(*args, **kwargs)
        return self.query_proposal(*args, **kwargs)

    def get_query(self, x):
        proposal_pos_embeds = F.interpolate(self.meta_pos_embed, size=512 // 16,
                                            mode="bilinear", align_corners=False) # 用于产生 query 的特征图尺寸是多少，第二个参数size就是多少
        query_features, query_pos_embeds, query_locations, proposal_cls_logits = self.query_proposal(
            x, proposal_pos_embeds
        )
        return query_features, query_pos_embeds, query_locations, proposal_cls_logits


@MODELS.register_module()
class MyMaskDecoder(SamMaskDecoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).mask_decoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        SamMaskDecoder.__init__(self, sam_config)
        # self.mask_decoder = SamMaskDecoder(sam_config)

    def forward(self, *args, **kwargs):
        # return self.mask_decoder(*args, **kwargs)
        image_embeddings = kwargs['image_embeddings']
        sparse_prompt_embeddings = kwargs['sparse_prompt_embeddings']
        image_pe = kwargs['image_positional_embeddings']
        dense_prompt_embeddings = kwargs['dense_prompt_embeddings']
        multimask_output = kwargs['multimask_output']
        output_attentions = kwargs.get('output_attentions', None)


        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_pe.repeat_interleave(point_batch_size, 0)

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, attentions = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=None,
            target_embedding=None,
            output_attentions=None,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1: (1 + self.num_mask_tokens), :]

        query = point_embedding[:, :, (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        outputs = (masks, iou_pred)

        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)

        outputs = (query,) + outputs
        return outputs


    # def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> dict:



@MODELS.register_module()
class Classifier(BaseModule):
    def __init__(self, in_features: int, num_classes: int):
        BaseModule.__init__(self)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, *args, **kwargs):
        cl = nn.Sequential(

        )
        return self.mask_decoder(*args, **kwargs)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.init_weight()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    def init_weight(self):
        for layer in self.layers:
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Prepare output
        return masks, iou_pred

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)  # (1, 256, 64, 64)

        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

def check_nan_inf(C: torch.Tensor, name: str):
    if torch.isnan(C).any() or torch.isinf(C).any():
        print(f"{name} has NaN or Inf values")