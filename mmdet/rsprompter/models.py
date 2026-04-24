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
from mmengine.structures import InstanceData
from peft import get_peft_config, get_peft_model
from torch import nn, Tensor
from torch.nn.init import sparse
from transformers import SamConfig
from transformers.models.sam.modeling_sam import SamVisionEncoder, SamMaskDecoder, SamPositionalEmbedding, \
    SamPromptEncoder, SamModel, SamVisionEncoderOutput
from typing import List, TypeVar, Tuple, Optional, Dict, Union
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

from mmpretrain.models import LayerNorm2d
import math
# from .utils import *
import matplotlib.pyplot as plt
import cv2

T = TypeVar('T')


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
class RSPrompterAnchor(MaskRCNN):
    def __init__(
            self,
            shared_image_embedding,
            decoder_freeze=True,
            *args,
            **kwargs):
        peft_config = kwargs.get('backbone', {}).get('peft_config', {})
        super().__init__(*args, **kwargs)
        self.shared_image_embedding = MODELS.build(shared_image_embedding)
        self.decoder_freeze = decoder_freeze

        '''
        不用的时候删除下面三行代码
        '''
        # from transformers import Swinv2Model
        # self.backbone = Swinv2Model.from_pretrained("/data2/yihan/MyProject/RSPrompter-release/swinv2-tiny-patch4-window16-256/")
        # self.from1024to256 = nn.Sequential(
        #     nn.Conv2d(
        #         768,
        #         256,
        #         kernel_size=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(256, eps=1e-6),
        #     nn.Conv2d(
        #         256,
        #         256,
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(256, eps=1e-6),
        # )

        ''' 删除 '''
        # # 自定义初始化函数
        # import torch.nn.init as init
        # def init_weights(m):
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         init.xavier_uniform_(m.weight)  # 使用 Xavier 初始化权重
        #         if m.bias is not None:
        #             init.zeros_(m.bias)  # 初始化偏置为零
        # # 对 Sequential 中的所有层应用初始化函数
        # self.from1024to256.apply(init_weights)

        self.frozen_modules = []
        # 真正使用 Lora是在 backbone中定义的
        if peft_config is None:  # 如果不使用 LORA微调，就冻结backbone
            self.frozen_modules += [self.backbone]
        if self.decoder_freeze:
            self.frozen_modules += [
                self.shared_image_embedding,
                self.roi_head.mask_head.mask_decoder,
                self.roi_head.mask_head.no_mask_embed,
            ]
        self._set_grad_false(self.frozen_modules)

    def _set_grad_false(self, module_list=[]):
        for module in module_list:
            module.eval()
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False

    def get_image_wide_positional_embeddings(self, size):
        target_device = self.shared_image_embedding.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        vision_outputs = self.backbone(batch_inputs)

        ''' 删掉下面这行 '''
        # vision_outputs = (vision_outputs['last_hidden_state'].reshape(int(batch_inputs.shape[0]), int(batch_inputs.shape[2]/32), int(batch_inputs.shape[3]/32), -1).permute(0, 3, 1, 2), )
        # vision_outputs = (vision_outputs, )


        if isinstance(vision_outputs, SamVisionEncoderOutput):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs[1]
        elif isinstance(vision_outputs, tuple):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs
        else:
            raise NotImplementedError

        ''' 删掉下面这行 '''
        # image_embeddings = self.from1024to256(image_embeddings)

        image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_embeddings.shape[-1])
        # repeat with batch size
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        x = self.neck(vision_hidden_states)

        return x, image_embeddings, image_positional_embeddings

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)

        losses = dict()
        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                          self.test_cfg.rpn)
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        # set cat_id of gt_labels to 0 in RPN
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)

        # loss_and_predict(): 依次调用self.rpn_head的forward() -> loss_by_feat() -> predict_by_feat()
        rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
            x, rpn_data_samples, proposal_cfg=proposal_cfg)
        # avoid get same name with roi_head loss
        keys = rpn_losses.keys()
        for key in list(keys):
            if 'loss' in key and 'rpn' not in key:
                rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        losses.update(rpn_losses)


        roi_losses = self.roi_head.loss(
            x, # tuple_5，5个不同尺度的特征图，H/4 ~ H/64，每个尺寸都是(batch=2，dim=256，H，W)
            rpn_results_list, # list_2，batch中2个图像的proposal，每个图像都有1K个proposal，包括1K个bboxes、labels、scores
            batch_data_samples, # list_2，batch中2个图像的实例标注GT信息
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor, # (b, c, h, w) = (2, 3, 512, 512)
                batch_data_samples: SampleList, # list列表，长度 = batch，列表中每个元素都是 DetDataSample 类的实例，该实例其中一个属性就是 gt_instances
                rescale: bool = True) -> SampleList:
        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)

        # plt.imshow(F.interpolate(x[2][:,:,:20,:], size=(380, 640), mode='bilinear', align_corners=False)[0, :, :, :].mean(0).detach().cpu().numpy(), cmap='jet')
        # plt.savefig(f"/data2/yihan/tmp/{batch_data_samples[0].get('img_path').split('/')[-1].split('.')[0]}.png", bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.close()

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        ) # results_list 是一个列表，表中元素是 InstanceData 类的实例，包含bboxes、labels、masks(原图尺寸origin_size)、scores
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list) # 给 DetDataSample 实例添加了一个 pred_instances属性
        return batch_data_samples


@MODELS.register_module()
class RSPrompterQuery(Mask2Former):
    def __init__(
            self,
            shared_image_embedding,
            decoder_freeze=True,
            *args,
            **kwargs):
        peft_config = kwargs.get('backbone', {}).get('peft_config', {})
        super().__init__(*args, **kwargs)
        self.decoder_freeze = decoder_freeze
        self.with_mask2formerhead = False if isinstance(self.panoptic_head, RSMask2FormerHead) else True
        self.shared_image_embedding = MODELS.build(shared_image_embedding)

        # model_dict = self.backbone.state_dict()
        # checkpoint = "/data2/yihan/MyProject/RSPrompter-release/sam-vit-base/pytorch_model.bin"
        # pretrained_dict = torch.load(checkpoint)
        # pretrained_dict = {k.replace('sem_seg_head.predictor.query_proposal.', ''): v for k, v in
        #                    pretrained_dict.items() if 'query_proposal' in k}
        #
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
        #                    k in model_dict and v.shape == model_dict[k].shape}
        # model_dict.update(pretrained_dict)
        # self.backbone.load_state_dict(model_dict)

        self.frozen_modules = []
        if peft_config is None:
            self.frozen_modules += [self.backbone]
        if self.decoder_freeze:
            self.frozen_modules += [
                self.shared_image_embedding,
                self.panoptic_head.mask_decoder,
            ]
        self._set_grad_false(self.frozen_modules)

    def _set_grad_false(self, module_list=[]):
        for module in module_list:
            module.eval()
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False

    def get_image_wide_positional_embeddings(self, size):
        target_device = self.shared_image_embedding.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        vision_outputs = self.backbone(batch_inputs)
        if isinstance(vision_outputs, SamVisionEncoderOutput):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs[1]
        elif isinstance(vision_outputs, tuple):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs
        else:
            raise NotImplementedError

        image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_embeddings.shape[-1])
        # repeat with batch size
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        x = self.neck(vision_hidden_states)
        return x, image_embeddings, image_positional_embeddings

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:

        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)

        if self.with_mask2formerhead:
            losses = self.panoptic_head.loss(x, batch_data_samples)
        else:
            losses = self.panoptic_head.loss(x, batch_data_samples,
                                             image_embeddings=image_embeddings,
                                             image_positional_embeddings=image_positional_embeddings)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:

        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)

        if self.with_mask2formerhead:
            mask_cls_results, mask_pred_results = self.panoptic_head.predict(x, batch_data_samples)
        else:
            mask_cls_results, mask_pred_results = self.panoptic_head.predict(
                x, batch_data_samples,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings
            )

        results_list = self.panoptic_fusion_head.predict(
            mask_cls_results,
            mask_pred_results,
            batch_data_samples,
            rescale=rescale)
        results = self.add_pred_to_datasample(batch_data_samples, results_list)

        return results

@MODELS.register_module()
class MySamMaskDecoder_Fastinst(SamMaskDecoder, BaseModule):
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

        ''' yihan 原版'''
        '''
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)
        '''
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        ''' yihan 原版 '''
        '''
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        '''
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

@MODELS.register_module()
class RSMask2FormerHead(Mask2FormerHead, BaseModule):
    def __init__(
            self,
            mask_decoder,
            decoder_plus,
            with_sincos=True,
            per_pointset_point=1,
            multimask_output=False,
            attention_similarity=None,
            target_embedding=None,
            output_attentions=None,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_plus = decoder_plus
        self.multimask_output = multimask_output
        self.attention_similarity = attention_similarity
        self.target_embedding = target_embedding
        self.output_attentions = output_attentions

        self.mask_decoder = MODELS.build(mask_decoder)

        prompt_encoder = dict(
            type='RSSamPromptEncoder',
            hf_pretrain_name=copy.deepcopy(mask_decoder.get('hf_pretrain_name')),
            init_cfg=copy.deepcopy(mask_decoder.get('init_cfg')),
        )
        prompt_encoder = MODELS.build(prompt_encoder)
        prompt_encoder.init_weights()
        if self.decoder_plus:
            self.sam_mask_embed = prompt_encoder.prompt_encoder.mask_embed
        else:
            self.no_mask_embed = prompt_encoder.prompt_encoder.no_mask_embed
            del self.mask_embed

        self.per_pointset_point = per_pointset_point
        self.with_sincos = with_sincos

        self.feat_channels = kwargs['feat_channels']
        self.out_channels = kwargs['out_channels']
        if with_sincos:
            num_sincos = 2
        else:
            num_sincos = 1
        self.point_emb = nn.Sequential(
            nn.Linear(self.feat_channels, self.feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels // 2, self.feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels // 2, self.out_channels * num_sincos * per_pointset_point)
        )
        del self.cls_embed
        self.cls_embed = nn.Sequential(
            nn.Linear(self.feat_channels, self.feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels, self.num_classes + 1))

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int],
                      image_embeddings=None,  # (2, 256, 16, 16)
                      image_positional_embeddings=None  # (2, 256, 16, 16)
                      ) -> Tuple[Tensor]:
        img_bs = image_embeddings.shape[0]  # 2
        image_embedding_size = image_embeddings.shape[-2:]

        decoder_out = self.transformer_decoder.post_norm(decoder_out)  # (2, 70, 128)
        # shape (batch_size, num_queries, c)
        cls_pred = self.cls_embed(decoder_out)  # (2, 70, 11)
        # shape (batch_size, num_queries, c)
        point_embedings = self.point_emb(decoder_out)  # (2, 70, 2560)

        point_embedings = einops.rearrange(point_embedings, 'b n_set (n_point c) -> b n_set n_point c',
                                           n_point=self.per_pointset_point)  # (2, 70, 5, 512)
        if self.with_sincos:
            point_embedings = torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2]  # (2, 70, 5, 256)

        # B, N_set, N_point, C => (B, N_set), 1, N_point, C
        sparse_embeddings = einops.rearrange(point_embedings,
                                             'b n_set n_point c -> (b n_set) n_point c')  # (140, 5, 256)
        sparse_embeddings = sparse_embeddings.unsqueeze(1)  # (140, 1, 5, 256)

        if self.decoder_plus:
            # shape (num_queries, batch_size, h, w)
            mask_embed = self.mask_embed(decoder_out)  # (2, 70, 256)
            mask_pred_plus = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)  # (2, 70, 64, 64)

            input_masks = mask_pred_plus.detach()  # (2, 70, 64, 64)
            input_masks = einops.repeat(input_masks, 'b n h w -> (b n) c h w', c=1)  # (140, 1, 64, 64)
            # (bs num_q) c h w
            dense_embeddings = self.sam_mask_embed(input_masks)  # (140, 256, 256/16=16, 16)
        else:
            mask_pred_plus = None
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(img_bs, -1,
                                                                                     image_embedding_size[0],
                                                                                     image_embedding_size[1])

        image_embeddings = torch.repeat_interleave(image_embeddings, repeats=self.num_queries,
                                                   dim=0)  # (140, 256, 16, 16)
        image_positional_embeddings = torch.repeat_interleave(image_positional_embeddings, repeats=self.num_queries,
                                                              dim=0)  # (140, 256, 16, 16)
        mask_pred, iou_predictions, mask_dencoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,  # (2*70=140, 256, 16, 16)
            image_positional_embeddings=image_positional_embeddings,  # (140, 256, 16, 16)
            sparse_prompt_embeddings=sparse_embeddings,  # (140, 1, 5, 256)，因为配置文件中 prompt_shape = (70, 5)
            dense_prompt_embeddings=dense_embeddings,  # (140, 256, 16, 16)
            multimask_output=self.multimask_output,  # False
            attention_similarity=self.attention_similarity,  # None
            target_embedding=self.target_embedding,  # None
            output_attentions=self.output_attentions,  # None
        )  # mask_pred=(140, 1, 1, 16*4=64, 64), iou_predictions=(140, 1, 1), mask_dencoder_attentions=None
        mask_pred = mask_pred.reshape(img_bs, -1, *mask_pred.shape[-2:])  # (2, 70, 64, 64)
        if not self.decoder_plus:  # 跳过
            h, w = mask_pred.shape[-2:]
            # shape (batch_size, num_queries, h, w)
            attn_mask_pred = mask_pred.reshape(img_bs, -1, h, w)
        else:  # 执行这里
            attn_mask_pred = mask_pred_plus  # (2, 70, 64, 64)
        attn_mask = F.interpolate(attn_mask_pred, attn_mask_target_size, mode='bilinear',
                                  align_corners=False)  # (2, 70, 4, 4)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()
        return cls_pred, mask_pred, attn_mask, mask_pred_plus  # (2, 70, 11), (2, 70, 64, 64), (16, 70, 4*4=16), (2, 70, 64, 64)

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList,
                image_embeddings=None,
                image_positional_embeddings=None
                ) -> Tuple[List[Tensor]]:
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(
            x)  # mask_features.shape=(2, 256, 64, 64), multi_scale_memorys是个list，包含三个形状的张量(2,128,4,4)、(2,128,8,8)、(2,128,16,16)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])  # (2, 128, 4, 4)
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)  # (2, 16, 128)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)  # (1, 1, 128)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)  # (2, 4, 4)
            decoder_positional_encoding = self.decoder_positional_encoding(mask).to(
                decoder_input.dtype)  # (2, 128, 4, 4)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)  # (2, 16, 128)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))  # (2, 70, 128)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))  # (2, 70, 128)

        cls_pred_list = []
        mask_pred_list = []
        mask_pred_plus_list = []
        attn_mask = None

        cls_pred, mask_pred, attn_mask, mask_pred_plus = self._forward_head(query_feat, mask_features,
                                                                            multi_scale_memorys[0].shape[-2:],
                                                                            image_embeddings,
                                                                            image_positional_embeddings)
        cls_pred_list.append(cls_pred)  # len=1
        mask_pred_list.append(mask_pred)  # len=1
        mask_pred_plus_list.append(mask_pred_plus)  # len=1

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            if attn_mask is not None:
                # if a mask is all True(all background), then set it all False.
                mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
                attn_mask = attn_mask & mask_sum

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask, mask_pred_plus = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
                image_embeddings, image_positional_embeddings)

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            mask_pred_plus_list.append(mask_pred_plus)
        return cls_pred_list, mask_pred_list, mask_pred_plus_list  # 三个list的len都是7

    def loss(
            self,
            x: Tuple[Tensor],
            batch_data_samples: SampleList,
            image_embeddings=None,  # (b, c, h, w)
            image_positional_embeddings=None  # 和image_embedding形状相同
    ) -> Dict[str, Tensor]:
        """Perform forward propagation and loss calculation of the panoptic
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_img_metas = []
        batch_gt_instances = []
        batch_gt_semantic_segs = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if 'gt_sem_seg' in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
            else:
                batch_gt_semantic_segs.append(None)

        # forward
        all_cls_scores, all_mask_preds, all_mask_preds_plus = self(x, batch_data_samples, image_embeddings,
                                                                   image_positional_embeddings)
        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(batch_gt_instances,
                                                batch_gt_semantic_segs)
        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds, all_mask_preds_plus,
                                   batch_gt_instances, batch_img_metas)
        return losses

    def loss_by_feat(self,
                     all_cls_scores: Tensor,
                     all_mask_preds: Tensor,
                     all_mask_preds_plus,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice, losses_mask_plus, losses_dice_plus = multi_apply(
            self._loss_by_feat_single,
            all_cls_scores, all_mask_preds,
            all_mask_preds_plus,
            batch_gt_instances_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        loss_dict['loss_mask_plus'] = losses_mask_plus[-1]
        loss_dict['loss_dice_plus'] = losses_dice_plus[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_mask_plus_i, loss_dice_plus_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], losses_mask_plus[:-1], losses_dice_plus[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            loss_dict[f'd{num_dec_layer}.loss_mask_plus'] = loss_mask_plus_i
            loss_dict[f'd{num_dec_layer}.loss_dice_plus'] = loss_dice_plus_i

            num_dec_layer += 1
        return loss_dict

    def _loss_by_feat_single(self,
                             cls_scores: Tensor,
                             mask_preds: Tensor,
                             mask_preds_plus,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        mask_preds_plus_list = [mask_preds_plus[i] for i in range(num_imgs)]

        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_plus_list,
                                        batch_gt_instances, batch_img_metas)

        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]
        mask_preds_plus = mask_preds_plus[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            loss_dice_plus = mask_preds_plus.sum()
            loss_mask_plus = mask_preds_plus.sum()
            return loss_cls, loss_mask, loss_dice, loss_mask_plus, loss_dice_plus

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # points_coords = points_coords.to(mask_preds.dtype)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).to(mask_preds.dtype), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)
        mask_point_preds_plus = point_sample(
            mask_preds_plus.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)
        loss_dice_plus = self.loss_dice(
            mask_point_preds_plus, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)

        mask_point_preds_plus = mask_point_preds_plus.reshape(-1)

        # loss_mask = self.loss_mask(
        #     mask_point_preds,
        #     mask_point_targets,
        #     avg_factor=num_total_masks * self.num_points)
        # to avoid nan in fp16 when num_total_masks * self.num_points
        loss_mask = self.loss_mask(mask_point_preds, mask_point_targets)
        loss_mask_plus = self.loss_mask(mask_point_preds_plus, mask_point_targets)
        return loss_cls, loss_mask, loss_dice, loss_mask_plus, loss_dice_plus

    def predict(self, x: Tuple[Tensor],
                batch_data_samples: SampleList,
                image_embeddings=None,
                image_positional_embeddings=None
                ) -> Tuple[Tensor]:
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        all_cls_scores, all_mask_preds, all_mask_preds_plus = self(
            x, batch_data_samples, image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        mask_pred_plus_results = all_mask_preds_plus[-1]
        # upsample masks
        try:
            img_shape = batch_img_metas[0]['batch_input_shape']
        except:
            img_shape = batch_img_metas[0]['pad_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        return mask_cls_results, mask_pred_results


# @MODELS.register_module()
# class MyMask2FormerHead(RSMask2FormerHead):
#     def __init__(
#             self,
#             loss_proposal: ConfigType,
#             mask_decoder,
#             *args,
#             **kwargs):
#         super().__init__(mask_decoder=mask_decoder,*args, **kwargs)
#
#         self.cls_embed = nn.Sequential(
#             nn.Linear(self.feat_channels, self.feat_channels),
#             nn.ReLU(inplace=True),
#             # nn.Linear(self.feat_channels, self.feat_channels),
#             # nn.ReLU(inplace=True),
#             nn.Linear(self.feat_channels, self.num_classes + 1)
#         )
#         # nn.init.xavier_uniform_(self.cls_embed[0].weight)
#         # nn.init.zeros_(self.cls_embed[0].bias)
#         # nn.init.xavier_uniform_(self.cls_embed[2].weight)
#         # nn.init.zeros_(self.cls_embed[2].bias)
#         ''' 删 '''
#         prompt_encoder_config = SamConfig.from_pretrained(copy.deepcopy(mask_decoder.get('hf_pretrain_name'))).prompt_encoder_config
#         positionEmbedding_config = SamConfig.from_pretrained(copy.deepcopy(mask_decoder.get('hf_pretrain_name'))).vision_config
#         # if extra_config is not None:
#         #     prompt_encoder_config.update(extra_config)
#         self.shared_patch_embedding = SamPositionalEmbedding(positionEmbedding_config)
#         self.prompt_encoder = SamPromptEncoder(prompt_encoder_config, shared_patch_embedding=self.shared_patch_embedding)
#         # self.pos_point = nn.Embedding(1, self.feat_channels)
#         meta_pos_size = int(round(math.sqrt(self.num_queries)))
#         self.meta_pos_embed = nn.Parameter(torch.empty(1, 256, meta_pos_size, meta_pos_size))
#         nn.init.xavier_uniform_(self.meta_pos_embed)  # 进行 Xavier 初始化
#
#         self.matcher = HungarianMatcher(
#             cost_class=2.0,
#             cost_mask=5.0,
#             cost_dice=5.0,
#             cost_location=1000.0,
#             num_points=12544,  # 采样 12544个点，减少计算量，并加速匹配过程
#         )
#         self.loss_proposal = MODELS.build(loss_proposal)
#
#         # self.query_proposal = QueryProposal(256, 70, 10)
#         self.query_proposal = QueryProposal(256, kwargs.get('num_queries'), 10)
#
#         ''' 删 '''
#         # 初始化 self.query_proposal
#         # model_dict = self.query_proposal.state_dict()
#         # checkpoint = "/data1/yihan/MyProject/FastInst-main/output/fastinst_r50_ppm-fpn_bs16_50ep_x3_640/model_0134999.pth"
#         # pretrained_dict = torch.load(checkpoint)['model']
#         # pretrained_dict = {k.replace('sem_seg_head.predictor.query_proposal.', ''): v for k, v in pretrained_dict.items() if 'query_proposal' in k}
#         #
#         # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
#         #                    k in model_dict and v.shape == model_dict[k].shape}
#         # model_dict.update(pretrained_dict)
#         # self.query_proposal.load_state_dict(model_dict)
#
#         ''' 删 '''
#         # model_dict = self.mask_decoder.state_dict()
#         # checkpoint = "/data2/yihan/MyProject/RSPrompter-release/sam-vit-base/pytorch_model.bin"
#         # pretrained_dict = torch.load(checkpoint)
#         # pretrained_dict = {k.replace('mask_decoder.', ''): v for k, v in
#         #                    pretrained_dict.items() if 'mask_decoder' in k}
#         #
#         # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
#         #                    k in model_dict and v.shape == model_dict[k].shape}
#         # model_dict.update(pretrained_dict)
#         # self.mask_decoder.load_state_dict(model_dict)
#
#
#     def _forward_head(self, query: Tensor, # (2, 70, 256)
#                       query_locations=None, # (2, 70, 2)
#                       image_embeddings=None, # (2, 256, 16, 16)
#                       image_positional_embeddings=None # (2, 256, 16, 16)
#                       ) -> Tuple[Tensor]:
#         img_bs = image_embeddings.shape[0] # 2
#         query_locations = query_locations.reshape(-1, 2).unsqueeze(1).unsqueeze(1) # (140, 1, 1, 2), input_points.shape=(batch_size, num_points, 2)
#         point_labels = torch.ones((img_bs*self.num_queries, 1, 1)).to(query_locations.device) # (140, 1, 1), input_labels.shape=(batch_size, point_batch_size, num_points)
#         sparse_embeddings_tmp, _ = self.prompt_encoder(query_locations, point_labels, None, None) # (140, 1, 2, 256)
#         image_embedding_size = image_embeddings.shape[-2:]
#
#         cls_pred = self.cls_embed(query)  # (2, 70, 11)
#
#         query = query.reshape(-1, 256).unsqueeze(1).unsqueeze(1) # (140, 1, 1, 256)
#         # decoder_out = self.transformer_decoder.post_norm(decoder_out) # (2, 70, 128)
#
#         # cls_pred = self.cls_embed(query) # (2, 70, 11)
#
#         '''
#         point_embedings = self.point_emb(decoder_out) # (2, 70, 2560)
#
#         point_embedings = einops.rearrange(point_embedings, 'b n_set (n_point c) -> b n_set n_point c',
#                                            n_point=self.per_pointset_point) # (2, 70, 5, 512)
#         if self.with_sincos:
#             point_embedings = torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2] # (2, 70, 5, 256)
#
#         # B, N_set, N_point, C => (B, N_set), 1, N_point, C
#         sparse_embeddings = einops.rearrange(point_embedings, 'b n_set n_point c -> (b n_set) n_point c') # (140, 5, 256)
#         sparse_embeddings = sparse_embeddings.unsqueeze(1)  # (140, 1, 5, 256)
#         '''
#         # point_embedings = einops.rearrange(point_embedings, 'b n_set n_point c -> (b n_set) n_point c') # (140, 5, 256)
#         # point_embedings = point_embedings.unsqueeze(1) # (140, 1, 5, 256)
#
#         # query = decoder_out.reshape(-1, 256).unsqueeze(1).unsqueeze(1) # (140, 1, 1, 256)
#         # sparse_embeddings = torch.cat((sparse_embeddings, query), dim=2) # (140, 1, 6, 256)
#         '''
#         sparse_embeddings = torch.cat((sparse_embeddings, sparse_embeddings_tmp), dim=2) # (140, 1, 7, 256)
#         '''
#         sparse_embeddings = torch.cat((query, sparse_embeddings_tmp), dim=2)  # (140, 1, 7, 256)
#         # query = decoder_out.reshape(-1, 256).unsqueeze(1).unsqueeze(1)  # (140, 1, 1, 256)
#         # sparse_embeddings = query
#
#         mask_pred_plus = None
#         dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(img_bs*self.num_queries, -1, image_embedding_size[0], image_embedding_size[1]) # (140, 256, 16, 16)
#         # dense_embeddings = dense_embeddings_tmp # (140, 256, 16, 16)
#         image_embeddings = torch.repeat_interleave(image_embeddings, repeats=self.num_queries, dim=0) # (140, 256, 16, 16) 对batch中每个张量重复num_queries次
#         image_positional_embeddings = torch.repeat_interleave(image_positional_embeddings, repeats=self.num_queries,dim=0) # (140, 256, 16, 16)
#         decoder_out, mask_pred, iou_predictions, mask_dencoder_attentions = self.mask_decoder(
#             image_embeddings=image_embeddings, # (2*70=140, 256, 16, 16)
#             image_positional_embeddings=image_positional_embeddings, # (140, 256, 16, 16)
#             sparse_prompt_embeddings=sparse_embeddings, # (140, 1, 5, 256)，因为配置文件中 prompt_shape = (70, 5)
#             dense_prompt_embeddings=dense_embeddings, # (140, 256, 16, 16)
#             multimask_output=self.multimask_output, # False
#             attention_similarity=self.attention_similarity, # None
#             target_embedding=self.target_embedding, # None
#             output_attentions=self.output_attentions, # None
#         ) # mask_pred=(140, 1, 1, 16*4=64, 64), iou_predictions=(140, 1, 1), mask_dencoder_attentions=None
#         mask_pred = mask_pred.reshape(img_bs, -1, *mask_pred.shape[-2:]) # (2, 70, 64, 64)
#
#         decoder_out = decoder_out.squeeze(1).reshape(img_bs, -1, 256) # (2, 70, 256)
#
#         # cls_pred = self.cls_embed(decoder_out)  # (2, 70, 11)
#
#         h, w = mask_pred.shape[-2:]
#         attn_mask_pred = mask_pred.reshape(img_bs, -1, h, w) # shape (batch_size, num_queries, h, w)
#
#
#         return cls_pred, mask_pred, mask_pred_plus # (2, 70, 11), (2, 70, 64, 64), None
#
#     def forward(self, x: List[Tensor],
#                 batch_data_samples: SampleList,
#                 image_embeddings=None,
#                 image_positional_embeddings=None,
#                 query_features=None,
#                 query_locations=None,
#                 ) -> Tuple[List[Tensor]]:
#         ''' 删 '''
#         # query_features, query_pos_embeds, query_locations, proposal_cls_logits = self.get_query(x[1])  # (2,256,70)、(2,256,70)、(2,70,2)、(2,11,32,32)
#         query_features, query_pos_embeds, query_locations, proposal_cls_logits = self.get_query(image_embeddings)  # (2,256,70)、(2,256,70)、(2,70,2)、(2,11,32,32)
#         query_features = query_features.permute(0, 2, 1)
#         query_pos_embeds = query_pos_embeds.permute(0, 2, 1)
#
#         query_features = query_features + query_pos_embeds # query_pos_embeds出现了9e10如此大的数，debug发现
#
#         if torch.isnan(query_features).sum()>0 or torch.isinf(query_features).sum()>0:
#             print(f'最大值{query_features.max()}，最小值{query_features.min()}，nan的个数{torch.isnan(query_features).sum()}，inf的个数{torch.isinf(query_features).sum()}')
#
#         batch_size = x[0].shape[0]
#         cls_pred_list = []
#         mask_pred_list = []
#         mask_pred_plus_list = []
#         attn_mask = None
#
#         cls_pred, mask_pred, mask_pred_plus = self._forward_head(query_features,
#                                                                     query_locations,
#                                                                     image_embeddings,
#                                                                     image_positional_embeddings)
#         cls_pred_list.append(cls_pred) # len=1
#         mask_pred_list.append(mask_pred) # len=1
#         mask_pred_plus_list.append(mask_pred_plus) # len=1
#
#         return cls_pred_list, mask_pred_list, mask_pred_plus_list, proposal_cls_logits # mask_pred_plus_list = [None]
#
#     def loss(
#             self,
#             x: Tuple[Tensor],
#             batch_data_samples: SampleList,
#             image_embeddings=None, # (b, c, h, w)
#             image_positional_embeddings=None # 和image_embedding形状相同
#     ) -> Dict[str, Tensor]:
#         """Perform forward propagation and loss calculation of the panoptic
#         head on the features of the upstream network.
#
#         Args:
#             x (tuple[Tensor]): Multi-level features from the upstream
#                 network, each is a 4D-tensor.
#             batch_data_samples (List[:obj:`DetDataSample`]): The Data
#                 Samples. It usually includes information such as
#                 `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
#
#         Returns:
#             dict[str, Tensor]: a dictionary of loss components
#         """
#         batch_img_metas = []
#         batch_gt_instances = []
#         batch_gt_semantic_segs = []
#         for data_sample in batch_data_samples:
#             batch_img_metas.append(data_sample.metainfo)
#             batch_gt_instances.append(data_sample.gt_instances)
#             if 'gt_sem_seg' in data_sample:
#                 batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
#             else:
#                 batch_gt_semantic_segs.append(None)
#
#         # forward
#         # query_features, query_pos_embeds, query_locations, proposal_cls_logits = self.get_query(x[1]) # (2,256,70)、(2,256,70)、(2,70,2)、(2,11,32,32)
#         # query_features = query_features.permute(0, 2, 1)
#         # query_pos_embeds = query_pos_embeds.permute(0, 2, 1)
#
#         all_cls_scores, all_mask_preds, all_mask_preds_plus, proposal_cls_logits = self(x, batch_data_samples, image_embeddings,
#                                                                    image_positional_embeddings)
#         ''' 删 '''
#         all_mask_preds = [F.interpolate(
#                             ele,
#                             size=(512, 512),
#                             mode='bilinear',
#                             align_corners=False
#                         ) for ele in all_mask_preds]
#
#         # preprocess ground truth
#         batch_gt_instances = self.preprocess_gt(batch_gt_instances,
#                                                 batch_gt_semantic_segs)
#         # loss
#         losses = self.loss_by_feat(all_cls_scores, all_mask_preds, all_mask_preds_plus,
#                                    batch_gt_instances, batch_img_metas)
#         # 再加上辅助分类头损失
#         targets=[]
#         for ele in batch_gt_instances:
#             e={}
#             e['labels'] = ele.labels
#             e['masks'] = ele.masks
#             targets.append(e)
#
#         output_proposals = {"proposal_cls_logits": proposal_cls_logits}
#         indices = self.matcher(output_proposals, targets)
#         proposal_loss_dict = self.loss_proposals(output_proposals, targets, indices)
#
#         losses.update(proposal_loss_dict)
#         return losses
#
#     def loss_proposals(self, output_proposals, targets, indices):
#         assert "proposal_cls_logits" in output_proposals
#
#         proposal_size = output_proposals["proposal_cls_logits"].shape[-2:]
#         proposal_cls_logits = output_proposals["proposal_cls_logits"].flatten(2).float()  # b, c, hw
#
#         target_classes = self.num_classes * torch.ones([proposal_cls_logits.shape[0],
#                                                         proposal_size[0] * proposal_size[1]],
#                                                        device=proposal_cls_logits.device) # (batch, 32*32)
#         target_classes = target_classes.to(torch.int64)
#
#         target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#         idx = self._get_src_permutation_idx(indices)
#         target_classes[idx] = target_classes_o # indices匹配到的位置，target_classes值是对应标签值，否则是10（表示背景）
#
#         # loss_proposal = F.cross_entropy(proposal_cls_logits, target_classes, ignore_index=-1)
#         loss_proposal = self.loss_proposal(proposal_cls_logits, target_classes) # (batch, 11, 4096)、(batch, 4096)
#         losses = {"loss_proposal": loss_proposal}
#
#         return losses
#
#     def _get_src_permutation_idx(self, indices):
#         # permute predictions following indices
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx
#
#     def loss_by_feat(self,
#                      all_cls_scores: Tensor,
#                      all_mask_preds: Tensor,
#                      all_mask_preds_plus,
#                      batch_gt_instances: List[InstanceData],
#                      batch_img_metas: List[dict]) -> Dict[str, Tensor]:
#         num_dec_layers = len(all_cls_scores)
#         batch_gt_instances_list = [
#             batch_gt_instances for _ in range(num_dec_layers)
#         ]
#         img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
#         # losses_cls, losses_mask, losses_dice, losses_mask_plus, losses_dice_plus = multi_apply(
#         #     self._loss_by_feat_single,
#         #     all_cls_scores, all_mask_preds,
#         #     all_mask_preds_plus,
#         #     batch_gt_instances_list, img_metas_list)
#         losses_cls, losses_mask, losses_dice = multi_apply(
#             self._loss_by_feat_single,
#             all_cls_scores,
#             all_mask_preds,
#             batch_gt_instances_list,
#             img_metas_list)
#         loss_dict = dict()
#         # loss from the last decoder layer
#         loss_dict['loss_cls'] = losses_cls[-1]
#         loss_dict['loss_mask'] = losses_mask[-1]
#         loss_dict['loss_dice'] = losses_dice[-1]
#         # loss from other decoder layers
#         num_dec_layer = 0
#         for loss_cls_i, loss_mask_i, loss_dice_i, loss_dice_plus_i in zip(
#                 losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
#             loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
#             loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
#             loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
#             loss_dict[f'd{num_dec_layer}.loss_dice_plus'] = loss_dice_plus_i
#
#             num_dec_layer += 1
#         return loss_dict
#
#     def _loss_by_feat_single(self,
#                              cls_scores: Tensor,
#                              mask_preds: Tensor, # (2, 70, 512, 512)
#                              batch_gt_instances: List[InstanceData],
#                              batch_img_metas: List[dict]) -> Tuple[Tensor]:
#         num_imgs = cls_scores.size(0)
#         cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
#         mask_preds_list = [mask_preds[i] for i in range(num_imgs)] # list_2
#
#         (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
#          avg_factor) = self.get_targets(cls_scores_list, mask_preds_list, batch_gt_instances, batch_img_metas)
#
#         # shape (batch_size, num_queries)
#         labels = torch.stack(labels_list, dim=0)
#         # shape (batch_size, num_queries)
#         label_weights = torch.stack(label_weights_list, dim=0)
#         # shape (num_total_gts, h, w)
#         mask_targets = torch.cat(mask_targets_list, dim=0)
#         # shape (batch_size, num_queries)
#         mask_weights = torch.stack(mask_weights_list, dim=0)
#
#         # classfication loss
#         # shape (batch_size * num_queries, )
#         cls_scores = cls_scores.flatten(0, 1)
#         labels = labels.flatten(0, 1)
#         label_weights = label_weights.flatten(0, 1)
#
#         class_weight = cls_scores.new_tensor(self.class_weight)
#         loss_cls = self.loss_cls(
#             cls_scores,
#             labels,
#             label_weights,
#             avg_factor=class_weight[labels].sum())
#
#         num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
#         num_total_masks = max(num_total_masks, 1)
#
#         # extract positive ones
#         # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
#         mask_preds = mask_preds[mask_weights > 0]
#
#         if mask_targets.shape[0] == 0:
#             # zero match
#             loss_dice = mask_preds.sum()
#             loss_mask = mask_preds.sum()
#
#             return loss_cls, loss_mask, loss_dice
#
#         with torch.no_grad():
#             points_coords = get_uncertain_point_coords_with_randomness(
#                 mask_preds.unsqueeze(1), None, self.num_points,
#                 self.oversample_ratio, self.importance_sample_ratio)
#             # points_coords = points_coords.to(mask_preds.dtype)
#             # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
#             mask_point_targets = point_sample(
#                 mask_targets.unsqueeze(1).to(mask_preds.dtype), points_coords).squeeze(1)
#         # shape (num_queries, h, w) -> (num_queries, num_points)
#         mask_point_preds = point_sample(
#             mask_preds.unsqueeze(1), points_coords).squeeze(1)
#
#         # dice loss
#         loss_dice = self.loss_dice(
#             mask_point_preds, mask_point_targets, avg_factor=num_total_masks)
#
#         # mask loss
#         # shape (num_queries, num_points) -> (num_queries * num_points, )
#         mask_point_preds = mask_point_preds.reshape(-1)
#         # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
#         mask_point_targets = mask_point_targets.reshape(-1)
#
#         # loss_mask = self.loss_mask(
#         #     mask_point_preds,
#         #     mask_point_targets,
#         #     avg_factor=num_total_masks * self.num_points)
#         # to avoid nan in fp16 when num_total_masks * self.num_points
#         loss_mask = self.loss_mask(mask_point_preds, mask_point_targets)
#         return loss_cls, loss_mask, loss_dice
#
#     def predict(self, x: Tuple[Tensor],
#                 batch_data_samples: SampleList,
#                 image_embeddings=None,
#                 image_positional_embeddings=None
#                 ) -> Tuple[Tensor]:
#         batch_img_metas = [
#             data_sample.metainfo for data_sample in batch_data_samples
#         ]
#         all_cls_scores, all_mask_preds, all_mask_preds_plus, _ = self(
#             x, batch_data_samples, image_embeddings=image_embeddings,
#             image_positional_embeddings=image_positional_embeddings)
#         mask_cls_results = all_cls_scores[-1]
#         mask_pred_results = all_mask_preds[-1]
#         # mask_pred_plus_results = all_mask_preds_plus[-1]
#         # upsample masks
#         try:
#             img_shape = batch_img_metas[0]['batch_input_shape']
#         except:
#             img_shape = batch_img_metas[0]['pad_shape']
#         mask_pred_results = F.interpolate(
#             mask_pred_results,
#             size=(img_shape[0], img_shape[1]),
#             mode='bilinear',
#             align_corners=False
#         )
#
#         return mask_cls_results, mask_pred_results
#
#     def get_query(self, x):
#         proposal_pos_embeds = F.interpolate(self.meta_pos_embed, size=x.shape[-1],
#                                             mode="bilinear", align_corners=False) # 用于产生 query 的特征图尺寸是多少，第二个参数size就是多少
#         query_features, query_pos_embeds, query_locations, proposal_cls_logits = self.query_proposal(
#             x, proposal_pos_embeds
#         )
#         return query_features, query_pos_embeds, query_locations, proposal_cls_logits
#
#

@MODELS.register_module()
class RSMaskFormerFusionHead(MaskFormerFusionHead):
    def predict(self,
                mask_cls_results: Tensor,
                mask_pred_results: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False,
                **kwargs) -> List[dict]:
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        panoptic_on = self.test_cfg.get('panoptic_on', True)
        semantic_on = self.test_cfg.get('semantic_on', False)
        instance_on = self.test_cfg.get('instance_on', False)
        assert not semantic_on, 'segmantic segmentation ' \
                                'results are not supported yet.'
        results = []
        for mask_cls_result, mask_pred_result, meta in zip(
                mask_cls_results, mask_pred_results, batch_img_metas):
            # remove padding
            img_height, img_width = meta['img_shape'][:2]
            ori_img_height, ori_img_width = meta['ori_shape'][:2]
            scale_factor = meta['scale_factor']
            ori_scaled_height = int(ori_img_height * scale_factor[1])
            ori_scaled_width = int(ori_img_width * scale_factor[0])
            mask_pred_result = mask_pred_result[:, :ori_scaled_height, :ori_scaled_width]

            if rescale:
                # return result in original resolution
                ori_height, ori_width = meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]

            result = dict()
            if panoptic_on:
                pan_results = self.panoptic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['pan_results'] = pan_results

            if instance_on:
                ins_results = self.instance_postprocess(
                    mask_cls_result, mask_pred_result)
                result['ins_results'] = ins_results

            if semantic_on:
                sem_results = self.semantic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['sem_results'] = sem_results

            results.append(result)

        return results


@MODELS.register_module()
class RSSamModel(BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name)
        if extra_config is not None:
            sam_config.update(extra_config)
        self.sam_model = SamModel(sam_config)

        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(self.sam_model, init_cfg.get('checkpoint'))
            self.sam_model.is_init = True

    def init_weights(self):
        pass

    def forward(self, *args, **kwargs):
        return self.sam_model(*args, **kwargs)


@MODELS.register_module()
class RSSamPositionalEmbedding(SamPositionalEmbedding, BaseModule):
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
class RSSamVisionEncoder(BaseModule):
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
        # a = torch.load(init_cfg.get('checkpoint'))
        # b = vision_encoder.state_dict()
        # load checkpoint
        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                vision_encoder,
                init_cfg.get('checkpoint'),
                map_location='cpu',
                revise_keys=[(r'^module\.', ''), (r'^vision_encoder\.', '')])
        # LoRA 微调（可选）
        if peft_config is not None and isinstance(peft_config, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                'target_modules': ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_config)
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            # 不用LORA时
            self.vision_encoder = vision_encoder
        self.vision_encoder.is_init = True

    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')

    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)


@MODELS.register_module()
class MMPretrainSamVisionEncoder(BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            img_size=1024,
            peft_config=None,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        vision_encoder_cfg = dict(
            type='mmpretrain.ViTSAM',
            arch=hf_pretrain_name.split('-')[-1].split('_')[-1],
            img_size=img_size,
            patch_size=16,
            out_channels=256,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
        )
        vision_encoder = MODELS.build(vision_encoder_cfg)
        # a = torch.load(init_cfg.get('checkpoint'))
        # b = vision_encoder.state_dict()
        # load checkpoint
        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                vision_encoder,
                init_cfg.get('checkpoint'),
                map_location='cpu',
                revise_keys=[
                    (r'^module\.', ''),
                    (r'^vision_encoder\.', ''),
                    (r'.layer_norm1.', '.ln1.'),
                    (r'.layer_norm2.', '.ln2.'),
                    (r'.mlp.lin1.', '.ffn.layers.0.0.'),
                    (r'.mlp.lin2.', '.ffn.layers.1.'),
                    (r'neck.conv1.', 'channel_reduction.0.'),
                    (r'neck.ln1.', 'channel_reduction.1.'),
                    (r'neck.conv2.', 'channel_reduction.2.'),
                    (r'neck.ln2.', 'channel_reduction.3.'),
                ]
            )

        if peft_config is not None and isinstance(peft_config, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                'target_modules': ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_config)
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder
        self.vision_encoder.is_init = True

    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')

    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)


@MODELS.register_module()
class RSSamPromptEncoder(SamPromptEncoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).prompt_encoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.prompt_encoder = SamPromptEncoder(sam_config, shared_patch_embedding=None)

    def forward(self, *args, **kwargs):
        return self.prompt_encoder(*args, **kwargs)


@MODELS.register_module()
class RSSamMaskDecoder(SamMaskDecoder, BaseModule):
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
        self.mask_decoder = SamMaskDecoder(sam_config)

    def forward(self, *args, **kwargs):
        return self.mask_decoder(*args, **kwargs)


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
        # # 上采样层，将特征图从 16x16 扩展到 64x64
        # self.channel_fusion =nn.Sequential(
        #     nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
        #
        # )

        # self.channel_fusion = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         in_channels,
        #         512,
        #         kernel_size=2,
        #         stride=2,
        #         padding=0,
        #         bias=False,
        #     ),
        #     LayerNorm2d(512, eps=1e-6),
        #     nn.GELU(),
        #     # nn.ConvTranspose2d(
        #     #     512,
        #     #     512,
        #     #     kernel_size=2,
        #     #     stride=2,
        #     #     padding=0,
        #     #     bias=False,
        #     # ),
        #     # LayerNorm2d(512, eps=1e-6),
        #     nn.ConvTranspose2d(
        #         512,
        #         256,
        #         kernel_size=2,
        #         stride=2,
        #         padding=0,
        #         bias=False,
        #     ),
        #     LayerNorm2d(256, eps=1e-6),
        #     nn.GELU(),
        # )

        ''' 删掉这两行 '''
        # self.channel_fusion = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels,
        #         out_channels,
        #         kernel_size=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(out_channels, eps=1e-6),
        #     nn.Conv2d(
        #         out_channels,
        #         out_channels,
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(out_channels, eps=1e-6),
        # )

        ''' 作者原版代码 '''
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
class RSFeatureAggregator(BaseModule):
    # RSPrompter论文中的 Fig.4实现的功能

    in_channels_dict = {
        'base': [768] * (12 + 1),  # [768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768]
        'large': [1024] * (24 + 1),
        'huge': [1280] * (32 + 1),
    }

    def __init__(
            self,
            in_channels,
            hidden_channels=64,
            out_channels=256,
            select_layers=range(1, 12, 2),  # range(1, 13, 2) = [1, 3, 5, 7, 9, 11]

            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, str)
        model_arch = 'base' if 'base' in in_channels else 'large' if 'large' in in_channels else 'huge'
        self.in_channels = self.in_channels_dict[model_arch]
        self.select_layers = select_layers

        self.downconvs = nn.ModuleList()
        for i_layer in self.select_layers:
            self.downconvs.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i_layer], hidden_channels, 1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.hidden_convs = nn.ModuleList()
        for _ in self.select_layers:
            self.hidden_convs.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = [einops.rearrange(x, 'b h w c -> b c h w') for x in inputs]

        features = []
        for idx, i_layer in enumerate(self.select_layers):
            features.append(self.downconvs[idx](inputs[i_layer]))

        x = None
        for hidden_state, hidden_conv in zip(features, self.hidden_convs):
            if x is not None:
                hidden_state = x + hidden_state
            residual = hidden_conv(hidden_state)
            x = hidden_state + residual
        x = self.fusion_conv(x)
        return x


@MODELS.register_module()
class SAMDet(BaseDetector):
    def __init__(
            self,
            detector,
            segmentor,
            data_preprocessor=None,
            test_cfg=None,
            init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.detector = MODELS.build(detector)
        self.segmentor = MODELS.build(segmentor)
        self.segmentor.eval()
        self.test_cfg = test_cfg
        for param in self.segmentor.parameters():
            param.requires_grad = False

    def extract_feat(self, batch_inputs: Tensor):
        pass

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        pass

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, tuple]:
        losses = self.detector.loss(batch_inputs, batch_data_samples)
        return losses

    def oracle_predict(self,
                       batch_inputs: Tensor,
                       batch_data_samples: SampleList,
                       rescale: bool = True):

        batch_data_samples = self.detector.predict(batch_inputs, batch_data_samples, rescale=rescale)
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        for input_img, data_sample, meta in zip(batch_inputs, batch_data_samples, batch_img_metas):
            pred_instance_data = InstanceData()
            pred_instance_data.bboxes = data_sample.gt_instances.bboxes
            pred_instance_data.labels = data_sample.gt_instances.labels
            pred_instance_data.scores = torch.ones_like(data_sample.gt_instances.labels, dtype=torch.float32,
                                                        device=data_sample.gt_instances.labels.device)

            bboxes = pred_instance_data.bboxes
            ori_img_shape = data_sample.ori_shape
            if len(bboxes) == 0:
                mask_pred_binary = torch.zeros(
                    0,
                    ori_img_shape[0],
                    ori_img_shape[1],
                    device=batch_inputs.device,
                    dtype=torch.bool)
            else:
                scale_factor = data_sample.scale_factor
                repeat_num = bboxes.size(-1) // 2
                scale_factor = bboxes.new_tensor(scale_factor).repeat((1, repeat_num))
                bboxes = bboxes * scale_factor

                input_img = input_img.unsqueeze(0)
                bboxes = bboxes.unsqueeze(0)
                outputs = self.segmentor(
                    pixel_values=input_img,
                    input_boxes=bboxes,
                    multimask_output=False,
                )
                mask_pred_result = outputs.pred_masks
                mask_pred_result = mask_pred_result[0]
                mask_pred_result = mask_pred_result.squeeze(1)

                ori_img_height, ori_img_width = meta['ori_shape'][:2]
                scale_factor = meta['scale_factor']
                ori_scaled_height = int(ori_img_height * scale_factor[1])
                ori_scaled_width = int(ori_img_width * scale_factor[0])

                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=meta['img_shape'],
                    mode='bilinear',
                    align_corners=False)[:, 0]

                mask_pred_result = mask_pred_result[:, :ori_scaled_height, :ori_scaled_width]
                # return result in original resolution
                ori_height, ori_width = meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]
                mask_pred_binary = (mask_pred_result > 0)
            pred_instance_data.masks = mask_pred_binary
            data_sample.pred_instances = pred_instance_data
        return batch_data_samples

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True):
        if self.test_cfg is not None and self.test_cfg.get('oracle_on', True):
            return self.oracle_predict(batch_inputs, batch_data_samples, rescale=rescale)

        batch_data_samples = self.detector.predict(batch_inputs, batch_data_samples, rescale=rescale)
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        for input_img, data_sample, meta in zip(batch_inputs, batch_data_samples, batch_img_metas):
            bboxes = data_sample.pred_instances.bboxes
            ori_img_shape = data_sample.ori_shape
            if len(bboxes) == 0:
                mask_pred_binary = torch.zeros(
                    0,
                    ori_img_shape[0],
                    ori_img_shape[1],
                    device=batch_inputs.device,
                    dtype=torch.bool)
            else:
                scale_factor = data_sample.scale_factor
                repeat_num = bboxes.size(-1) // 2
                scale_factor = bboxes.new_tensor(scale_factor).repeat((1, repeat_num))
                bboxes = bboxes * scale_factor

                input_img = input_img.unsqueeze(0)
                bboxes = bboxes.unsqueeze(0)
                outputs = self.segmentor(
                    pixel_values=input_img,
                    input_boxes=bboxes,
                    multimask_output=False,
                )
                mask_pred_result = outputs.pred_masks
                mask_pred_result = mask_pred_result[0]
                mask_pred_result = mask_pred_result.squeeze(1)

                ori_img_height, ori_img_width = meta['ori_shape'][:2]
                scale_factor = meta['scale_factor']
                ori_scaled_height = int(ori_img_height * scale_factor[1])
                ori_scaled_width = int(ori_img_width * scale_factor[0])

                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=meta['img_shape'],
                    mode='bilinear',
                    align_corners=False)[:, 0]

                mask_pred_result = mask_pred_result[:, :ori_scaled_height, :ori_scaled_width]
                # return result in original resolution
                ori_height, ori_width = meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]
                mask_pred_binary = (mask_pred_result > 0)
            data_sample.pred_instances.masks = mask_pred_binary

        return batch_data_samples


@MODELS.register_module()
class SAMSegMaskRCNN(MaskRCNN):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        peft_config = kwargs.get('backbone', {}).get('peft_config', {})
        super().__init__(*args, **kwargs)

        if peft_config is None:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        vision_outputs = self.backbone(batch_inputs)
        if isinstance(vision_outputs, SamVisionEncoderOutput):
            image_embeddings = vision_outputs.last_hidden_state
            vision_hidden_states = vision_outputs.hidden_states
        elif isinstance(vision_outputs, tuple):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs
        else:
            raise NotImplementedError
        x = self.neck(vision_hidden_states)
        return x


@MODELS.register_module()
class SAMSegMask2Former(Mask2Former):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        peft_config = kwargs.get('backbone', {}).get('peft_config', {})
        super().__init__(*args, **kwargs)

        if peft_config is None:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        vision_outputs = self.backbone(batch_inputs)
        if isinstance(vision_outputs, SamVisionEncoderOutput):
            image_embeddings = vision_outputs.last_hidden_state
            vision_hidden_states = vision_outputs.hidden_states
        elif isinstance(vision_outputs, tuple):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs
        else:
            raise NotImplementedError

        x = self.neck(vision_hidden_states)
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
            '''
            self.lateral_convs：
            ModuleList(
              (0): ConvModule(
                (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm_layer): LN2d()
              )
              (1): ConvModule(
                (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm_layer): LN2d()
              )
              (2-3): 2 x ConvModule(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm_layer): LN2d()
              )
            )

            self.fpn_convs:
            ModuleList(
              (0-3): 4 x ConvModule(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (norm_layer): LN2d()
              )
            )
            '''
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
class RSPrompterAnchorRoIPromptHead(StandardRoIHead):
    def __init__(
            self,
            with_extra_pe=False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if with_extra_pe:
            out_channels = self.bbox_roi_extractor.out_channels
            positional_encoding = dict(
                num_feats=out_channels // 2,
                normalize=True,
            )
            self.extra_pe = SinePositionalEncoding(**positional_encoding)

    def _mask_forward(self,
                      x: Tuple[Tensor],
                      rois: Tensor = None,
                      pos_inds: Optional[Tensor] = None,
                      bbox_feats: Optional[Tensor] = None,
                      image_embeddings=None,
                      image_positional_embeddings=None,
                      ) -> dict:
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois) # (17, 256, 14, 14)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_preds, iou_predictions = self.mask_head(
            mask_feats,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            roi_img_ids=rois[:, 0] if rois is not None else None,
        )
        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats, iou_predictions=iou_predictions)
        return mask_results

    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult], bbox_feats: Tensor,
                  batch_gt_instances: InstanceList,
                  image_embeddings=None,
                  image_positional_embeddings=None,
                  ) -> dict:
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results]) # (17, 5)，5代表[batch_ind, x1, y1, x2, y2]，17代表batch中的两个图像总共有多少个pos_priors
            if len(pos_rois) == 0:
                print('no pos rois')
                return dict(loss_mask=dict(loss_mask=0 * x[0].sum()))
            mask_results = self._mask_forward(
                x, pos_rois,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'], # (17, 1, 128, 128)
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)

        mask_results.update(loss_mask=mask_loss_and_target['loss_mask'])
        return mask_results

    def loss(self,
             x: Tuple[Tensor],
             rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample],
             # extra inputs
             image_embeddings=None,
             image_positional_embeddings=None,
             ) -> dict:
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        if hasattr(self, 'extra_pe'):
            bs, _, h, w = x[0].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.extra_pe(mask_pe)
            outputs = []
            for i in range(len(x)):
                output = x[i] + F.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
                outputs.append(output)
            x = tuple(outputs)

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i] # 1000个proposal
            rpn_results.priors = rpn_results.pop('bboxes') # 将边界框信息存储在新的属性 priors 中，后续的代码中使用 priors

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i]
            ) # 边界框分配器，用于预测的候选框和真实的GT框进行匹配。通常一个预测框（proposal）会与多个GT框进行比较，最终通过某种规则（例如 IoU）选择与之最匹配的GT框
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x]
            ) # 边界框采样器，用于根据边界框分配的结果来选择哪些框作为训练样本。常见的采样策略包括：从正样本中选择一定数量的框（通常是 IoU 较高的框）、从负样本中选择一定数量的框（通常是 IoU 较低的框）、平衡正负样本的数量
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(
                x, sampling_results, bbox_results['bbox_feats'], batch_gt_instances,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
            losses.update(mask_results['loss_mask'])

        return losses

    def predict_mask(
            self,
            x: Tuple[Tensor],
            batch_img_metas: List[dict],
            results_list: InstanceList,
            rescale: bool = False,
            image_embeddings=None,
            image_positional_embeddings=None,
    ) -> InstanceList:

        # don't need to consider aug_test.
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        mask_results = self._mask_forward(
            x, mask_rois,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings)

        mask_preds = mask_results['mask_preds']
        # split batch mask prediction back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)

        # TODO: Handle the case where rescale is false
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        return results_list

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False,
                # extra inputs
                image_embeddings=None,
                image_positional_embeddings=None,
                ) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        if hasattr(self, 'extra_pe'):
            bs, _, h, w = x[0].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.extra_pe(mask_pe)
            outputs = []
            for i in range(len(x)):
                output = x[i] + F.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
                outputs.append(output)
            x = tuple(outputs)

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale)

        if self.with_mask:
            results_list = self.predict_mask(
                x, batch_img_metas, results_list, rescale=rescale,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
        return results_list


@MODELS.register_module()
class RSPrompterAnchorMaskHead(FCNMaskHead, BaseModule):
    def __init__(
            self,
            mask_decoder,
            in_channels,
            roi_feat_size=14,
            per_pointset_point=5,
            with_sincos=True,
            multimask_output=False,
            attention_similarity=None,
            target_embedding=None,
            output_attentions=None,
            class_agnostic=False,
            loss_mask: ConfigType = dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            init_cfg=None,
            *args,
            **kwargs):
        BaseModule.__init__(self, init_cfg=init_cfg)

        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size
        self.per_pointset_point = per_pointset_point
        self.with_sincos = with_sincos
        self.multimask_output = multimask_output
        self.attention_similarity = attention_similarity
        self.target_embedding = target_embedding
        self.output_attentions = output_attentions

        self.mask_decoder = MODELS.build(mask_decoder)

        prompt_encoder = dict(
            type='RSSamPromptEncoder',
            hf_pretrain_name=copy.deepcopy(mask_decoder.get('hf_pretrain_name')),
            init_cfg=copy.deepcopy(mask_decoder.get('init_cfg')),
        )
        prompt_encoder = MODELS.build(prompt_encoder)
        prompt_encoder.init_weights()
        self.no_mask_embed = prompt_encoder.prompt_encoder.no_mask_embed

        if with_sincos:
            num_sincos = 2
        else:
            num_sincos = 1
        self.point_emb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),  # (batch, 256, 7, 7)
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),  # (batch, 256*7*7)
            nn.Linear(in_channels * roi_feat_size ** 2 // 4, in_channels),  # (batch, 256)
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),  # (batch, 256)
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels * num_sincos * per_pointset_point)  # (batch, 256*2*5)
        )

        self.loss_mask = MODELS.build(loss_mask)
        self.class_agnostic = class_agnostic

    def init_weights(self) -> None:
        BaseModule.init_weights(self)

    def forward(self,
                x, # x.shape=(17, 256, 14, 14)，batch中的 2个图像共有 17个大小为 14x14的RoI
                image_embeddings,
                image_positional_embeddings,
                roi_img_ids=None,
                ):
        # x.shape = (17, 256, 14, 14)，batch中的 2个图像共有 17个大小为 14x14的RoI
        img_bs = image_embeddings.shape[0] # 2
        roi_bs = x.shape[0] # 17
        image_embedding_size = image_embeddings.shape[-2:]

        point_embedings = self.point_emb(x) # (17, 256*5*2)
        point_embedings = einops.rearrange(point_embedings, 'b (n c) -> b n c', n=self.per_pointset_point) # (17, 5, 512)
        if self.with_sincos:
            point_embedings = torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2] # (17, 5, 256)，这里的5是指每个RoI区域生成5个提示

        # (B * N_set), N_point, C
        sparse_embeddings = point_embedings.unsqueeze(1) # （17，1，5，256）
        num_roi_per_image = torch.bincount(roi_img_ids.long()) # len(roi_img_ids)=17, num_roi_per_image=tensor([9, 8])
        # deal with the case that there is no roi in an image
        '''
        什么时候需要处理没有RoI的情况？
        1、这个问题要结合torch.bincount()来看
        2、假设batch=2，roi_img_ids中保存的是tensor([0., 0., 0., 0., 0., 1., 1., 1., 1.])，代表每个RoI属于batch中哪个图像
        3、如果第一张图像没有RoI，那么num_roi_per_image = tensor([0, 17])，此时不会有问题
        4、但如果第二张图像没有RoI，那么num_roi_per_image = tensor([17])，这样向量就少了一个维度，需要用0填充，就是下面这段代码要做的
        5、根本原因是：如果x中的最大值是n，则返回的torch.bincount结果张量的大小将是n + 1，表示从0到n每个值的计数。
        '''
        num_roi_per_image = torch.cat([num_roi_per_image,
                                       torch.zeros(img_bs - len(num_roi_per_image), device=num_roi_per_image.device,
                                                   dtype=num_roi_per_image.dtype)])

        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(roi_bs, -1, image_embedding_size[0],
                                                                                 image_embedding_size[1]) # (17, 256, 32, 32)
        # get image embeddings with num_roi_per_image
        # batch=2，第一个图像重复 9 次，第二个图像重复 8 次
        image_embeddings = image_embeddings.repeat_interleave(num_roi_per_image, dim=0) # (17, 256, 32, 32)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)

        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings, # (batch=17, C=256, H=32, W=32)
            image_positional_embeddings=image_positional_embeddings, # (batch=17, C=256, H=32, W=32)
            sparse_prompt_embeddings=sparse_embeddings, # (batch=17, 1, 5, 256)
            dense_prompt_embeddings=dense_embeddings, # (batch=17, 256, 32, 32)
            multimask_output=self.multimask_output,
            attention_similarity=self.attention_similarity,
            target_embedding=self.target_embedding,
            output_attentions=self.output_attentions,
        )
        '''
        low_res_masks, (17, 1, 1, 32*4=128, 128)
        iou_predictions, (17, 1, 1)
        mask_decoder_attentions, None
        '''
        h, w = low_res_masks.shape[-2:]
        low_res_masks = low_res_masks.reshape(roi_bs, -1, h, w) # (17, 1, 128, 128)
        iou_predictions = iou_predictions.reshape(roi_bs, -1) # (17, 1)
        return low_res_masks, iou_predictions

    def get_targets(self, sampling_results: List[SamplingResult],
                    batch_gt_instances: InstanceList,
                    rcnn_train_cfg: ConfigDict) -> Tensor:
        pos_proposals = [res.pos_priors for res in sampling_results] # 17个正样本的bbox坐标
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ] # 17个正样本，每一个对应的GT下标（下标范围 0~gt_nums-1，不包含背景，因为已是正样本），注意GT下标(gt_inds)和GT标签(gt_labels)不一样
        '''
        GT下标(gt_inds)：取值[0, 当前图像拥有的GT数量-1]
        GT_label：取值[0, num_classes-1]
        '''
        gt_masks = [res.masks for res in batch_gt_instances] # 真值掩码
        mask_targets_list = []
        mask_size = rcnn_train_cfg.mask_size # (512, 512)，取自配置文件中的crop_size
        device = pos_proposals[0].device
        for pos_gt_inds, gt_mask in zip(pos_assigned_gt_inds, gt_masks):
            if len(pos_gt_inds) == 0:
                mask_targets = torch.zeros((0,) + mask_size, device=device, dtype=torch.float32)
            else:
                mask_targets = gt_mask[pos_gt_inds.cpu()].to_tensor(dtype=torch.float32, device=device)
            mask_targets_list.append(mask_targets)
        mask_targets = torch.cat(mask_targets_list)
        return mask_targets

    def loss_and_target(self, mask_preds: Tensor,
                        sampling_results: List[SamplingResult],
                        batch_gt_instances: InstanceList,
                        rcnn_train_cfg: ConfigDict) -> dict:
        mask_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg) # (17, 512, 512)，17个 mask_pred对应的 gt_mask

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results]) # (17), 17个 mask_pred对应的 gt_labels
        # resize to mask_targets size，双线性插值到 crop_size的尺寸
        mask_preds = F.interpolate(mask_preds, size=mask_targets.shape[-2:], mode='bilinear', align_corners=False) # (17, 1, 512, 512)

        loss = dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            if self.class_agnostic: # 执行这里，因为 mask decoder预测的是二值掩码，而类别在生成 RoI时就已经分配好了
                loss_mask = self.loss_mask(mask_preds, mask_targets,
                                           torch.zeros_like(pos_labels))
            else: # 跳过
                loss_mask = self.loss_mask(mask_preds, mask_targets,
                                           pos_labels)
        loss['loss_mask'] = loss_mask
        return dict(loss_mask=loss, mask_targets=mask_targets)

    def _predict_by_feat_single(self,
                                mask_preds: Tensor,
                                bboxes: Tensor,
                                labels: Tensor,
                                img_meta: dict,
                                rcnn_test_cfg: ConfigDict,
                                rescale: bool = False,
                                activate_map: bool = False) -> Tensor:
        scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
            (1, 2))
        img_h, img_w = img_meta['ori_shape'][:2]
        if not activate_map:
            mask_preds = mask_preds.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_preds = bboxes.new_tensor(mask_preds)

        if rescale:  # in-placed rescale the bboxes
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)
        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = F.interpolate(mask_preds, size=img_meta['batch_input_shape'], mode='bilinear',
                                align_corners=False).squeeze(1)

        scale_factor_w, scale_factor_h = img_meta['scale_factor']
        ori_rescaled_size = (img_h * scale_factor_h, img_w * scale_factor_w)
        im_mask = im_mask[:, :int(ori_rescaled_size[0]), :int(ori_rescaled_size[1])]

        h, w = img_meta['ori_shape']
        im_mask = F.interpolate(im_mask.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)

        if threshold >= 0:
            im_mask = im_mask >= threshold
        else:
            # for visualization and debugging
            im_mask = (im_mask * 255).to(dtype=torch.uint8)
        return im_mask


@MODELS.register_module()
class MMPretrainSwinTransformer(BaseModule):
    def __init__(
            self,
            hf_pretrain_name=None,
            img_size=224,
            peft_config=None,
            init_cfg=None,
            extra_config=None,
    ):
        super().__init__(init_cfg=init_cfg)
        vision_encoder_cfg = dict(
            type='mmpretrain.SwinTransformer',
            arch='base',
            img_size=img_size,
        )
        vision_encoder = MODELS.build(vision_encoder_cfg)
        # load checkpoint
        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                vision_encoder,
                init_cfg.get('checkpoint'),
                map_location='cpu',
                # revise_keys=[
                #     (r'^module\.', ''),
                #     (r'^vision_encoder\.', ''),
                #     (r'.layer_norm1.', '.ln1.'),
                #     (r'.layer_norm2.', '.ln2.'),
                #     (r'.mlp.lin1.', '.ffn.layers.0.0.'),
                #     (r'.mlp.lin2.', '.ffn.layers.1.'),
                #     (r'neck.conv1.', 'channel_reduction.0.'),
                #     (r'neck.ln1.', 'channel_reduction.1.'),
                #     (r'neck.conv2.', 'channel_reduction.2.'),
                #     (r'neck.ln2.', 'channel_reduction.3.'),
                # ]
            )

        self.vision_encoder = vision_encoder
        self.vision_encoder.is_init = True

    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')

    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)


@MODELS.register_module()
class MMPretrainSwinTransformerV2(BaseModule):
    def __init__(
            self,
            hf_pretrain_name=None,
            img_size=224,
            peft_config=None,
            init_cfg=None,
            extra_config=None,
    ):
        super().__init__(init_cfg=init_cfg)
        vision_encoder_cfg = dict(
            type='mmpretrain.SwinTransformerV2',
            arch='base',
            img_size=img_size,
        )
        vision_encoder = MODELS.build(vision_encoder_cfg)
        tmp = vision_encoder.state_dict()
        # load checkpoint
        if init_cfg is not None:
            # checkpoint = torch.load(init_cfg.get('checkpoint'))
            checkpoint = torch.load(init_cfg.get('checkpoint'))['model']
            # vision_encoder.load_state_dict(checkpoint, strict=False)
            # del checkpoint['classifier.weight']
            # del checkpoint['classifier.bias']

            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                vision_encoder,
                init_cfg.get('checkpoint'),
                map_location='cpu',
                revise_keys=[
                    (r'^module\.', ''),
                    (r'^swinv2\.', ''), # 移除 'swinv2.' 前缀
                    (r'^encoder\.', ''), # 移除 'encoder.' 前缀
                    # (r'^embeddings\.', '.ln1.'), # 移除 'embeddings.' 前缀
                    (r'embeddings.patch_embeddings', 'patch_embed'),
                    (r'embeddings', 'patch_embed'),

                    (r'layers', 'stage'),
                    (r'attention', 'attn'),


                    (r'.layer_norm1.', '.ln1.'),
                    (r'.layer_norm2.', '.ln2.'),
                    (r'.mlp.lin1.', '.ffn.layers.0.0.'),
                    (r'.mlp.lin2.', '.ffn.layers.1.'),
                    (r'neck.conv1.', 'channel_reduction.0.'),
                    (r'neck.ln1.', 'channel_reduction.1.'),
                    (r'neck.conv2.', 'channel_reduction.2.'),
                    (r'neck.ln2.', 'channel_reduction.3.'),
                ]
            )
        a = vision_encoder.state_dict()
        self.vision_encoder = vision_encoder
        self.vision_encoder.is_init = True

    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')

    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)


@MODELS.register_module()
class RS_Swin_FeatureAggregator(BaseModule):
    # RSPrompter论文中的 Fig.4实现的功能

    in_channels_dict = {
        'base': [768] * (12 + 1),  # [768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768]
        'large': [1024] * (24 + 1),
        'huge': [1280] * (32 + 1),
    }

    def __init__(
            self,
            in_channels,
            hidden_channels=64,
            out_channels=256,
            select_layers=[0, 1, 2, 3],  # range(1, 13, 2) = [1, 3, 5, 7, 9, 11]

            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, str)
        model_arch = 'base'
        # self.in_channels = self.in_channels_dict[model_arch]
        self.in_channels = [128, 256, 512, 1024]
        self.select_layers = [0, 1, 2, 3]

        self.downconvs = nn.ModuleList()
        for i_layer in self.select_layers:
            self.downconvs.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i_layer], hidden_channels, 1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.hidden_convs = nn.ModuleList()
        for _ in self.select_layers:
            self.hidden_convs.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = [einops.rearrange(x, 'b h w c -> b c h w') for x in inputs]

        features = []
        for idx, i_layer in enumerate(self.select_layers):
            features.append(self.downconvs[idx](inputs[i_layer]))

        x = None
        for hidden_state, hidden_conv in zip(features, self.hidden_convs):
            if x is not None:
                hidden_state = x + hidden_state
            residual = hidden_conv(hidden_state)
            x = hidden_state + residual
        x = self.fusion_conv(x)
        return x

# class QueryProposal(nn.Module):
#
#     def __init__(self, num_features, num_queries, num_classes):
#         super().__init__()
#         self.topk = num_queries
#         self.num_classes = num_classes
#
#         self.conv_proposal_cls_logits = nn.Sequential(
#             nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features, num_classes + 1, kernel_size=1, stride=1, padding=0),
#         )
#
#     @torch.no_grad()
#     def compute_coordinates(self, x):
#         h, w = x.size(2), x.size(3)
#         y_loc = torch.linspace(0, 1, h, device=x.device)
#         x_loc = torch.linspace(0, 1, w, device=x.device)
#         y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
#         locations = torch.stack([x_loc, y_loc], 0).unsqueeze(0)
#         return locations
#
#     def seek_local_maximum(self, x, epsilon=1e-6):
#         """
#         inputs:
#             x: torch.tensor, shape [b, c, h, w]
#         return:
#             torch.tensor, shape [b, c, h, w]
#         """
#         x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
#         # top, bottom, left, right, top-left, top-right, bottom-left, bottom-right
#         maximum = (x >= x_pad[:, :, :-2, 1:-1]) & \
#                   (x >= x_pad[:, :, 2:, 1:-1]) & \
#                   (x >= x_pad[:, :, 1:-1, :-2]) & \
#                   (x >= x_pad[:, :, 1:-1, 2:]) & \
#                   (x >= x_pad[:, :, :-2, :-2]) & \
#                   (x >= x_pad[:, :, :-2, 2:]) & \
#                   (x >= x_pad[:, :, 2:, :-2]) & \
#                   (x >= x_pad[:, :, 2:, 2:]) & \
#                   (x >= epsilon)
#         return maximum.to(x)
#
#     def forward(self, x, pos_embeddings):
#
#         proposal_cls_logits = self.conv_proposal_cls_logits(x)  # b, c, h, w
#         proposal_cls_probs = proposal_cls_logits.softmax(dim=1)  # b, c, h, w
#         proposal_cls_one_hot = F.one_hot(proposal_cls_probs[:, :-1, :, :].max(1)[1],
#                                          num_classes=self.num_classes + 1).permute(0, 3, 1, 2)  # b, c, h, w
#         proposal_cls_probs = proposal_cls_probs.mul(proposal_cls_one_hot)
#         proposal_local_maximum_map = self.seek_local_maximum(proposal_cls_probs)  # b, c, h, w
#         proposal_cls_probs = proposal_cls_probs + proposal_local_maximum_map  # b, c, h, w
#
#         # top-k indices
#         # x的分辨率是（30, 54），展平后相当于有 1620个索引代表（30，54）中的每个像素位置
#         # 而 topk_indices就是topk个取值范围在 0~1620 内的数，代表的是前topk个最大类别概率(logits)的像素位置
#         topk_indices = torch.topk(proposal_cls_probs[:, :-1, :, :].flatten(2).max(1)[0], self.topk, dim=1)[1]  # b, q
#         topk_indices = topk_indices.unsqueeze(1)  # b, 1, q
#
#         # topk queries
#         topk_proposals = torch.gather(x.flatten(2), dim=2, index=topk_indices.repeat(1, x.shape[1], 1))  # b, c, q
#         pos_embeddings = pos_embeddings.repeat(x.shape[0], 1, 1, 1).flatten(2)
#         topk_pos_embeddings = torch.gather(
#             pos_embeddings, dim=2, index=topk_indices.repeat(1, pos_embeddings.shape[1], 1)
#         )  # b, c, q
#         '''
#         原版
#         if self.training:
#             locations = self.compute_coordinates(x).repeat(x.shape[0], 1, 1, 1)
#             topk_locations = torch.gather(
#                 locations.flatten(2), dim=2, index=topk_indices.repeat(1, locations.shape[1], 1)
#             )
#             topk_locations = topk_locations.transpose(-1, -2)  # b, q, 2
#         else:
#             topk_locations = None
#         '''
#
#         ''' 删除 '''
#         locations = self.compute_coordinates(x).repeat(x.shape[0], 1, 1, 1)
#         topk_locations = torch.gather(
#             locations.flatten(2), dim=2, index=topk_indices.repeat(1, locations.shape[1], 1)
#         )
#         topk_locations = topk_locations.transpose(-1, -2)  # b, q, 2
#         return topk_proposals, topk_pos_embeddings, topk_locations, proposal_cls_logits



















# import copy
# import warnings
# import einops
# import numpy as np
# import torch
# from mmcv.cnn import build_norm_layer, ConvModule
# from mmcv.ops import point_sample
# from mmengine import ConfigDict
# from mmengine.dist import is_main_process
# from mmengine.model import BaseModule
# from mmengine.structures import InstanceData
# from peft import get_peft_config, get_peft_model
# from torch import nn, Tensor
# from transformers import SamConfig
# from transformers.models.sam.modeling_sam import SamVisionEncoder, SamMaskDecoder, SamPositionalEmbedding, \
#     SamPromptEncoder, SamModel, SamVisionEncoderOutput
# from typing import List, TypeVar, Tuple, Optional, Dict, Union
# from mmdet.models import MaskRCNN, StandardRoIHead, FCNMaskHead, SinePositionalEncoding, Mask2Former, Mask2FormerHead, \
#     MaskFormerFusionHead, BaseDetector
# from mmdet.models.task_modules import SamplingResult
# from mmdet.models.utils import unpack_gt_instances, empty_instances, multi_apply, \
#     get_uncertain_point_coords_with_randomness
# from mmdet.registry import MODELS
# from mmdet.structures import SampleList, DetDataSample, OptSampleList
# from mmdet.structures.bbox import bbox2roi
# from mmdet.utils import OptConfigType, MultiConfig, ConfigType, InstanceList, reduce_mean
# import torch.nn.functional as F
#
# from mmpretrain.models import LayerNorm2d
#
# T = TypeVar('T')
#
#
# @MODELS.register_module(force=True)
# class LN2d(nn.Module):
#     """A LayerNorm variant, popularized by Transformers, that performs
#     pointwise mean and variance normalization over the channel dimension for
#     inputs that have shape (batch_size, channels, height, width)."""
#
#     def __init__(self, normalized_shape, eps=1e-6):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.normalized_shape = (normalized_shape,)
#
#     def forward(self, x):
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)
#         x = self.weight[:, None, None] * x + self.bias[:, None, None]
#         return x
#
#
# @MODELS.register_module()
# class RSPrompterAnchor(MaskRCNN):
#     def __init__(
#             self,
#             shared_image_embedding,
#             decoder_freeze=True,
#             *args,
#             **kwargs):
#         peft_config = kwargs.get('backbone', {}).get('peft_config', {})
#         super().__init__(*args, **kwargs)
#         self.shared_image_embedding = MODELS.build(shared_image_embedding)
#         self.decoder_freeze = decoder_freeze
#
#         '''
#         不用的时候删除下面三行代码
#         '''
#         # from transformers import Swinv2Model
#         # self.backbone = Swinv2Model.from_pretrained("/data2/yihan/MyProject/RSPrompter-release/swinv2-tiny-patch4-window16-256/")
#         # self.from1024to256 = nn.Sequential(
#         #     nn.Conv2d(
#         #         768,
#         #         256,
#         #         kernel_size=1,
#         #         bias=False,
#         #     ),
#         #     LayerNorm2d(256, eps=1e-6),
#         #     nn.Conv2d(
#         #         256,
#         #         256,
#         #         kernel_size=3,
#         #         padding=1,
#         #         bias=False,
#         #     ),
#         #     LayerNorm2d(256, eps=1e-6),
#         # )
#
#         ''' 删除 '''
#         # # 自定义初始化函数
#         # import torch.nn.init as init
#         # def init_weights(m):
#         #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         #         init.xavier_uniform_(m.weight)  # 使用 Xavier 初始化权重
#         #         if m.bias is not None:
#         #             init.zeros_(m.bias)  # 初始化偏置为零
#         # # 对 Sequential 中的所有层应用初始化函数
#         # self.from1024to256.apply(init_weights)
#
#         self.frozen_modules = []
#         # 真正使用 Lora是在 backbone中定义的
#         if peft_config is None:  # 如果不使用 LORA微调，就冻结backbone
#             self.frozen_modules += [self.backbone]
#         if self.decoder_freeze:
#             self.frozen_modules += [
#                 self.shared_image_embedding,
#                 self.roi_head.mask_head.mask_decoder,
#                 self.roi_head.mask_head.no_mask_embed,
#             ]
#         self._set_grad_false(self.frozen_modules)
#
#     def _set_grad_false(self, module_list=[]):
#         for module in module_list:
#             module.eval()
#             if isinstance(module, nn.Parameter):
#                 module.requires_grad = False
#             for param in module.parameters():
#                 param.requires_grad = False
#
#     def get_image_wide_positional_embeddings(self, size):
#         target_device = self.shared_image_embedding.shared_image_embedding.positional_embedding.device
#         target_dtype = self.shared_image_embedding.shared_image_embedding.positional_embedding.dtype
#         grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
#         y_embed = grid.cumsum(dim=0) - 0.5
#         x_embed = grid.cumsum(dim=1) - 0.5
#         y_embed = y_embed / size
#         x_embed = x_embed / size
#
#         positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
#         return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width
#
#     def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
#         vision_outputs = self.backbone(batch_inputs)
#
#         ''' 删掉下面这行 '''
#         # vision_outputs = (vision_outputs['last_hidden_state'].reshape(int(batch_inputs.shape[0]), int(batch_inputs.shape[2]/32), int(batch_inputs.shape[3]/32), -1).permute(0, 3, 1, 2), )
#         # vision_outputs = (vision_outputs, )
#
#         if isinstance(vision_outputs, SamVisionEncoderOutput):
#             image_embeddings = vision_outputs[0]
#             vision_hidden_states = vision_outputs[1]
#         elif isinstance(vision_outputs, tuple):
#             image_embeddings = vision_outputs[0]
#             vision_hidden_states = vision_outputs
#         else:
#             raise NotImplementedError
#
#         ''' 删掉下面这行 '''
#         # image_embeddings = self.from1024to256(image_embeddings)
#
#         image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_embeddings.shape[-1])
#         # repeat with batch size
#         batch_size = image_embeddings.shape[0]
#         image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)
#
#         x = self.neck(vision_hidden_states)
#
#         return x, image_embeddings, image_positional_embeddings
#
#     def loss(self, batch_inputs: Tensor,
#              batch_data_samples: SampleList) -> dict:
#         x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)
#
#         losses = dict()
#         # RPN forward and loss
#         proposal_cfg = self.train_cfg.get('rpn_proposal',
#                                           self.test_cfg.rpn)
#         rpn_data_samples = copy.deepcopy(batch_data_samples)
#         # set cat_id of gt_labels to 0 in RPN
#         for data_sample in rpn_data_samples:
#             data_sample.gt_instances.labels = \
#                 torch.zeros_like(data_sample.gt_instances.labels)
#
#         # loss_and_predict(): 依次调用self.rpn_head的forward() -> loss_by_feat() -> predict_by_feat()
#         rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
#             x, rpn_data_samples, proposal_cfg=proposal_cfg)
#         # avoid get same name with roi_head loss
#         keys = rpn_losses.keys()
#         for key in list(keys):
#             if 'loss' in key and 'rpn' not in key:
#                 rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
#         losses.update(rpn_losses)
#
#         roi_losses = self.roi_head.loss(
#             x,  # tuple_5，5个不同尺度的特征图，H/4 ~ H/64，每个尺寸都是(batch=2，dim=256，H，W)
#             rpn_results_list,  # list_2，batch中2个图像的proposal，每个图像都有1K个proposal，包括1K个bboxes、labels、scores
#             batch_data_samples,  # list_2，batch中2个图像的实例标注GT信息
#             image_embeddings=image_embeddings,
#             image_positional_embeddings=image_positional_embeddings,
#         )
#         losses.update(roi_losses)
#
#         return losses
#
#     def predict(self,
#                 batch_inputs: Tensor,  # (b, c, h, w) = (2, 3, 512, 512)
#                 batch_data_samples: SampleList,
#                 # list列表，长度 = batch，列表中每个元素都是 DetDataSample 类的实例，该实例其中一个属性就是 gt_instances
#                 rescale: bool = True) -> SampleList:
#         x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)
#
#         # If there are no pre-defined proposals, use RPN to get proposals
#         if batch_data_samples[0].get('proposals', None) is None:
#             rpn_results_list = self.rpn_head.predict(
#                 x, batch_data_samples, rescale=False)
#         else:
#             rpn_results_list = [
#                 data_sample.proposals for data_sample in batch_data_samples
#             ]
#
#         results_list = self.roi_head.predict(
#             x, rpn_results_list, batch_data_samples, rescale=rescale,
#             image_embeddings=image_embeddings,
#             image_positional_embeddings=image_positional_embeddings,
#         )  # results_list 是一个列表，表中元素是 InstanceData 类的实例，包含bboxes、labels、masks(原图尺寸origin_size)、scores
#         batch_data_samples = self.add_pred_to_datasample(
#             batch_data_samples, results_list)  # 给 DetDataSample 实例添加了一个 pred_instances属性
#         return batch_data_samples
#
#
# @MODELS.register_module()
# class RSPrompterQuery(Mask2Former):
#     def __init__(
#             self,
#             shared_image_embedding,
#             decoder_freeze=True,
#             *args,
#             **kwargs):
#         peft_config = kwargs.get('backbone', {}).get('peft_config', {})
#         super().__init__(*args, **kwargs)
#         self.decoder_freeze = decoder_freeze
#         self.with_mask2formerhead = False if isinstance(self.panoptic_head, RSMask2FormerHead) else True
#         self.shared_image_embedding = MODELS.build(shared_image_embedding)
#
#         self.frozen_modules = []
#         if peft_config is None:
#             self.frozen_modules += [self.backbone]
#         if self.decoder_freeze:
#             self.frozen_modules += [
#                 self.shared_image_embedding,
#                 self.panoptic_head.mask_decoder,
#             ]
#         self._set_grad_false(self.frozen_modules)
#
#     def _set_grad_false(self, module_list=[]):
#         for module in module_list:
#             module.eval()
#             if isinstance(module, nn.Parameter):
#                 module.requires_grad = False
#             for param in module.parameters():
#                 param.requires_grad = False
#
#     def get_image_wide_positional_embeddings(self, size):
#         target_device = self.shared_image_embedding.shared_image_embedding.positional_embedding.device
#         target_dtype = self.shared_image_embedding.shared_image_embedding.positional_embedding.dtype
#         grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
#         y_embed = grid.cumsum(dim=0) - 0.5
#         x_embed = grid.cumsum(dim=1) - 0.5
#         y_embed = y_embed / size
#         x_embed = x_embed / size
#
#         positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
#         return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width
#
#     def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
#         vision_outputs = self.backbone(batch_inputs)
#         if isinstance(vision_outputs, SamVisionEncoderOutput):
#             image_embeddings = vision_outputs[0]
#             vision_hidden_states = vision_outputs[1]
#         elif isinstance(vision_outputs, tuple):
#             image_embeddings = vision_outputs[0]
#             vision_hidden_states = vision_outputs
#         else:
#             raise NotImplementedError
#
#         image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_embeddings.shape[-1])
#         # repeat with batch size
#         batch_size = image_embeddings.shape[0]
#         image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)
#
#         x = self.neck(vision_hidden_states)
#         return x, image_embeddings, image_positional_embeddings
#
#     def loss(self, batch_inputs: Tensor,
#              batch_data_samples: SampleList) -> Dict[str, Tensor]:
#
#         x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)
#
#         if self.with_mask2formerhead:
#             losses = self.panoptic_head.loss(x, batch_data_samples)
#         else:
#             losses = self.panoptic_head.loss(x, batch_data_samples,
#                                              image_embeddings=image_embeddings,
#                                              image_positional_embeddings=image_positional_embeddings)
#         return losses
#
#     def predict(self,
#                 batch_inputs: Tensor,
#                 batch_data_samples: SampleList,
#                 rescale: bool = True) -> SampleList:
#
#         x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)
#
#         if self.with_mask2formerhead:
#             mask_cls_results, mask_pred_results = self.panoptic_head.predict(x, batch_data_samples)
#         else:
#             mask_cls_results, mask_pred_results = self.panoptic_head.predict(
#                 x, batch_data_samples,
#                 image_embeddings=image_embeddings,
#                 image_positional_embeddings=image_positional_embeddings
#             )
#
#         results_list = self.panoptic_fusion_head.predict(
#             mask_cls_results,
#             mask_pred_results,
#             batch_data_samples,
#             rescale=rescale)
#         results = self.add_pred_to_datasample(batch_data_samples, results_list)
#
#         return results
#
#
# @MODELS.register_module()
# class RSMask2FormerHead(Mask2FormerHead, BaseModule):
#     def __init__(
#             self,
#             mask_decoder,
#             decoder_plus,
#             with_sincos=True,
#             per_pointset_point=1,
#             multimask_output=False,
#             attention_similarity=None,
#             target_embedding=None,
#             output_attentions=None,
#             *args,
#             **kwargs):
#         super().__init__(*args, **kwargs)
#         self.decoder_plus = decoder_plus
#         self.multimask_output = multimask_output
#         self.attention_similarity = attention_similarity
#         self.target_embedding = target_embedding
#         self.output_attentions = output_attentions
#
#         self.mask_decoder = MODELS.build(mask_decoder)
#
#         prompt_encoder = dict(
#             type='RSSamPromptEncoder',
#             hf_pretrain_name=copy.deepcopy(mask_decoder.get('hf_pretrain_name')),
#             init_cfg=copy.deepcopy(mask_decoder.get('init_cfg')),
#         )
#         prompt_encoder = MODELS.build(prompt_encoder)
#         prompt_encoder.init_weights()
#         if self.decoder_plus:
#             self.sam_mask_embed = prompt_encoder.prompt_encoder.mask_embed
#         else:
#             self.no_mask_embed = prompt_encoder.prompt_encoder.no_mask_embed
#             del self.mask_embed
#
#         self.per_pointset_point = per_pointset_point
#         self.with_sincos = with_sincos
#
#         self.feat_channels = kwargs['feat_channels']
#         self.out_channels = kwargs['out_channels']
#         if with_sincos:
#             num_sincos = 2
#         else:
#             num_sincos = 1
#         self.point_emb = nn.Sequential(
#             nn.Linear(self.feat_channels, self.feat_channels // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.feat_channels // 2, self.feat_channels // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.feat_channels // 2, self.out_channels * num_sincos * per_pointset_point)
#         )
#         del self.cls_embed
#         self.cls_embed = nn.Sequential(
#             nn.Linear(self.feat_channels, self.feat_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.feat_channels, self.num_classes + 1))
#
#     def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
#                       attn_mask_target_size: Tuple[int, int],
#                       image_embeddings=None,  # (2, 256, 16, 16)
#                       image_positional_embeddings=None  # (2, 256, 16, 16)
#                       ) -> Tuple[Tensor]:
#         img_bs = image_embeddings.shape[0]  # 2
#         image_embedding_size = image_embeddings.shape[-2:]
#
#         decoder_out = self.transformer_decoder.post_norm(decoder_out)  # (2, 70, 128)
#         # shape (batch_size, num_queries, c)
#         cls_pred = self.cls_embed(decoder_out)  # (2, 70, 11)
#         # shape (batch_size, num_queries, c)
#         point_embedings = self.point_emb(decoder_out)  # (2, 70, 2560)
#
#         point_embedings = einops.rearrange(point_embedings, 'b n_set (n_point c) -> b n_set n_point c',
#                                            n_point=self.per_pointset_point)  # (2, 70, 5, 512)
#         if self.with_sincos:
#             point_embedings = torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2]  # (2, 70, 5, 256)
#
#         # B, N_set, N_point, C => (B, N_set), 1, N_point, C
#         sparse_embeddings = einops.rearrange(point_embedings,
#                                              'b n_set n_point c -> (b n_set) n_point c')  # (140, 5, 256)
#         sparse_embeddings = sparse_embeddings.unsqueeze(1)  # (140, 1, 5, 256)
#
#         if self.decoder_plus:
#             # shape (num_queries, batch_size, h, w)
#             mask_embed = self.mask_embed(decoder_out)  # (2, 70, 256)
#             mask_pred_plus = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)  # (2, 70, 64, 64)
#
#             input_masks = mask_pred_plus.detach()  # (2, 70, 64, 64)
#             input_masks = einops.repeat(input_masks, 'b n h w -> (b n) c h w', c=1)  # (140, 1, 64, 64)
#             # (bs num_q) c h w
#             dense_embeddings = self.sam_mask_embed(input_masks)  # (140, 256, 256/16=16, 16)
#         else:
#             mask_pred_plus = None
#             dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(img_bs, -1,
#                                                                                      image_embedding_size[0],
#                                                                                      image_embedding_size[1])
#
#         image_embeddings = torch.repeat_interleave(image_embeddings, repeats=self.num_queries,
#                                                    dim=0)  # (140, 256, 16, 16)
#         image_positional_embeddings = torch.repeat_interleave(image_positional_embeddings, repeats=self.num_queries,
#                                                               dim=0)  # (140, 256, 16, 16)
#         mask_pred, iou_predictions, mask_dencoder_attentions = self.mask_decoder(
#             image_embeddings=image_embeddings,  # (2*70=140, 256, 16, 16)
#             image_positional_embeddings=image_positional_embeddings,  # (140, 256, 16, 16)
#             sparse_prompt_embeddings=sparse_embeddings,  # (140, 1, 5, 256)，因为配置文件中 prompt_shape = (70, 5)
#             dense_prompt_embeddings=dense_embeddings,  # (140, 256, 16, 16)
#             multimask_output=self.multimask_output,  # False
#             attention_similarity=self.attention_similarity,  # None
#             target_embedding=self.target_embedding,  # None
#             output_attentions=self.output_attentions,  # None
#         )  # mask_pred=(140, 1, 1, 16*4=64, 64), iou_predictions=(140, 1, 1), mask_dencoder_attentions=None
#         mask_pred = mask_pred.reshape(img_bs, -1, *mask_pred.shape[-2:])  # (2, 70, 64, 64)
#         if not self.decoder_plus:  # 跳过
#             h, w = mask_pred.shape[-2:]
#             # shape (batch_size, num_queries, h, w)
#             attn_mask_pred = mask_pred.reshape(img_bs, -1, h, w)
#         else:  # 执行这里
#             attn_mask_pred = mask_pred_plus  # (2, 70, 64, 64)
#         attn_mask = F.interpolate(attn_mask_pred, attn_mask_target_size, mode='bilinear',
#                                   align_corners=False)  # (2, 70, 4, 4)
#         # shape (num_queries, batch_size, h, w) ->
#         #   (batch_size * num_head, num_queries, h, w)
#         attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
#             (1, self.num_heads, 1, 1)).flatten(0, 1)
#         attn_mask = attn_mask.sigmoid() < 0.5
#         attn_mask = attn_mask.detach()
#         return cls_pred, mask_pred, attn_mask, mask_pred_plus  # (2, 70, 11), (2, 70, 64, 64), (16, 70, 4*4=16), (2, 70, 64, 64)
#
#     def forward(self, x: List[Tensor],
#                 batch_data_samples: SampleList,
#                 image_embeddings=None,
#                 image_positional_embeddings=None
#                 ) -> Tuple[List[Tensor]]:
#         batch_size = x[0].shape[0]
#         mask_features, multi_scale_memorys = self.pixel_decoder(
#             x)  # mask_features.shape=(2, 256, 64, 64), multi_scale_memorys是个list，包含三个形状的张量(2,128,4,4)、(2,128,8,8)、(2,128,16,16)
#         # multi_scale_memorys (from low resolution to high resolution)
#         decoder_inputs = []
#         decoder_positional_encodings = []
#         for i in range(self.num_transformer_feat_level):
#             decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])  # (2, 128, 4, 4)
#             # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
#             decoder_input = decoder_input.flatten(2).permute(0, 2, 1)  # (2, 16, 128)
#             level_embed = self.level_embed.weight[i].view(1, 1, -1)  # (1, 1, 128)
#             decoder_input = decoder_input + level_embed
#             # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
#             mask = decoder_input.new_zeros(
#                 (batch_size,) + multi_scale_memorys[i].shape[-2:],
#                 dtype=torch.bool)  # (2, 4, 4)
#             decoder_positional_encoding = self.decoder_positional_encoding(mask).to(
#                 decoder_input.dtype)  # (2, 128, 4, 4)
#             decoder_positional_encoding = decoder_positional_encoding.flatten(
#                 2).permute(0, 2, 1)  # (2, 16, 128)
#             decoder_inputs.append(decoder_input)
#             decoder_positional_encodings.append(decoder_positional_encoding)
#         # shape (num_queries, c) -> (batch_size, num_queries, c)
#         query_feat = self.query_feat.weight.unsqueeze(0).repeat(
#             (batch_size, 1, 1))  # (2, 70, 128)
#         query_embed = self.query_embed.weight.unsqueeze(0).repeat(
#             (batch_size, 1, 1))  # (2, 70, 128)
#
#         cls_pred_list = []
#         mask_pred_list = []
#         mask_pred_plus_list = []
#         attn_mask = None
#
#         cls_pred, mask_pred, attn_mask, mask_pred_plus = self._forward_head(query_feat, mask_features,
#                                                                             multi_scale_memorys[0].shape[-2:],
#                                                                             image_embeddings,
#                                                                             image_positional_embeddings)
#         cls_pred_list.append(cls_pred)  # len=1
#         mask_pred_list.append(mask_pred)  # len=1
#         mask_pred_plus_list.append(mask_pred_plus)  # len=1
#
#         for i in range(self.num_transformer_decoder_layers):
#             level_idx = i % self.num_transformer_feat_level
#             if attn_mask is not None:
#                 # if a mask is all True(all background), then set it all False.
#                 mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
#                 attn_mask = attn_mask & mask_sum
#
#             # cross_attn + self_attn
#             layer = self.transformer_decoder.layers[i]
#             query_feat = layer(
#                 query=query_feat,
#                 key=decoder_inputs[level_idx],
#                 value=decoder_inputs[level_idx],
#                 query_pos=query_embed,
#                 key_pos=decoder_positional_encodings[level_idx],
#                 cross_attn_mask=attn_mask,
#                 query_key_padding_mask=None,
#                 # here we do not apply masking on padded region
#                 key_padding_mask=None)
#             cls_pred, mask_pred, attn_mask, mask_pred_plus = self._forward_head(
#                 query_feat, mask_features, multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
#                 image_embeddings, image_positional_embeddings)
#
#             cls_pred_list.append(cls_pred)
#             mask_pred_list.append(mask_pred)
#             mask_pred_plus_list.append(mask_pred_plus)
#         return cls_pred_list, mask_pred_list, mask_pred_plus_list  # 三个list的len都是7
#
#     def loss(
#             self,
#             x: Tuple[Tensor],
#             batch_data_samples: SampleList,
#             image_embeddings=None,  # (b, c, h, w)
#             image_positional_embeddings=None  # 和image_embedding形状相同
#     ) -> Dict[str, Tensor]:
#         """Perform forward propagation and loss calculation of the panoptic
#         head on the features of the upstream network.
#
#         Args:
#             x (tuple[Tensor]): Multi-level features from the upstream
#                 network, each is a 4D-tensor.
#             batch_data_samples (List[:obj:`DetDataSample`]): The Data
#                 Samples. It usually includes information such as
#                 `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
#
#         Returns:
#             dict[str, Tensor]: a dictionary of loss components
#         """
#         batch_img_metas = []
#         batch_gt_instances = []
#         batch_gt_semantic_segs = []
#         for data_sample in batch_data_samples:
#             batch_img_metas.append(data_sample.metainfo)
#             batch_gt_instances.append(data_sample.gt_instances)
#             if 'gt_sem_seg' in data_sample:
#                 batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
#             else:
#                 batch_gt_semantic_segs.append(None)
#
#         # forward
#         all_cls_scores, all_mask_preds, all_mask_preds_plus = self(x, batch_data_samples, image_embeddings,
#                                                                    image_positional_embeddings)
#         # preprocess ground truth
#         batch_gt_instances = self.preprocess_gt(batch_gt_instances,
#                                                 batch_gt_semantic_segs)
#         # loss
#         losses = self.loss_by_feat(all_cls_scores, all_mask_preds, all_mask_preds_plus,
#                                    batch_gt_instances, batch_img_metas)
#         return losses
#
#     def loss_by_feat(self,
#                      all_cls_scores: Tensor,
#                      all_mask_preds: Tensor,
#                      all_mask_preds_plus,
#                      batch_gt_instances: List[InstanceData],
#                      batch_img_metas: List[dict]) -> Dict[str, Tensor]:
#         num_dec_layers = len(all_cls_scores)
#         batch_gt_instances_list = [
#             batch_gt_instances for _ in range(num_dec_layers)
#         ]
#         img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
#         losses_cls, losses_mask, losses_dice, losses_mask_plus, losses_dice_plus = multi_apply(
#             self._loss_by_feat_single,
#             all_cls_scores, all_mask_preds,
#             all_mask_preds_plus,
#             batch_gt_instances_list, img_metas_list)
#
#         loss_dict = dict()
#         # loss from the last decoder layer
#         loss_dict['loss_cls'] = losses_cls[-1]
#         loss_dict['loss_mask'] = losses_mask[-1]
#         loss_dict['loss_dice'] = losses_dice[-1]
#         loss_dict['loss_mask_plus'] = losses_mask_plus[-1]
#         loss_dict['loss_dice_plus'] = losses_dice_plus[-1]
#         # loss from other decoder layers
#         num_dec_layer = 0
#         for loss_cls_i, loss_mask_i, loss_dice_i, loss_mask_plus_i, loss_dice_plus_i in zip(
#                 losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], losses_mask_plus[:-1], losses_dice_plus[:-1]):
#             loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
#             loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
#             loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
#             loss_dict[f'd{num_dec_layer}.loss_mask_plus'] = loss_mask_plus_i
#             loss_dict[f'd{num_dec_layer}.loss_dice_plus'] = loss_dice_plus_i
#
#             num_dec_layer += 1
#         return loss_dict
#
#     def _loss_by_feat_single(self,
#                              cls_scores: Tensor,
#                              mask_preds: Tensor,
#                              mask_preds_plus,
#                              batch_gt_instances: List[InstanceData],
#                              batch_img_metas: List[dict]) -> Tuple[Tensor]:
#         num_imgs = cls_scores.size(0)
#         cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
#         mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
#         mask_preds_plus_list = [mask_preds_plus[i] for i in range(num_imgs)]
#
#         (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
#          avg_factor) = self.get_targets(cls_scores_list, mask_preds_plus_list,
#                                         batch_gt_instances, batch_img_metas)
#
#         # shape (batch_size, num_queries)
#         labels = torch.stack(labels_list, dim=0)
#         # shape (batch_size, num_queries)
#         label_weights = torch.stack(label_weights_list, dim=0)
#         # shape (num_total_gts, h, w)
#         mask_targets = torch.cat(mask_targets_list, dim=0)
#         # shape (batch_size, num_queries)
#         mask_weights = torch.stack(mask_weights_list, dim=0)
#
#         # classfication loss
#         # shape (batch_size * num_queries, )
#         cls_scores = cls_scores.flatten(0, 1)
#         labels = labels.flatten(0, 1)
#         label_weights = label_weights.flatten(0, 1)
#
#         class_weight = cls_scores.new_tensor(self.class_weight)
#         loss_cls = self.loss_cls(
#             cls_scores,
#             labels,
#             label_weights,
#             avg_factor=class_weight[labels].sum())
#
#         num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
#         num_total_masks = max(num_total_masks, 1)
#
#         # extract positive ones
#         # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
#         mask_preds = mask_preds[mask_weights > 0]
#         mask_preds_plus = mask_preds_plus[mask_weights > 0]
#
#         if mask_targets.shape[0] == 0:
#             # zero match
#             loss_dice = mask_preds.sum()
#             loss_mask = mask_preds.sum()
#             loss_dice_plus = mask_preds_plus.sum()
#             loss_mask_plus = mask_preds_plus.sum()
#             return loss_cls, loss_mask, loss_dice, loss_mask_plus, loss_dice_plus
#
#         with torch.no_grad():
#             points_coords = get_uncertain_point_coords_with_randomness(
#                 mask_preds.unsqueeze(1), None, self.num_points,
#                 self.oversample_ratio, self.importance_sample_ratio)
#             # points_coords = points_coords.to(mask_preds.dtype)
#             # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
#             mask_point_targets = point_sample(
#                 mask_targets.unsqueeze(1).to(mask_preds.dtype), points_coords).squeeze(1)
#         # shape (num_queries, h, w) -> (num_queries, num_points)
#         mask_point_preds = point_sample(
#             mask_preds.unsqueeze(1), points_coords).squeeze(1)
#         mask_point_preds_plus = point_sample(
#             mask_preds_plus.unsqueeze(1), points_coords).squeeze(1)
#
#         # dice loss
#         loss_dice = self.loss_dice(
#             mask_point_preds, mask_point_targets, avg_factor=num_total_masks)
#         loss_dice_plus = self.loss_dice(
#             mask_point_preds_plus, mask_point_targets, avg_factor=num_total_masks)
#
#         # mask loss
#         # shape (num_queries, num_points) -> (num_queries * num_points, )
#         mask_point_preds = mask_point_preds.reshape(-1)
#         # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
#         mask_point_targets = mask_point_targets.reshape(-1)
#
#         mask_point_preds_plus = mask_point_preds_plus.reshape(-1)
#
#         # loss_mask = self.loss_mask(
#         #     mask_point_preds,
#         #     mask_point_targets,
#         #     avg_factor=num_total_masks * self.num_points)
#         # to avoid nan in fp16 when num_total_masks * self.num_points
#         loss_mask = self.loss_mask(mask_point_preds, mask_point_targets)
#         loss_mask_plus = self.loss_mask(mask_point_preds_plus, mask_point_targets)
#         return loss_cls, loss_mask, loss_dice, loss_mask_plus, loss_dice_plus
#
#     def predict(self, x: Tuple[Tensor],
#                 batch_data_samples: SampleList,
#                 image_embeddings=None,
#                 image_positional_embeddings=None
#                 ) -> Tuple[Tensor]:
#         batch_img_metas = [
#             data_sample.metainfo for data_sample in batch_data_samples
#         ]
#         all_cls_scores, all_mask_preds, all_mask_preds_plus = self(
#             x, batch_data_samples, image_embeddings=image_embeddings,
#             image_positional_embeddings=image_positional_embeddings)
#         mask_cls_results = all_cls_scores[-1]
#         mask_pred_results = all_mask_preds[-1]
#         mask_pred_plus_results = all_mask_preds_plus[-1]
#         # upsample masks
#         try:
#             img_shape = batch_img_metas[0]['batch_input_shape']
#         except:
#             img_shape = batch_img_metas[0]['pad_shape']
#         mask_pred_results = F.interpolate(
#             mask_pred_results,
#             size=(img_shape[0], img_shape[1]),
#             mode='bilinear',
#             align_corners=False)
#
#         return mask_cls_results, mask_pred_results
#
#
# @MODELS.register_module()
# class RSMaskFormerFusionHead(MaskFormerFusionHead):
#     def predict(self,
#                 mask_cls_results: Tensor,
#                 mask_pred_results: Tensor,
#                 batch_data_samples: SampleList,
#                 rescale: bool = False,
#                 **kwargs) -> List[dict]:
#         batch_img_metas = [
#             data_sample.metainfo for data_sample in batch_data_samples
#         ]
#         panoptic_on = self.test_cfg.get('panoptic_on', True)
#         semantic_on = self.test_cfg.get('semantic_on', False)
#         instance_on = self.test_cfg.get('instance_on', False)
#         assert not semantic_on, 'segmantic segmentation ' \
#                                 'results are not supported yet.'
#         results = []
#         for mask_cls_result, mask_pred_result, meta in zip(
#                 mask_cls_results, mask_pred_results, batch_img_metas):
#             # remove padding
#             img_height, img_width = meta['img_shape'][:2]
#             ori_img_height, ori_img_width = meta['ori_shape'][:2]
#             scale_factor = meta['scale_factor']
#             ori_scaled_height = int(ori_img_height * scale_factor[1])
#             ori_scaled_width = int(ori_img_width * scale_factor[0])
#             mask_pred_result = mask_pred_result[:, :ori_scaled_height, :ori_scaled_width]
#
#             if rescale:
#                 # return result in original resolution
#                 ori_height, ori_width = meta['ori_shape'][:2]
#                 mask_pred_result = F.interpolate(
#                     mask_pred_result[:, None],
#                     size=(ori_height, ori_width),
#                     mode='bilinear',
#                     align_corners=False)[:, 0]
#
#             result = dict()
#             if panoptic_on:
#                 pan_results = self.panoptic_postprocess(
#                     mask_cls_result, mask_pred_result)
#                 result['pan_results'] = pan_results
#
#             if instance_on:
#                 ins_results = self.instance_postprocess(
#                     mask_cls_result, mask_pred_result)
#                 result['ins_results'] = ins_results
#
#             if semantic_on:
#                 sem_results = self.semantic_postprocess(
#                     mask_cls_result, mask_pred_result)
#                 result['sem_results'] = sem_results
#
#             results.append(result)
#
#         return results
#
#
# @MODELS.register_module()
# class RSSamModel(BaseModule):
#     def __init__(
#             self,
#             hf_pretrain_name,
#             extra_config=None,
#             init_cfg=None,
#     ):
#         BaseModule.__init__(self, init_cfg=init_cfg)
#         sam_config = SamConfig.from_pretrained(hf_pretrain_name)
#         if extra_config is not None:
#             sam_config.update(extra_config)
#         self.sam_model = SamModel(sam_config)
#
#         if init_cfg is not None:
#             from mmengine.runner.checkpoint import load_checkpoint
#             load_checkpoint(self.sam_model, init_cfg.get('checkpoint'))
#             self.sam_model.is_init = True
#
#     def init_weights(self):
#         pass
#
#     def forward(self, *args, **kwargs):
#         return self.sam_model(*args, **kwargs)
#
#
# @MODELS.register_module()
# class RSSamPositionalEmbedding(SamPositionalEmbedding, BaseModule):
#     def __init__(
#             self,
#             hf_pretrain_name,
#             extra_config=None,
#             init_cfg=None,
#     ):
#         BaseModule.__init__(self, init_cfg=init_cfg)
#         sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
#         if extra_config is not None:
#             sam_config.update(extra_config)
#         self.shared_image_embedding = SamPositionalEmbedding(sam_config)
#
#     def forward(self, *args, **kwargs):
#         return self.shared_image_embedding(*args, **kwargs)
#
#
# @MODELS.register_module()
# class RSSamVisionEncoder(BaseModule):
#     def __init__(
#             self,
#             hf_pretrain_name,
#             extra_config=None,
#             peft_config=None,
#             init_cfg=None,
#     ):
#         BaseModule.__init__(self, init_cfg=init_cfg)
#         sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
#         if extra_config is not None:
#             sam_config.update(extra_config)
#         vision_encoder = SamVisionEncoder(sam_config)
#         # a = torch.load(init_cfg.get('checkpoint'))
#         # b = vision_encoder.state_dict()
#         # load checkpoint
#         if init_cfg is not None:
#             from mmengine.runner.checkpoint import load_checkpoint
#             load_checkpoint(
#                 vision_encoder,
#                 init_cfg.get('checkpoint'),
#                 map_location='cpu',
#                 revise_keys=[(r'^module\.', ''), (r'^vision_encoder\.', '')])
#         # LoRA 微调（可选）
#         if peft_config is not None and isinstance(peft_config, dict):
#             config = {
#                 "peft_type": "LORA",
#                 "r": 16,
#                 'target_modules': ["qkv"],
#                 "lora_alpha": 32,
#                 "lora_dropout": 0.05,
#                 "bias": "none",
#                 "inference_mode": False,
#             }
#             config.update(peft_config)
#             peft_config = get_peft_config(config)
#             self.vision_encoder = get_peft_model(vision_encoder, peft_config)
#             if is_main_process():
#                 self.vision_encoder.print_trainable_parameters()
#         else:
#             # 不用LORA时
#             self.vision_encoder = vision_encoder
#         self.vision_encoder.is_init = True
#
#     def init_weights(self):
#         if is_main_process():
#             print('the vision encoder has been initialized')
#
#     def forward(self, *args, **kwargs):
#         return self.vision_encoder(*args, **kwargs)
#
#
# @MODELS.register_module()
# class MMPretrainSamVisionEncoder(BaseModule):
#     def __init__(
#             self,
#             hf_pretrain_name,
#             img_size=1024,
#             peft_config=None,
#             init_cfg=None,
#     ):
#         super().__init__(init_cfg=init_cfg)
#         vision_encoder_cfg = dict(
#             type='mmpretrain.ViTSAM',
#             arch=hf_pretrain_name.split('-')[-1].split('_')[-1],
#             img_size=img_size,
#             patch_size=16,
#             out_channels=256,
#             use_abs_pos=True,
#             use_rel_pos=True,
#             window_size=14,
#         )
#         vision_encoder = MODELS.build(vision_encoder_cfg)
#         # a = torch.load(init_cfg.get('checkpoint'))
#         # b = vision_encoder.state_dict()
#         # load checkpoint
#         if init_cfg is not None:
#             from mmengine.runner.checkpoint import load_checkpoint
#             load_checkpoint(
#                 vision_encoder,
#                 init_cfg.get('checkpoint'),
#                 map_location='cpu',
#                 revise_keys=[
#                     (r'^module\.', ''),
#                     (r'^vision_encoder\.', ''),
#                     (r'.layer_norm1.', '.ln1.'),
#                     (r'.layer_norm2.', '.ln2.'),
#                     (r'.mlp.lin1.', '.ffn.layers.0.0.'),
#                     (r'.mlp.lin2.', '.ffn.layers.1.'),
#                     (r'neck.conv1.', 'channel_reduction.0.'),
#                     (r'neck.ln1.', 'channel_reduction.1.'),
#                     (r'neck.conv2.', 'channel_reduction.2.'),
#                     (r'neck.ln2.', 'channel_reduction.3.'),
#                 ]
#             )
#
#         if peft_config is not None and isinstance(peft_config, dict):
#             config = {
#                 "peft_type": "LORA",
#                 "r": 16,
#                 'target_modules': ["qkv"],
#                 "lora_alpha": 32,
#                 "lora_dropout": 0.05,
#                 "bias": "none",
#                 "inference_mode": False,
#             }
#             config.update(peft_config)
#             peft_config = get_peft_config(config)
#             self.vision_encoder = get_peft_model(vision_encoder, peft_config)
#             if is_main_process():
#                 self.vision_encoder.print_trainable_parameters()
#         else:
#             self.vision_encoder = vision_encoder
#         self.vision_encoder.is_init = True
#
#     def init_weights(self):
#         if is_main_process():
#             print('the vision encoder has been initialized')
#
#     def forward(self, *args, **kwargs):
#         return self.vision_encoder(*args, **kwargs)
#
#
# @MODELS.register_module()
# class RSSamPromptEncoder(SamPromptEncoder, BaseModule):
#     def __init__(
#             self,
#             hf_pretrain_name,
#             extra_config=None,
#             init_cfg=None,
#     ):
#         BaseModule.__init__(self, init_cfg=init_cfg)
#         sam_config = SamConfig.from_pretrained(hf_pretrain_name).prompt_encoder_config
#         if extra_config is not None:
#             sam_config.update(extra_config)
#         self.prompt_encoder = SamPromptEncoder(sam_config, shared_patch_embedding=None)
#
#     def forward(self, *args, **kwargs):
#         return self.prompt_encoder(*args, **kwargs)
#
#
# @MODELS.register_module()
# class RSSamMaskDecoder(SamMaskDecoder, BaseModule):
#     def __init__(
#             self,
#             hf_pretrain_name,
#             extra_config=None,
#             init_cfg=None,
#     ):
#         BaseModule.__init__(self, init_cfg=init_cfg)
#         sam_config = SamConfig.from_pretrained(hf_pretrain_name).mask_decoder_config
#         if extra_config is not None:
#             sam_config.update(extra_config)
#         self.mask_decoder = SamMaskDecoder(sam_config)
#
#     def forward(self, *args, **kwargs):
#         return self.mask_decoder(*args, **kwargs)
#
#
# @MODELS.register_module()
# class RSFPN(BaseModule):
#     def __init__(
#             self,
#             feature_aggregator=None,
#             feature_spliter=None,
#             init_cfg=None,
#     ):
#         super().__init__(init_cfg=init_cfg)
#         if feature_aggregator is not None:
#             self.feature_aggregator = MODELS.build(feature_aggregator)
#         if feature_spliter is not None:
#             self.feature_spliter = MODELS.build(feature_spliter)
#
#     def forward(self, inputs):
#         if hasattr(self, 'feature_aggregator'):
#             x = self.feature_aggregator(inputs)
#         else:
#             x = inputs
#         if hasattr(self, 'feature_spliter'):
#             x = self.feature_spliter(x)
#         else:
#             x = (x,)
#         return x
#
#
# @MODELS.register_module()
# class PseudoFeatureAggregator(BaseModule):
#     def __init__(
#             self,
#             in_channels,  # rsprompter_anchor_LIACi，256，固定的，不受图像分辨率影响
#             hidden_channels=64,  # 512，固定
#             out_channels=256,  # 256，固定
#             init_cfg=None,
#     ):
#         super().__init__(init_cfg=init_cfg)
#         '''
#         实际上就是直接经过三层卷积，对SAM的image encoder输出的特征图做融合操作，连残差链接都没有
#         '''
#         # # 上采样层，将特征图从 16x16 扩展到 64x64
#         # self.channel_fusion =nn.Sequential(
#         #     nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
#         #
#         # )
#
#         # self.channel_fusion = nn.Sequential(
#         #     nn.ConvTranspose2d(
#         #         in_channels,
#         #         512,
#         #         kernel_size=2,
#         #         stride=2,
#         #         padding=0,
#         #         bias=False,
#         #     ),
#         #     LayerNorm2d(512, eps=1e-6),
#         #     nn.GELU(),
#         #     # nn.ConvTranspose2d(
#         #     #     512,
#         #     #     512,
#         #     #     kernel_size=2,
#         #     #     stride=2,
#         #     #     padding=0,
#         #     #     bias=False,
#         #     # ),
#         #     # LayerNorm2d(512, eps=1e-6),
#         #     nn.ConvTranspose2d(
#         #         512,
#         #         256,
#         #         kernel_size=2,
#         #         stride=2,
#         #         padding=0,
#         #         bias=False,
#         #     ),
#         #     LayerNorm2d(256, eps=1e-6),
#         #     nn.GELU(),
#         # )
#
#         ''' 删掉这两行 '''
#         # self.channel_fusion = nn.Sequential(
#         #     nn.Conv2d(
#         #         in_channels,
#         #         out_channels,
#         #         kernel_size=1,
#         #         bias=False,
#         #     ),
#         #     LayerNorm2d(out_channels, eps=1e-6),
#         #     nn.Conv2d(
#         #         out_channels,
#         #         out_channels,
#         #         kernel_size=3,
#         #         padding=1,
#         #         bias=False,
#         #     ),
#         #     LayerNorm2d(out_channels, eps=1e-6),
#         # )
#
#         ''' 作者原版代码 '''
#         self.channel_fusion = nn.Sequential(
#             nn.Conv2d(
#                 in_channels,
#                 hidden_channels,
#                 kernel_size=1,
#                 bias=False,
#             ),
#             LayerNorm2d(hidden_channels, eps=1e-6),
#             nn.Conv2d(
#                 hidden_channels,
#                 hidden_channels,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),
#             LayerNorm2d(hidden_channels, eps=1e-6),
#             nn.Conv2d(
#                 hidden_channels,
#                 out_channels,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),
#             LayerNorm2d(out_channels, eps=1e-6),
#         )
#
#     def forward(self, inputs):
#         assert len(inputs) == 1  # 要求inputs是个tuple
#         x = inputs[0]
#         x = self.channel_fusion(x)
#         return x
#
#
# @MODELS.register_module()
# class RSFeatureAggregator(BaseModule):
#     # RSPrompter论文中的 Fig.4实现的功能
#
#     in_channels_dict = {
#         'base': [768] * (12 + 1),  # [768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768]
#         'large': [1024] * (24 + 1),
#         'huge': [1280] * (32 + 1),
#     }
#
#     def __init__(
#             self,
#             in_channels,
#             hidden_channels=64,
#             out_channels=256,
#             select_layers=range(1, 12, 2),  # range(1, 13, 2) = [1, 3, 5, 7, 9, 11]
#
#             init_cfg=None,
#     ):
#         super().__init__(init_cfg=init_cfg)
#         assert isinstance(in_channels, str)
#         model_arch = 'base' if 'base' in in_channels else 'large' if 'large' in in_channels else 'huge'
#         self.in_channels = self.in_channels_dict[model_arch]
#         self.select_layers = select_layers
#
#         self.downconvs = nn.ModuleList()
#         for i_layer in self.select_layers:
#             self.downconvs.append(
#                 nn.Sequential(
#                     nn.Conv2d(self.in_channels[i_layer], hidden_channels, 1),
#                     nn.BatchNorm2d(hidden_channels),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
#                     nn.BatchNorm2d(hidden_channels),
#                     nn.ReLU(inplace=True),
#                 )
#             )
#
#         self.hidden_convs = nn.ModuleList()
#         for _ in self.select_layers:
#             self.hidden_convs.append(
#                 nn.Sequential(
#                     nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
#                     nn.BatchNorm2d(hidden_channels),
#                     nn.ReLU(inplace=True),
#                 )
#             )
#
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(hidden_channels, out_channels, 1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#         )
#
#     def forward(self, inputs):
#         assert len(inputs) == len(self.in_channels)
#         inputs = [einops.rearrange(x, 'b h w c -> b c h w') for x in inputs]
#
#         features = []
#         for idx, i_layer in enumerate(self.select_layers):
#             features.append(self.downconvs[idx](inputs[i_layer]))
#
#         x = None
#         for hidden_state, hidden_conv in zip(features, self.hidden_convs):
#             if x is not None:
#                 hidden_state = x + hidden_state
#             residual = hidden_conv(hidden_state)
#             x = hidden_state + residual
#         x = self.fusion_conv(x)
#         return x
#
#
# @MODELS.register_module()
# class SAMDet(BaseDetector):
#     def __init__(
#             self,
#             detector,
#             segmentor,
#             data_preprocessor=None,
#             test_cfg=None,
#             init_cfg=None):
#         super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
#         self.detector = MODELS.build(detector)
#         self.segmentor = MODELS.build(segmentor)
#         self.segmentor.eval()
#         self.test_cfg = test_cfg
#         for param in self.segmentor.parameters():
#             param.requires_grad = False
#
#     def extract_feat(self, batch_inputs: Tensor):
#         pass
#
#     def _forward(self,
#                  batch_inputs: Tensor,
#                  batch_data_samples: OptSampleList = None):
#         pass
#
#     def loss(self, batch_inputs: Tensor,
#              batch_data_samples: SampleList) -> Union[dict, tuple]:
#         losses = self.detector.loss(batch_inputs, batch_data_samples)
#         return losses
#
#     def oracle_predict(self,
#                        batch_inputs: Tensor,
#                        batch_data_samples: SampleList,
#                        rescale: bool = True):
#
#         batch_data_samples = self.detector.predict(batch_inputs, batch_data_samples, rescale=rescale)
#         batch_img_metas = [
#             data_sample.metainfo for data_sample in batch_data_samples
#         ]
#         for input_img, data_sample, meta in zip(batch_inputs, batch_data_samples, batch_img_metas):
#             pred_instance_data = InstanceData()
#             pred_instance_data.bboxes = data_sample.gt_instances.bboxes
#             pred_instance_data.labels = data_sample.gt_instances.labels
#             pred_instance_data.scores = torch.ones_like(data_sample.gt_instances.labels, dtype=torch.float32,
#                                                         device=data_sample.gt_instances.labels.device)
#
#             bboxes = pred_instance_data.bboxes
#             ori_img_shape = data_sample.ori_shape
#             if len(bboxes) == 0:
#                 mask_pred_binary = torch.zeros(
#                     0,
#                     ori_img_shape[0],
#                     ori_img_shape[1],
#                     device=batch_inputs.device,
#                     dtype=torch.bool)
#             else:
#                 scale_factor = data_sample.scale_factor
#                 repeat_num = bboxes.size(-1) // 2
#                 scale_factor = bboxes.new_tensor(scale_factor).repeat((1, repeat_num))
#                 bboxes = bboxes * scale_factor
#
#                 input_img = input_img.unsqueeze(0)
#                 bboxes = bboxes.unsqueeze(0)
#                 outputs = self.segmentor(
#                     pixel_values=input_img,
#                     input_boxes=bboxes,
#                     multimask_output=False,
#                 )
#                 mask_pred_result = outputs.pred_masks
#                 mask_pred_result = mask_pred_result[0]
#                 mask_pred_result = mask_pred_result.squeeze(1)
#
#                 ori_img_height, ori_img_width = meta['ori_shape'][:2]
#                 scale_factor = meta['scale_factor']
#                 ori_scaled_height = int(ori_img_height * scale_factor[1])
#                 ori_scaled_width = int(ori_img_width * scale_factor[0])
#
#                 mask_pred_result = F.interpolate(
#                     mask_pred_result[:, None],
#                     size=meta['img_shape'],
#                     mode='bilinear',
#                     align_corners=False)[:, 0]
#
#                 mask_pred_result = mask_pred_result[:, :ori_scaled_height, :ori_scaled_width]
#                 # return result in original resolution
#                 ori_height, ori_width = meta['ori_shape'][:2]
#                 mask_pred_result = F.interpolate(
#                     mask_pred_result[:, None],
#                     size=(ori_height, ori_width),
#                     mode='bilinear',
#                     align_corners=False)[:, 0]
#                 mask_pred_binary = (mask_pred_result > 0)
#             pred_instance_data.masks = mask_pred_binary
#             data_sample.pred_instances = pred_instance_data
#         return batch_data_samples
#
#     def predict(self,
#                 batch_inputs: Tensor,
#                 batch_data_samples: SampleList,
#                 rescale: bool = True):
#         if self.test_cfg is not None and self.test_cfg.get('oracle_on', True):
#             return self.oracle_predict(batch_inputs, batch_data_samples, rescale=rescale)
#
#         batch_data_samples = self.detector.predict(batch_inputs, batch_data_samples, rescale=rescale)
#         batch_img_metas = [
#             data_sample.metainfo for data_sample in batch_data_samples
#         ]
#         for input_img, data_sample, meta in zip(batch_inputs, batch_data_samples, batch_img_metas):
#             bboxes = data_sample.pred_instances.bboxes
#             ori_img_shape = data_sample.ori_shape
#             if len(bboxes) == 0:
#                 mask_pred_binary = torch.zeros(
#                     0,
#                     ori_img_shape[0],
#                     ori_img_shape[1],
#                     device=batch_inputs.device,
#                     dtype=torch.bool)
#             else:
#                 scale_factor = data_sample.scale_factor
#                 repeat_num = bboxes.size(-1) // 2
#                 scale_factor = bboxes.new_tensor(scale_factor).repeat((1, repeat_num))
#                 bboxes = bboxes * scale_factor
#
#                 input_img = input_img.unsqueeze(0)
#                 bboxes = bboxes.unsqueeze(0)
#                 outputs = self.segmentor(
#                     pixel_values=input_img,
#                     input_boxes=bboxes,
#                     multimask_output=False,
#                 )
#                 mask_pred_result = outputs.pred_masks
#                 mask_pred_result = mask_pred_result[0]
#                 mask_pred_result = mask_pred_result.squeeze(1)
#
#                 ori_img_height, ori_img_width = meta['ori_shape'][:2]
#                 scale_factor = meta['scale_factor']
#                 ori_scaled_height = int(ori_img_height * scale_factor[1])
#                 ori_scaled_width = int(ori_img_width * scale_factor[0])
#
#                 mask_pred_result = F.interpolate(
#                     mask_pred_result[:, None],
#                     size=meta['img_shape'],
#                     mode='bilinear',
#                     align_corners=False)[:, 0]
#
#                 mask_pred_result = mask_pred_result[:, :ori_scaled_height, :ori_scaled_width]
#                 # return result in original resolution
#                 ori_height, ori_width = meta['ori_shape'][:2]
#                 mask_pred_result = F.interpolate(
#                     mask_pred_result[:, None],
#                     size=(ori_height, ori_width),
#                     mode='bilinear',
#                     align_corners=False)[:, 0]
#                 mask_pred_binary = (mask_pred_result > 0)
#             data_sample.pred_instances.masks = mask_pred_binary
#
#         return batch_data_samples
#
#
# @MODELS.register_module()
# class SAMSegMaskRCNN(MaskRCNN):
#     def __init__(
#             self,
#             *args,
#             **kwargs,
#     ):
#         peft_config = kwargs.get('backbone', {}).get('peft_config', {})
#         super().__init__(*args, **kwargs)
#
#         if peft_config is None:
#             self.backbone.eval()
#             for param in self.backbone.parameters():
#                 param.requires_grad = False
#
#     def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
#         vision_outputs = self.backbone(batch_inputs)
#         if isinstance(vision_outputs, SamVisionEncoderOutput):
#             image_embeddings = vision_outputs.last_hidden_state
#             vision_hidden_states = vision_outputs.hidden_states
#         elif isinstance(vision_outputs, tuple):
#             image_embeddings = vision_outputs[0]
#             vision_hidden_states = vision_outputs
#         else:
#             raise NotImplementedError
#         x = self.neck(vision_hidden_states)
#         return x
#
#
# @MODELS.register_module()
# class SAMSegMask2Former(Mask2Former):
#     def __init__(
#             self,
#             *args,
#             **kwargs,
#     ):
#         peft_config = kwargs.get('backbone', {}).get('peft_config', {})
#         super().__init__(*args, **kwargs)
#
#         if peft_config is None:
#             self.backbone.eval()
#             for param in self.backbone.parameters():
#                 param.requires_grad = False
#
#     def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
#         vision_outputs = self.backbone(batch_inputs)
#         if isinstance(vision_outputs, SamVisionEncoderOutput):
#             image_embeddings = vision_outputs.last_hidden_state
#             vision_hidden_states = vision_outputs.hidden_states
#         elif isinstance(vision_outputs, tuple):
#             image_embeddings = vision_outputs[0]
#             vision_hidden_states = vision_outputs
#         else:
#             raise NotImplementedError
#
#         x = self.neck(vision_hidden_states)
#         return x
#
#
# @MODELS.register_module()
# class RSSimpleFPN(BaseModule):
#     #
#     def __init__(self,
#                  backbone_channel: int,
#                  in_channels: List[int],
#                  out_channels: int,
#                  num_outs: int,
#                  conv_cfg: OptConfigType = None,
#                  norm_cfg: OptConfigType = None,
#                  act_cfg: OptConfigType = None,
#                  init_cfg: MultiConfig = None) -> None:
#         super().__init__(init_cfg=init_cfg)
#         assert isinstance(in_channels, list)
#         self.backbone_channel = backbone_channel  # 256
#         self.in_channels = in_channels  # [64, 128, 256, 256]
#         self.out_channels = out_channels  # 256
#         self.num_ins = len(in_channels)  # 4
#         self.num_outs = num_outs  # 5
#
#         self.fpn1 = nn.Sequential(
#             nn.ConvTranspose2d(self.backbone_channel,
#                                self.backbone_channel // 2, 2, 2),
#             build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
#             nn.GELU(),
#             nn.ConvTranspose2d(self.backbone_channel // 2,
#                                self.backbone_channel // 4, 2, 2))
#         self.fpn2 = nn.Sequential(
#             nn.ConvTranspose2d(self.backbone_channel,
#                                self.backbone_channel // 2, 2, 2))
#         self.fpn3 = nn.Sequential(nn.Identity())  # 直接返回输入，不对输入进行任何改变
#         self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.lateral_convs = nn.ModuleList()
#         self.fpn_convs = nn.ModuleList()
#
#         for i in range(self.num_ins):  # self.num_ins=4
#             l_conv = ConvModule(
#                 in_channels[i],
#                 out_channels,
#                 1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg,
#                 inplace=False)
#             fpn_conv = ConvModule(
#                 out_channels,
#                 out_channels,
#                 3,
#                 padding=1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg,
#                 inplace=False)
#             '''
#             self.lateral_convs：
#             ModuleList(
#               (0): ConvModule(
#                 (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 (norm_layer): LN2d()
#               )
#               (1): ConvModule(
#                 (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 (norm_layer): LN2d()
#               )
#               (2-3): 2 x ConvModule(
#                 (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 (norm_layer): LN2d()
#               )
#             )
#
#             self.fpn_convs:
#             ModuleList(
#               (0-3): 4 x ConvModule(
#                 (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#                 (norm_layer): LN2d()
#               )
#             )
#             '''
#             self.lateral_convs.append(l_conv)
#             self.fpn_convs.append(fpn_conv)
#
#     def forward(self, input: Tensor) -> tuple:
#         """Forward function.
#
#         Args:
#             inputs (Tensor): Features from the upstream network, 4D-tensor
#         Returns:
#             tuple: Feature maps, each is a 4D-tensor.
#         """
#         # build FPN
#         inputs = []
#         inputs.append(self.fpn1(input))
#         inputs.append(self.fpn2(input))
#         inputs.append(self.fpn3(input))
#         inputs.append(self.fpn4(input))
#
#         # build laterals
#         laterals = [
#             lateral_conv(inputs[i])
#             for i, lateral_conv in enumerate(self.lateral_convs)
#         ]
#
#         # build outputs
#         # part 1: from original levels
#         outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]
#
#         # part 2: add extra levels
#         if self.num_outs > len(outs):
#             for i in range(self.num_outs - self.num_ins):
#                 outs.append(F.max_pool2d(outs[-1], 1, stride=2))
#         return tuple(outs)
#
#
# @MODELS.register_module()
# class RSPrompterAnchorRoIPromptHead(StandardRoIHead):
#     def __init__(
#             self,
#             with_extra_pe=False,
#             *args,
#             **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         if with_extra_pe:
#             out_channels = self.bbox_roi_extractor.out_channels
#             positional_encoding = dict(
#                 num_feats=out_channels // 2,
#                 normalize=True,
#             )
#             self.extra_pe = SinePositionalEncoding(**positional_encoding)
#
#     def _mask_forward(self,
#                       x: Tuple[Tensor],
#                       rois: Tensor = None,
#                       pos_inds: Optional[Tensor] = None,
#                       bbox_feats: Optional[Tensor] = None,
#                       image_embeddings=None,
#                       image_positional_embeddings=None,
#                       ) -> dict:
#         assert ((rois is not None) ^
#                 (pos_inds is not None and bbox_feats is not None))
#         if rois is not None:
#             mask_feats = self.mask_roi_extractor(
#                 x[:self.mask_roi_extractor.num_inputs], rois)
#             if self.with_shared_head:
#                 mask_feats = self.shared_head(mask_feats)
#         else:
#             assert bbox_feats is not None
#             mask_feats = bbox_feats[pos_inds]
#
#         mask_preds, iou_predictions = self.mask_head(
#             mask_feats,
#             image_embeddings=image_embeddings,
#             image_positional_embeddings=image_positional_embeddings,
#             roi_img_ids=rois[:, 0] if rois is not None else None,
#         )
#         mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats, iou_predictions=iou_predictions)
#         return mask_results
#
#     def mask_loss(self, x: Tuple[Tensor],
#                   sampling_results: List[SamplingResult], bbox_feats: Tensor,
#                   batch_gt_instances: InstanceList,
#                   image_embeddings=None,
#                   image_positional_embeddings=None,
#                   ) -> dict:
#         if not self.share_roi_extractor:
#             pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
#             if len(pos_rois) == 0:
#                 print('no pos rois')
#                 return dict(loss_mask=dict(loss_mask=0 * x[0].sum()))
#             mask_results = self._mask_forward(
#                 x, pos_rois,
#                 image_embeddings=image_embeddings,
#                 image_positional_embeddings=image_positional_embeddings,
#             )
#         else:
#             pos_inds = []
#             device = bbox_feats.device
#             for res in sampling_results:
#                 pos_inds.append(
#                     torch.ones(
#                         res.pos_priors.shape[0],
#                         device=device,
#                         dtype=torch.uint8))
#                 pos_inds.append(
#                     torch.zeros(
#                         res.neg_priors.shape[0],
#                         device=device,
#                         dtype=torch.uint8))
#             pos_inds = torch.cat(pos_inds)
#
#             mask_results = self._mask_forward(
#                 x, pos_inds=pos_inds, bbox_feats=bbox_feats)
#
#         mask_loss_and_target = self.mask_head.loss_and_target(
#             mask_preds=mask_results['mask_preds'],
#             sampling_results=sampling_results,
#             batch_gt_instances=batch_gt_instances,
#             rcnn_train_cfg=self.train_cfg)
#
#         mask_results.update(loss_mask=mask_loss_and_target['loss_mask'])
#         return mask_results
#
#     def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
#              batch_data_samples: List[DetDataSample],
#              # extra inputs
#              image_embeddings=None,
#              image_positional_embeddings=None,
#              ) -> dict:
#         assert len(rpn_results_list) == len(batch_data_samples)
#         outputs = unpack_gt_instances(batch_data_samples)
#         batch_gt_instances, batch_gt_instances_ignore, _ = outputs
#
#         if hasattr(self, 'extra_pe'):
#             bs, _, h, w = x[0].shape
#             mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
#             img_feats_pe = self.extra_pe(mask_pe)
#             outputs = []
#             for i in range(len(x)):
#                 output = x[i] + F.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
#                 outputs.append(output)
#             x = tuple(outputs)
#
#         # assign gts and sample proposals
#         num_imgs = len(batch_data_samples)
#         sampling_results = []
#         for i in range(num_imgs):
#             # rename rpn_results.bboxes to rpn_results.priors
#             rpn_results = rpn_results_list[i]  # 1000个proposal
#             rpn_results.priors = rpn_results.pop('bboxes')  # 将边界框信息存储在新的属性 priors 中，后续的代码中使用 priors
#
#             assign_result = self.bbox_assigner.assign(
#                 rpn_results, batch_gt_instances[i],
#                 batch_gt_instances_ignore[i])
#             sampling_result = self.bbox_sampler.sample(
#                 assign_result,
#                 rpn_results,
#                 batch_gt_instances[i],
#                 feats=[lvl_feat[i][None] for lvl_feat in x])
#             sampling_results.append(sampling_result)
#
#         losses = dict()
#         # bbox head loss
#         if self.with_bbox:
#             bbox_results = self.bbox_loss(x, sampling_results)
#             losses.update(bbox_results['loss_bbox'])
#
#         # mask head forward and loss
#         if self.with_mask:
#             mask_results = self.mask_loss(
#                 x, sampling_results, bbox_results['bbox_feats'], batch_gt_instances,
#                 image_embeddings=image_embeddings,
#                 image_positional_embeddings=image_positional_embeddings,
#             )
#             losses.update(mask_results['loss_mask'])
#
#         return losses
#
#     def predict_mask(
#             self,
#             x: Tuple[Tensor],
#             batch_img_metas: List[dict],
#             results_list: InstanceList,
#             rescale: bool = False,
#             image_embeddings=None,
#             image_positional_embeddings=None,
#     ) -> InstanceList:
#
#         # don't need to consider aug_test.
#         bboxes = [res.bboxes for res in results_list]
#         mask_rois = bbox2roi(bboxes)
#         if mask_rois.shape[0] == 0:
#             results_list = empty_instances(
#                 batch_img_metas,
#                 mask_rois.device,
#                 task_type='mask',
#                 instance_results=results_list,
#                 mask_thr_binary=self.test_cfg.mask_thr_binary)
#             return results_list
#
#         mask_results = self._mask_forward(
#             x, mask_rois,
#             image_embeddings=image_embeddings,
#             image_positional_embeddings=image_positional_embeddings)
#
#         mask_preds = mask_results['mask_preds']
#         # split batch mask prediction back to each image
#         num_mask_rois_per_img = [len(res) for res in results_list]
#         mask_preds = mask_preds.split(num_mask_rois_per_img, 0)
#
#         # TODO: Handle the case where rescale is false
#         results_list = self.mask_head.predict_by_feat(
#             mask_preds=mask_preds,
#             results_list=results_list,
#             batch_img_metas=batch_img_metas,
#             rcnn_test_cfg=self.test_cfg,
#             rescale=rescale)
#         return results_list
#
#     def predict(self,
#                 x: Tuple[Tensor],
#                 rpn_results_list: InstanceList,
#                 batch_data_samples: SampleList,
#                 rescale: bool = False,
#                 # extra inputs
#                 image_embeddings=None,
#                 image_positional_embeddings=None,
#                 ) -> InstanceList:
#         batch_img_metas = [
#             data_samples.metainfo for data_samples in batch_data_samples
#         ]
#
#         if hasattr(self, 'extra_pe'):
#             bs, _, h, w = x[0].shape
#             mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
#             img_feats_pe = self.extra_pe(mask_pe)
#             outputs = []
#             for i in range(len(x)):
#                 output = x[i] + F.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
#                 outputs.append(output)
#             x = tuple(outputs)
#
#         # If it has the mask branch, the bbox branch does not need
#         # to be scaled to the original image scale, because the mask
#         # branch will scale both bbox and mask at the same time.
#         bbox_rescale = rescale if not self.with_mask else False
#         results_list = self.predict_bbox(
#             x,
#             batch_img_metas,
#             rpn_results_list,
#             rcnn_test_cfg=self.test_cfg,
#             rescale=bbox_rescale)
#
#         if self.with_mask:
#             results_list = self.predict_mask(
#                 x, batch_img_metas, results_list, rescale=rescale,
#                 image_embeddings=image_embeddings,
#                 image_positional_embeddings=image_positional_embeddings,
#             )
#         return results_list
#
#
# @MODELS.register_module()
# class RSPrompterAnchorMaskHead(FCNMaskHead, BaseModule):
#     def __init__(
#             self,
#             mask_decoder,
#             in_channels,
#             roi_feat_size=14,
#             per_pointset_point=5,
#             with_sincos=True,
#             multimask_output=False,
#             attention_similarity=None,
#             target_embedding=None,
#             output_attentions=None,
#             class_agnostic=False,
#             loss_mask: ConfigType = dict(
#                 type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
#             init_cfg=None,
#             *args,
#             **kwargs):
#         BaseModule.__init__(self, init_cfg=init_cfg)
#
#         self.in_channels = in_channels
#         self.roi_feat_size = roi_feat_size
#         self.per_pointset_point = per_pointset_point
#         self.with_sincos = with_sincos
#         self.multimask_output = multimask_output
#         self.attention_similarity = attention_similarity
#         self.target_embedding = target_embedding
#         self.output_attentions = output_attentions
#
#         self.mask_decoder = MODELS.build(mask_decoder)
#
#         prompt_encoder = dict(
#             type='RSSamPromptEncoder',
#             hf_pretrain_name=copy.deepcopy(mask_decoder.get('hf_pretrain_name')),
#             init_cfg=copy.deepcopy(mask_decoder.get('init_cfg')),
#         )
#         prompt_encoder = MODELS.build(prompt_encoder)
#         prompt_encoder.init_weights()
#         self.no_mask_embed = prompt_encoder.prompt_encoder.no_mask_embed
#
#         if with_sincos:
#             num_sincos = 2
#         else:
#             num_sincos = 1
#         self.point_emb = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),  # (batch, 7, 7, 256)
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Flatten(),  # (batch, 7*7*256)
#             nn.Linear(in_channels * roi_feat_size ** 2 // 4, in_channels),  # (batch, 256)
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels, in_channels),  # (batch, 256)
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels, in_channels * num_sincos * per_pointset_point)  # (batch, 256*2*5)
#         )
#
#         self.loss_mask = MODELS.build(loss_mask)
#         self.class_agnostic = class_agnostic
#
#     def init_weights(self) -> None:
#         BaseModule.init_weights(self)
#
#     def forward(self,
#                 x,  # x.shape=(16, 256, 14, 14)，batch中的 2个图像共有 16个大小为 14x14的RoI
#                 image_embeddings,
#                 image_positional_embeddings,
#                 roi_img_ids=None,
#                 ):
#         # x.shape = (16, 256, 14, 14)，batch中的 2个图像共有 16个大小为 14x14的RoI
#         img_bs = image_embeddings.shape[0]  # 2
#         roi_bs = x.shape[0]  # 16
#         image_embedding_size = image_embeddings.shape[-2:]
#
#         point_embedings = self.point_emb(x)  # (16, 256*5*2)
#         point_embedings = einops.rearrange(point_embedings, 'b (n c) -> b n c',
#                                            n=self.per_pointset_point)  # (16, 5, 512)
#         if self.with_sincos:
#             point_embedings = torch.sin(point_embedings[..., ::2]) + point_embedings[...,
#                                                                      1::2]  # (16, 5, 256)，这里的5是指每个RoI区域生成5个提示
#
#         # (B * N_set), N_point, C
#         sparse_embeddings = point_embedings.unsqueeze(1)  # （16，1，5，256）
#         num_roi_per_image = torch.bincount(roi_img_ids.long())  # len(roi_img_ids)=16, num_roi_per_image=tensor([9, 7])
#         # deal with the case that there is no roi in an image
#         num_roi_per_image = torch.cat([num_roi_per_image,
#                                        torch.zeros(img_bs - len(num_roi_per_image), device=num_roi_per_image.device,
#                                                    dtype=num_roi_per_image.dtype)])
#
#         dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(roi_bs, -1, image_embedding_size[0],
#                                                                                  image_embedding_size[
#                                                                                      1])  # (16, 256, 32, 32)
#         # get image embeddings with num_roi_per_image
#         # batch=2，第一个图像重复 9 次，第二个图像重复 7 次
#         image_embeddings = image_embeddings.repeat_interleave(num_roi_per_image, dim=0)  # (16, 256, 32, 32)
#         image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)
#
#         low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
#             image_embeddings=image_embeddings,  # (batch=16, C=256, H=32, W=32)
#             image_positional_embeddings=image_positional_embeddings,  # (batch=16, C=256, H=32, W=32)
#             sparse_prompt_embeddings=sparse_embeddings,  # (batch=16, 1, 5, 256)
#             dense_prompt_embeddings=dense_embeddings,  # (batch=16, 256, 32, 32)
#             multimask_output=self.multimask_output,
#             attention_similarity=self.attention_similarity,
#             target_embedding=self.target_embedding,
#             output_attentions=self.output_attentions,
#         )
#         h, w = low_res_masks.shape[-2:]
#         low_res_masks = low_res_masks.reshape(roi_bs, -1, h, w)
#         iou_predictions = iou_predictions.reshape(roi_bs, -1)
#         return low_res_masks, iou_predictions
#
#     def get_targets(self, sampling_results: List[SamplingResult],
#                     batch_gt_instances: InstanceList,
#                     rcnn_train_cfg: ConfigDict) -> Tensor:
#         pos_proposals = [res.pos_priors for res in sampling_results]
#         pos_assigned_gt_inds = [
#             res.pos_assigned_gt_inds for res in sampling_results
#         ]
#         gt_masks = [res.masks for res in batch_gt_instances]
#         mask_targets_list = []
#         mask_size = rcnn_train_cfg.mask_size
#         device = pos_proposals[0].device
#         for pos_gt_inds, gt_mask in zip(pos_assigned_gt_inds, gt_masks):
#             if len(pos_gt_inds) == 0:
#                 mask_targets = torch.zeros((0,) + mask_size, device=device, dtype=torch.float32)
#             else:
#                 mask_targets = gt_mask[pos_gt_inds.cpu()].to_tensor(dtype=torch.float32, device=device)
#             mask_targets_list.append(mask_targets)
#         mask_targets = torch.cat(mask_targets_list)
#         return mask_targets
#
#     def loss_and_target(self, mask_preds: Tensor,
#                         sampling_results: List[SamplingResult],
#                         batch_gt_instances: InstanceList,
#                         rcnn_train_cfg: ConfigDict) -> dict:
#         mask_targets = self.get_targets(
#             sampling_results=sampling_results,
#             batch_gt_instances=batch_gt_instances,
#             rcnn_train_cfg=rcnn_train_cfg)
#
#         pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
#         # resize to mask_targets size
#         mask_preds = F.interpolate(mask_preds, size=mask_targets.shape[-2:], mode='bilinear', align_corners=False)
#
#         loss = dict()
#         if mask_preds.size(0) == 0:
#             loss_mask = mask_preds.sum()
#         else:
#             if self.class_agnostic:
#                 loss_mask = self.loss_mask(mask_preds, mask_targets,
#                                            torch.zeros_like(pos_labels))
#             else:
#                 loss_mask = self.loss_mask(mask_preds, mask_targets,
#                                            pos_labels)
#         loss['loss_mask'] = loss_mask
#         return dict(loss_mask=loss, mask_targets=mask_targets)
#
#     def _predict_by_feat_single(self,
#                                 mask_preds: Tensor,
#                                 bboxes: Tensor,
#                                 labels: Tensor,
#                                 img_meta: dict,
#                                 rcnn_test_cfg: ConfigDict,
#                                 rescale: bool = False,
#                                 activate_map: bool = False) -> Tensor:
#         scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
#             (1, 2))
#         img_h, img_w = img_meta['ori_shape'][:2]
#         if not activate_map:
#             mask_preds = mask_preds.sigmoid()
#         else:
#             # In AugTest, has been activated before
#             mask_preds = bboxes.new_tensor(mask_preds)
#
#         if rescale:  # in-placed rescale the bboxes
#             bboxes /= scale_factor
#         else:
#             w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
#             img_h = np.round(img_h * h_scale.item()).astype(np.int32)
#             img_w = np.round(img_w * w_scale.item()).astype(np.int32)
#         threshold = rcnn_test_cfg.mask_thr_binary
#         im_mask = F.interpolate(mask_preds, size=img_meta['batch_input_shape'], mode='bilinear',
#                                 align_corners=False).squeeze(1)
#
#         scale_factor_w, scale_factor_h = img_meta['scale_factor']
#         ori_rescaled_size = (img_h * scale_factor_h, img_w * scale_factor_w)
#         im_mask = im_mask[:, :int(ori_rescaled_size[0]), :int(ori_rescaled_size[1])]
#
#         h, w = img_meta['ori_shape']
#         im_mask = F.interpolate(im_mask.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
#
#         if threshold >= 0:
#             im_mask = im_mask >= threshold
#         else:
#             # for visualization and debugging
#             im_mask = (im_mask * 255).to(dtype=torch.uint8)
#         return im_mask
#
#
# @MODELS.register_module()
# class MMPretrainSwinTransformer(BaseModule):
#     def __init__(
#             self,
#             hf_pretrain_name=None,
#             img_size=224,
#             peft_config=None,
#             init_cfg=None,
#             extra_config=None,
#     ):
#         super().__init__(init_cfg=init_cfg)
#         vision_encoder_cfg = dict(
#             type='mmpretrain.SwinTransformer',
#             arch='base',
#             img_size=img_size,
#         )
#         vision_encoder = MODELS.build(vision_encoder_cfg)
#         # load checkpoint
#         if init_cfg is not None:
#             from mmengine.runner.checkpoint import load_checkpoint
#             load_checkpoint(
#                 vision_encoder,
#                 init_cfg.get('checkpoint'),
#                 map_location='cpu',
#                 # revise_keys=[
#                 #     (r'^module\.', ''),
#                 #     (r'^vision_encoder\.', ''),
#                 #     (r'.layer_norm1.', '.ln1.'),
#                 #     (r'.layer_norm2.', '.ln2.'),
#                 #     (r'.mlp.lin1.', '.ffn.layers.0.0.'),
#                 #     (r'.mlp.lin2.', '.ffn.layers.1.'),
#                 #     (r'neck.conv1.', 'channel_reduction.0.'),
#                 #     (r'neck.ln1.', 'channel_reduction.1.'),
#                 #     (r'neck.conv2.', 'channel_reduction.2.'),
#                 #     (r'neck.ln2.', 'channel_reduction.3.'),
#                 # ]
#             )
#
#         self.vision_encoder = vision_encoder
#         self.vision_encoder.is_init = True
#
#     def init_weights(self):
#         if is_main_process():
#             print('the vision encoder has been initialized')
#
#     def forward(self, *args, **kwargs):
#         return self.vision_encoder(*args, **kwargs)
#
#
# @MODELS.register_module()
# class MMPretrainSwinTransformerV2(BaseModule):
#     def __init__(
#             self,
#             hf_pretrain_name=None,
#             img_size=224,
#             peft_config=None,
#             init_cfg=None,
#             extra_config=None,
#     ):
#         super().__init__(init_cfg=init_cfg)
#         vision_encoder_cfg = dict(
#             type='mmpretrain.SwinTransformerV2',
#             arch='base',
#             img_size=img_size,
#         )
#         vision_encoder = MODELS.build(vision_encoder_cfg)
#         tmp = vision_encoder.state_dict()
#         # load checkpoint
#         if init_cfg is not None:
#             # checkpoint = torch.load(init_cfg.get('checkpoint'))
#             checkpoint = torch.load(init_cfg.get('checkpoint'))['model']
#             # vision_encoder.load_state_dict(checkpoint, strict=False)
#             # del checkpoint['classifier.weight']
#             # del checkpoint['classifier.bias']
#
#             from mmengine.runner.checkpoint import load_checkpoint
#             load_checkpoint(
#                 vision_encoder,
#                 init_cfg.get('checkpoint'),
#                 map_location='cpu',
#                 revise_keys=[
#                     (r'^module\.', ''),
#                     (r'^swinv2\.', ''),  # 移除 'swinv2.' 前缀
#                     (r'^encoder\.', ''),  # 移除 'encoder.' 前缀
#                     # (r'^embeddings\.', '.ln1.'), # 移除 'embeddings.' 前缀
#                     (r'embeddings.patch_embeddings', 'patch_embed'),
#                     (r'embeddings', 'patch_embed'),
#
#                     (r'layers', 'stage'),
#                     (r'attention', 'attn'),
#
#                     (r'.layer_norm1.', '.ln1.'),
#                     (r'.layer_norm2.', '.ln2.'),
#                     (r'.mlp.lin1.', '.ffn.layers.0.0.'),
#                     (r'.mlp.lin2.', '.ffn.layers.1.'),
#                     (r'neck.conv1.', 'channel_reduction.0.'),
#                     (r'neck.ln1.', 'channel_reduction.1.'),
#                     (r'neck.conv2.', 'channel_reduction.2.'),
#                     (r'neck.ln2.', 'channel_reduction.3.'),
#                 ]
#             )
#         a = vision_encoder.state_dict()
#         self.vision_encoder = vision_encoder
#         self.vision_encoder.is_init = True
#
#     def init_weights(self):
#         if is_main_process():
#             print('the vision encoder has been initialized')
#
#     def forward(self, *args, **kwargs):
#         return self.vision_encoder(*args, **kwargs)
#
#
# @MODELS.register_module()
# class RS_Swin_FeatureAggregator(BaseModule):
#     # RSPrompter论文中的 Fig.4实现的功能
#
#     in_channels_dict = {
#         'base': [768] * (12 + 1),  # [768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768]
#         'large': [1024] * (24 + 1),
#         'huge': [1280] * (32 + 1),
#     }
#
#     def __init__(
#             self,
#             in_channels,
#             hidden_channels=64,
#             out_channels=256,
#             select_layers=[0, 1, 2, 3],  # range(1, 13, 2) = [1, 3, 5, 7, 9, 11]
#
#             init_cfg=None,
#     ):
#         super().__init__(init_cfg=init_cfg)
#         assert isinstance(in_channels, str)
#         model_arch = 'base'
#         # self.in_channels = self.in_channels_dict[model_arch]
#         self.in_channels = [128, 256, 512, 1024]
#         self.select_layers = [0, 1, 2, 3]
#
#         self.downconvs = nn.ModuleList()
#         for i_layer in self.select_layers:
#             self.downconvs.append(
#                 nn.Sequential(
#                     nn.Conv2d(self.in_channels[i_layer], hidden_channels, 1),
#                     nn.BatchNorm2d(hidden_channels),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
#                     nn.BatchNorm2d(hidden_channels),
#                     nn.ReLU(inplace=True),
#                 )
#             )
#
#         self.hidden_convs = nn.ModuleList()
#         for _ in self.select_layers:
#             self.hidden_convs.append(
#                 nn.Sequential(
#                     nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
#                     nn.BatchNorm2d(hidden_channels),
#                     nn.ReLU(inplace=True),
#                 )
#             )
#
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(hidden_channels, out_channels, 1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#         )
#
#     def forward(self, inputs):
#         assert len(inputs) == len(self.in_channels)
#         inputs = [einops.rearrange(x, 'b h w c -> b c h w') for x in inputs]
#
#         features = []
#         for idx, i_layer in enumerate(self.select_layers):
#             features.append(self.downconvs[idx](inputs[i_layer]))
#
#         x = None
#         for hidden_state, hidden_conv in zip(features, self.hidden_convs):
#             if x is not None:
#                 hidden_state = x + hidden_state
#             residual = hidden_conv(hidden_state)
#             x = hidden_state + residual
#         x = self.fusion_conv(x)
#         return x
