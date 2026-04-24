import copy
import warnings
import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from click import prompt
from mmcv.cnn import build_norm_layer, ConvModule
from mmcv.ops import point_sample
from mmengine import ConfigDict
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from peft import get_peft_config, get_peft_model
# from sympy.polys.polyconfig import query
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
from mmdet.models.roi_heads import Shared2FCBBoxHead
from peft import get_peft_config, get_peft_model

from mmpretrain.models import LayerNorm2d
import math

from . import RSSamPromptEncoder, RSSamMaskDecoder, RSPrompterAnchorMaskHead
from .utils import *
from mmdet.rsprompter import MMPretrainSamVisionEncoder
from .models_DVT import Denoiser
from .anchor import *
import os

# import sys
# sys.path.append("/data2/yihan/MyProject/RSPrompter-release/")
# from segment_anything import
from .my_utils import *
from transformers import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
# from mmdet.rsprompter import RSPrompterAnchorMaskHead
from transformers.activations import ACT2FN

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
class SAM_Prompt_Encoder(RSSamPromptEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        shared_patch_embedding = SamPositionalEmbedding(SamConfig.from_pretrained(kwargs.get('hf_pretrain_name')).vision_config)
        prompt_config = SamConfig.from_pretrained(kwargs.get('hf_pretrain_name')).prompt_encoder_config
        prompt_config.update(kwargs.get('extra_config', None))
        # self.prompt_encoder = SamPromptEncoder(prompt_config, shared_patch_embedding=shared_patch_embedding)
        self.prompt_encoder = MySamPromptEncoder(prompt_config, shared_patch_embedding=shared_patch_embedding)

        # pretrained_dict = torch.load(kwargs['init_cfg'].checkpoint)
        # model_dict = self.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
        #                    k in model_dict and v.shape == model_dict[k].shape}
        # model_dict.update(pretrained_dict)
        # self.load_state_dict(model_dict)

@MODELS.register_module()
class SAM_Mask_Decoder(RSSamMaskDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.flag=1
        # pretrained_dict = torch.load(kwargs['init_cfg'].checkpoint)
        # model_dict = self.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
        #                    k in model_dict and v.shape == model_dict[k].shape}
        # model_dict.update(pretrained_dict)
        # self.load_state_dict(model_dict)

        # peft_mask_decoder=False
        # # peft_mask_decoder=kwargs.get('peft_mask_decoder',False)
        # if peft_mask_decoder:
        #     config = {
        #         "peft_type": "LORA",
        #         "r": 16,
        #         'target_modules': ["qkv"],
        #         "lora_alpha": 32,
        #         "lora_dropout": 0.05,
        #         "bias": "none",
        #         "inference_mode": False,
        #     }
        #
        #     peft_config = get_peft_config(config)
        #     self.mask_decoder = get_peft_model(self.mask_decoder, peft_config)
        #     print(1)
    ''' 删除，将hyper_in @ upscaled_embedding修改为了余弦相似度 start '''

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_positional_embeddings: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
            output_attentions: Optional[bool] = None,
            attention_similarity: torch.Tensor = None,
            target_embedding: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`):
                the embeddings from the image encoder
            image_positional_embedding (`torch.Tensor`):
                positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (`torch.Tensor`):
                The embeddings of the points and boxes
            dense_prompt_embeddings (`torch.Tensor`):
                the embeddings of the mask inputs
            multimask_output (bool):
                Whether to return multiple masks or a single mask.
            output_attentions (bool, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat([self.mask_decoder.iou_token.weight, self.mask_decoder.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.mask_decoder.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, attentions = self.mask_decoder.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1: (1 + self.mask_decoder.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )
        # '''delete begin'''
        # save_dir = "/data2/yihan/tmp"
        # os.makedirs(save_dir, exist_ok=True)
        #
        # # ==== 生成保存路径 ====
        # base_path = os.path.join(save_dir, f"{self.flag}.png")
        # if os.path.exists(base_path):
        #     save_path = os.path.join(save_dir, f"{self.flag}_1.png")
        #     self.flag += 1
        # else:
        #     save_path = base_path
        #
        # # ==== 生成图像并保存 ====
        # plt.imshow(
        #     F.interpolate(image_embeddings[:, :, :23, :], size=(380, 640), mode='bilinear', align_corners=False)[-1, :, :, :]
        #     .mean(0).detach().cpu().numpy(),
        #     cmap='jet'
        # )
        # plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.close()
        #
        # '''delete end'''
        upscaled_embedding = self.mask_decoder.upscale_conv1(image_embeddings)
        upscaled_embedding = self.mask_decoder.activation(self.mask_decoder.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.mask_decoder.activation(self.mask_decoder.upscale_conv2(upscaled_embedding))

        # plt.show()

        prompt = upscaled_embedding

        hyper_in_list = []
        for i in range(self.mask_decoder.num_mask_tokens):
            current_mlp = self.mask_decoder.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.mask_decoder.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        outputs = (masks, iou_pred)

        # if output_attentions:
        #     outputs = outputs + (attentions,)
        # else:
        #     outputs = outputs + (None,)

        return outputs + (prompt, )
    ''' 删除 end '''


@MODELS.register_module()
class SAM_Anchor_Prompt(MaskRCNN):
    def __init__(
            self,
            shared_image_embedding: ConfigDict,
            # prompt_encoder: ConfigDict,
            loss_proposal: ConfigDict,
            num_queries: int,
            num_classes: int,
            decoder_freeze=True,
            adapter=None,
            *args,
            **kwargs):
        peft_config = kwargs.get('backbone', {}).get('peft_config', {})
        super().__init__(*args, **kwargs)
        self.shared_image_embedding = MODELS.build(shared_image_embedding)
        self.decoder_freeze = decoder_freeze

        # self.prompt_encoder = MODELS.build(prompt_encoder)

        # 2025-02-15
        self.adapter = False
        if adapter is not None:
            self.adapter = MODELS.build(adapter)

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

        # self.AttnFusion = AttentionFusion(256,256)
        # self.Depth_proj = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        # self.edge_attn = EdgeAttention(in_channels=256, out_channels=1)
        # self.pixel_attn = PixelAttention(in_channels=256, out_channels=256)
        # self.alpha = nn.Parameter(torch.tensor(0.5))
        # self.backbone.vision_encoder.base_model.model.out_indices = [2,11]
        # self.backbone.vision_encoder.base_model.model.output_inter = True
        # self.CBAM = CBAM(256,256)
        # self.WCAF = WCAF(embed_dims=768)


        for name, param in self.named_parameters():
            print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        print(1)
        ''' yihan '''
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params / 1e6:.2f}M")

        # 只统计可训练参数
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
        print(1)

        # 保存模型的参数，运行一次后就注释掉
        # torch.save(self.state_dict(), "/data1/yihan/MyProject/tmp/tmp.pth")
        # print(1)
        # import sys
        # sys.exit()


    def loss_proposals(self, output_proposals, targets, indices):
        assert "proposal_cls_logits" in output_proposals

        proposal_size = output_proposals["proposal_cls_logits"].shape[-2:]
        proposal_cls_logits = output_proposals["proposal_cls_logits"].flatten(2).float()  # b, c, hw

        target_classes = self.num_classes * torch.ones([proposal_cls_logits.shape[0],
                                                        proposal_size[0] * proposal_size[1]],
                                                       device=proposal_cls_logits.device) # (batch, 32*32)
        target_classes = target_classes.to(torch.int64)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        idx = self._get_src_permutation_idx(indices)
        target_classes[idx] = target_classes_o # indices匹配到的位置，target_classes值是对应标签值，否则是10（表示背景）

        # loss_proposal = F.cross_entropy(proposal_cls_logits, target_classes, ignore_index=-1)
        loss_proposal = self.loss_proposal(proposal_cls_logits, target_classes) # (batch, 11, 4096)、(batch, 4096)
        losses = {"loss_proposal": loss_proposal}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_query(self, x):
        proposal_pos_embeds = F.interpolate(self.meta_pos_embed, size=x.shape[-1],
                                            mode="bilinear", align_corners=False) # 用于产生 query 的特征图尺寸是多少，第二个参数size就是多少
        query_features, query_pos_embeds, query_locations, proposal_cls_logits = self.query_proposal(
            x, proposal_pos_embeds
        )
        return query_features, query_pos_embeds, query_locations, proposal_cls_logits

    def _set_grad_false(self, module_list=[]):
        for module in module_list:
            module.eval()
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False

    def get_image_wide_positional_embeddings(self, size_h, size_w):
        target_device = self.shared_image_embedding.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size_h, size_w), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size_h
        x_embed = x_embed / size_w

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        # rgb_inputs = batch_inputs[:, :3, :, :] # (2, 3, 512, 512)
        # depth_inputs = batch_inputs[:, 3:, :, :] # (2, 1, 512, 512)
        # depth_inputs = self.Depth_proj(depth_inputs) # (2, 3, 512, 512)

        # rgb_outputs = self.backbone(rgb_inputs) # tuple_1, (2, 256, 32, 32)
        # depth_outputs = self.backbone(depth_inputs) # tuple_1, (2, 256, 32, 32)

        # vision_outputs = self.AttnFusion(rgb_outputs[0], depth_outputs[0]) # (2, 256, 32, 32)
        '''
        [0][:]代表C=768的输出，其中[0][0]是第一个全局注意力块的输出，[0][1]是最后一个全局注意力块的输出
        [1][:]代表C=256的输出，其中[1][0]是第一个全局注意力块的输出，[1][1]是最后一个全局注意力块的输出
        '''

        '''
        edge_mask = self.edge_attn(depth_outputs[0])
        pixel_mask = self.pixel_attn(depth_outputs[0])
        vision_outputs = self.alpha * (rgb_outputs[0] * edge_mask) + (1 - self.alpha) * (rgb_outputs[0] * pixel_mask)
        '''
        # vision_outputs = self.WCAF(depth_outputs[0][1], rgb_outputs[0][1])
        # vision_outputs = self.backbone.vision_encoder.base_model.model.channel_reduction(vision_outputs)
        # vision_outputs = (vision_outputs,)
        rgb_outputs = self.backbone(batch_inputs)
        vision_outputs = (rgb_outputs[1][3],)
        # plt.subplot(1, 2, 1)
        # plt.title('edge_mask')
        # plt.imshow(edge_mask[0,0,:,:].detach().cpu().numpy(), cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.title('pixel_mask')
        # plt.imshow(pixel_mask[0,0,:,:].detach().cpu().numpy(), cmap='gray')
        # plt.show()
        # vision_outputs = self.backbone(batch_inputs)

        ''' 删掉下面这行 '''
        # vision_outputs = (vision_outputs['last_hidden_state'].reshape(int(batch_inputs.shape[0]), int(batch_inputs.shape[2]/32), int(batch_inputs.shape[3]/32), -1).permute(0, 3, 1, 2), )
        # vision_outputs = (vision_outputs, )


        if isinstance(vision_outputs, SamVisionEncoderOutput): # 跳过
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs[1]
        elif isinstance(vision_outputs, tuple): # 执行
            image_embeddings = vision_outputs[0] # (b, 256, 32, 32)
            vision_hidden_states = vision_outputs # tuple_1, 1个(b, 256, 32, 32)
        else: # 跳过
            raise NotImplementedError

        ''' 删掉下面这行 '''
        # image_embeddings = self.from1024to256(image_embeddings)

        # image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_embeddings.shape)
        image_positional_embeddings = self.get_image_wide_positional_embeddings(size_h=image_embeddings.shape[-2], size_w=image_embeddings.shape[-1])
        # repeat with batch size
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        # vision_hidden_states是tuple(image_embeddings,), image_embeddings是vision_hidden_states[0]
        x = self.neck(vision_hidden_states)

        vit_features = rgb_outputs[0][1] # (b, 768, 32, 32)，本想用layer_2的输出，但此处是layer11的输出，更改后再通过实验验证下
        return x, image_embeddings, image_positional_embeddings, vit_features

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        x, image_embeddings, image_positional_embeddings, vit_features = self.extract_feat(batch_inputs)

        # 在这一步之前，完成特征融合的操作
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

        ''' 删 '''
        # query_features, query_pos_embeds, query_locations, proposal_cls_logits = self.get_query(
        #     image_embeddings)  # (2,256,70)、(2,256,70)、(2,70,2)、(2,11,32,32)
        # query_features = query_features.permute(0, 2, 1)
        # query_pos_embeds = query_pos_embeds.permute(0, 2, 1)
        # query_features = query_features + query_pos_embeds

        # # 再加上辅助分类头损失
        # targets = []
        # for ele in batch_data_samples:
        #     e = {}
        #     e['labels'] = ele.gt_instances.get('labels')
        #     e['masks'] = torch.tensor(ele.gt_instances.get('masks').masks, dtype=torch.bool)
        #     targets.append(e)
        #
        # output_proposals = {"proposal_cls_logits": proposal_cls_logits}
        # indices = self.matcher(output_proposals, targets)
        # proposal_loss_dict = self.loss_proposals(output_proposals, targets, indices)
        #
        # losses.update(proposal_loss_dict)
        #
        # img_bs = query_locations.shape[0]
        # num_queries = query_locations.shape[1]
        # query_locations = query_locations.reshape(-1, 2).unsqueeze(1).unsqueeze(1)  # (140, 1, 1, 2), input_points.shape=(batch_size, num_points, 2)
        # point_labels = torch.ones((img_bs * num_queries, 1, 1)).to(query_locations.device)  # (140, 1, 1), input_labels.shape=(batch_size, point_batch_size, num_points)
        # query_locations, _ = self.prompt_encoder(query_locations, point_labels, None, None)  # (140, 1, 2, 256)

        roi_losses = self.roi_head.loss(
            x, # tuple_5，5个不同尺度的特征图，H/4 ~ H/64，每个尺寸都是(batch=2，dim=256，H，W)
            rpn_results_list, # list_2，batch中2个图像的proposal，每个图像都有1K个proposal，包括1K个bboxes、labels、scores
            batch_data_samples, # list_2，batch中2个图像的实例标注GT信息
            # query_features=query_features,
            # query_locations=query_locations,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            vit_feature=vit_features
        )
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor, # (b, c, h, w) = (2, 3, 512, 512)
                batch_data_samples: SampleList, # list列表，长度 = batch，列表中每个元素都是 DetDataSample 类的实例，该实例其中一个属性就是 gt_instances
                rescale: bool = True) -> SampleList:
        x, image_embeddings, image_positional_embeddings, vit_features = self.extract_feat(batch_inputs)

        # ## os.makedirs("/data2/yihan/tmp/", exist_ok=True)
        # plt.imshow(F.interpolate(x[2][:, :, :, :], size=(380, 640), mode='bilinear', align_corners=False)[0, :, :, :].mean(0).detach().cpu().numpy(), cmap='jet')
        # plt.savefig(f"/data2/yihan/tmp/{batch_data_samples[0].get('img_path').split('/')[-1].split('.')[0]}.png",bbox_inches='tight', pad_inches=0, dpi=300)
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
            vit_feature=vit_features,
        ) # results_list 是一个列表，表中元素是 InstanceData 类的实例，包含bboxes、labels、masks(原图尺寸origin_size)、scores
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list) # 给 DetDataSample 实例添加了一个 pred_instances属性
        return batch_data_samples

@MODELS.register_module()
class MyPrompterAnchorRoIPromptHead(StandardRoIHead):
    def __init__(
            self,
            with_extra_pe=False,
            decoder_layers=0,
            *args,
            **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.decoder_layers = decoder_layers
        if with_extra_pe:
            out_channels = self.bbox_roi_extractor.out_channels
            positional_encoding = dict(
                num_feats=out_channels // 2,
                normalize=True,
            )
            self.extra_pe = SinePositionalEncoding(**positional_encoding)
        vit_dim=768
        transformer_dim=256
        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LN2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LN2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )
        # self.selfprompter = SelfPrompter(transformer_dim // 8)
        self.self_prompter = SelfPrompter(32)
        self.mask_prompt_embedding = MaskPromptEmbedding(32)

    def _mask_forward(self,
                      x: Tuple[Tensor],
                      rois: Tensor = None,
                      pos_inds: Optional[Tensor] = None,
                      bbox_feats: Optional[Tensor] = None,
                      # query_features=None,
                      # query_locations=None,
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

        mask_preds, iou_predictions, prompt = self.mask_head(
            mask_feats, # (17, 256, 14, 14)
            # query_features=query_features,
            # query_locations=query_locations,
            image_embeddings=image_embeddings, # (2, 256, 32, 32)
            image_positional_embeddings=image_positional_embeddings, # (2, 256, 32, 32)
            roi_img_ids=rois[:, 0] if rois is not None else None, # (17)
        )
        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats, iou_predictions=iou_predictions, prompt=prompt)
        return mask_results

    def _mask_forward_2(self,
          x: Tuple[Tensor],
          rois: Tensor = None,
          mask_as_prompt: Optional[Tensor] = None,
          mask_bi: Optional[Tensor] = None,
          image_embeddings=None,
          image_positional_embeddings=None,
      ) -> dict:
        mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois) # (17, 256, 14, 14)
        mask_as_prompt_preds, iou_predictions = self.mask_head.mask_as_prompt_forward(
            mask_feats, # (17, 256, 14, 14)
            mask_as_prompt, # (17,256,32,32)
            mask_bi,
            image_embeddings=image_embeddings, # (2, 256, 32, 32)
            image_positional_embeddings=image_positional_embeddings, # (2, 256, 32, 32)
            roi_img_ids=rois[:, 0] if rois is not None else None, # (17)
        ) # (17, 1, 512/4=128, 128), (17, 1)
        # mask_results = dict(mask_as_prompt_preds=mask_as_prompt_preds, mask_feats=mask_feats, iou_predictions=iou_predictions)
        mask_results = dict(mask_preds=mask_as_prompt_preds, mask_feats=mask_feats, iou_predictions=iou_predictions)
        return mask_results

    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult], bbox_feats: Tensor,
                  batch_gt_instances: InstanceList,
                  # query_features=None,
                  # query_locations=None,
                  image_embeddings=None,
                  image_positional_embeddings=None,
                  vit_feature=None,
                  ) -> dict:
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results]) # (17, 5)，5代表[batch_ind, x1, y1, x2, y2]，17代表batch中的两个图像总共有多少个pos_priors
            if len(pos_rois) == 0:
                print('no pos rois')
                # return dict(loss_mask=dict(loss_mask=0 * x[0].sum()))
                tmp = dict()
                l = 0 * x[0].sum()
                tmp.update({'loss_mask': l})
                if hasattr(self.mask_head, 'loss_boundary'):
                    tmp.update({'loss_boundary': l})
                for i in range(2, self.decoder_layers + 1):
                    tmp.update({f'loss_mask{i}': l})
                    if hasattr(self.mask_head, 'loss_boundary'):
                        tmp.update({f'loss_boundary{i}': l})
                # return dict(loss_mask=dict(loss_mask=0 * x[0].sum()))
                return tmp
            mask_results = self._mask_forward(
                x, # tuple_5
                pos_rois, # (17, 5)
                # query_features=query_features,
                # query_locations=query_locations,
                image_embeddings=image_embeddings, # (2, 256, 32, 32)
                image_positional_embeddings=image_positional_embeddings, # (2, 256, 32, 32)
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

        ''' yihan '''
        if 'loss_boundary' in mask_loss_and_target.get('loss_mask', {}):
            mask_results.update({
                'loss_mask': mask_loss_and_target['loss_mask']['loss_mask'],
                'loss_boundary': mask_loss_and_target['loss_mask']['loss_boundary'],
            })
        else:
            mask_results.update({
                'loss_mask': mask_loss_and_target['loss_mask']['loss_mask'],
            })
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])

        # 要根据pos_rois[:, 0]来确定batch中的每个特征图重复多少次
        roi_img_ids = pos_rois[:, 0] # tensor([0., 0.])
        img_bs = image_embeddings.shape[0]
        num_roi_per_image = torch.bincount(roi_img_ids.long())  # len(roi_img_ids)=17, num_roi_per_image=tensor([9, 8])
        num_roi_per_image = torch.cat([num_roi_per_image,torch.zeros(img_bs - len(num_roi_per_image),device=num_roi_per_image.device,dtype=num_roi_per_image.dtype)])
        image_embeddings_tmp = image_embeddings.repeat_interleave(num_roi_per_image, dim=0)  # (17, 256, 32, 32)
        vit_feature = vit_feature.repeat_interleave(num_roi_per_image, dim=0)  # (17, 768, 32, 32)

        hq_feature = self.embedding_encoder(image_embeddings_tmp) + self.compress_vit_feat(vit_feature) # (17,32,128,128)
        prompt_matrix = self.self_prompter(hq_feature, mask_results['prompt']) # (17,32,128,128)，对应笔记中的 prompt和 hq_feature融合
        mask_prompt = self.mask_prompt_embedding(prompt_matrix) # (17,256,32,32)
        '''
        可视化 Spatial Prompt
        tmp = mask_results['prompt'] # (17,32,128,128)
        b, c, h, w = tmp.shape
        center_vector = tmp[:, :, 20, 20]  # 取中心点特征
        feature_map_flat = tmp.view(b, c, -1)  # 变为 (b, 32, 128*128)
        similarity_map = torch.einsum('bc, bcn -> bn', center_vector, feature_map_flat)  # 点积相似度 (b, 128*128)
        similarity_map = similarity_map.view(b, h, w)  # 变回 (b, 128, 128)
        similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min())

        tmp2 = prompt_matrix  # (17,32,128,128)
        b, c, h, w = tmp2.shape
        center_vector = tmp2[:, :, 20, 20]  # 取中心点特征
        feature_map_flat = tmp2.view(b, c, -1)  # 变为 (b, 32, 128*128)
        similarity_map2 = torch.einsum('bc, bcn -> bn', center_vector, feature_map_flat)  # 点积相似度 (b, 128*128)
        similarity_map2 = similarity_map2.view(b, h, w)  # 变回 (b, 128, 128)
        similarity_map2 = (similarity_map2 - similarity_map2.min()) / (similarity_map2.max() - similarity_map2.min())

        # 4. 可视化相似度图
        plt.subplot(1,2,1)
        plt.imshow(similarity_map[0].cpu().detach().numpy(), cmap='jet')
        # plt.colorbar()
        plt.title("prompt Similarity Map")
        plt.scatter(20, 20, color='red', s=100, marker='o')
        plt.subplot(1, 2, 2)
        plt.imshow(similarity_map2[0].cpu().detach().numpy(), cmap='jet')
        # plt.colorbar()
        plt.title("Feature Similarity Map")
        plt.scatter(20, 20, color='red', s=100, marker='o')
        plt.show()
        '''

        # 在此之前，写自提示逻辑
        for i in range(2, self.decoder_layers + 1):
            mask_results_2 = self._mask_forward_2(
                x=x,  # tuple_5
                rois=pos_rois, # (17, 5)
                # mask_as_prompt=mask_results['mask_preds'], # (17, 1, 128, 128)
                mask_as_prompt=mask_prompt, # (17,256,32,32)
                mask_bi=mask_results['mask_preds'],
                image_embeddings=image_embeddings, # (2, 256, 32, 32)
                image_positional_embeddings=image_positional_embeddings, # (2, 256, 32, 32)
            )
            mask_loss_and_target_2 = self.mask_head.loss_and_target(
                mask_preds=mask_results_2['mask_preds'], # (17, 1, 128, 128)
                sampling_results=sampling_results,
                batch_gt_instances=batch_gt_instances,
                rcnn_train_cfg=self.train_cfg)
            mask_results.update(mask_results_2) # 更新 mask_preds、iou_predictions
            # mask_results.update({
            #         f'loss_mask{i}': mask_loss_and_target_2['loss_mask']['loss_mask'],
            #         f'loss_boundary{i}': mask_loss_and_target_2['loss_mask']['loss_boundary'],
            # })

            if 'loss_boundary' in mask_loss_and_target_2.get('loss_mask', {}):
                mask_results.update({
                    f'loss_mask{i}': mask_loss_and_target_2['loss_mask']['loss_mask'],
                    f'loss_boundary{i}': mask_loss_and_target_2['loss_mask']['loss_boundary'],
                })
            else:
                mask_results.update({
                    f'loss_mask{i}': mask_loss_and_target_2['loss_mask']['loss_mask'],
                })



        # mask_results_2 = self._mask_forward_2(
        #     x=x, # tuple_5
        #     rois=pos_rois, # (17, 5)
        #     mask_as_prompt=mask_results['mask_preds'], # (17, 1, 128, 128)
        #     image_embeddings=image_embeddings, # (2, 256, 32, 32)
        #     image_positional_embeddings=image_positional_embeddings, # (2, 256, 32, 32)
        # )
        # mask_loss_and_target_2 = self.mask_head.loss_and_target(
        #     mask_preds=mask_results_2['mask_as_prompt_preds'], # (17, 1, 128, 128)
        #     sampling_results=sampling_results,
        #     batch_gt_instances=batch_gt_instances,
        #     rcnn_train_cfg=self.train_cfg)


        # mask_results.update(loss_mask=mask_loss_and_target['loss_mask'])
        # mask_results.update(loss_mask2=mask_loss_and_target_2['loss_mask'])
        # mask_results.update(loss_boundary=mask_loss_and_target_2['loss_mask'])
        # mask_results_2.update(loss_mask2=mask_loss_and_target_2['loss_mask'])
        return mask_results
    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois) # 根据 RoI区域面积的大小匹配相应的特征图（面积大的 RoI代表大目标，匹配低分辨率特征图），从中提取特征
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = bbox2roi([res.priors for res in sampling_results]) # (num_boxes, 4 + 1)，4+1中的 "1" 可以区分roi来自batch中哪张图像
        bbox_results = self._bbox_forward(x, rois)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results

    def loss(self,
             x: Tuple[Tensor],
             rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample],
             # query_features=None,
             # query_locations=None,
             # extra inputs
             image_embeddings=None,
             image_positional_embeddings=None,
             vit_feature=None,
             ) -> dict:
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        if hasattr(self, 'extra_pe'):
            bs, _, h, w = x[0].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.extra_pe(mask_pe)
            outputs = []
            for i in range(len(x)): # x中所有分辨率的特征图都添加 bilinear后的 extra_pe
                output = x[i] + F.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
                outputs.append(output)
            x = tuple(outputs)

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples) # num_imgs=batch_size
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
            losses.update(bbox_results['loss_bbox']) # dict_keys(['loss_cls', 'acc', 'loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(
                x, # 多尺度特征图
                sampling_results,
                bbox_results['bbox_feats'], # 根据 RoI，从多尺度特征图中提取出的特征矩阵(512, 256, 7, 7)，不过预测 mask时要重新根据RoI提取特征图(17, 256, 14, 14)，bbox_feats似乎没用了
                batch_gt_instances,
                # query_features=query_features,
                # query_locations=query_locations,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                vit_feature=vit_feature,
            )
            losses.update({'loss_mask': mask_results['loss_mask']})
            if 'loss_boundary' in mask_results:
                losses.update({'loss_boundary': mask_results['loss_boundary']})


            for i in range(2, self.decoder_layers + 1):
                losses.update({
                    f'loss_mask{i}': mask_results[f'loss_mask{i}']
                })

                if f'loss_boundary{i}' in mask_results:
                    losses.update({
                        f'loss_boundary{i}': mask_results[f'loss_boundary{i}']
                    })

            # losses.update(mask_results['loss_mask'])
            # if mask_results.get('loss_mask2') == None: # 避免no pos rois的情况
            #     losses.update({'loss_mask2': 0 * x[0].sum()}) # tensor(0., device='cuda:2')，损失必须是个tensor，且在cuda上，写成0 * x[0].sum()很方便
            #     losses.update({'loss_boundary2': 0 * x[0].sum()}) # tensor(0., device='cuda:2')，损失必须是个tensor，且在cuda上，写成0 * x[0].sum()很方便
            #     losses.update({'loss_boundary': 0 * x[0].sum()}) # tensor(0., device='cuda:2')，损失必须是个tensor，且在cuda上，写成0 * x[0].sum()很方便
            # else:
            #     losses.update({'loss_mask2': mask_results['loss_mask2']['loss_mask']})
            #     losses.update({'loss_boundary2': mask_results['loss_mask2']['loss_boundary']})

        return losses

    def predict_mask(
            self,
            x: Tuple[Tensor],
            batch_img_metas: List[dict],
            results_list: InstanceList,
            rescale: bool = False,
            image_embeddings=None,
            image_positional_embeddings=None,
            vit_feature=None,
    ) -> InstanceList:

        # don't need to consider aug_test.
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes) # (n, 5)
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

        ''' 删除_start '''
        # 要根据pos_rois[:, 0]来确定batch中的每个特征图重复多少次
        roi_img_ids = mask_rois[:, 0]  # tensor([0., 0.])
        img_bs = image_embeddings.shape[0]
        num_roi_per_image = torch.bincount(roi_img_ids.long())  # len(roi_img_ids)=17, num_roi_per_image=tensor([9, 8])
        num_roi_per_image = torch.cat([num_roi_per_image,torch.zeros(img_bs - len(num_roi_per_image), device=num_roi_per_image.device,dtype=num_roi_per_image.dtype)])
        image_embeddings_tmp = image_embeddings.repeat_interleave(num_roi_per_image, dim=0)  # (17, 256, 32, 32)
        vit_feature = vit_feature.repeat_interleave(num_roi_per_image, dim=0)  # (17, 768, 32, 32)

        hq_feature = self.embedding_encoder(image_embeddings_tmp) + self.compress_vit_feat(
            vit_feature)  # (17,32,128,128)
        prompt_matrix = self.self_prompter(hq_feature, mask_results['prompt'])  # (17,32,128,128)
        mask_prompt = self.mask_prompt_embedding(prompt_matrix)  # (17,256,32,32)

        # os.makedirs("/data2/yihan/tmp/", exist_ok=True)
        # plt.imshow(mask_results['mask_preds'][0, 0, :90, :].detach().cpu().numpy(), cmap='jet')
        # plt.axis('off')
        # plt.savefig(f"/data2/yihan/tmp/{batch_img_metas[0].get('img_path').split('/')[-1].split('.')[0]}.png",
        #             bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.close()

        for i in range(2, self.decoder_layers + 1):
            mask_results_2 = self._mask_forward_2(
                x=x,  # tuple_5
                rois=mask_rois,  # (17, 5)
                # mask_as_prompt=mask_results['mask_preds'], # (17, 1, 128, 128)
                mask_as_prompt=mask_prompt, # (17,256,32,32)
                mask_bi=mask_results['mask_preds'],  # (17, 1, 128, 128)
                image_embeddings=image_embeddings, # (2, 256, 32, 32)
                image_positional_embeddings=image_positional_embeddings, # (2, 256, 32, 32)
            )
            mask_results.update(mask_results_2) # 更新 mask_preds、iou_predictions
        ''' 删除_end '''

        # print(1)
        # os.makedirs("/data2/yihan/tmp/", exist_ok=True)
        # plt.imshow(mask_results['mask_preds'][0, 0, :, :].detach().cpu().numpy(), cmap='jet')
        # plt.axis('off')
        # plt.savefig(f"/data2/yihan/tmp/{batch_img_metas[0].get('img_path').split('/')[-1].split('.')[0]}.png",
        #             bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.close()

        # mask_preds = mask_results['mask_preds']
        # mask_preds = mask_results_2['mask_as_prompt_preds']
        mask_preds = mask_results['mask_preds']

        ''' 删除 start '''
        # from matplotlib import pyplot as plt
        # # for i in range(63):
        # #     plt.subplot(9,7,i+1)
        # #     plt.axis('off')
        # #     plt.imshow(mask_preds[i,0,:,:].detach().cpu().numpy()>0,cmap='gray')
        # plt.subplot(1,2,1)
        # plt.imshow(mask_preds[1,0,:,:].detach().cpu().numpy()>0,cmap='gray')
        # tmp = F.interpolate(
        #     mask_preds[1:2],
        #     size=(512, 512),
        #     mode='bilinear',
        #     align_corners=False
        # )
        # plt.subplot(1,2,2)
        # plt.imshow(tmp[0,0,:,:].detach().cpu().numpy()>0, cmap='gray')
        # plt.show()
        ''' 删除 end '''

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
                vit_feature=None,
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
                vit_feature=vit_feature,
            )
        return results_list

@MODELS.register_module()
class MyBoxHead(Shared2FCBBoxHead):

    def __init__(self, queries_num=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queries_num=queries_num

    def forward(self, x: Tuple[Tensor],
                # query_features=None,
                # query_locations=None
                ) -> tuple:

        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

@MODELS.register_module()
class MyMaskHead(RSPrompterAnchorMaskHead):
    def __init__(self, prompt_encoder=None, loss_boundary=None, decoder_layers=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_encoder = MODELS.build(prompt_encoder)
        # 注释掉后，代表 decoder用共享参数
        # self.mask_decoder2 = MODELS.build(kwargs.get('mask_decoder', None))

        ''' yihan '''
        # self.mask_decoder2 = self.mask_decoder # 2025-01-06，解除注释
        self.decoder_layers = decoder_layers
        # for i in range(2, self.decoder_layers+1):
        #     setattr(self, f"mask_decoder{i}", self.mask_decoder)

        if loss_boundary != None:
            self.loss_boundary = MODELS.build(loss_boundary)

        self.atten_fuse = AttentionFusion(256, 256)
        # self.atten_fuse = CBAM(256, 256)
        # self.alpha = nn.Parameter(torch.tensor(0.5))
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

        low_res_masks, iou_predictions, prompt = self.mask_decoder(
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
        return low_res_masks, iou_predictions, prompt

    def mask_as_prompt_forward(self,
        x,  # x.shape=(17, 256, 14, 14)，batch中的 2个图像共有 17个大小为 14x14的RoI
        mask_as_prompt_forward, # (17,256,32,32)
        mask_bi, # (17,1,128,128)
        image_embeddings,
        image_positional_embeddings,
        roi_img_ids=None,
    ):
        # x.shape = (17, 256, 14, 14)，batch中的 2个图像共有 17个大小为 14x14的RoI
        img_bs = image_embeddings.shape[0]  # 2
        roi_bs = x.shape[0]  # 17
        image_embedding_size = image_embeddings.shape[-2:]
        point_embedings = self.point_emb(x)  # (17, 256*5*2)
        point_embedings = einops.rearrange(point_embedings, 'b (n c) -> b n c',
                                           n=self.per_pointset_point)  # (17, 5, 512)
        if self.with_sincos:
            point_embedings = torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2]  # (17, 5, 256)，这里的5是指每个RoI区域生成5个提示

        # (B * N_set), N_point, C
        sparse_embeddings = point_embedings.unsqueeze(1)  # （17，1，5，256）

        num_roi_per_image = torch.bincount(
            roi_img_ids.long())  # len(roi_img_ids)=17, num_roi_per_image=tensor([9, 8])
        '''
        什么时候需要处理没有RoI的情况？
        1、这个问题要结合torch.bincount()来看
        2、假设batch=2，roi_img_ids中保存的是tensor([0., 0., 0., 0., 0., 1., 1., 1., 1.])，代表每个RoI属于batch中哪个图像
        3、如果第一张图像没有RoI，那么num_roi_per_image = tensor([0, 17])，此时不会有问题
        4、但如果第二张图像没有RoI，那么num_roi_per_image = tensor([17])，这样向量就少了一个维度，需要用0填充，就是下面这段代码要做的
        5、根本原因是：如果x中的最大值是n，则返回的torch.bincount结果张量的大小将是n + 1，表示从0到n每个值的计数。
        '''
        # deal with the case that there is no roi in an image
        num_roi_per_image = torch.cat([num_roi_per_image, torch.zeros(img_bs - len(num_roi_per_image), device=num_roi_per_image.device, dtype=num_roi_per_image.dtype)])

        # dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(roi_bs, -1, image_embedding_size[0], image_embedding_size[1])  # (17, 256, 32, 32)
        # mask_as_prompt_forward = (mask_as_prompt_forward > 0.5).to(torch.float32)
        _, dense_embeddings_bi = self.prompt_encoder(
            input_points=None,
            input_labels=None,
            input_boxes=None,
            input_masks=mask_bi,
        ) # (17, 256, 32, 32)
        '''
        可视化Semantic Prompt
        tmp = dense_embeddings_bi  # (17,32,128,128)
        b, c, h, w = tmp.shape
        center_vector = tmp[:, :, 15, 15]  # 取中心点特征
        feature_map_flat = tmp.view(b, c, -1)  # 变为 (b, 32, 128*128)
        similarity_map = torch.einsum('bc, bcn -> bn', center_vector, feature_map_flat)  # 点积相似度 (b, 128*128)
        similarity_map = similarity_map.view(b, h, w)  # 变回 (b, 128, 128)
        similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min())

        # plt.subplot(1, 2, 1)
        plt.imshow(similarity_map[0].cpu().detach().numpy(), cmap='jet')

        plt.title("Feature Similarity Map")
        plt.scatter(15, 15, color='red', s=100, marker='o')
        plt.show()
        '''
        # dense_embeddings=mask_as_prompt_forward + dense_embeddings_bi
        dense_embeddings=self.atten_fuse(mask_as_prompt_forward, dense_embeddings_bi)
        # dense_embeddings=self.alpha*mask_as_prompt_forward + (1-self.alpha)*dense_embeddings_bi # work_dirs/RGB_Depth/enhance_prompt_2测试这种做法
        # get image embeddings with num_roi_per_image
        # batch=2，第一个图像重复 9 次，第二个图像重复 8 次
        image_embeddings = image_embeddings.repeat_interleave(num_roi_per_image, dim=0) # (17, 256, 32, 32)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)

        # self.mask_decoder()代表两次decoder共享参数，self.mask_decoder2()代表两次decoder的参数是独立的
        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
        # low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder2(
            image_embeddings=image_embeddings,  # (batch=17, C=256, H=32, W=32)
            image_positional_embeddings=image_positional_embeddings,  # (batch=17, C=256, H=32, W=32)
            sparse_prompt_embeddings=sparse_embeddings,  # (batch=17, 1, 5, 256)
            dense_prompt_embeddings=dense_embeddings,  # (batch=17, 256, 32, 32)
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
        low_res_masks = low_res_masks.reshape(roi_bs, -1, h, w)  # (17, 1, 128, 128)
        iou_predictions = iou_predictions.reshape(roi_bs, -1)  # (17, 1)
        return low_res_masks, iou_predictions

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

        ''' yihan '''
        if hasattr(self, 'loss_boundary'): # 判断是否有 self.loss_boundary
            loss_boundary = self.loss_boundary(mask_preds, mask_targets)
            loss['loss_boundary'] = loss_boundary

        # return dict(loss_mask=loss, mask_targets=mask_targets)
        return dict(loss_mask=loss, mask_targets=mask_targets)


@MODELS.register_module()
class MyPretrainEncoder_DVT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # 获取sam的配置文件
        sam_config = SamConfig.from_pretrained("/data2/yihan/MyProject/RSPrompter-release/sam-vit-base/")
        sam_config.vision_config.update(dict(image_size=512, output_hidden_states=True))
        sam = SamModel(sam_config)
        self.vision_encoder = sam.vision_encoder
        del sam
        # torch.cuda.empty_cache()

        # vit_dict = torch.load("/data2/yihan/MyProject/RSPrompter-release/work_dirs/mask_pred_peft_vitb_loss-boundary_200/epoch_200.pth")['state_dict']
        vit_dict = torch.load("/data2/yihan/MyProject/RSPrompter-release/work_dirs/mask_pred_peft_vitb_loss-boundary_200/epoch_200.pth", map_location='cpu')['state_dict']
        vit_dict = {
            k.replace('.ln1.', '.layer_norm1.')
            .replace('.ln2.', '.layer_norm2.')
            .replace('.ffn.layers.0.0.', '.mlp.lin1.')
            .replace('.ffn.layers.1.', '.mlp.lin2.')
            .replace('channel_reduction.0.', 'neck.conv1.')
            .replace('channel_reduction.1.', 'neck.layer_norm1.')
            .replace('channel_reduction.2.', 'neck.conv2.')
            .replace('channel_reduction.3.', 'neck.layer_norm2.')
            : v
            for k, v in vit_dict.items()}

        vit_dict = {k.replace('backbone.vision_encoder.',''):v for k,v in vit_dict.items()}
        vit_dict = {k.replace('base_model.model.',''):v for k,v in vit_dict.items()}
        vit_dict = {k.replace('base_layer.',''):v for k,v in vit_dict.items()}
        sum = 0
        model_dict = self.vision_encoder.state_dict()
        for k, v in model_dict.items():
            if k in vit_dict and v.shape == vit_dict[k].shape:
                if 'lora' not in k:
                    model_dict[k] = vit_dict[k]
                    sum += 1
        # assert sum==model_dict.__len__(), f'vision encoder预训练参数未完全加载成功，进度{sum}/{model_dict.__len__()}'
        print(f'self.vision_encoder预训练参数加载进度{sum}/{len(self.vision_encoder.state_dict())}')
        self.vision_encoder.load_state_dict(model_dict)

        peft_config = {
            "peft_type": "LORA",
            "r": 16,
            'target_modules': ["qkv"],
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "inference_mode": False,
        }
        peft_config = get_peft_config(peft_config)
        self.vision_encoder = get_peft_model(self.vision_encoder, peft_config)
        self.vision_encoder.is_init = True

        self.dvt = Denoiser(
            noise_map_height=32,
            noise_map_width=32,
            feat_dim=768,
            vit=None,
            num_blocks=1,
        )
        # dvt_checkpoint = "/data1/yihan/MyProject/Denoising-ViT-main/work_dirs_stage2/denosing-vit/debug/checkpoints/ckpt_039999.pth"
        dvt_checkpoint = "/data1/yihan/MyProject/Denoising-ViT-main/work_dirs_stage2/vitb/denosing-vit/debug/checkpoints/ckpt_039999.pth"

        # vit_dict = torch.load(vision_encoder_checkpoint)['state_dict']
        # dvt_dict = torch.load(dvt_checkpoint)['denoiser']
        dvt_dict = torch.load(dvt_checkpoint, map_location=torch.device('cpu'))['denoiser']
        # vit_dict = {k.replace('backbone.vision_encoder.',''):v for k,v in vit_dict.items() if 'backbone.vision_encoder.' in k}

        sum = 0
        for k, v in self.dvt.state_dict().items():
            if k in dvt_dict and dvt_dict[k].shape == v.shape:
                sum += 1
        print(f'self.dvt预训练权重加载进度{sum}/{len(self.dvt.state_dict())}')
        self.dvt.load_state_dict(dvt_dict)

        # frozen_modules=[self.vision_encoder, self.dvt]
        frozen_modules=[self.dvt]
        self._set_grad_false(frozen_modules)
        # 单独解冻 self.vision_encoder.neck 的参数，不要解冻！！！
        # for param in self.vision_encoder.neck.parameters():
        #     param.requires_grad = True

        # 解冻与 LoRA 相关的参数
        # for name, param in self.vision_encoder.named_parameters():
        #     if 'lora' in name:
        #         param.requires_grad = True

        # state_dict() 不反映 requires_grad 状态，必须像下面这样打印出来才能看到
        for name, param in self.vision_encoder.named_parameters():
            print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        for name, param in self.dvt.named_parameters():
            print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

        print(1)

    def forward(self, *args, **kwargs):
        # with torch.no_grad():
        dvt_output = self.dvt(self.vision_encoder(*args, **kwargs).hidden_states[-1])

        ''' 删除，给去噪器产生的特征图添加一个固定的正弦位置嵌入 '''
        # dvt_output += get_sinusoidal_positional_embeddings(32*32, 768).reshape(32,32,-1).unsqueeze(0).repeat(dvt_output.shape[0],1,1,1).to(dvt_output)

        return (self.vision_encoder.neck(dvt_output), )

    def _set_grad_false(self, module_list=[]):
        for module in module_list:
            module.eval()
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False

    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')

def get_sinusoidal_positional_embeddings(seq_len, d_model):
    """
    Generate sinusoidal positional embeddings.

    Args:
        seq_len (int): The length of the sequence or number of positions.
        d_model (int): The dimensionality of the embedding.

    Returns:
        torch.Tensor: A tensor of shape (seq_len, d_model), containing the sinusoidal positional embeddings.
    """
    positions = torch.arange(seq_len).float().unsqueeze(1)  # Shape: (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # Shape: (d_model / 2,)
    pe = torch.zeros(seq_len, d_model)  # Initialize tensor to store the embeddings
    pe[:, 0::2] = torch.sin(positions * div_term)  # Apply sine to even indices
    pe[:, 1::2] = torch.cos(positions * div_term)  # Apply cosine to odd indices

    return pe

@MODELS.register_module()
class SAM_Mask_Decoder_tmp(RSSamMaskDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # pretrained_dict = torch.load(kwargs['init_cfg'].checkpoint)
        # model_dict = self.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
        #                    k in model_dict and v.shape == model_dict[k].shape}
        # model_dict.update(pretrained_dict)
        # self.load_state_dict(model_dict)

        # peft_mask_decoder=False
        # # peft_mask_decoder=kwargs.get('peft_mask_decoder',False)
        # if peft_mask_decoder:
        #     config = {
        #         "peft_type": "LORA",
        #         "r": 16,
        #         'target_modules': ["qkv"],
        #         "lora_alpha": 32,
        #         "lora_dropout": 0.05,
        #         "bias": "none",
        #         "inference_mode": False,
        #     }
        #
        #     peft_config = get_peft_config(config)
        #     self.mask_decoder = get_peft_model(self.mask_decoder, peft_config)
        #     print(1)
    ''' 删除，将hyper_in @ upscaled_embedding修改为了余弦相似度 start '''
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        attention_similarity: torch.Tensor = None,
        target_embedding: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat([self.mask_decoder.iou_token.weight, self.mask_decoder.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.mask_decoder.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, attentions = self.mask_decoder.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.mask_decoder.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.mask_decoder.upscale_conv1(image_embeddings)
        upscaled_embedding = self.mask_decoder.activation(self.mask_decoder.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.mask_decoder.activation(self.mask_decoder.upscale_conv2(upscaled_embedding))

        hyper_in_list = []
        for i in range(self.mask_decoder.num_mask_tokens):
            current_mlp = self.mask_decoder.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)

        ''' 原版 '''
        # masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        ''' 余弦相似度 '''
        masks = (F.normalize(hyper_in, p=2, dim=-1) @ F.normalize(upscaled_embedding, p=2, dim=-2)).reshape(batch_size, point_batch_size, -1, height, width)
        # Generate mask quality predictions
        iou_pred = self.mask_decoder.iou_prediction_head(iou_token_out)

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

        return outputs

@MODELS.register_module()
class New_Mask_Decoder(SamMaskDecoder):
    def __init__(self, *args, **kwargs):
        config = SamConfig.from_pretrained(kwargs.get('hf_pretrain_name',''))
        assert config, 'config不能为空'
        # super().__init__(*args, **kwargs)
        super().__init__(config=config.mask_decoder_config)
        self.mask_tokens = nn.Embedding(1, 256)
        del self.iou_token

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_positional_embeddings: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
            output_attentions: Optional[bool] = None,
            attention_similarity: torch.Tensor = None,
            target_embedding: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`):
                the embeddings from the image encoder
            image_positional_embedding (`torch.Tensor`):
                positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (`torch.Tensor`):
                The embeddings of the points and boxes
            dense_prompt_embeddings (`torch.Tensor`):
                the embeddings of the mask inputs
            multimask_output (bool):
                Whether to return multiple masks or a single mask.
            output_attentions (bool, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat([self.mask_tokens.weight, ], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.mask_tokens.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, attentions = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        ) # point_embedding.shpe=(17, 1, 6, 256)，其中6 = 1个mask_token + 6个sparse_embedding
        mask_tokens_out = point_embedding[:, :, :1, :] # (17, 1, 1, 256)

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        ) # (17, 256, 32, 32)

        upscaled_embedding = self.upscale_conv1(image_embeddings) # (17, 64, 64, 64)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding)) # (17, 64, 64, 64)
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding)) # (17, 32, 128, 128)

        hyper_in_list = []
        current_mlp = self.output_hypernetworks_mlps[0]
        hyper_in_list += [current_mlp(mask_tokens_out[:, :, 0, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2) # (17, 1, 1, 32)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width) # (17, 1, 32, 128*128)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width) # (17, 1, 1, 128, 128)

        # Generate mask quality predictions

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]

        outputs = (masks, torch.zeros((masks.shape[:3]), dtype=torch.float32))

        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)

        return outputs


from .sam import UAViTEncoder

@MODELS.register_module()
class MyPretrainEncoder_Adapter(BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config,
            peft_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        vision_encoder = UAViTEncoder(sam_config)
        # load checkpoint
        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                vision_encoder,
                init_cfg.get('checkpoint'),
                map_location='cpu',
                revise_keys=[(r'^module\.', ''), (r'^vision_encoder\.', '')])

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
class AttentionFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels):
        super().__init__()
        # 步骤1：将RGB和深度特征拼接后降维
        self.fusion_conv = nn.Conv2d(rgb_channels + depth_channels, rgb_channels, kernel_size=1)

        # 步骤2：通道注意力（SE Block）
        self.se_block = nn.Sequential(
            # Squeeze: 全局平均池化，将HxW的特征图压缩为1x1
            nn.AdaptiveAvgPool2d(1),
            # Excitation: 两层全连接生成权重
            nn.Conv2d(rgb_channels, rgb_channels // 16, kernel_size=1),  # 降维
            nn.ReLU(),
            nn.Conv2d(rgb_channels // 16, rgb_channels, kernel_size=1),  # 升维
            nn.Sigmoid()  # 输出0~1的权重
        )

    def forward(self, rgb_feat, depth_feat):
        # 1. 拼接RGB和深度特征
        fused = torch.cat([rgb_feat, depth_feat], dim=1)  # [B, rgb_c+depth_c, H, W]
        # 2. 通过1x1卷积减少通道数
        fused = self.fusion_conv(fused)  # [B, rgb_c, H, W]
        # 3. 生成注意力权重
        attention = self.se_block(fused)  # [B, rgb_c, 1, 1]
        # 4. 应用注意力权重：特征图每个通道乘以对应的权重
        weighted_feature = fused * attention  # [B, rgb_c, H, W]

        return weighted_feature

# 边缘级注意力
class EdgeAttention(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # 使用3x3卷积提取边缘（替代传统边缘检测），这里使用了可学习的边缘提取
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            LN2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(), # 将输出压缩到[0,1]作为注意力权重
        )

    def forward(self, depth_map):
        # depth_map: [B,1,H,W]
        edge_attn = self.edge_conv(depth_map)  # [B,1,H,W]
        return edge_attn

# 像素级注意力
class PixelAttention(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        # 通过1x1卷积生成像素级权重
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            LN2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, depth_map):
        # depth_map: [B,1,H,W]
        pixel_attn = self.conv(depth_map)  # [B,1,H,W]
        return pixel_attn

class SelfPrompter(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        t = int(abs((math.log(channels, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.CBR = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, bias=False),
            # nn.BatchNorm2d(channels),
            LN2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hq_feature, prompt_feature):
        x = hq_feature + prompt_feature # (b,32,128,128)
        x = self.CBR(x) # (b,32,128,128)
        x = self.conv1(x) # (b,32,128,128)

        w = self.avg_pool(x)  # (b,32,1,1)
        w = self.conv1d(w.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # (b,32,1,1)，注意区分conv1d和conv2d
        w = self.sigmoid(w)  # (b,32,1,1)

        x = x * w  # (b,32,128,128)

        return x  # (b,32,128,128)

class MaskPromptEmbedding(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.channels = channels

        self.activation = nn.GELU()
        self.conv1 = nn.Conv2d(channels, channels*2, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(channels*2, channels*4, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(channels*4, channels*8, kernel_size=1)
        self.layer_norm1 = LN2d(channels*2)
        self.layer_norm2 = LN2d(channels*4)

    def forward(self, masks):
        hidden_states = self.conv1(masks) # (b,32,128,128)->(b,64,64,64)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.conv2(hidden_states) # (b,64,64,64)->(b,128,32,32)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        dense_embeddings = self.conv3(hidden_states) # (b,128,32,32)->(b,256,32,32)
        return dense_embeddings

class WCAF(nn.Module):
    def __init__(
        self,
        embed_dims: int = 768,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.qkv_bias = True
        # 特征对齐（增加 LayerNorm）
        # self.norm_SRC = nn.LayerNorm(embed_dims)
        # self.norm_TGT = nn.LayerNorm(embed_dims)
        self.K_proj_SRC = nn.Linear(embed_dims, embed_dims, bias=self.qkv_bias)
        self.V_proj_SRC = nn.Linear(embed_dims, embed_dims, bias=self.qkv_bias)
        self.Q_proj_TGT = nn.Linear(embed_dims, embed_dims, bias=self.qkv_bias)
        self.V_proj_TGT = nn.Linear(embed_dims, embed_dims, bias=self.qkv_bias)
        # self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=0.0, batch_first=True)
        self.scale = embed_dims ** -0.5
        self.GAP = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.Sig = nn.Sigmoid()
        # 可学习的特征缩放因子
        # self.gamma_SRC = nn.Parameter(torch.ones(1))
        # self.gamma_TGT = nn.Parameter(torch.ones(1))


    def forward(self, SRC, TGT):
        B, C, H, W = SRC.shape  # (B, 768, 32, 32)
        # **变换输入形状: (B, C, H, W) → (B, H×W, C)**
        SRC = SRC.flatten(2).permute(0, 2, 1)  # (B, seq_len, C) where seq_len = H × W
        TGT = TGT.flatten(2).permute(0, 2, 1)  # (B, seq_len, C)
        # 特征对齐（LayerNorm）
        # SRC = self.norm_SRC(SRC)
        # TGT = self.norm_TGT(TGT)

        # **计算 Q, K, V**
        K = self.K_proj_SRC(SRC)  # (B, seq_len, C)
        Q = self.Q_proj_TGT(TGT)  # (B, seq_len, C)
        Vs = self.V_proj_SRC(SRC)  # (B, seq_len, C)
        Vt = self.V_proj_TGT(TGT)  # (B, seq_len, C)

        # **计算注意力权重**
        # _, As = self.attn(Q, K, Vs)  # As: (B, seq_len, seq_len)
        # _, At = self.attn(Q, K, Vt)  # At: (B, seq_len, seq_len)
        attn = (Q * self.scale) @ K.transpose(-2, -1) # (B, seq_len, seq_len)
        attn = attn.softmax(dim=-1)
        # **计算加权 V**
        x1 = attn @ Vs  # (B, seq_len, C)
        x2 = attn @ Vt  # (B, seq_len, C)

        # **计算全局特征**
        x1 = self.Sig(x1.mean(dim=-1, keepdim=True))  # (B, seq_len, 1)
        x2 = self.Sig(x2.mean(dim=-1, keepdim=True))  # (B, seq_len, 1)

        # **逐元素相乘**
        x1 = x1 * SRC  # (B, seq_len, C)
        x2 = x2 * TGT  # (B, seq_len, C)

        # **恢复回原始形状 (B, C, H, W)**
        out = x1 + x2  # (B, seq_len, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)

        return out

class FusionModule(nn.Module):
    def __init__(
        self,
        embed_dims: int = 768,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.RGB_WCAF = WCAF(embed_dims)
        self.Depth_WCAF = WCAF(embed_dims)


    def forward(self, rgb_feature, depth_feature):
        new_rgb_feature = self.RGB_WCAF(SRC=depth_feature, TGT=rgb_feature)
        new_depth_feature = self.Depth_WCAF(SRC=rgb_feature, TGT=depth_feature)

        return new_rgb_feature, new_depth_feature

from mmpretrain.models import ViTSAM
from mmpretrain.models.utils import resize_pos_embed
from mmengine.model import ModuleList
from mmpretrain.models.backbones.vit_sam import TransformerEncoderLayer
from mmpretrain.models.backbones.vit_sam import window_partition, window_unpartition
from typing import Type
from mmcv.cnn.bricks.transformer import FFN
@MODELS.register_module()
class UCTNet_PretrainSamViT(BaseModule):
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
            # type='UCTNet_ViTSAM',
            arch=hf_pretrain_name.split('-')[-1].split('_')[-1],
            img_size=img_size,
            patch_size=16,
            out_channels=256,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
        )
        vision_encoder1 = MODELS.build(vision_encoder_cfg)
        vision_encoder2 = MODELS.build(vision_encoder_cfg)

        # a = torch.load(init_cfg.get('checkpoint'))
        # b = vision_encoder.state_dict()
        # load checkpoint
        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                vision_encoder1,
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
            load_checkpoint(
                vision_encoder2,
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
            self.vision_encoder_rgb = get_peft_model(vision_encoder1, peft_config)
            self.vision_encoder_depth = get_peft_model(vision_encoder2, peft_config) # 开启深度分支LoRA微调
            # self.vision_encoder_depth = vision_encoder2 # 深度分支参数冻结
            # self.vision_encoder_depth = self.vision_encoder_rgb # RGB分支和深度分支共享参数
            if is_main_process():
                self.vision_encoder_rgb.print_trainable_parameters()
                # self.vision_encoder_depth.print_trainable_parameters()
        else:
            self.vision_encoder_rgb = vision_encoder1
            # self.vision_encoder_depth = vision_encoder2
        self.vision_encoder_rgb.is_init = True
        self.vision_encoder_depth.is_init = True

        self.FusionModules = nn.ModuleList([FusionModule() for _ in range(4)])
        self.out_indices=[2,5,8,11]
        self.output_inter = True
        # 冻结深度特征提取分支
        # self._set_grad_false([self.vision_encoder_depth,])

    def _set_grad_false(self, module_list=[]):
        for module in module_list:
            module.eval()
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False
    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')

    def forward(self, batch_inputs):
        rgb_inputs = batch_inputs[:, :3, :, :]  # (b, 3, 512, 512)
        depth_inputs = batch_inputs[:, 3:, :, :]  # (b, 3, 512, 512)
        B = rgb_inputs.shape[0]
        rgb, rgb_patch_resolution = self.vision_encoder_rgb.patch_embed(rgb_inputs) # (b,32*32,768)
        depth, depth_patch_resolution = self.vision_encoder_depth.patch_embed(depth_inputs) # (b,32*32,768)
        rgb = rgb.view(B, rgb_patch_resolution[0], rgb_patch_resolution[1], self.vision_encoder_rgb.embed_dims) # (b,32,32,768)
        depth = depth.view(B, depth_patch_resolution[0], depth_patch_resolution[1], self.vision_encoder_depth.embed_dims) # (b,32,32,768)

        if self.vision_encoder_rgb.use_abs_pos:
            # 'resize_pos_embed' only supports 'pos_embed' with ndim==3, but
            # in ViTSAM, the 'pos_embed' has 4 dimensions (1, H, W, C), so it
            # is flattened. Besides, ViTSAM doesn't have any extra token.
            resized_pos_embed = resize_pos_embed(
                self.vision_encoder_rgb.pos_embed.flatten(1, 2),
                self.vision_encoder_rgb.patch_resolution,
                rgb_patch_resolution,
                mode=self.vision_encoder_rgb.interpolate_mode,
                num_extra_tokens=0)
            rgb = rgb + resized_pos_embed.view(1, *rgb_patch_resolution, self.vision_encoder_rgb.embed_dims)
            rgb = self.vision_encoder_rgb.drop_after_pos(rgb)

        if self.vision_encoder_depth.use_abs_pos:
            # 'resize_pos_embed' only supports 'pos_embed' with ndim==3, but
            # in ViTSAM, the 'pos_embed' has 4 dimensions (1, H, W, C), so it
            # is flattened. Besides, ViTSAM doesn't have any extra token.
            resized_pos_embed = resize_pos_embed(
                self.vision_encoder_depth.pos_embed.flatten(1, 2),
                self.vision_encoder_depth.patch_resolution,
                depth_patch_resolution,
                mode=self.vision_encoder_depth.interpolate_mode,
                num_extra_tokens=0)
            depth = depth + resized_pos_embed.view(1, *depth_patch_resolution, self.vision_encoder_depth.embed_dims)
            depth = self.vision_encoder_depth.drop_after_pos(depth)

        outs = []
        inter_embeddings = []
        for i, (layer_rgb,layer_depth) in enumerate(zip(self.vision_encoder_rgb.layers, self.vision_encoder_depth.layers)):
            rgb = layer_rgb(rgb) # bhwc, (b,32,32,768)
            depth = layer_depth(depth) # bhwc, (b,32,32,768)

            if i in self.out_indices: # (2,5,8,11)
                # (B, H, W, C) -> (B, C, H, W)
                rgb_reshape = rgb.permute(0, 3, 1, 2) # (b,768,32,32)
                depth_reshape = depth.permute(0, 3, 1, 2) # (b,768,32,32)
                fusion_i = i // 3 # (2,5,8,11) // 3 = (0,1,2,3)
                new_rgb_feature, new_depth_feature = self.FusionModules[fusion_i](rgb_feature=rgb_reshape, depth_feature=depth_reshape)

                rgb = new_rgb_feature.permute(0,2,3,1) # (b,32,32,768)
                depth = new_depth_feature.permute(0,2,3,1) # (b,32,32,768)

                inter_embeddings.append(new_rgb_feature)
                if self.vision_encoder_rgb.out_channels > 0:
                    new_rgb_feature_reduced = self.vision_encoder_rgb.channel_reduction(new_rgb_feature)
                outs.append(self.vision_encoder_rgb._format_output(new_rgb_feature_reduced))

        if getattr(self, "output_inter", None) is None:
            return tuple(outs)
        else:
            return tuple(inter_embeddings), tuple(outs)
        # return self.vision_encoder(*args, **kwargs)

class MySamPromptEncoder(nn.Module):
    def __init__(self, config: SamPromptEncoderConfig, shared_patch_embedding):
        super().__init__()
        self.shared_embedding = shared_patch_embedding
        self.mask_embed = SamMaskEmbedding(config)
        self.no_mask_embed = nn.Embedding(1, config.hidden_size)

        # self.image_embedding_size = (config.image_embedding_size, config.image_embedding_size)
        self.image_embedding_size = config.image_embedding_size
        self.input_image_size = config.image_size

        self.point_embed = nn.ModuleList(
            [nn.Embedding(1, config.hidden_size) for i in range(config.num_point_embeddings)]
        )
        self.hidden_size = config.hidden_size
        self.not_a_point_embed = nn.Embedding(1, config.hidden_size)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            target_point_shape = (points.shape[0], points.shape[1], 1, points.shape[-1])
            target_labels_shape = (points.shape[0], points.shape[1], 1)
            padding_point = torch.zeros(target_point_shape, device=points.device)
            padding_label = -torch.ones(target_labels_shape, device=labels.device)
            points = torch.cat([points, padding_point], dim=2)
            labels = torch.cat([labels, padding_label], dim=2)
        # input_shape = (self.input_image_size, self.input_image_size)
        input_shape = self.input_image_size
        point_embedding = self.shared_embedding(points, input_shape)

        # torch.where and expanding the labels tensor is required by the ONNX export
        point_embedding = torch.where(labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding)

        # This is required for the ONNX export. The dtype, device need to be explicitely
        # specificed as otherwise torch.onnx.export interprets as double
        point_embedding = torch.where(
            labels[..., None] != -10,
            point_embedding,
            torch.tensor(0.0, dtype=point_embedding.dtype, device=point_embedding.device),
        )

        point_embedding = torch.where(
            (labels == 0)[:, :, :, None],
            point_embedding + self.point_embed[0].weight[None, None, :, :],
            point_embedding,
        )

        point_embedding = torch.where(
            (labels == 1)[:, :, :, None],
            point_embedding + self.point_embed[1].weight[None, None, :, :],
            point_embedding,
        )

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        batch_size, nb_boxes = boxes.shape[:2]
        coords = boxes.reshape(batch_size, nb_boxes, 2, 2)
        # input_shape = (self.input_image_size, self.input_image_size)
        input_shape = self.input_image_size
        corner_embedding = self.shared_embedding(coords, input_shape)
        corner_embedding[:, :, 0, :] += self.point_embed[2].weight
        corner_embedding[:, :, 1, :] += self.point_embed[3].weight
        return corner_embedding

    def forward(
        self,
        input_points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        input_labels: Optional[torch.Tensor],
        input_boxes: Optional[torch.Tensor],
        input_masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (`torch.Tensor`, *optional*):
                point coordinates and labels to embed.
            boxes (`torch.Tensor`, *optional*):
                boxes to embed
            masks (`torch.Tensor`, *optional*):
                masks to embed
        """
        sparse_embeddings = None
        batch_size = 1
        target_device = self.shared_embedding.positional_embedding.device
        if input_points is not None:
            batch_size, point_batch_size = input_points.shape[:2]
            if input_labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            point_embeddings = self._embed_points(input_points, input_labels, pad=(input_boxes is None))
            sparse_embeddings = point_embeddings
        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            box_embeddings = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=2)
        if input_masks is not None:
            dense_embeddings = self.mask_embed(input_masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        if sparse_embeddings is None:
            sparse_embeddings = torch.zeros((batch_size, 1, 1, self.hidden_size), device=target_device)

        return sparse_embeddings, dense_embeddings

class SamMaskEmbedding(nn.Module):
    def __init__(self, config: SamPromptEncoderConfig):
        super().__init__()
        self.mask_input_channels = config.mask_input_channels // 4
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.mask_input_channels, config.mask_input_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1)
        self.layer_norm1 = SamLayerNorm(
            self.mask_input_channels, eps=config.layer_norm_eps, data_format="channels_first"
        )
        self.layer_norm2 = SamLayerNorm(
            self.mask_input_channels * 4, eps=config.layer_norm_eps, data_format="channels_first"
        )

    def forward(self, masks):
        hidden_states = self.conv1(masks)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        dense_embeddings = self.conv3(hidden_states)
        return dense_embeddings

class SamLayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x