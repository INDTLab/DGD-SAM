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
from.anchor import *

# import sys
# sys.path.append("/data2/yihan/MyProject/RSPrompter-release/")
# from segment_anything import


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
        self.prompt_encoder = SamPromptEncoder(prompt_config, shared_patch_embedding=shared_patch_embedding)
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
    # def forward(
    #     self,
    #     image_embeddings: torch.Tensor,
    #     image_positional_embeddings: torch.Tensor,
    #     sparse_prompt_embeddings: torch.Tensor,
    #     dense_prompt_embeddings: torch.Tensor,
    #     multimask_output: bool,
    #     output_attentions: Optional[bool] = None,
    #     attention_similarity: torch.Tensor = None,
    #     target_embedding: torch.Tensor = None,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #
    #     batch_size, num_channels, height, width = image_embeddings.shape
    #     point_batch_size = sparse_prompt_embeddings.shape[1]
    #     # Concatenate output tokens
    #     output_tokens = torch.cat([self.mask_decoder.iou_token.weight, self.mask_decoder.mask_tokens.weight], dim=0)
    #     output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)
    #
    #     if sparse_prompt_embeddings.sum().item() != 0:
    #         tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
    #     else:
    #         tokens = output_tokens
    #     point_embeddings = tokens.to(self.mask_decoder.iou_token.weight.dtype)
    #
    #     # Expand per-image data in batch direction to be per-point
    #     image_embeddings = image_embeddings + dense_prompt_embeddings
    #     image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
    #     image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)
    #
    #     # Run the transformer, image_positional_embedding are consumed
    #     point_embedding, image_embeddings, attentions = self.mask_decoder.transformer(
    #         point_embeddings=point_embeddings,
    #         image_embeddings=image_embeddings,
    #         image_positional_embeddings=image_positional_embeddings,
    #         attention_similarity=attention_similarity,
    #         target_embedding=target_embedding,
    #         output_attentions=output_attentions,
    #     )
    #     iou_token_out = point_embedding[:, :, 0, :]
    #     mask_tokens_out = point_embedding[:, :, 1 : (1 + self.mask_decoder.num_mask_tokens), :]
    #
    #     # Upscale mask embeddings and predict masks using the mask tokens
    #     image_embeddings = image_embeddings.transpose(2, 3).reshape(
    #         batch_size * point_batch_size, num_channels, height, width
    #     )
    #
    #     upscaled_embedding = self.mask_decoder.upscale_conv1(image_embeddings)
    #     upscaled_embedding = self.mask_decoder.activation(self.mask_decoder.upscale_layer_norm(upscaled_embedding))
    #     upscaled_embedding = self.mask_decoder.activation(self.mask_decoder.upscale_conv2(upscaled_embedding))
    #
    #     hyper_in_list = []
    #     for i in range(self.mask_decoder.num_mask_tokens):
    #         current_mlp = self.mask_decoder.output_hypernetworks_mlps[i]
    #         hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
    #     hyper_in = torch.stack(hyper_in_list, dim=2)
    #
    #     _, num_channels, height, width = upscaled_embedding.shape
    #     upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
    #
    #     ''' 原版 '''
    #     # masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)
    #
    #     ''' 余弦相似度 '''
    #     masks = (F.normalize(hyper_in, p=2, dim=-1) @ F.normalize(upscaled_embedding, p=2, dim=-2)).reshape(batch_size, point_batch_size, -1, height, width)
    #     # Generate mask quality predictions
    #     iou_pred = self.mask_decoder.iou_prediction_head(iou_token_out)
    #
    #     # Select the correct mask or masks for output
    #     if multimask_output:
    #         mask_slice = slice(1, None)
    #     else:
    #         mask_slice = slice(0, 1)
    #     masks = masks[:, :, mask_slice, :, :]
    #     iou_pred = iou_pred[:, :, mask_slice]
    #
    #     outputs = (masks, iou_pred)
    #
    #     if output_attentions:
    #         outputs = outputs + (attentions,)
    #     else:
    #         outputs = outputs + (None,)
    #
    #     return outputs
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
        self.corr_layer6 = CorrelationLayer(feat_channel=256)
        self.decoder6 = decoder(256, 256)

        for name, param in self.named_parameters():
            print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        print(1)
        ''' yihan '''
        # 保存模型的参数，运行一次后就注释掉
        # torch.save(self.state_dict(), "/data1/yihan/MyProject/tmp/tmp.pth")
        # print(1)
        # import sys
        # sys.exit()

        # self.num_classes = num_classes
        # meta_pos_size = int(round(math.sqrt(num_queries)))
        # self.meta_pos_embed = nn.Parameter(torch.empty(1, 256, meta_pos_size, meta_pos_size))
        # nn.init.xavier_uniform_(self.meta_pos_embed)  # 进行 Xavier 初始化
        # self.loss_proposal = MODELS.build(loss_proposal)
        # # self.query_proposal = QueryProposal(256, 70, 10)
        # self.query_proposal = QueryProposal(256, num_queries, self.num_classes)
        #
        # self.matcher = HungarianMatcher(
        #     cost_class=2.0,
        #     cost_mask=5.0,
        #     cost_dice=5.0,
        #     cost_location=1000.0,
        #     num_points=12544,  # 采样 12544个点，减少计算量，并加速匹配过程
        # )
        # self.cls_embed = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.ReLU(inplace=True),
        #     # nn.Linear(self.feat_channels, self.feat_channels),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(256, self.num_classes + 1)
        # )

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
        rgb_inputs = batch_inputs[:, :3, :, :]  # (2, 3, 512, 512)
        depth_inputs = batch_inputs[:, 3:, :, :]  # (2, 1, 512, 512)
        # depth_inputs = self.Depth_proj(depth_inputs) # (2, 3, 512, 512)

        rgb_outputs = self.backbone(rgb_inputs)  # tuple_1, (2, 256, 32, 32)
        depth_outputs = self.backbone(depth_inputs)  # tuple_1, (2, 256, 32, 32)

        # vision_outputs = self.AttnFusion(rgb_outputs[0], depth_outputs[0])  # (2, 256, 32, 32)
        lamda = 0.01
        Fca, Fd = self.corr_layer6((rgb_outputs[0], depth_outputs[0]))  # corr_fea_6对应论文中的Fca，depth_feat6对应论文中的Fd（亦是fd）
        vision_outputs = self.decoder6(Fca, Fd * lamda)
        vision_outputs = (vision_outputs,)
        # vision_outputs = self.backbone(batch_inputs)

        ''' 删掉下面这行 '''
        # vision_outputs = (vision_outputs['last_hidden_state'].reshape(int(batch_inputs.shape[0]), int(batch_inputs.shape[2]/32), int(batch_inputs.shape[3]/32), -1).permute(0, 3, 1, 2), )
        # vision_outputs = (vision_outputs, )

        if isinstance(vision_outputs, SamVisionEncoderOutput):  # 跳过
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs[1]
        elif isinstance(vision_outputs, tuple):  # 执行
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs
        else: # 跳过
            raise NotImplementedError

        ''' 删掉下面这行 '''
        # image_embeddings = self.from1024to256(image_embeddings)

        image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_embeddings.shape[-1])
        # repeat with batch size
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        # vision_hidden_states是tuple(image_embeddings,), image_embeddings是vision_hidden_states[0]
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
        )
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor, # (b, c, h, w) = (2, 3, 512, 512)
                batch_data_samples: SampleList, # list列表，长度 = batch，列表中每个元素都是 DetDataSample 类的实例，该实例其中一个属性就是 gt_instances
                rescale: bool = True) -> SampleList:
        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)

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

        mask_preds, iou_predictions = self.mask_head(
            mask_feats, # (17, 256, 14, 14)
            # query_features=query_features,
            # query_locations=query_locations,
            image_embeddings=image_embeddings, # (2, 256, 32, 32)
            image_positional_embeddings=image_positional_embeddings, # (2, 256, 32, 32)
            roi_img_ids=rois[:, 0] if rois is not None else None, # (17)
        )
        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats, iou_predictions=iou_predictions)
        return mask_results

    def _mask_forward_2(self,
          x: Tuple[Tensor],
          rois: Tensor = None,
          mask_as_prompt: Optional[Tensor] = None,
          image_embeddings=None,
          image_positional_embeddings=None,
      ) -> dict:
        mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois) # (17, 256, 14, 14)
        mask_as_prompt_preds, iou_predictions = self.mask_head.mask_as_prompt_forward(
            mask_feats, # (17, 256, 14, 14)
            mask_as_prompt, # (17, 1, 128, 128)
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
                  ) -> dict:
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results]) # (17, 5)，5代表[batch_ind, x1, y1, x2, y2]，17代表batch中的两个图像总共有多少个pos_priors
            if len(pos_rois) == 0:
                print('no pos rois')
                # return dict(loss_mask=dict(loss_mask=0 * x[0].sum()))
                tmp = dict()
                l = 0 * x[0].sum()
                tmp.update({'loss_mask': l})
                tmp.update({'loss_boundary': l})
                for i in range(2, self.decoder_layers + 1):
                    tmp.update({f'loss_mask{i}': l})
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
        mask_results.update({
                f'loss_mask': mask_loss_and_target['loss_mask']['loss_mask'],
                f'loss_boundary': mask_loss_and_target['loss_mask']['loss_boundary'],
        })
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])

        for i in range(2, self.decoder_layers + 1):
            mask_results_2 = self._mask_forward_2(
                x=x,  # tuple_5
                rois=pos_rois, # (17, 5)
                mask_as_prompt=mask_results['mask_preds'], # (17, 1, 128, 128)
                image_embeddings=image_embeddings, # (2, 256, 32, 32)
                image_positional_embeddings=image_positional_embeddings, # (2, 256, 32, 32)
            )
            mask_loss_and_target_2 = self.mask_head.loss_and_target(
                mask_preds=mask_results_2['mask_preds'], # (17, 1, 128, 128)
                sampling_results=sampling_results,
                batch_gt_instances=batch_gt_instances,
                rcnn_train_cfg=self.train_cfg)
            mask_results.update(mask_results_2) # 更新 mask_preds、iou_predictions
            mask_results.update({
                    f'loss_mask{i}': mask_loss_and_target_2['loss_mask']['loss_mask'],
                    f'loss_boundary{i}': mask_loss_and_target_2['loss_mask']['loss_boundary'],
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
            losses.update(bbox_results['loss_bbox']) # dict_keys(['loss_cls', 'acc', 'loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(
                x, # 多尺度特征图
                sampling_results,
                bbox_results['bbox_feats'], # 根据RoI，从多尺度特征图中提取出的特征矩阵(512, 256, 7, 7)，不过预测mask时要重新根据RoI提取特征图(17, 256, 14, 14)，bbox_feats似乎没用了
                batch_gt_instances,
                # query_features=query_features,
                # query_locations=query_locations,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
            losses.update({'loss_mask': mask_results['loss_mask']})
            losses.update({'loss_boundary': mask_results['loss_boundary']})
            for i in range(2, self.decoder_layers + 1):
                losses.update({
                    f'loss_mask{i}': mask_results[f'loss_mask{i}']
                })
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

        ''' 删除_start '''
        # mask_results_2 = self._mask_forward_2(
        #     x=x, # tuple_5
        #     rois=mask_rois, # (17, 5)
        #     mask_as_prompt=mask_results['mask_preds'], # (17, 1, 128, 128)
        #     image_embeddings=image_embeddings,  # (2, 256, 32, 32)
        #     image_positional_embeddings=image_positional_embeddings, # (2, 256, 32, 32)
        # )
        for i in range(2, self.decoder_layers + 1):
            mask_results_2 = self._mask_forward_2(
                x=x,  # tuple_5
                rois=mask_rois,  # (17, 5)
                mask_as_prompt=mask_results['mask_preds'], # (17, 1, 128, 128)
                image_embeddings=image_embeddings, # (2, 256, 32, 32)
                image_positional_embeddings=image_positional_embeddings, # (2, 256, 32, 32)
            )
            mask_results.update(mask_results_2) # 更新 mask_preds、iou_predictions
        ''' 删除_end '''

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

    def mask_as_prompt_forward(self,
        x,  # x.shape=(17, 256, 14, 14)，batch中的 2个图像共有 17个大小为 14x14的RoI
        mask_as_prompt_forward, # (17, 1, 128, 128)
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
        _, dense_embeddings = self.prompt_encoder(
            input_points=None,
            input_labels=None,
            input_boxes=None,
            input_masks=mask_as_prompt_forward
        )
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

with_corr = True
with_pac = True
corr_size = 32
max_norm = 5
class EncDecFusing(nn.Module):
    def __init__(self, in_channels):
        super(EncDecFusing, self).__init__()
        self.enc_fea_proc = nn.Sequential(
            # nn.InstanceNorm2d(in_channels),
            LayerNorm2d(in_channels, eps=1e-6),
            nn.ReLU(inplace=True),
            # nn.GELU(),
        )
        self.fusing_layer = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, enc_fea, dec_fea):
        enc_fea = self.enc_fea_proc(enc_fea)

        if dec_fea.size(2) != enc_fea.size(2):
            dec_fea = F.upsample(dec_fea, size=[enc_fea.size(2), enc_fea.size(3)], mode='bilinear',
                                 align_corners=True)

        enc_fea = torch.cat([enc_fea, dec_fea], dim=1)
        output = self.fusing_layer(enc_fea)
        return output
def _neg_idx(idx):
    return None if idx == 0 else -idx


def np_gaussian_2d(width, sigma=-1):
    '''Truncated 2D Gaussian filter'''
    assert width % 2 == 1
    if sigma <= 0:
        sigma = float(width) / 4

    r = np.arange(-(width // 2), (width // 2) + 1, dtype=np.float32)
    gaussian_1d = np.exp(-0.5 * r * r / (sigma * sigma))
    gaussian_2d = gaussian_1d.reshape(-1, 1) * gaussian_1d
    gaussian_2d /= gaussian_2d.sum()

    return gaussian_2d


def nd2col(input_nd, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, transposed=False,
           use_pyinn_if_possible=False):
    """
    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, *kernel_size, *L_{out})` where
          :math:`L_{out} = floor((L_{in} + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)` for non-transposed
          :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding` for transposed
    """
    n_dims = len(input_nd.shape[2:])
    kernel_size = (kernel_size,) * n_dims if isinstance(kernel_size, Number) else kernel_size
    stride = (stride,) * n_dims if isinstance(stride, Number) else stride
    padding = (padding,) * n_dims if isinstance(padding, Number) else padding
    output_padding = (output_padding,) * n_dims if isinstance(output_padding, Number) else output_padding
    dilation = (dilation,) * n_dims if isinstance(dilation, Number) else dilation

    if transposed:
        assert n_dims == 2, 'Only 2D is supported for fractional strides.'
        w_one = input_nd.new_ones(1, 1, 1, 1)
        pad = [(k - 1) * d - p for (k, d, p) in zip(kernel_size, dilation, padding)]
        input_nd = F.conv_transpose2d(input_nd, w_one, stride=stride)
        input_nd = F.pad(input_nd, (pad[1], pad[1] + output_padding[1], pad[0], pad[0] + output_padding[0]))
        stride = _pair(1)
        padding = _pair(0)

    (bs, nch), in_sz = input_nd.shape[:2], input_nd.shape[2:]
    out_sz = tuple([((i + 2 * p - d * (k - 1) - 1) // s + 1)
                    for (i, k, d, p, s) in zip(in_sz, kernel_size, dilation, padding, stride)])
    # Use PyINN if possible (about 15% faster) TODO confirm the speed-up
    if n_dims == 2 and dilation == 1 and has_pyinn and torch.cuda.is_available() and use_pyinn_if_possible:
        output = P.im2col(input_nd, kernel_size, stride, padding)
    else:
        output = F.unfold(input_nd, kernel_size, dilation, padding, stride)
        out_shape = (bs, nch) + tuple(kernel_size) + out_sz
        output = output.view(*out_shape).contiguous()
    return output

import torch
from torch import nn
import numpy as np
import math
from numbers import Number
from itertools import repeat

import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair

try:
    import pyinn as P

    has_pyinn = True
except ImportError:
    P = None
    has_pyinn = False
    pass
class GaussKernel2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, dilation, channel_wise):
        ctx.kernel_size = _pair(kernel_size)
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        bs, ch, in_h, in_w = input.shape
        out_h = (in_h + 2 * ctx.padding[0] - ctx.dilation[0] * (ctx.kernel_size[0] - 1) - 1) // ctx.stride[0] + 1
        out_w = (in_w + 2 * ctx.padding[1] - ctx.dilation[1] * (ctx.kernel_size[1] - 1) - 1) // ctx.stride[1] + 1
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_h, out_w)
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        feat_0 = cols.contiguous()[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :]
        diff_sq = (cols - feat_0).pow(2)
        if not channel_wise:
            diff_sq = diff_sq.sum(dim=1, keepdim=True)
        output = torch.exp(-0.5 * diff_sq)
        # ctx._backend = type2backend[input.type()]
        ctx.save_for_backward(input, output)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        bs, ch, in_h, in_w = input.shape
        out_h, out_w = output.shape[-2:]
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_h, out_w)
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        feat_0 = cols.contiguous()[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :]
        diff = cols - feat_0
        grad = -0.5 * grad_output * output
        grad_diff = grad.expand_as(cols) * (2 * diff)
        grad_diff[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :] -= \
            grad_diff.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        # grad_input = grad_output.new()
        # ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state,
        #                                     grad_diff.view(bs, ch * ctx.kernel_size[0] * ctx.kernel_size[1], -1),
        #                                     grad_input,
        #                                     in_h, in_w,
        #                                     ctx.kernel_size[0], ctx.kernel_size[1],
        #                                     ctx.dilation[0], ctx.dilation[1],
        #                                     ctx.padding[0], ctx.padding[1],
        #                                     ctx.stride[0], ctx.stride[1])

        grad_input = F.fold(grad_diff.view(bs, ch * ctx.kernel_size[0] * ctx.kernel_size[1], -1),
                            (in_h, in_w), ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)

        return grad_input, None, None, None, None, None


class PacConv2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel, weight, bias=None, stride=1, padding=0, dilation=1, shared_filters=False):
        (bs, ch), in_sz = input.shape[:2], input.shape[2:]
        if kernel.size(1) > 1:
            raise ValueError('Non-singleton channel is not allowed for kernel.')
        ctx.input_size = in_sz
        ctx.in_ch = ch
        ctx.kernel_size = tuple(weight.shape[-2:])
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        ctx.shared_filters = shared_filters
        ctx.save_for_backward(input if (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]) else None,
                              kernel if (ctx.needs_input_grad[0] or ctx.needs_input_grad[2]) else None,
                              weight if (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]) else None)
        # ctx._backend = type2backend[input.type()]

        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)

        in_mul_k = cols.view(bs, ch, *kernel.shape[2:]) * kernel

        # matrix multiplication, written as an einsum to avoid repeated view() and permute()
        if shared_filters:
            output = torch.einsum('ijklmn,zykl->ijmn', (in_mul_k.float(), weight.float()))
        else:
            output = torch.einsum('ijklmn,ojkl->iomn', (in_mul_k.float(), weight.float()))

        if bias is not None:
            output += bias.view(1, -1, 1, 1)

        return output.clone()  # TODO understand why a .clone() is needed here

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_input = grad_kernel = grad_weight = grad_bias = None
        (bs, out_ch), out_sz = grad_output.shape[:2], grad_output.shape[2:]
        in_ch = ctx.in_ch

        input, kernel, weight = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            if ctx.shared_filters:
                grad_in_mul_k = grad_output.view(bs, out_ch, 1, 1, out_sz[0], out_sz[1]) \
                                * weight.view(ctx.kernel_size[0], ctx.kernel_size[1], 1, 1)
            else:
                grad_in_mul_k = torch.einsum('iomn,ojkl->ijklmn', (grad_output.float(), weight.float()))
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            in_cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
            in_cols = in_cols.view(bs, in_ch, ctx.kernel_size[0], ctx.kernel_size[1], out_sz[0], out_sz[1])
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.new()
            grad_im2col_output = grad_in_mul_k * kernel
            grad_im2col_output = grad_im2col_output.view(bs, -1, out_sz[0] * out_sz[1])
            # ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state,
            #                                     grad_im2col_output,
            #                                     grad_input,
            #                                     ctx.input_size[0], ctx.input_size[1],
            #                                     ctx.kernel_size[0], ctx.kernel_size[1],
            #                                     ctx.dilation[0], ctx.dilation[1],
            #                                     ctx.padding[0], ctx.padding[1],
            #                                     ctx.stride[0], ctx.stride[1])
            grad_input = F.fold(grad_im2col_output,
                                ctx.input_size[:2], ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)

        if ctx.needs_input_grad[1]:
            grad_kernel = in_cols * grad_in_mul_k
            grad_kernel = grad_kernel.sum(dim=1, keepdim=True)
        if ctx.needs_input_grad[2]:
            in_mul_k = in_cols * kernel
            if ctx.shared_filters:
                grad_weight = torch.einsum('ijmn,ijklmn->kl', (grad_output.float(), in_mul_k.float()))
                grad_weight = grad_weight.view(1, 1, ctx.kernel_size[0], ctx.kernel_size[1]).contiguous()
            else:
                grad_weight = torch.einsum('iomn,ijklmn->ojkl', (grad_output.float(), in_mul_k.float()))
        if ctx.needs_input_grad[3]:
            grad_bias = torch.einsum('iomn->o', (grad_output.float(),))

        return grad_input, grad_kernel, grad_weight, grad_bias, None, None, None, None
def packernel2d(input, mask=None, kernel_size=0, stride=1, padding=0, output_padding=0, dilation=1,
                kernel_type='gaussian', smooth_kernel_type='none', smooth_kernel=None, inv_alpha=None, inv_lambda=None,
                channel_wise=False, normalize_kernel=False, transposed=False, native_impl=False):
    kernel_size = _pair(kernel_size)
    dilation = _pair(dilation)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    stride = _pair(stride)
    output_mask = False if mask is None else True
    norm = None

    if mask is not None and mask.dtype != input.dtype:
        mask = torch.tensor(mask, dtype=input.dtype, device=input.device)

    if transposed:
        in_sz = tuple(int((o - op - 1 - (k - 1) * d + 2 * p) // s) + 1 for (o, k, s, p, op, d) in
                      zip(input.shape[-2:], kernel_size, stride, padding, output_padding, dilation))
    else:
        in_sz = input.shape[-2:]

    if mask is not None or normalize_kernel:
        mask_pattern = input.new_ones(1, 1, *in_sz)
        mask_pattern = nd2col(mask_pattern, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                              dilation=dilation, transposed=transposed)
        if mask is not None:
            mask = nd2col(mask, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                          dilation=dilation, transposed=transposed)
            if not normalize_kernel:
                norm = mask.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) \
                       / mask_pattern.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        else:
            mask = mask_pattern

    if transposed:
        stride = _pair(1)
        padding = tuple((k - 1) * d // 2 for (k, d) in zip(kernel_size, dilation))

    # native_impl = True
    if native_impl:
        bs, k_ch, in_h, in_w = input.shape

        x = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)
        x = x.view(bs, k_ch, -1, *x.shape[-2:]).contiguous()

        if smooth_kernel_type == 'none':
            self_idx = kernel_size[0] * kernel_size[1] // 2
            feat_0 = x[:, :, self_idx:self_idx + 1, :, :]
        else:
            smooth_kernel_size = smooth_kernel.shape[2:]
            smooth_padding = (int(padding[0] - (kernel_size[0] - smooth_kernel_size[0]) / 2),
                              int(padding[1] - (kernel_size[1] - smooth_kernel_size[1]) / 2))
            crop = tuple(-1 * np.minimum(0, smooth_padding))
            input_for_kernel_crop = input.view(-1, 1, in_h, in_w)[:, :,
                                    crop[0]:_neg_idx(crop[0]), crop[1]:_neg_idx(crop[1])]
            smoothed = F.conv2d(input_for_kernel_crop, smooth_kernel,
                                stride=stride, padding=tuple(np.maximum(0, smooth_padding)))
            feat_0 = smoothed.view(bs, k_ch, 1, *x.shape[-2:])

        x = torch.cosine_similarity(x, feat_0, dim=1).unsqueeze(1)
        # x = x - feat_0
        # if kernel_type.find('_asym') >= 0:
        #     x = F.relu(x, inplace=True)
        # # x.pow_(2)  # this causes an autograd issue in pytorch>0.4
        # x = x * x
        # if not channel_wise:
        #     x = torch.sum(x, dim=1, keepdim=True)
        # if kernel_type == 'gaussian':
        #     x = torch.exp_(x.mul_(-0.5))  # TODO profiling for identifying the culprit of 5x slow down
        #     # x = torch.exp(-0.5 * x)
        # elif kernel_type.startswith('inv_'):
        #     epsilon = 1e-4
        #     x = inv_alpha.view(1, -1, 1, 1, 1) \
        #         + torch.pow(x + epsilon, 0.5 * inv_lambda.view(1, -1, 1, 1, 1))
        # else:
        #     raise ValueError()
        output = x.view(*(x.shape[:2] + tuple(kernel_size) + x.shape[-2:])).contiguous()
    else:
        assert (smooth_kernel_type == 'none' and
                kernel_type == 'gaussian')
        output = GaussKernel2dFn.apply(input, kernel_size, stride, padding, dilation, channel_wise)

    if mask is not None:
        output = output * mask  # avoid numerical issue on masked positions

    if normalize_kernel:
        norm = output.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

    if norm is not None:
        empty_mask = (norm == 0)
        output = output / (norm + torch.tensor(empty_mask, dtype=input.dtype, device=input.device))
        output_mask = (1 - empty_mask) if output_mask else None
    else:
        output_mask = None

    return output, output_mask


def pacconv2d(input, kernel, weight, bias=None, stride=1, padding=0, dilation=1, shared_filters=False,
              native_impl=False):
    kernel_size = tuple(weight.shape[-2:])
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    if native_impl:
        # im2col on input
        im_cols = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)

        # main computation
        if shared_filters:
            output = torch.einsum('ijklmn,zykl->ijmn', (im_cols.float() * kernel.float(), weight.float()))
        else:
            output = torch.einsum('ijklmn,ojkl->iomn', (im_cols.float() * kernel.float(), weight.float()))

        if bias is not None:
            output += bias.view(1, -1, 1, 1)
    else:
        output = PacConv2dFn.apply(input, kernel, weight, bias, stride, padding, dilation, shared_filters)

    return output
class _PacConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, bias,
                 pool_only, kernel_type, smooth_kernel_type,
                 channel_wise, normalize_kernel, shared_filters, filler):
        super(_PacConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.pool_only = pool_only
        self.kernel_type = kernel_type
        self.smooth_kernel_type = smooth_kernel_type
        self.channel_wise = channel_wise
        self.normalize_kernel = normalize_kernel
        self.shared_filters = shared_filters
        self.filler = filler
        if any([k % 2 != 1 for k in kernel_size]):
            raise ValueError('kernel_size only accept odd numbers')
        if smooth_kernel_type.find('_') >= 0 and int(smooth_kernel_type[smooth_kernel_type.rfind('_') + 1:]) % 2 != 1:
            raise ValueError('smooth_kernel_type only accept kernels of odd widths')
        if shared_filters:
            assert in_channels == out_channels, 'when specifying shared_filters, number of channels should not change'
        if any([p > d * (k - 1) / 2 for (p, d, k) in zip(padding, dilation, kernel_size)]):
            # raise ValueError('padding ({}) too large'.format(padding))
            pass  # TODO verify that this indeed won't cause issues
        if not pool_only:
            if self.filler in {'pool', 'crf_pool'}:
                assert shared_filters
                self.register_buffer('weight', torch.ones(1, 1, *kernel_size))
                if self.filler == 'crf_pool':
                    self.weight[(0, 0) + tuple(k // 2 for k in kernel_size)] = 0  # Eq.5, DenseCRF
            elif shared_filters:
                self.weight = Parameter(torch.Tensor(1, 1, *kernel_size))
            elif transposed:
                self.weight = Parameter(torch.Tensor(in_channels, out_channels, *kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            if bias:
                self.bias = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)
        if kernel_type.startswith('inv_'):
            self.inv_alpha_init = float(kernel_type.split('_')[1])
            self.inv_lambda_init = float(kernel_type.split('_')[2])
            if self.channel_wise and kernel_type.find('_fixed') < 0:
                if out_channels <= 0:
                    raise ValueError('out_channels needed for channel_wise {}'.format(kernel_type))
                inv_alpha = self.inv_alpha_init * torch.ones(out_channels)
                inv_lambda = self.inv_lambda_init * torch.ones(out_channels)
            else:
                inv_alpha = torch.tensor(float(self.inv_alpha_init))
                inv_lambda = torch.tensor(float(self.inv_lambda_init))
            if kernel_type.find('_fixed') < 0:
                self.register_parameter('inv_alpha', Parameter(inv_alpha))
                self.register_parameter('inv_lambda', Parameter(inv_lambda))
            else:
                self.register_buffer('inv_alpha', inv_alpha)
                self.register_buffer('inv_lambda', inv_lambda)
        elif kernel_type != 'gaussian':
            raise ValueError('kernel_type set to invalid value ({})'.format(kernel_type))
        if smooth_kernel_type.startswith('full_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            self.smooth_kernel = Parameter(torch.Tensor(1, 1, *repeat(smooth_kernel_size, len(kernel_size))))
        elif smooth_kernel_type == 'gaussian':
            smooth_1d = torch.tensor([.25, .5, .25])
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0).unsqueeze(0))
        elif smooth_kernel_type.startswith('average_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            smooth_1d = torch.tensor((1.0 / smooth_kernel_size,) * smooth_kernel_size)
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0).unsqueeze(0))
        elif smooth_kernel_type != 'none':
            raise ValueError('smooth_kernel_type set to invalid value ({})'.format(smooth_kernel_type))

        self.reset_parameters()

    def reset_parameters(self):
        if not (self.pool_only or self.filler in {'pool', 'crf_pool'}):
            if self.filler == 'uniform':
                n = self.in_channels
                for k in self.kernel_size:
                    n *= k
                stdv = 1. / math.sqrt(n)
                if self.shared_filters:
                    stdv *= self.in_channels
                self.weight.data.uniform_(-stdv, stdv)
                if self.bias is not None:
                    self.bias.data.uniform_(-stdv, stdv)
            elif self.filler == 'linear':
                effective_kernel_size = tuple(2 * s - 1 for s in self.stride)
                pad = tuple(int((k - ek) // 2) for k, ek in zip(self.kernel_size, effective_kernel_size))
                assert self.transposed and self.in_channels == self.out_channels
                assert all(k >= ek for k, ek in zip(self.kernel_size, effective_kernel_size))
                w = 1.0
                for i, (p, s, k) in enumerate(zip(pad, self.stride, self.kernel_size)):
                    d = len(pad) - i - 1
                    w = w * (np.array((0.0,) * p + tuple(range(1, s)) + tuple(range(s, 0, -1)) + (0,) * p) / s).reshape(
                        (-1,) + (1,) * d)
                    if self.normalize_kernel:
                        w = w * np.array(tuple(((k - j - 1) // s) + (j // s) + 1.0 for j in range(k))).reshape(
                            (-1,) + (1,) * d)
                self.weight.data.fill_(0.0)
                for c in range(1 if self.shared_filters else self.in_channels):
                    self.weight.data[c, c, :] = torch.tensor(w)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            elif self.filler in {'crf', 'crf_perturbed'}:
                assert len(self.kernel_size) == 2 and self.kernel_size[0] == self.kernel_size[1] \
                       and self.in_channels == self.out_channels
                perturb_range = 0.001
                n_classes = self.in_channels
                gauss = np_gaussian_2d(self.kernel_size[0]) * self.kernel_size[0] * self.kernel_size[0]
                gauss[self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
                if self.shared_filters:
                    self.weight.data[0, 0, :] = torch.tensor(gauss)
                else:
                    compat = 1.0 - np.eye(n_classes, dtype=np.float32)
                    self.weight.data[:] = torch.tensor(compat.reshape(n_classes, n_classes, 1, 1) * gauss)
                if self.filler == 'crf_perturbed':
                    self.weight.data.add_((torch.rand_like(self.weight.data) - 0.5) * perturb_range)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            else:
                raise ValueError('Initialization method ({}) not supported.'.format(self.filler))
        if hasattr(self, 'inv_alpha') and isinstance(self.inv_alpha, Parameter):
            self.inv_alpha.data.fill_(self.inv_alpha_init)
            self.inv_lambda.data.fill_(self.inv_lambda_init)
        if hasattr(self, 'smooth_kernel') and isinstance(self.smooth_kernel, Parameter):
            self.smooth_kernel.data.fill_(1.0 / np.multiply.reduce(self.smooth_kernel.shape))

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', kernel_type={kernel_type}')
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.bias is None:
            s += ', bias=False'
        if self.smooth_kernel_type != 'none':
            s += ', smooth_kernel_type={smooth_kernel_type}'
        if self.channel_wise:
            s += ', channel_wise=True'
        if self.normalize_kernel:
            s += ', normalize_kernel=True'
        if self.shared_filters:
            s += ', shared_filters=True'
        return s.format(**self.__dict__)

class PacConv2d(_PacConvNd):
    r"""
    Args (in addition to those of Conv2d):
        kernel_type (str): 'gaussian' | 'inv_{alpha}_{lambda}[_asym][_fixed]'. Default: 'gaussian'
        smooth_kernel_type (str): 'none' | 'gaussian' | 'average_{sz}' | 'full_{sz}'. Default: 'none'
        normalize_kernel (bool): Default: False
        shared_filters (bool): Default: False
        filler (str): 'uniform'. Default: 'uniform'

    Note:
        - kernel_size only accepts odd numbers
        - padding should not be larger than :math:`dilation * (kernel_size - 1) / 2`
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 kernel_type='gaussian', smooth_kernel_type='none', normalize_kernel=False, shared_filters=False,
                 filler='uniform', native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PacConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, False, _pair(0), bias,
            False, kernel_type, smooth_kernel_type, False, normalize_kernel, shared_filters, filler)

        self.native_impl = native_impl

    def compute_kernel(self, input_for_kernel, input_mask=None):
        return packernel2d(input_for_kernel, input_mask,
                           kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                           dilation=self.dilation, kernel_type=self.kernel_type,
                           smooth_kernel_type=self.smooth_kernel_type,
                           smooth_kernel=self.smooth_kernel if hasattr(self, 'smooth_kernel') else None,
                           inv_alpha=self.inv_alpha if hasattr(self, 'inv_alpha') else None,
                           inv_lambda=self.inv_lambda if hasattr(self, 'inv_lambda') else None,
                           channel_wise=False, normalize_kernel=self.normalize_kernel, transposed=False,
                           native_impl=self.native_impl)

    def forward(self, input_2d, input_for_kernel, kernel=None, mask=None):
        output_mask = None
        if kernel is None:
            kernel, output_mask = self.compute_kernel(input_for_kernel, mask)

        output = pacconv2d(input_2d, kernel, self.weight, self.bias, self.stride, self.padding, self.dilation,
                           self.shared_filters, self.native_impl)

        return output if output_mask is None else (output, output_mask)

class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()

        if with_pac:
            self.pac = PacConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            # self.norm = nn.InstanceNorm2d(in_channels)
            self.norm = LayerNorm2d(in_channels, eps=1e-6)
        self.decoding = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.InstanceNorm2d(out_channels),
            LayerNorm2d(out_channels, eps=1e-6),
            nn.ReLU(inplace=True))
            # nn.GELU())

    def forward(self, feat, guide):
        if with_pac:
            feat = self.norm(self.pac(feat, guide)).relu()
            # feat = nn.GELU()(self.norm(self.pac(feat, guide)))
        output = self.decoding(feat)
        return output

def corr_fun(Kernel_tmp, Feature):
    size = Kernel_tmp.size() # (4, 512, 32, 32)
    CORR = []
    Kernel = []
    for i in range(len(Feature)):
        ker = Kernel_tmp[i:i + 1] # (1, 512, 32, 32)，猜测[i:i + 1]这样写是为了保留原始维度，因为写成Kernel_tmp[i]就会缺少第一维，而i:i+1不会这样
        fea = Feature[i:i + 1] # (1, 512, 32, 32)
        ker = ker.view(size[1], size[2] * size[3]).transpose(0, 1) # (1024, 512), 即(H*W, s^2)
        ker = ker.unsqueeze(2).unsqueeze(3) # (1024, 512, 1, 1)

        co = F.conv2d(fea, ker.contiguous()) # (1, 1024, 32, 32)
        CORR.append(co)
        ker = ker.unsqueeze(0)
        Kernel.append(ker)
    corr = torch.cat(CORR, 0)
    Kernel = torch.cat(Kernel, 0)
    return corr, Kernel


class CorrelationLayer(nn.Module):
    def __init__(self, feat_channel):
        super(CorrelationLayer, self).__init__()

        self.pool_layer = nn.AdaptiveAvgPool2d((corr_size, corr_size))

        self.corr_reduce = nn.Sequential(
            nn.Conv2d(corr_size * corr_size, feat_channel, kernel_size=1),
            # nn.InstanceNorm2d(feat_channel),
            LayerNorm2d(feat_channel, eps=1e-6),
            nn.ReLU(),
            # nn.GELU(),
            nn.Conv2d(feat_channel, feat_channel, 3, padding=1),
            LayerNorm2d(feat_channel, eps=1e-6),
        )
        # self.Dnorm = nn.InstanceNorm2d(feat_channel)
        self.Dnorm = LayerNorm2d(feat_channel, eps=1e-6)

        self.feat_adapt = nn.Sequential(
            nn.Conv2d(feat_channel * 2, feat_channel, 1),
            # nn.InstanceNorm2d(feat_channel),
            LayerNorm2d(feat_channel, eps=1e-6),
            nn.ReLU(inplace=True)
            # nn.GELU(),
        )

    def forward(self, x):
        # calculate correlation map
        RGB_feat_downsize = F.normalize(self.pool_layer(x[0])) # (b, 512, 32, 32)
        RGB_feat_norm = F.normalize(x[0]) # (b, 512, 32, 32)
        RGB_corr, _ = corr_fun(RGB_feat_downsize, RGB_feat_norm)

        Depth_feat_downsize = F.normalize(self.pool_layer(x[1]))
        Depth_feat_norm = F.normalize(x[1])
        Depth_corr, _ = corr_fun(Depth_feat_downsize, Depth_feat_norm)

        corr = (RGB_corr + Depth_corr) / 2 # (b, 1024, 32, 32)
        Red_corr = self.corr_reduce(corr) # (b, 512, 32, 32), self.corr_reduce是1x1卷积降维

        # beta cond
        new_feat = torch.cat([x[0], Red_corr], 1) # (b, 1024, 32, 32)
        new_feat = self.feat_adapt(new_feat) # (b, 512, 32, 32), self.feat_adapt是1x1卷积降维, 这一步的输出对应论文中的Fca

        Depth_feat = self.Dnorm(x[1]) # (b, 512, 32, 32)
        return new_feat, Depth_feat

class ImageDecoder(nn.Module):
    def __init__(self):

        super(ImageDecoder, self).__init__()
        channels = [64, 128, 256, 512, 512, 512]
        self.with_corr = with_corr

        # feature fusing: encoder feature and decoder feature
        self.enc_dec_fusing5 = EncDecFusing(channels[4])
        self.enc_dec_fusing4 = EncDecFusing(channels[3])
        self.enc_dec_fusing3 = EncDecFusing(channels[2])
        self.enc_dec_fusing2 = EncDecFusing(channels[1])
        self.enc_dec_fusing1 = EncDecFusing(channels[0])

        # correlation calculate
        self.corr_layer6 = CorrelationLayer(feat_channel=channels[5])
        self.corr_layer5 = CorrelationLayer(feat_channel=channels[4])
        self.corr_layer4 = CorrelationLayer(feat_channel=channels[3])
        self.corr_layer3 = CorrelationLayer(feat_channel=channels[2])
        self.corr_layer2 = CorrelationLayer(feat_channel=channels[1])
        self.corr_layer1 = CorrelationLayer(feat_channel=channels[0])

        # decoder modules
        self.decoder6 = decoder(channels[5], channels[4])
        self.decoder5 = decoder(channels[4], channels[3])
        self.decoder4 = decoder(channels[3], channels[2])
        self.decoder3 = decoder(channels[2], channels[1])
        self.decoder2 = decoder(channels[1], channels[0])
        self.decoder1 = decoder(channels[0], channels[0])

        # predict layers
        self.conv_loss6 = nn.Conv2d(in_channels=channels[4], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss5 = nn.Conv2d(in_channels=channels[3], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss4 = nn.Conv2d(in_channels=channels[2], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss3 = nn.Conv2d(in_channels=channels[1], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss2 = nn.Conv2d(in_channels=channels[0], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss1 = nn.Conv2d(in_channels=channels[0], out_channels=1, kernel_size=3, padding=1)

    def forward(self, image_feats, depth_feats):

        encoder_conv1, encoder_conv2, encoder_conv3, encoder_conv4, encoder_conv5, x6 = image_feats
        depth_feat1, depth_feat2, depth_feat3, depth_feat4, depth_feat5, depth_feat6 = depth_feats

        lamda = 0.01
        corr_fea_6, depth_feat6 = self.corr_layer6((x6, depth_feat6)) # corr_fea_6对应论文中的Fca，depth_feat6对应论文中的Fd
        dec_fea_6 = self.decoder6(corr_fea_6, depth_feat6 * lamda) # 论文中的PAC操作在self.decoder_i中实现，
        mask6 = self.conv_loss6(dec_fea_6)

        fus_fea_5 = self.enc_dec_fusing5(encoder_conv5, dec_fea_6)
        corr_fea_5, depth_feat5 = self.corr_layer5((fus_fea_5, depth_feat5))
        dec_fea_5 = self.decoder5(corr_fea_5, depth_feat5 * lamda)
        mask5 = self.conv_loss5(dec_fea_5)

        fus_fea_4 = self.enc_dec_fusing4(encoder_conv4, dec_fea_5)
        corr_fea_4, depth_feat4 = self.corr_layer4((fus_fea_4, depth_feat4))
        dec_fea_4 = self.decoder4(corr_fea_4, depth_feat4 * lamda)
        mask4 = self.conv_loss4(dec_fea_4)

        # image (decoder3)
        fus_fea_3 = self.enc_dec_fusing3(encoder_conv3, dec_fea_4)
        corr_fea_3, depth_feat3 = self.corr_layer3((fus_fea_3, depth_feat3))
        dec_fea_3 = self.decoder3(corr_fea_3, depth_feat3 * lamda)
        mask3 = self.conv_loss3(dec_fea_3)

        # image (decoder2)
        fus_fea_2 = self.enc_dec_fusing2(encoder_conv2, dec_fea_3)
        corr_fea_2, depth_feat2 = self.corr_layer2((fus_fea_2, depth_feat2))
        dec_fea_2 = self.decoder2(corr_fea_2, depth_feat2 * lamda)
        mask2 = self.conv_loss2(dec_fea_2)

        # image (decoder1)
        fus_fea_1 = self.enc_dec_fusing1(encoder_conv1, dec_fea_2)
        corr_fea_1, depth_feat1 = self.corr_layer1((fus_fea_1, depth_feat1))
        dec_fea_1 = self.decoder1(corr_fea_1, depth_feat1 * lamda)
        mask1 = self.conv_loss1(dec_fea_1)

        return mask6, mask5, mask4, mask3, mask2, mask1