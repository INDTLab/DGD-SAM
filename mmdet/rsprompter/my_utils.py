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

# import sys
# sys.path.append("/data2/yihan/MyProject/RSPrompter-release/")
# from segment_anything import


T = TypeVar('T')

class Fuse_Block(nn.Module):
    '''
    用的时候，先把待融合的特征图concat，再传给Fuse_block的实例
    '''
    def __init__(self,in_c=256*3,out_c=256): # in_channel主要看需要融合的特征图concat后，得到的通道数是几
        super().__init__()
        self.pw1 = Conv(in_c, out_c, 1, 1)
        self.se = senet(c = out_c)
        self.pw2 = Conv(out_c,out_c,1,1)
    def forward(self,x):
        x = self.pw1(x)
        x = self.se(x)
        x = self.pw2(x)
        #x = self.se(x)
        return x

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c_in, c_out, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class senet(nn.Module):
    def __init__(self,c=256,r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(c,c//r,1,1,0,bias=True),nn.ReLU(),nn.Conv2d(c//r,c,1,1,0,bias=True))
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias,std=1e-6)
        self.apply(_init_weights)

    def forward(self,x):
        res = x
        b,c,h,w=x.size()
        #x = x.view(b,c,h*w)
        avg_out = self.fc(self.avg_pool(x)) # self.avg_pool(x)=(b,256,1,1), avg_out=(b,256,1,1)
        max_out = self.fc(self.max_pool(x))
        out = avg_out+max_out
        x = x*self.sigmoid(out)
        #x = x.view(b,c,h,w)
        return x+res

class ChannelAttention(nn.Module):
    """
    通道注意力模块
    通过全局平均池化和全局最大池化生成两个描述向量，
    然后经过共享 MLP 得到通道注意力权重。
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 全局平均池化和最大池化：将 HxW 维度压缩到 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的 MLP（用 1x1 卷积实现），先降维再升维
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x 的形状: (B, C, H, W)
        avg_out = self.shared_MLP(self.avg_pool(x))  # (B, C, 1, 1)
        max_out = self.shared_MLP(self.max_pool(x))  # (B, C, 1, 1)
        # 两种池化结果相加
        out = avg_out + max_out
        # 通过 Sigmoid 得到归一化的通道注意力权重
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    利用通道维度上的最大池化和平均池化生成两个 2D 特征图，
    然后拼接后通过一个卷积层生成空间注意力图。
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 保证 kernel_size 合法：3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2  # 计算 padding 保持尺寸不变

        # 卷积层用于提取空间注意力
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x 的形状: (B, C, H, W)
        # 在通道维度上计算平均池化和最大池化，得到 (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接两个特征图，得到 (B, 2, H, W)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 通过卷积提取空间注意力，再经过 Sigmoid 激活
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    CBAM 模块
    将通道注意力和空间注意力按顺序串联，
    分别对输入特征进行加权，从而增强关键特征。
    """

    def __init__(self, in_channels=256, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, feat):
        # 先通过通道注意力
        out = feat * self.channel_attention(feat)
        # 再通过空间注意力
        out = out * self.spatial_attention(out)
        return out

class SEBlock(nn.Module):
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