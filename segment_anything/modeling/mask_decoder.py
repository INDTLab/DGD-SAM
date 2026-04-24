# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


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
            multimask_output: bool,
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



        import torch.nn.functional as F
        from matplotlib import pyplot as plt
        import numpy as np
        for i in range(4):
            mask = masks[0, i, :, :].detach().cpu()
            # mask = mask*(256.**-0.5)
            # mask = torch.sigmoid(mask)
            mask = torch.relu(mask)
            # mask = F.softmax(mask, dim=-1) # 不能用softmax，因为mask此时会有不少17.69这种值，e^17是很大的，很可能导致softmax时分母溢出
            mask = mask.numpy()
            # mask = (mask * 255).astype('uint8')
            # 找到最小值和最大值
            x_min = np.min(mask)
            x_max = np.max(mask)
            # 线性归一化到 0~255
            mask = (mask - x_min) / (x_max - x_min) * 255
            plt.imshow(mask, cmap='viridis')  # 使用灰度图进行可视化
            # plt.axis('off')  # 不显示坐标轴
            plt.show()

        # 注释掉这里，输出的mask就包括4个
        # # Select the correct mask or masks for output
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]  # 64x3x256x256
        # iou_pred = iou_pred[:, mask_slice]  # 64x3


        # if(torch.equal(masks[0:2], masks[2:4])):
        #     print("batch=2")
        # else:
        #     print("不等于")
        # import torch.nn.functional as F
        # # for i in range(64):
        # mask = masks[0, 0, :, :].detach().cpu()
        # mask = torch.sigmoid(mask)
        # # mask=F.gelu(mask)
        # mask = mask.numpy()
        # mask = (mask * 255).astype('uint8')
        # from matplotlib import pyplot as plt
        # plt.imshow(mask, cmap='gray')  # 使用灰度图进行可视化
        # plt.axis('off')  # 不显示坐标轴
        # plt.show()

        # Prepare output
        return masks, iou_pred


    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        from matplotlib import pyplot as plt
        abs_pos = image_pe.permute(0, 2, 3, 1)
        sum = 0
        abs_pos = abs_pos[0]
        a = abs_pos[::21, ::21, :]

        for i in range(4):
            for j in range(4):
                c = a[i, j, :]
                b = (c @ abs_pos.permute(2, 0, 1).view(256, -1)).view(64, 64)
                plt.subplot(4, 4, sum + 1)
                plt.axis('off')
                plt.imshow(b, cmap='viridis')
                sum += 1

        plt.show()

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
        src = src.transpose(1, 2).view(b, c, h, w) # (1, 256, 64, 64)

        '''
        # 查看未从256维降到32维时的注意力权重矩阵的热力图
        import torch.nn.functional as F
        from matplotlib import pyplot as plt
        # mask = (hs[0, 1+self.num_mask_tokens, :] @ src.view(b, c, h * w)).view(h, w)
        mask = (hs[0, 0, :] @ src.view(b, c, h * w)).view(h, w)
        mask = mask * (256. ** -0.5)
        # mask = torch.sigmoid(mask)
        # mask = torch.relu(mask)
        mask = F.softmax(mask, dim=-1)
        mask = mask.numpy()
        plt.imshow(mask, cmap='hot') # 使用热力图进行可视化
        plt.show()
        
        import torch.nn.functional as F
        # 第一个mask token与经过交叉注意力之后的特征图做自注意力，得到相似度矩阵
        tmp = F.interpolate(
            src,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )  # tmp.shape=(1,256,1024,1024)
        tmp = tmp[..., 0:722, 0:1024]  # tmp.shape=(1,256,722,1024)，原图大小是722*1024
        tmp = F.interpolate(tmp, (767, 1024), mode="bilinear", align_corners=False)  # tmp.shape=(1,256,767,1024)
        tmp = tmp[0]
        a = hs[0, 1, :] # hs.shape = (1, 7, 256) # 取出不需要输出多粒度分割结果时的mask token
        ans = (a @ tmp.view(-1, 767 * 1024)).view(-1, 767, 1024)
        plt.title('before mask token down to dim 32')
        plt.imshow(ans[0], cmap='viridis')
        plt.show()
        '''

        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        import torch.nn.functional as F
        # 第一个mask token与经过交叉注意力之后的特征图做自注意力，得到相似度矩阵
        tmp = F.interpolate(
            upscaled_embedding,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )  # tmp.shape=(1,256,1024,1024)
        tmp = tmp[..., 0:722, 0:1024]  # tmp.shape=(1,256,722,1024)，原图大小是722*1024
        tmp = F.interpolate(tmp, (767, 1024), mode="bilinear", align_corners=False)  # tmp.shape=(1,256,767,1024)
        tmp = tmp[0]
        a = hyper_in[0, 0, :]  # hyper_in.shape = (1, 4, 32)
        ans = (a @ tmp.view(-1, 767 * 1024)).view(-1, 767, 1024)
        plt.title('mask token dim=32')
        plt.imshow(ans[0], cmap='viridis')
        plt.show()

        # m=F.relu(masks[0,0,:,:])
        # m=F.sigmoid(masks[0,0,:,:])
        # prompt_predict.py中的单点提示
        # from matplotlib import pyplot as plt
        # plt.subplot(2, 2, 1)
        # plt.imshow(F.sigmoid(masks[0,0,:,:]))
        # plt.subplot(2, 2, 2)
        # plt.imshow(F.sigmoid(masks[0,1,:,:]))
        # plt.subplot(2, 2, 3)
        # plt.imshow(F.sigmoid(masks[0,2,:,:]))
        # plt.subplot(2, 2, 4)
        # plt.imshow(F.sigmoid(masks[0,3,:,:]))
        # plt.show()

        # from matplotlib import pyplot as plt
        # sum = 1
        # for i in range(8):
        #     for j in range(8):
        #         plt.subplot(8, 8, sum)
        #         plt.imshow(F.sigmoid(masks[sum - 1, 1, :, :]))
        #         sum += 1
        # plt.show()

        # 打断点
        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
