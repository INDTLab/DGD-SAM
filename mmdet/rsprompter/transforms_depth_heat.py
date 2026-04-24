import numpy as np
from mmcv.image import imread
from mmdet.registry import TRANSFORMS
import mmcv
from mmdet.registry import MODELS
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmdet.models import DetDataPreprocessor
import torch
from mmcv.transforms import RandomResize
from mmdet.datasets.transforms import RandomCrop, PackDetInputs, RandomFlip, Resize, Pad
import os.path as osp
from mmengine.model.utils import stack_batch
from mmdet.datasets.transforms import CopyPaste

@TRANSFORMS.register_module()
class LoadDepthFromFile:
    def __init__(self,
                 depth_root: str,  # 必须声明此参数
                 suffix: str = '.png',
                 to_float32: bool = False):
        self.depth_root = depth_root # 初始化时保存参数
        self.suffix = suffix
        self.to_float32 = to_float32

    def __call__(self, results: dict) -> dict:
        # 从RGB路径推导深度图路径
        img_path = results['img_path']
        # filename = osp.splitext(osp.basename(img_path))[0] # image_0002
        # depth_filename = f"{filename}{self.suffix}"
        depth_path = results['depth_path']

        # 加载深度图（示例使用16位PNG）
        depth = mmcv.imread(depth_path, flag='unchanged') # 保持原始位深
        depth = depth[:,:,[2, 1, 0]] # 由 BGR转换成 RGB格式
        # 转换为float32（可选）
        if self.to_float32:
            depth = depth.astype(np.float32)

        results['depth'] = depth
        results['img'] = np.dstack((results['img'], depth))
        return results

@MODELS.register_module()
class DualDetDataPreprocessor(DetDataPreprocessor):
    """支持RGB和深度图的双模态预处理"""

    def __init__(self, depth_mean=0.0, depth_std=1.0, **kwargs):
        super().__init__(**kwargs)
        # 深度图归一化参数
        self.depth_mean = torch.tensor(depth_mean, dtype=torch.float32)
        self.depth_std = torch.tensor(depth_std, dtype=torch.float32)

    def forward(self, data, training=False):
        # 原始预处理流程
        tmp = super().forward(data, training) # DetDataPreprocessor只会处理 data中的 img的前三个通道，所以这里可以直接调用super.forward()
        batch_inputs, batch_data_samples = tmp['inputs'], tmp['data_samples']

        depth_inputs = []

        depth = data['inputs']
        for _depth in depth:
            _depth = _depth[3:, :, :] # (3, 512, 512)
            # 将深度图转移到相同设备
            self.depth_mean = self.depth_mean.to(_depth.device).view(3, 1, 1)
            self.depth_std = self.depth_std.to(_depth.device).view(3, 1, 1)

            # 执行归一化 (x - mean) / std
            _depth = (_depth - self.depth_mean) / self.depth_std

            # 添加到batch_inputs
            # batch_inputs = torch.cat([batch_inputs, _depth.unsqueeze(1)], dim=1)
            # depth_inputs.append(_depth.unsqueeze(0))
            depth_inputs.append(_depth)
        '''
        stack_batch接收list(tensor(C, H, W), )，输出是tensor(b, C, H, W)，list中有几个张量，输出的b就是几
        '''
        batch_depth_inputs = stack_batch(depth_inputs,32,0).to(batch_inputs.device) # (b, 1, H, W)

        # batch_depth_inputs = torch.stack(depth_inputs, dim=0).to(batch_inputs.device)  # (b, 1, 512, 512)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                batch_depth_inputs, _ = batch_aug(batch_depth_inputs, None)
        final_inputs = torch.cat((batch_inputs, batch_depth_inputs), dim=1).to(batch_inputs.device)

        '''
        from matplotlib import pyplot as plt
        plt.subplot(1, 2, 1)
        plt.title('origin')
        plt.imshow(final_inputs[0,:3,:,:].permute(1,2,0).cpu().numpy())
        plt.subplot(1, 2, 2)
        plt.title('depth')
        plt.imshow(final_inputs[0, 3:, :, :].permute(1,2,0).cpu().numpy(), cmap='gray')
        plt.show()
        '''
        return {'inputs': final_inputs, 'data_samples': batch_data_samples}

@TRANSFORMS.register_module()
class DualPad(Pad):
    def __init__(self, pad_val=dict(img=0, depth=0), **kwargs):
        super().__init__(pad_val=pad_val, **kwargs)
        self.depth_pad_val = pad_val.get('depth', 0)

    def _pad_dual(self, results):
        if 'depth' in results:
            depth = results['depth']
            padded_depth = mmcv.impad(
                img=depth,
                shape=results['pad_shape'][:2],  # 使用与RGB相同的pad_shape
                pad_val=self.depth_pad_val
            )
            results['depth'] = padded_depth
        return results

    def __call__(self, results):
        results['depth'] = results['img'][:, :, 3:] # 经历过 Resize，所以要先更新下 results['depth']
        results['img'] = results['img'][:, :, :3]

        results = super().__call__(results) # 先 pad原图
        results = self._pad_dual(results)
        depth = results['depth']
        results['img'] = np.dstack((results['img'], depth))
        return results
