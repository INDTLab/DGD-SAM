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
        self.depth_root = depth_root  # 初始化时保存参数
        self.suffix = suffix
        self.to_float32 = to_float32

    def __call__(self, results: dict) -> dict:
        # 从RGB路径推导深度图路径
        img_path = results['img_path']
        # filename = osp.splitext(osp.basename(img_path))[0] # image_0002
        # depth_filename = f"{filename}{self.suffix}"
        depth_path = results['depth_path']

        # 加载深度图（示例使用16位PNG）
        depth = mmcv.imread(depth_path, flag='unchanged')  # 保持原始位深

        # 转换为float32（可选）
        if self.to_float32:
            depth = depth.astype(np.float32)

        results['depth'] = depth
        # 删除
        results['img'] = np.dstack((results['img'], depth))
        return results

@TRANSFORMS.register_module()
class DualRandomResize(RandomResize):
    def __init__(self, depth_interp='nearest', **kwargs):
        super().__init__(**kwargs)
        self.depth_interp = depth_interp

    def _resize_dual(self, results):
        if 'depth' in results:
            h, w = results['img'].shape[:2]
            depth = mmcv.imresize(
                results['depth'],
                (w, h),  # 保持与RGB相同的尺寸
                interpolation=self.depth_interp
            )
            results['depth'] = depth
        return results

    def __call__(self, results):
        results = super().__call__(results)
        return self._resize_dual(results)


@TRANSFORMS.register_module()
class DualRandomCrop(RandomCrop):
    def __init__(self, depth_pad_val=0, **kwargs):
        super().__init__(**kwargs)
        self.depth_pad_val = depth_pad_val

    def _crop_dual(self, results):
        if 'depth' in results:
            crop_y1, crop_y2, crop_x1, crop_x2 = self._crop_area
            depth = results['depth']

            # 执行裁剪
            depth_cropped = depth[crop_y1:crop_y2, crop_x1:crop_x2]

            # 处理负坐标情况（allow_negative_crop=True时）
            if crop_y1 < 0 or crop_x1 < 0:
                pad_h = max(0, -crop_y1)
                pad_w = max(0, -crop_x1)
                depth_cropped = mmcv.impad(
                    depth_cropped,
                    padding=(pad_w, pad_h, 0, 0),
                    pad_val=self.depth_pad_val)

            results['depth'] = depth_cropped
        return results

    def __call__(self, results):
        results = super().__call__(results)
        return self._crop_dual(results)


@TRANSFORMS.register_module()
class DualRandomFlip(RandomFlip):
    """同步处理RGB图像和深度图的翻转增强"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _flip_dual(self, results: dict):
        """处理深度图的翻转"""
        if 'depth' in results:
            # 获取翻转方向和状态
            flip_direction = results.get('flip_direction', None)
            do_flip = results.get('flip', False)

            if do_flip and (flip_direction is not None):
                depth = results['depth']

                # 执行翻转（与RGB相同方向）
                depth_flipped = mmcv.imflip(
                    depth,
                    direction=flip_direction,
                    # 深度图推荐使用非插值方式
                    # interpolation='nearest'
                )
                results['depth'] = depth_flipped
        return results

    def __call__(self, results):
        results = super().__call__(results)
        return self._flip_dual(results)

@TRANSFORMS.register_module()
class DualPackDetInputs(PackDetInputs):
    def __init__(self, meta_keys=(), **kwargs):
        super().__init__(meta_keys=meta_keys, **kwargs)
        self.meta_keys = meta_keys + ('depth_path',)  # 添加深度路径到meta

    def transform(self, results):
        packed_results = super().transform(results)

        # 添加深度数据
        if 'depth' in results:
            depth = results['depth']
            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, -1)  # 添加通道维度
            # packed_results['inputs']['depth'] = torch.from_numpy(depth.transpose(2, 0, 1))
            packed_results['depth'] = torch.from_numpy(depth.transpose(2, 0, 1))

        return packed_results


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
        tmp = super().forward(data, training) # DetDataPreprocessor只会处理 data中的 img的前三个通道，所以这里直接调用super.forward()就可以
        batch_inputs, batch_data_samples = tmp['inputs'], tmp['data_samples']

        # # 处理深度图
        # depth_inputs = []
        # if 'depth' in data:
        #     depth = data['depth']
        #     # if not isinstance(depth, torch.Tensor):
        #     #     depth = torch.tensor(depth)
        #     for _depth in depth:
        #         # 将深度图转移到相同设备
        #         self.depth_mean = self.depth_mean.to(_depth.device)
        #         self.depth_std = self.depth_std.to(_depth.device)
        #
        #         # 执行归一化 (x - mean) / std
        #         _depth = (_depth - self.depth_mean) / self.depth_std
        #
        #         # 添加到batch_inputs
        #         # batch_inputs = torch.cat([batch_inputs, _depth.unsqueeze(1)], dim=1)
        #         depth_inputs.append(_depth)
        # # batch_depth_inputs = torch.cat(depth_inputs, dim=0).to(batch_inputs.device) # (b, 512, 512)
        # batch_depth_inputs = torch.stack(depth_inputs, dim=0).to(batch_inputs.device) # (b, 1, 512, 512)
        # final_inputs = torch.cat((batch_inputs, batch_depth_inputs), dim=1).to(batch_inputs.device)
        # # return {'inputs': batch_inputs, 'data_samples': batch_data_samples, 'depth_inputs': batch_depth_inputs}
        # return {'inputs': final_inputs, 'data_samples': batch_data_samples}

        depth_inputs = []

        depth = data['inputs']
        for _depth in depth:
            _depth = _depth[3, :, :]
            # 将深度图转移到相同设备
            self.depth_mean = self.depth_mean.to(_depth.device)
            self.depth_std = self.depth_std.to(_depth.device)

            # 执行归一化 (x - mean) / std
            _depth = (_depth - self.depth_mean) / self.depth_std

            # 添加到batch_inputs
            # batch_inputs = torch.cat([batch_inputs, _depth.unsqueeze(1)], dim=1)
            depth_inputs.append(_depth.unsqueeze(0))
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
        plt.imshow(final_inputs[0, 3, :, :].cpu().numpy(), cmap='gray')
        plt.show()
        '''
        return {'inputs': final_inputs, 'data_samples': batch_data_samples}

@TRANSFORMS.register_module()
class DualResize(Resize):
    """测试阶段确定性缩放"""
    def __init__(self, depth_interp='nearest', **kwargs):
        super().__init__(**kwargs)
        self.depth_interp = depth_interp

    def _resize_dual(self, results):
        if 'depth' in results:
            h, w = results['img'].shape[:2]
            depth = mmcv.imresize(
                results['depth'],
                (w, h),  # 保持与RGB相同的目标尺寸
                interpolation=self.depth_interp
            )
            results['depth'] = depth
        return results

    def __call__(self, results):
        results = super().__call__(results)
        return self._resize_dual(results)

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
        results = super().__call__(results)
        return self._pad_dual(results)

