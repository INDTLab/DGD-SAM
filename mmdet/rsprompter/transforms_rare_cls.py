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
class RareClassCopyPaste(CopyPaste):
    # 针对稀有类应用定向的 CopyPaste 增强，增加数据集中稀有类别的出现次数
    def __init__(self, rare_class_ids, **kwargs):
        super().__init__(**kwargs)
        self.rare_class_ids = rare_class_ids

    def transform(self, results):
        assert 'mix_results' in results, 'Expected mix_results in input dict'
        assert len(results['mix_results']) == 1, 'Only 1 mix image supported in CopyPaste'

        paste_result = results['mix_results'][0]
        paste_labels = paste_result.get('gt_bboxes_labels', None)

        if paste_labels is None:
            return results  # 粘贴图没有标签，跳过

        # 找出稀有类目标的索引
        keep_inds = [i for i, label in enumerate(paste_labels) if label in self.rare_class_ids]
        if not keep_inds:
            return results  # 粘贴图中不含稀有类，跳过

        # 只保留稀有类目标
        for key in ['gt_bboxes', 'gt_bboxes_labels', 'gt_masks', 'gt_area']:
            if key in paste_result:
                paste_result[key] = paste_result[key][keep_inds]

        # 更新回 mix_results[0]
        results['mix_results'][0] = paste_result

        # 调用父类的 _copy_paste 执行粘贴操作
        return self._copy_paste(results, paste_result)