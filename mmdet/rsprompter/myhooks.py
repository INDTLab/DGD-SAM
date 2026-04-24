from typing import Optional

import torch
import os
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS
from mmengine.logging import MMLogger
from typing import Dict, Optional, Sequence, Union
import copy

@HOOKS.register_module()
class NaNCheckHook(Hook):
    def __init__(self):
        super(NaNCheckHook, self).__init__()

    def before_train_epoch(self, runner):
        # 在每个训练轮开始前检查模型参数
        for name, param in runner.model.named_parameters():
            if param.requires_grad:
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"NaN or Inf detected in parameters before epoch {runner.epoch}: {name}")

    def after_train_iter(self, runner):
        # 在每次训练迭代后检查损失和模型输出
        if torch.isnan(runner.outputs['loss']).any() or torch.isinf(runner.outputs['loss']).any():
            print(f"NaN or Inf detected in loss after iteration {runner.iter}: {runner.outputs['loss']}")

        # 检查输出
        outputs = runner.outputs['logits']  # 假设你有一个logits输出
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f"NaN or Inf detected in outputs after iteration {runner.iter}: {outputs}")

@HOOKS.register_module()
class EpochInfoHook(Hook):
    def before_train_epoch(self, runner):
        # 获取当前 epoch 数
        current_epoch = runner.epoch
        # 获取总的 epoch 数
        max_epochs = runner.max_epochs
        # 将 epoch 信息传递给模型
        runner.model.current_epoch = current_epoch
        runner.model.max_epochs = max_epochs

# @HOOKS.register_module()
# class CustomFileLoggerHook(Hook):
#     def __init__(self, file_path="/data2/yihan/MyProject/RSPrompter-release/mmdet/rsprompter/__init__.py"):
#         self.file_path = file_path
#
#     def before_run(self, runner):
#         """在训练开始前打印 .py 文件内容"""
#         logger = MMLogger.get_current_instance()
#         if os.path.exists(self.file_path):
#             with open(self.file_path, 'r', encoding='utf-8') as f:
#                 file_content = f.read()
#                 logger.info(f"========== Printing {self.file_path} ==========\n{file_content}\n")
#         else:
#             logger.warning(f"File {self.file_path} does not exist!")

@HOOKS.register_module()
class CustomFileLoggerHook(Hook):
    def __init__(self, file_path="/data2/yihan/MyProject/mmdetection-main/mmdet/models/detectors/__init__.py"):
        self.file_path = file_path

    def before_run(self, runner):
        """在训练开始前打印 .py 文件内容"""
        # logger = MMLogger.get_current_instance()
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                # logger.info(f"========== Printing {self.file_path} ==========\n{file_content}\n")
                runner.logger.info(f"========== Printing {self.file_path} ==========\n{file_content}\n")
        else:
            # logger.warning(f"File {self.file_path} does not exist!")
            runner.logger.info(f"File {self.file_path} does not exist!")

@HOOKS.register_module()
class IgnoreOOMHook(Hook):
    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self._oom = False

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self._oom:
            runner.logger.warning(f"OOM occurred at batch {batch_idx}, skipping...")

    def before_backward(self, runner, loss):
        try:
            loss.backward()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                runner.logger.warning("OOM caught in backward(), skipping batch.")
                self._oom = True
                torch.cuda.empty_cache()
                # 让优化器不执行 step
                raise runner.CancelForwardException  # 控制流程跳过后续
            else:
                raise e

@HOOKS.register_module()
class EvalTestHook(Hook):
    def __init__(self, dataloader):
        self.test_dataloader = dataloader

    def after_train_epoch(self, runner: Runner):
    # def after_val_epoch(self, runner: Runner):
        # 评估 test 数据集
        # 初始化 test_loop（只做一次）
        if runner.test_loop is None:
            runner.test_loop = runner.build_test_loop(runner._test_loop)

        runner.logger.info(f'[EvalTestLoopHook] Running test after epoch {runner.epoch + 1}...')
        test_metrics = runner.test_loop.run()
        runner.logger.info(f'[EvalTestLoopHook] Test metrics: {test_metrics}')


# @HOOKS.register_module()
# class get_Binary_mIoU_Hook(Hook):
#     def __init__(self, dataloader):
#         self.test_dataloader = dataloader
#
#     def after_test(self, runner: Runner):
#     # def after_val_epoch(self, runner: Runner):
#         # 评估 test 数据集
#         # 初始化 test_loop（只做一次）
#         if runner.test_loop is None:
#             runner.test_loop = runner.build_test_loop(runner._test_loop)
#
#         runner.logger.info(f'[EvalTestLoopHook] Running test after epoch {runner.epoch + 1}...')
#         test_metrics = runner.test_loop.run()
#         runner.logger.info(f'[EvalTestLoopHook] Test metrics: {test_metrics}')
