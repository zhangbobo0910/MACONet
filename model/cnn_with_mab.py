import torch.nn as nn
from .cnn import MaskCNN_1, MaskCNN_2          # 原始 SDM+BFM 实现
from .mab_wrapper import MABWrapper            # 刚刚保存的包装器


class MaskCNN_1_MAB(MaskCNN_1):
    """将 MAB 串联到 MaskCNN_1 的输出。"""
    def __init__(self, input_channels, output_channels,
                 kernel_size=3, depth=3, theta=1):
        super().__init__(input_channels, output_channels,
                         kernel_size, depth, theta)
        # forward() 最终返回 2×input_channels 通道
        self.mab = MABWrapper(input_channels * 2)

    def forward(self, x, mask, training):
        x = super().forward(x, mask, training)
        return self.mab(x)                      # 串联 MAB


class MaskCNN_2_MAB(MaskCNN_2):
    """将 MAB 串联到 MaskCNN_2 的输出。"""
    def __init__(self, input_channels, output_channels,
                 kernel_size=3, depth=3, theta=1):
        super().__init__(input_channels, output_channels,
                         kernel_size, depth, theta)
        self.mab = MABWrapper(input_channels * 2)

    def forward(self, x, mask, training):
        x = super().forward(x, mask, training)
        return self.mab(x)
