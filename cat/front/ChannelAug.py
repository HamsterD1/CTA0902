# Copyright 2020 Tsinghua SPMI Lab 
# Apache 2.0.
# Author: Xiangzhu Kong(kongxiangzhu99@gmail.com)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAugment(nn.Module):
    def __init__(self, min_channels=2, max_channels=8,selected_channels=None):
        """
        初始化 ChannelAugment 模块。
        
        Args:
            min_channels (int): 保留的最少通道数。
            max_channels (int): 保留的最多通道数。
        """
        super(ChannelAugment, self).__init__()
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.selected_channels = selected_channels
        if self.selected_channels is not None:
            print(f"selected_channels:{self.selected_channels}")

    def forward(self, x, Random_Flag):
        """
        前向传播逻辑。
        
        Args:
            x (Tensor): 输入张量，形状为 (B, C, T)。
            Random_Flag (bool): 是否进行随机选取通道。
        
        Returns:
            Tensor: 输出张量，形状与输入相同。
        """
        B, C, T = x.shape
        
        if not C == self.max_channels:
            x = x[:,:self.max_channels, :]

        if Random_Flag and self.selected_channels is None:
            # 随机选择需要保留的通道数
            num_channels_to_keep = torch.randint(
                self.min_channels, self.max_channels + 1, (1,)
            ).item()
            # 随机选择保留的通道索引
            self.selected_channels = torch.randperm(C)[:num_channels_to_keep]

        if self.selected_channels is not None:
            # 创建与输入形状一致的 mask
            mask = torch.zeros(C, device=x.device)
            mask[self.selected_channels] = 1
            mask = mask.view(1, C, 1)  # 调整形状以便广播
            return x * mask
        else:
            # 不进行处理，选择所有通道
            return x

# 示例用法
if __name__ == "__main__":
    B, C, T = 4, 16, 128  # Batch size, channels, time steps
    x = torch.randn(B, C, T)

    augment = ChannelAugment()
    Random_Flag = True
    output = augment(x, Random_Flag)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
