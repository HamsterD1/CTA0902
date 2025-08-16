# Copyright 2020 Tsinghua SPMI Lab 
# Apache 2.0.
# Author: Xiangzhu Kong(kongxiangzhu99@gmail.com)

import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output

class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual


class SACC_TC(nn.Module):
    def __init__(self, fre_input_dim, fre_attention_dim=256):
        super(SACC_TC, self).__init__()
        
        self.attention_dim = fre_attention_dim
        
        # Linear layers for Q, K, V
        self.Q = nn.Sequential(
            nn.Linear(fre_input_dim, fre_attention_dim),
            nn.ReLU()
        )
        self.K = nn.Sequential(
            nn.Linear(fre_input_dim, fre_attention_dim),
            nn.ReLU()
        )
        self.V = nn.Sequential(
            nn.Linear(fre_input_dim, 1),
            nn.ReLU()
        )
    
    def forward(self, Xmag, ilen):
        # B,T,C,F
        
        # Log magnitude
        Xmag = torch.clamp(Xmag, min=1e-10)  # 设置一个较小的阈值来避免零值

        Xmag_log = torch.log(Xmag)
        
        # Mean-variance normalization
        epsilon = 1e-10  # 微小的偏置值
        Xmag_log_std = Xmag_log.std(dim=-1, unbiased=False, keepdim=True)
        Xmag_log = (Xmag_log - Xmag_log.mean(dim=-1, keepdim=True)) / (Xmag_log_std + epsilon)
        
        # Calculate Q, K, V
        Q = self.Q(Xmag_log)
        K = self.K(Xmag_log)
        V = self.V(Xmag_log)
        
        # Attention calculation
        P = torch.matmul(Q, K.transpose(-2, -1)) / (self.attention_dim ** 0.5)
        attention_weights = torch.nn.functional.softmax(P, dim=-1)
        # B, T, C, 1
        W = torch.matmul(attention_weights, V)
        # Original SACC have softmax on C axis, 可以将其注释掉，使得滤波器权重系数大于1
        # W = torch.nn.functional.softmax(W, dim=2)
        
        # Broadcast W to match the shape of Xmag
        W = W.expand_as(Xmag)
        
        # Multiply broadcasted W with original Xmag
        S = W * Xmag
        
        # Sum over the C axis
        S = S.sum(dim=2)
        
        # Final output size: (B, T, F)
        return S,ilen,W  

class AttentionMaskNet(nn.Module):
    def __init__(self, n_fft, d_model=64, n_heads=2, num_layers=1):
        super(AttentionMaskNet, self).__init__()
        
        self.n_fft = n_fft
        self.d_model = d_model
        
        # 输入线性层
        self.input_layer = nn.Linear(n_fft // 2 + 1, d_model)
        
        # 自注意力层
        self.transformer_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model, n_heads) for _ in range(num_layers)]
        )
        
        # 输出线性层
        self.output_layer = nn.Linear(d_model, n_fft // 2 + 1)
        
    def forward(self, input):
        # 输入 x 的形状: (B, T, F)
        B, T, F = input.shape
        
        # 转换输入
        x = self.input_layer(input)  # (B, T, d_model)
        
        # 转换形状为 (T, B, d_model) 以适应 Transformer
        x = x.permute(1, 0, 2)
        
        # 经过多个 Transformer 层
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 还原形状回 (B, T, d_model)
        x = x.permute(1, 0, 2)
        
        # 输出层
        mask = torch.sigmoid(self.output_layer(x))  # (B, T, F)，生成掩码
        
        # 增强后的输出
        enhanced_output = mask * input  # 应用掩码
        
        return enhanced_output


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.relu = nn.ReLU()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x

        B,C,H,W = x.size()
        x_h = self.pool_h(x) # 压缩水平方向: (B, C, H, W) --> (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # 压缩垂直方向: (B, C, H, W) --> (B, C, 1, W) --> (B,C,W,1)

        # 坐标注意力生成
        y = torch.cat([x_h, x_w], dim=2) # 拼接水平和垂直方向的向量: (B,C,H+W,1)
        y = self.conv1(y) # 通过Conv进行变换,并降维: (B,C,H+W,1)--> (B,d,H+W,1)
        y = self.bn1(y)   # BatchNorm操作: (B,d,H+W,1)
        y = self.relu(y)  # Relu操作: (B,d,H+W,1)
        
        x_h, x_w = torch.split(y, [H, W], dim=2) # 沿着空间方向重新分割为两部分: (B,d,H+W,1)--> x_h:(B,d,H,1); x_w:(B,d,W,1)
        x_w = x_w.permute(0, 1, 3, 2) # x_w: (B,d,W,1)--> (B,d,1,W)

        a_h = self.conv_h(x_h).sigmoid() # 恢复与输入相同的通道数,并生成垂直方向的权重: (B,d,H,1)-->(B,C,H,1)
        a_w = self.conv_w(x_w).sigmoid() # 恢复与输入相同的通道数,并生成水平方向的权重: (B,d,1,W)-->(B,C,1,W)

        #out = identity * a_w * a_h # 将垂直、水平方向权重应用于输入,从而反映感兴趣的对象是否存在于相应的行和列中: (B,C,H,W) * (B,C,1,W) * (B,C,H,1) = (B,C,H,W)
        out = a_w * a_h * identity
        
        return out + identity


class CSA_Beamformer(nn.Module):
    def __init__(self,
                 channel = 8,
                 n_fft = 512,
                 kernel_size_1 = 9,
                 kernel_size_2 = 7,
                 kernel_size_3 = 5,
                 kernel_size_4 = 3,
                 reduction = 2
                 ):
        super().__init__()
        #print("Version 27")
        #reduction = 1
        print(f"CSA reduction: {reduction}")
        self.co1=CoordAtt(inp=channel, oup=channel, reduction=reduction)
        self.co2=CoordAtt(inp=channel, oup=channel, reduction=reduction)
        self.ca_1 = CBAMBlock(channel=channel,reduction=reduction,kernel_size=kernel_size_1)
        self.ca_2 = CBAMBlock(channel=channel,reduction=reduction,kernel_size=kernel_size_2)
        self.ca_3 = CBAMBlock(channel=channel,reduction=reduction,kernel_size=kernel_size_3)
        self.ca_4 = CBAMBlock(channel=channel,reduction=reduction,kernel_size=kernel_size_4)
        self.sa = SACC_TC(
                        fre_input_dim=n_fft//2 + 1,
                        fre_attention_dim=n_fft//2)
        
        self.post = AttentionMaskNet(n_fft=n_fft)
    
    def forward(self, input, ilens=None):
        B, T, C, F, _ = input.size()

        input = input.permute(0,2,1,3,4)
        
        # Calculate power spectrogram Xmag
        Xmag = torch.pow(input, 2).sum(dim=-1)  # assuming X is complex, calculate power spectrogram

        Xmag_1 = self.ca_1(Xmag)
        Xmag_2 = self.ca_2(Xmag_1)
        Xmag_2 = self.co1(Xmag_2)
        Xmag_3 = self.ca_3(Xmag_2)
        Xmag = self.ca_4(Xmag_3)
        Xmag = self.co2(Xmag)
        #Xmag = torch.cat([Xmag_1, Xmag_2, Xmag_3], dim=1)
        # B,C,T,F -> B,T,C,F
        Xmag = Xmag.permute(0,2,1,3)

        output, _, _ = self.sa(Xmag,None)
        output = self.post(output)
        
        return output, ilens, None
        
# 计算模型大小
def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    # 每个参数占用4字节
    model_size_mb = total_params * 4 / (1024 ** 2)
    return model_size_mb 

from scipy.signal import savgol_filter

def plot_softmax_channels(W, save_path=None, channels_to_plot=None, smooth_window=None):
    """
    绘制 W 的 softmax 结果曲线，并可选地对曲线进行平滑处理。
    
    参数:
        W (torch.Tensor): 输入张量，形状为 [1, 时间帧, 通道数, 1]。
        save_path (str): 图像保存路径，如果为 None，则不保存。
        channels_to_plot (list): 指定要绘制的通道索引列表，如果为 None，则绘制所有通道。
        smooth_window (int): 平滑窗口大小，如果为 None，则不进行平滑处理。
    """
    # # 检查输入张量的形状
    # if W.shape != torch.Size([1, 104, 25, 1]):
    #     raise ValueError("输入张量的形状必须为 [1, 104, 25, 1]")
    
    # 将张量移动到 CPU（如果它在 GPU 上）
    W = W.cpu()
    
    # 对通道维度进行 softmax
    W_softmax = F.softmax(W, dim=2)  # 形状仍为 [1, 104, 25, 1]
    
    # 去掉多余的维度
    W_softmax = W_softmax.squeeze()  # 形状变为 [104, 25]
    
    # 获取时间帧和通道数
    time_frames = W_softmax.shape[0]
    num_channels = W_softmax.shape[1]
    
    # 如果未指定通道，则绘制所有通道
    if channels_to_plot is None:
        channels_to_plot = range(num_channels)
    
    # 创建绘图
    plt.figure(figsize=(12, 6))
    
    # 绘制每个通道的曲线
    for channel in channels_to_plot:
        # 获取当前通道的数据
        data = W_softmax[:, channel].numpy()
        
        # 如果指定了平滑窗口，则对数据进行平滑处理
        if smooth_window is not None:
            data = savgol_filter(data, window_length=smooth_window, polyorder=2)
        
        # 绘制曲线
        plt.plot(range(time_frames), data, label=f'Channel {channel}')
    
    # 添加标题和标签
    plt.title('Softmax Output Over Time Frames (Smoothed)' if smooth_window else 'Softmax Output Over Time Frames')
    plt.xlabel('Time Frames')
    plt.ylabel('Softmax Value')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"图像已保存到 {save_path}")
    
    # 显示图像
    plt.show()

if __name__ == '__main__':
    # B, T, C, F, _
    input=torch.randn(50,54,8,257,2)
    kernel_size=3
    #cbam = CBAMBlock(channel=8,reduction=2,kernel_size=kernel_size)
    cbam = CSA_Beamformer(channel=4, n_fft = 512)
    output,_,_=cbam(input,None)
    print(output.shape)

    model_size = get_model_size(cbam)
    print(f"Model Size: {model_size:.2f} MB")

    