# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Xiangzhu Kong(kongxiangzhu99@gmail.com), Keyu An, Huahuan Zheng
__all__ = ["UnifiedAMTrainer", "build_model", "_parser", "main"]

from .train_raw_multi_SACC import AMTrainer, build_model as am_builder, main_worker as basic_worker
from ..shared import coreutils
from ..shared.simu_net import SimuNet

import os
import argparse
from typing import *

import torch
import torch.nn as nn
import random
import math
import numpy as np
from torch.cuda.amp import autocast

import matplotlib.pyplot as plt




def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace):
    basic_worker(gpu, ngpus_per_node, args, func_build_model=build_model)


class UnifiedAMTrainer(AMTrainer):
    def __init__(
        self,
        # chunk related parameters
        # configure according to the encoder
        downsampling_ratio: int = 4,
        chunk_size: int = 40,
        context_size_left: int = 40,
        context_size_right: int = 40,
        # jitter is applied after the downsampling
        jitter_range: int = 2,
        mel_dim: int = 80,
        simu: bool = False,
        simu_loss_weight: float = 1.0,
        
        # 数据调度概率
        #p_multi: float = 0.5,

        **kwargs,
    ):
        super().__init__(**kwargs)

        self.simu = simu
        if self.simu:
            self.simu_net = SimuNet(
                mel_dim=mel_dim, out_len=context_size_right, hdim=256, rnn_num_layers=3
            )
            self.simu_loss = nn.L1Loss()
        
        self.simu_loss_weight = simu_loss_weight
        self.chunk_size = chunk_size
        self.context_size_left = context_size_left
        self.context_size_right = context_size_right
        self.jitter_range = jitter_range
        self.downsampling_ratio = downsampling_ratio
        
        #self.p_multi = p_multi
        
    def chunk_beamforming(self,inputs, in_lens):
        
        inputs = inputs.permute(0,2,1)
        if self.SHT is not None:
            inputs = self.SHT(inputs,Random_Flag = self.training)
        else:
            # Optionally handle the case where SHT is None
            #print("SHT is None, skipping spherical harmonics processing")
            pass
        inputs = inputs.permute(0,2,1)
        
        inputs, flens = self.stft(inputs, in_lens)
        
        chunk_size = self.chunk_size
        
        max_input_length = int(
            chunk_size * (math.ceil(float(inputs.shape[1]) / chunk_size))
        )
        inputs = map(lambda x: pad_to_len(x, max_input_length, 0), inputs)
        inputs = list(inputs)
        inputs = torch.stack(inputs, dim=0)

        left_context_size = self.context_size_left

        N_chunks = inputs.size(1) // chunk_size
        
        BC = inputs.size(0) * N_chunks
        Channel = inputs.size(2)
        Fre = inputs.size(3)
        RI = inputs.size(4)
        inputs = inputs.view(BC, chunk_size, Channel,Fre,RI)

        left_context = torch.zeros(
            BC, left_context_size, Channel, Fre,RI,device=inputs.device
            )

        if left_context_size > chunk_size:
            N = left_context_size // chunk_size
            for idx in range(N):
                left_context[
                    N - idx :, idx * chunk_size : (idx + 1) * chunk_size, ...
                ] = inputs[: -N + idx, :, ...]
            for idx in range(N):
                left_context[idx::N_chunks, : (N - idx) * chunk_size, ...] = 0
        else:
            left_context[1:, :, ...] = inputs[:-1, -left_context_size:, ...]
            left_context[0::N_chunks, :, ...] = 0

        if self.context_size_right > 0:
            # right_context = torch.zeros(inputs.size()[0], self.right_context_size, inputs.size()[2]).to(inputs.get_device())
            right_context = torch.zeros(
                BC, self.context_size_right, Channel, Fre,RI,device=inputs.device
            )
            if self.context_size_right > chunk_size:
                right_context[:-1, :chunk_size, ...] = inputs[1:, :, ...]
                right_context[:-2, chunk_size:, ...] = inputs[
                    2:, : self.context_size_right - chunk_size, ...
                ]
                right_context[N_chunks - 1 :: N_chunks, :, ...] = 0
                right_context[N_chunks - 2 :: N_chunks, chunk_size:, ...] = 0
            else:
                right_context[:-1, :, ...] = inputs[1:, : self.context_size_right, ...]
                right_context[N_chunks - 1 :: N_chunks, :, ...] = 0
            inputs_with_context = torch.cat(
                (left_context, inputs, right_context), dim=1
            )
        else:
            inputs_with_context = torch.cat((left_context, inputs), dim=1)
        
        # front end
        flens_chunk = torch.full([inputs_with_context.size(0)], inputs_with_context.size(1))
            
        inputs_with_context, flens1,weights = self.beamformer(
            inputs_with_context,
            flens_chunk,
            )
        assert (flens_chunk == flens1).all()
        
        if self.use_SACC or self.use_WSJCA:
            # utt cat 
            samples = inputs_with_context[:, left_context_size:chunk_size + left_context_size, :]
            samples = samples.contiguous().view(samples.size(0)//N_chunks, 
                                                samples.size(1)*N_chunks, 
                                                samples.size(2)
                                                )
            # utt feature exaction
            input_power = samples
        else:
            # utt cat 
            samples = inputs_with_context[:, left_context_size:chunk_size + left_context_size, :]
            samples = samples.contiguous().view(samples.size(0)//N_chunks, 
                                                samples.size(1)*N_chunks, 
                                                samples.size(2), 
                                                samples.size(3))
            # utt feature exaction
            input_power = samples[..., 0] ** 2 + samples[..., 1] ** 2 
        
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))

        samples, _ = self.logmel(input_amp, flens)
        
        return samples, flens
    
    def bf_chunk_infer_bak(
        self, inputs: torch.FloatTensor, in_lens: torch.LongTensor
    ) -> torch.FloatTensor:
        inputs = inputs.permute(0,2,1)
        if self.SHT is not None:
            inputs = self.SHT(inputs,Random_Flag = self.training)
        else:
            # Optionally handle the case where SHT is None
            #print("SHT is None, skipping spherical harmonics processing")
            pass
        inputs = inputs.permute(0,2,1)
        
        # time domain to fre domain
        inputs, flens = self.stft(inputs, in_lens)
        
        chunk_size = self.chunk_size
        
        max_input_length = int(
            chunk_size * (math.ceil(float(inputs.shape[1]) / chunk_size))
        )
        inputs = map(lambda x: pad_to_len(x, max_input_length, 0), inputs)
        inputs = list(inputs)
        inputs = torch.stack(inputs, dim=0)

        left_context_size = self.context_size_left
        if self.simu:
            simu_right_context = self.simu_net(inputs.clone(), chunk_size)

        N_chunks = inputs.size(1) // chunk_size
        
        BC = inputs.size(0) * N_chunks
        Channel = inputs.size(2)
        Fre = inputs.size(3)
        RI = inputs.size(4)
        inputs = inputs.view(BC, chunk_size, Channel,Fre,RI)

        left_context = torch.zeros(
            BC, left_context_size, Channel, Fre,RI,device=inputs.device
            )

        if left_context_size > chunk_size:
            N = left_context_size // chunk_size
            for idx in range(N):
                left_context[
                    N - idx :, idx * chunk_size : (idx + 1) * chunk_size, ...
                ] = inputs[: -N + idx, :, ...]
            for idx in range(N):
                left_context[idx::N_chunks, : (N - idx) * chunk_size, ...] = 0
        else:
            left_context[1:, :, ...] = inputs[:-1, -left_context_size:, ...]
            left_context[0::N_chunks, :, ...] = 0

        if self.context_size_right > 0:
            if self.simu:
                right_context = simu_right_context
                inputs_with_context = torch.cat(
                    (left_context, inputs, right_context), dim=1
                )
            else:
                # right_context = torch.zeros(inputs.size()[0], self.context_size_right, inputs.size()[2]).to(inputs.get_device())
                
                # right_context = torch.zeros(
                #     BC, self.context_size_right, Channel, Fre,RI,device=inputs.device
                # )
                # if self.context_size_right > chunk_size:
                #     right_context[:-1, :chunk_size, ...] = inputs[1:, :, ...]
                #     right_context[:-2, chunk_size:, ...] = inputs[
                #         2:, : self.context_size_right - chunk_size, ...
                #     ]
                #     right_context[N_chunks - 1 :: N_chunks, :, ...] = 0
                #     right_context[N_chunks - 2 :: N_chunks, chunk_size:, ...] = 0
                # else:
                #     right_context[:-1, :, ...] = inputs[1:, : self.context_size_right, ...]
                #     right_context[N_chunks - 1 :: N_chunks, :, ...] = 0
                # inputs_with_context = torch.cat((left_context, inputs, right_context), dim=1)
                
                # # 不使用right context
                inputs_with_context = torch.cat((left_context, inputs), dim=1)
        else:
            inputs_with_context = torch.cat((left_context, inputs), dim=1)
            
        
        # front end
        flens_chunk = torch.full([inputs_with_context.size(0)], inputs_with_context.size(1))
            
        inputs_with_context, flens1,_ = self.beamformer(
            inputs_with_context,
            flens_chunk,
            )
        assert (flens_chunk == flens1).all()
        
        # feature exaction
        if self.use_SACC or self.use_WSJCA:
            input_power = inputs_with_context
        else:
            input_power = inputs_with_context[..., 0] ** 2 + inputs_with_context[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        inputs_with_context, _ = self.logmel(input_amp, flens_chunk)    
            
        # ASR    
        enc_out_with_context, _ = self.encoder(
            inputs_with_context,
            flens_chunk,
        )
        enc_out = enc_out_with_context[
            :,
            left_context_size
            // self.downsampling_ratio : (chunk_size + left_context_size)
            // self.downsampling_ratio,
            :,
        ]
        enc_out = enc_out.contiguous().view(
            enc_out.size(0) // N_chunks, enc_out.size(1) * N_chunks, -1
        )

        out_lens = torch.div(
            chunk_size * torch.ceil(flens / chunk_size),
            self.downsampling_ratio,
            rounding_mode="floor",
        )
        return enc_out, out_lens
    
    def bf_chunk_infer(
        self, inputs: torch.FloatTensor, in_lens: torch.LongTensor
    ) -> torch.FloatTensor:
        inputs = inputs.permute(0,2,1)
        if self.SHT is not None:
            inputs = self.SHT(inputs, Random_Flag = self.training)
        else:
            # Optionally handle the case where SHT is None
            #print("SHT is None, skipping spherical harmonics processing")
            #pass
            _, C, _ = audio.shape
            if not C == 8:
                 audio = audio[:,:8, :]
        
        if self.use_kaldi:
            inputs, flens = self.trans.cal_stft(inputs, in_lens)
            # (B, C, T, F, 2) --> (B, T, C, F, 2)
            inputs = inputs.permute(0,2,1,3,4)

        else:
            inputs = inputs.permute(0,2,1)
            # time domain to fre domain
            inputs, flens = self.stft(inputs, in_lens)
        
        
        chunk_size = self.chunk_size
        
        max_input_length = int(
            chunk_size * (math.ceil(float(inputs.shape[1]) / chunk_size))
        )
        inputs = map(lambda x: pad_to_len(x, max_input_length, 0), inputs)
        inputs = list(inputs)
        inputs = torch.stack(inputs, dim=0)

        left_context_size = self.context_size_left

        num_chunks = inputs.size(1) // chunk_size
        
        assert num_chunks > 0
        
        BC = inputs.size(0) * num_chunks
        Channel = inputs.size(2)
        Fre = inputs.size(3)
        RI = inputs.size(4)
        inputs = inputs.view(BC, chunk_size, Channel,Fre,RI)

        left_context = torch.zeros(
            BC, left_context_size, Channel, Fre,RI,device=inputs.device
            )

        # fill first left chunk with zeros
        if left_context_size > chunk_size:
            N = left_context_size // chunk_size
            for idx in range(N):
                left_context[
                    N - idx :, idx * chunk_size : (idx + 1) * chunk_size, ...
                ] = inputs[: -N + idx, :, ...]
            for idx in range(N):
                left_context[idx::num_chunks, : (N - idx) * chunk_size, ...] = 0
        else:
            left_context[1:, :, ...] = inputs[:-1, -left_context_size:, ...]
            left_context[0::num_chunks, :, ...] = 0

        # if self.context_size_right > 0:
        #     right_context = torch.zeros(
        #         BC, self.context_size_right, Channel, Fre,RI, device=inputs.device
        #     )
        #     if self.context_size_right > chunk_size:
        #         right_context[:-1, :chunk_size, ...] = inputs[1:, :, ...]
        #         right_context[:-2, chunk_size:, ...] = inputs[
        #             2:, : self.context_size_right - chunk_size, ...
        #         ]
        #         right_context[num_chunks - 1 :: num_chunks, :, ...] = 0
        #         right_context[num_chunks - 2 :: num_chunks, chunk_size:, ...] = 0
        #     else:
        #         right_context[:-1, :, ...] = inputs[1:, : self.context_size_right, ...]
        #         right_context[num_chunks - 1 :: num_chunks, :, ...] = 0

        #     if self.simu:
        #         if self.training :
        #             if np.random.rand() < 0.5:
        #                 contexted_inputs = (left_context, inputs)
        #             else:
        #                 contexted_inputs = (left_context, inputs, right_context)
        #         else:
        #             contexted_inputs = (left_context, inputs)
        #     else:
        #         #simu_loss = 0
        #         if self.training and np.random.rand() < 0.5:
        #             contexted_inputs = (left_context, inputs, right_context)
        #         else:
        #             #contexted_inputs = (left_context, inputs, right_context)
        #             contexted_inputs = (left_context, inputs)
            
        #     inputs_with_context = torch.cat(contexted_inputs, dim=1)
        
        if self.context_size_right > 0:
            right_context = torch.zeros(
                BC, self.context_size_right, Channel, Fre, RI, device=inputs.device
            )
            if self.context_size_right > chunk_size:
                # 预计算需要复制的分块数
                num_right_chunks = (self.context_size_right + chunk_size - 1) // chunk_size
                
                # 构建全局偏移索引
                idx = torch.arange(BC, device=inputs.device).view(-1, 1)  # (BC, 1)
                shift = torch.arange(num_right_chunks, device=inputs.device) + 1  # (num_right_chunks,)
                global_idx = idx + shift  # (BC, num_right_chunks)
                global_idx = global_idx.clamp(max=inputs.size(0)-1)  # 避免越界
                
                # 提取所有需要的分块数据 (BC, num_right_chunks, chunk_size, ...)
                chunks = inputs[global_idx, :, ...]  # 利用广播机制
                
                # 将分块拼接成连续上下文 (BC, num_right_chunks*chunk_size, ...)
                chunks_flat = chunks.view(BC, -1, Channel, Fre, RI)
                
                # 裁剪到实际需要的长度
                right_context_data = chunks_flat[:, :self.context_size_right, ...]
                
                # 将数据写入右上下文
                right_context[:, :self.context_size_right, ...] = right_context_data
                
                # 处理边界分块 (最后几个分块需要置零)
                mask = (global_idx >= inputs.size(0)).any(dim=1)  # (BC,)
                right_context[mask] = 0
            else:
                # 单分块处理
                right_context[:-1, :, ...] = inputs[1:, :self.context_size_right, ...]
                right_context[num_chunks - 1 :: num_chunks, :, ...] = 0
            
            if self.simu:
                if self.training :
                    if np.random.rand() < 0.5:
                        contexted_inputs = (left_context, inputs)
                    else:
                        contexted_inputs = (left_context, inputs, right_context)
                else:
                    contexted_inputs = (left_context, inputs)
            else:
                simu_loss = 0
                if self.training and np.random.rand() < 0.5:
                    contexted_inputs = (left_context, inputs, right_context)
                else:
                    contexted_inputs = (left_context, inputs)
            
            inputs_with_context = torch.cat(contexted_inputs, dim=1)
        
        
        
        else:
            inputs_with_context = torch.cat((left_context, inputs), dim=1)
        
        flens1 = torch.full([inputs_with_context.size(0)], inputs_with_context.size(1))
        flens_chunk = torch.full([inputs_with_context.size(0)], inputs_with_context.size(1))
        
        inputs_with_context, flens1,_ = self.beamformer(
            inputs_with_context,
            flens_chunk,
            )
        assert (flens_chunk == flens1).all()
        
        # feature exaction
        if self.use_SACC or self.use_WSJCA:
            input_power = inputs_with_context
        else:
            input_power = inputs_with_context[..., 0] ** 2 + inputs_with_context[..., 1] ** 2
        
        if self.use_kaldi:
            stream_chunk, _ = self.trans.spectrum_to_fbank(input_power,flens_chunk)
        else:
            input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
            stream_chunk, _ = self.logmel(input_amp, flens_chunk)    
        
        if self.simu:
            # FIXME: maybe .clone() is not required
            left_context = stream_chunk[:, :left_context_size, :]
            inputs_without_context = stream_chunk[:, left_context_size:chunk_size + left_context_size, :]
            samples = inputs_without_context.contiguous().view(
                                            inputs_without_context.size(0)//num_chunks, 
                                            inputs_without_context.size(1)*num_chunks, 
                                            inputs_without_context.size(2))
            simu_right_context = self.simu_net(samples, chunk_size)

            mel_dim = stream_chunk.size(2)
            
            if self.context_size_right > 0:
                right_context = torch.zeros(
                    BC, self.context_size_right, mel_dim, device=inputs_without_context.device
                )
                if self.context_size_right > chunk_size:
                    right_context[:-1, :chunk_size, :] = inputs_without_context[1:, :, :]
                    right_context[:-2, chunk_size:, :] = inputs_without_context[
                        2:, : self.context_size_right - chunk_size, :
                    ]
                    right_context[num_chunks - 1 :: num_chunks, :, :] = 0
                    right_context[num_chunks - 2 :: num_chunks, chunk_size:, :] = 0
                else:
                    right_context[:-1, :, :] = inputs_without_context[1:, : self.context_size_right, :]
                    right_context[num_chunks - 1 :: num_chunks, :, :] = 0

                
                #simu_loss = self.simu_loss(simu_right_context, right_context.detach())
                if self.training:
                    if np.random.rand() < 0.5:
                        contexted_inputs = (left_context, inputs_without_context, simu_right_context)
                    elif np.random.rand() < 0.5:
                        contexted_inputs = (left_context, inputs_without_context)
                    else:
                        contexted_inputs = (left_context, inputs_without_context, right_context)
                else:
                    #contexted_inputs = (left_context, inputs_without_context, simu_right_context)
                    #contexted_inputs = (left_context, inputs_without_context, right_context)
                    contexted_inputs = (left_context, inputs_without_context)
                
                stream_chunk = torch.cat(contexted_inputs, dim=1)
            else:
                stream_chunk = torch.cat((left_context, inputs_without_context), dim=1)    
            
        # ASR    
        enc_out_with_context, _ = self.encoder(
            stream_chunk,
            flens1,
        )
        enc_out = enc_out_with_context[
            :,
            left_context_size
            // self.downsampling_ratio : (chunk_size + left_context_size)
            // self.downsampling_ratio,
            :,
        ]
        enc_out = enc_out.contiguous().view(
            enc_out.size(0) // num_chunks, enc_out.size(1) * num_chunks, -1
        )

        out_lens = torch.div(
            chunk_size * torch.ceil(flens / chunk_size),
            self.downsampling_ratio,
            rounding_mode="floor",
        )
        return enc_out, out_lens
  
    def get_model_flops(self, model=None, 
                        time=10, 
                        time_steps=120, 
                        channels=None, 
                        freq_bins=257,
                        streaming=False,
                        verbose=False
                        ):
        # 假设 10s 的语音片段， 1000帧，对应chunk大小为80 + 40，一共有9个 chunk
        
        # 输入形状 (batch_size, time_steps, channels, freq_bins, _)
        # batch_size, time_steps, channels, freq_bins, _ = 1, 100, 25, 257, 2
        if self.SHT is not None and channels is None:
            channels = (self.sph_order + 1) ** 2
        elif channels is None:
            channels = 8
        
        if streaming:
            # 分 chunk 的逻辑
            max_input_length = int(
                time_steps * (math.ceil(float(time * 100) / time_steps))
            )
            batch_size = max_input_length // time_steps
        else:
            # 不分 chunk，直接使用整个输入
            max_input_length = time * 100  # 假设每秒100帧
            batch_size = 1  # 不分 chunk，batch_size 为 1

        input_shape = (batch_size, time_steps, channels, freq_bins, 2)  
        
        if model is None:
            model = self.beamformer

        # from thop import profile
        # # 转换为张量并指定设备
        #dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
        #flens = torch.full([dummy_input.size(0)], dummy_input.size(1)).to(next(model.parameters()).device)
        # flops, params = profile(model, inputs=(dummy_input, flens), verbose=verbose)
        # print(f"Model FLOPs: {flops / 1e9:.2f} GFLOPs")
        # print(f"Model Parameters: {params / 1e6:.2f} M")
        
        # return flops, params
        # dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
    
        # 使用 calflops 计算 FLOPs 和 参数量，并打印详细信息
        from .calflops import calculate_flops
        print("Detailed FLOPs per layer:")
        calculate_flops(model, input_shape=input_shape, print_results=True, print_detailed=True)

        return None  # 或者返回其他需要的信息
        

    def forward(self, audio, lx, labels, ly):
        
        # if audio.size(0) * audio.size(1) > 200000:
        #     #print(torch.cuda.memory_allocated() / (1024 * 1024))
        #     print(lx)
        #     torch.cuda.empty_cache()
        audio = audio.permute(0,2,1)
        if self.SHT is not None:
            audio = self.SHT(audio,Random_Flag = self.training)
        else:
            # Optionally handle the case where SHT is None
            #print("SHT is None, skipping spherical harmonics processing")
            pass
        audio = audio.permute(0,2,1)
        
        
        # Chunk divide
        
        jitter = self.downsampling_ratio * random.randint(
            -self.jitter_range, self.jitter_range
        )
        chunk_size = self.chunk_size + jitter
        
        if self.use_kaldi:
            audio = audio.permute(0,2,1)
            inputs, flens = self.trans.cal_stft(audio, lx)
            # (B, C, T, F, 2) --> (B, T, C, F, 2)
            inputs = inputs.permute(0,2,1,3,4)
        else:
            inputs, flens = self.stft(audio, lx)
        
        
        max_input_length = int(
            chunk_size * (math.ceil(float(inputs.shape[1]) / chunk_size))
        )
        
        inputs = map(lambda x: pad_to_len(x, max_input_length, 0), inputs)
        inputs = list(inputs)
        inputs = torch.stack(inputs, dim=0)

        num_chunks = inputs.size(1) // chunk_size
        
        assert num_chunks > 0
        
        BC = inputs.size(0) * num_chunks
        Channel = inputs.size(2)
        Fre = inputs.size(3)
        RI = inputs.size(4)
        inputs = inputs.view(BC, chunk_size, Channel,Fre,RI)
        

        # setup left context
        left_context_size = self.context_size_left + jitter * (
            self.context_size_left // self.chunk_size
        )
        left_context = torch.zeros(BC, left_context_size, Channel, Fre,RI,device=inputs.device)
        # fill first left chunk with zeros
        if left_context_size > chunk_size:
            N = left_context_size // chunk_size
            for idx in range(N):
                left_context[
                    N - idx :, idx * chunk_size : (idx + 1) * chunk_size, ...
                ] = inputs[: -N + idx, :, ...]
            for idx in range(N):
                left_context[idx::num_chunks, : (N - idx) * chunk_size, ...] = 0
        else:
            left_context[1:, :, ...] = inputs[:-1, -left_context_size:, ...]
            left_context[0::num_chunks, :, ...] = 0

        # if self.context_size_right > 0:
        #     right_context = torch.zeros(
        #         BC, self.context_size_right, Channel, Fre,RI, device=inputs.device
        #     )
        #     if self.context_size_right > chunk_size:
        #         right_context[:-1, :chunk_size, ...] = inputs[1:, :, ...]
        #         right_context[:-2, chunk_size:, ...] = inputs[
        #             2:, : self.context_size_right - chunk_size, ...
        #         ]
        #         right_context[num_chunks - 1 :: num_chunks, :, ...] = 0
        #         right_context[num_chunks - 2 :: num_chunks, chunk_size:, ...] = 0
        #     else:
        #         right_context[:-1, :, ...] = inputs[1:, : self.context_size_right, ...]
        #         right_context[num_chunks - 1 :: num_chunks, :, ...] = 0

        #     if self.simu:
        #         if self.training :
        #             if np.random.rand() < 0.5:
        #                 contexted_inputs = (left_context, inputs)
        #             else:
        #                 contexted_inputs = (left_context, inputs, right_context)
        #         else:
        #             contexted_inputs = (left_context, inputs)
        #     else:
        #         simu_loss = 0
        #         if self.training and np.random.rand() < 0.5:
        #             contexted_inputs = (left_context, inputs)
        #         else:
        #             contexted_inputs = (left_context, inputs, right_context)
            
        #     inputs_with_context = torch.cat(contexted_inputs, dim=1)
        
        if self.context_size_right > 0:
            right_context = torch.zeros(
                BC, self.context_size_right, Channel, Fre, RI, device=inputs.device
            )
            if self.context_size_right > chunk_size:
                # 预计算需要复制的分块数
                num_right_chunks = (self.context_size_right + chunk_size - 1) // chunk_size
                
                # 构建全局偏移索引
                idx = torch.arange(BC, device=inputs.device).view(-1, 1)  # (BC, 1)
                shift = torch.arange(num_right_chunks, device=inputs.device) + 1  # (num_right_chunks,)
                global_idx = idx + shift  # (BC, num_right_chunks)
                global_idx = global_idx.clamp(max=inputs.size(0)-1)  # 避免越界
                
                # 提取所有需要的分块数据 (BC, num_right_chunks, chunk_size, ...)
                chunks = inputs[global_idx, :, ...]  # 利用广播机制
                
                # 将分块拼接成连续上下文 (BC, num_right_chunks*chunk_size, ...)
                chunks_flat = chunks.view(BC, -1, Channel, Fre, RI)
                
                # 裁剪到实际需要的长度
                right_context_data = chunks_flat[:, :self.context_size_right, ...]
                
                # 将数据写入右上下文
                right_context[:, :self.context_size_right, ...] = right_context_data
                
                # 处理边界分块 (最后几个分块需要置零)
                mask = (global_idx >= inputs.size(0)).any(dim=1)  # (BC,)
                right_context[mask] = 0
            else:
                # 单分块处理
                right_context[:-1, :, ...] = inputs[1:, :self.context_size_right, ...]
                right_context[num_chunks - 1 :: num_chunks, :, ...] = 0
            
            if self.simu:
                if self.training :
                    if np.random.rand() < 0.5:
                        contexted_inputs = (left_context, inputs)
                    else:
                        contexted_inputs = (left_context, inputs, right_context)
                else:
                    contexted_inputs = (left_context, inputs)
            else:
                simu_loss = 0
                if self.training and np.random.rand() < 0.5:
                    contexted_inputs = (left_context, inputs)
                else:
                    contexted_inputs = (left_context, inputs, right_context)
            
            inputs_with_context = torch.cat(contexted_inputs, dim=1)
            
        else:
            inputs_with_context = torch.cat((left_context, inputs), dim=1)
        
        flens1 = torch.full([inputs_with_context.size(0)], inputs_with_context.size(1))
        
        # Stream Front end 
        inputs_with_context, flens1,_ = self.beamformer(
                inputs_with_context,
                flens1,
                )

        #self.get_model_flops(self.beamformer)
        
        if torch.isnan(inputs_with_context).any():
            print("feats array contains NaN values!")
        
        if self.use_SACC or self.use_WSJCA:
            # utt cat 
            samples = inputs_with_context[:, left_context_size:chunk_size + left_context_size, :]
            samples = samples.contiguous().view(samples.size(0)//num_chunks, 
                                                samples.size(1)*num_chunks, 
                                                samples.size(2)
                                                )
                
            
            # utt feature exaction
            input_power = samples
        else:
            # utt cat 
            samples = inputs_with_context[:, left_context_size:chunk_size + left_context_size, :]
            samples = samples.contiguous().view(samples.size(0)//num_chunks, 
                                                samples.size(1)*num_chunks, 
                                                samples.size(2), 
                                                samples.size(3))
                
            
            # utt feature exaction
            input_power = samples[..., 0] ** 2 + samples[..., 1] ** 2 
        
        if self.use_kaldi:
            samples, _ = self.trans.spectrum_to_fbank(input_power,flens1)
        else:
            input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))

            samples, _ = self.logmel(input_amp, flens)
        
        # utt ASR
        logits, lx = self.encoder(samples, flens)
        logits = torch.log_softmax(logits, dim=-1)
        
        # utt loss
        labels = labels.cpu()
        lx = lx.cpu()
        ly = ly.cpu()
        if self.is_crf:
            if self._crf_ctx is None:
                # lazy init
                self.register_crf_ctx(self.den_lm)
            with autocast(enabled=False):
                loss = self.criterion(
                    logits.float(),
                    labels.to(torch.int),
                    lx.to(torch.int),
                    ly.to(torch.int),
                )
        else:
            # [N, T, C] -> [T, N, C]
            logits = logits.transpose(0, 1)
            loss = self.criterion(
                logits, labels.to(torch.int), lx.to(torch.int), ly.to(torch.int)
            )

        if self.use_SACC or self.use_WSJCA:
            # chunk feature exaction
            input_power = inputs_with_context
            if self.use_kaldi:
                stream_chunk, _ = self.trans.spectrum_to_fbank(input_power,flens1)
            else:
                input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
                stream_chunk, _ = self.logmel(input_amp, flens1)
        else:
            # chunk feature exaction
            input_power = inputs_with_context[..., 0] ** 2 + inputs_with_context[..., 1] ** 2
            input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
            stream_chunk, _ = self.logmel(input_amp, flens1)
        
        # Chunk simu or not
        if self.simu:
            # FIXME: maybe .clone() is not required
            left_context = stream_chunk[:, :left_context_size, :]
            inputs_without_context = stream_chunk[:, left_context_size:chunk_size + left_context_size, :]
            simu_right_context = self.simu_net(samples, chunk_size)

            mel_dim = stream_chunk.size(2)
            
            if self.context_size_right > 0:
                right_context = torch.zeros(
                    BC, self.context_size_right, mel_dim, device=inputs_without_context.device
                )
                if self.context_size_right > chunk_size:
                    right_context[:-1, :chunk_size, :] = inputs_without_context[1:, :, :]
                    right_context[:-2, chunk_size:, :] = inputs_without_context[
                        2:, : self.context_size_right - chunk_size, :
                    ]
                    right_context[num_chunks - 1 :: num_chunks, :, :] = 0
                    right_context[num_chunks - 2 :: num_chunks, chunk_size:, :] = 0
                else:
                    right_context[:-1, :, :] = inputs_without_context[1:, : self.context_size_right, :]
                    right_context[num_chunks - 1 :: num_chunks, :, :] = 0

                
                simu_loss = self.simu_loss(simu_right_context, right_context.detach())
                if self.training:
                    if np.random.rand() < 0.5:
                        contexted_inputs = (left_context, inputs_without_context, simu_right_context)
                    elif np.random.rand() < 0.5:
                        contexted_inputs = (left_context, inputs_without_context)
                    else:
                        contexted_inputs = (left_context, inputs_without_context, right_context)
                else:
                    contexted_inputs = (left_context, inputs_without_context, simu_right_context)
                
                stream_chunk = torch.cat(contexted_inputs, dim=1)
            else:
                stream_chunk = torch.cat((left_context, inputs_without_context), dim=1)    
                    
        # Chunk ASR
        enc_out_with_context, _ = self.encoder(
            stream_chunk,
            flens1,
        )
        enc_out = enc_out_with_context[
            :,
            left_context_size
            // self.downsampling_ratio : (chunk_size + left_context_size)
            // self.downsampling_ratio,
            :,
        ]
        enc_out = enc_out.contiguous().view(
            enc_out.size(0) // num_chunks, enc_out.size(1) * num_chunks, -1
        )
        
        
        enc_out = enc_out[:, : lx[0].int(), :]
        chunk_logits = torch.log_softmax(enc_out, dim=-1)

        # Chunk loss
        if self.is_crf:
            with autocast(enabled=False):
                chunk_loss = self.criterion(
                    chunk_logits.float(),
                    labels.to(torch.int),
                    lx.to(torch.int),
                    ly.to(torch.int),
                )
        else:
            # [N, T, C] -> [T, N, C]
            chunk_logits = chunk_logits.transpose(0, 1)
            chunk_loss = self.criterion(
                chunk_logits, labels.to(torch.int), lx.to(torch.int), ly.to(torch.int)
            )

        
        if self.simu:
            if float('inf') == loss + chunk_loss + simu_loss :
                print("loss + chunk_loss + simu_loss  is inf!")
            
            if math.isnan(loss + chunk_loss + simu_loss ):
                print("loss + chunk_loss + simu_loss is NaN!")    
            return loss + chunk_loss + (simu_loss * self.simu_loss_weight), loss, chunk_loss, simu_loss
        
        else:
            simu_loss = 0
            return loss + chunk_loss + (simu_loss * self.simu_loss_weight),loss,chunk_loss


def pad_to_len(t: torch.Tensor, pad_len: int, dim: int):
    """Pad the tensor `t` at `dim` to the length `pad_len` with right padding zeros."""
    if t.size(dim) == pad_len:
        return t
    else:
        pad_size = list(t.shape)
        pad_size[dim] = pad_len - t.size(dim)
        return torch.cat(
            [t, torch.zeros(*pad_size, dtype=t.dtype, device=t.device)], dim=dim
        )


def build_model(
    cfg: dict, args: Optional[argparse.Namespace] = None, dist: bool = True
):
    """
    cfg: refer to UnifiedAMTrainer.__init__()
    """
    assert "trainer" in cfg, f"missing 'trainer' in field:"
    cfg["trainer"]["encoder"] = am_builder(cfg, args, dist=False, wrapper=False)
    model = UnifiedAMTrainer(**cfg["trainer"])
    if not dist:
        return model

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[args.gpu],
        find_unused_parameters=False
        )

    return model


def _parser():
    return coreutils.basic_trainer_parser("Unified streaming/offline CTC/CRF Trainer")


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    coreutils.setup_path(args)
    coreutils.main_spawner(args, main_worker)

def plot_and_save_spectrogram(tensor, ori, logs, save_path):
    """
    Plots the spectrogram of a tensor and saves it to the specified path.

    Args:
        tensor (torch.Tensor): Input tensor of size (T, F), where T is the time axis and F is the frequency axis.
        save_path (str): The file path where the spectrogram image will be saved.

    Raises:
        ValueError: If the input tensor is not 2-dimensional.
    """
    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2-dimensional (T, F).")

    # Move tensor to CPU and normalize it
    tensor = tensor.cpu()
    if not ori:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    # Convert tensor to numpy array for plotting
    spectrogram = tensor.numpy()

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram.T, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Spectrogram')

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
