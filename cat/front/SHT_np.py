# Copyright 2020 Tsinghua SPMI Lab 
# Apache 2.0.
# Author: Xiangzhu Kong(kongxiangzhu99@gmail.com)

import torch
import spaudiopy as spa
import numpy as np
from spaudiopy.sph import sh_matrix

class SphericalHarmonicsProcessor:
    def __init__(self, 
                 mic_type="aishell4", 
                 sph_order=3, 
                 selected_channels=None, 
                 rand_selected_channels=True,
                 requires_grad=False,  # 新增参数
                 need_stft=False
                 ):
        print("mic_type:", mic_type)
        print("sph_order:", sph_order)
        print("SHT requires_grad:", requires_grad)
        if selected_channels is not None:
            print("selected_channels:", selected_channels)
        if rand_selected_channels:
            print("rand_selected_channels:", rand_selected_channels)
        self.need_stft = need_stft
        
        
        self.requires_grad = requires_grad  # 保存参数
        self.sph_order = sph_order
        self.rand_selected_channels = rand_selected_channels
        self.selected_channels = selected_channels
            
        # 选择麦克风阵列位置
        if mic_type == "aishell4":
            self.mic_positions_spherical = np.array([
                [0.05, 0., 1.57079633],
                [0.05, -0.78539819, 1.57079633],
                [0.05, -1.57079637, 1.57079633],
                [0.05, -2.3561945, 1.57079633],
                [0.05, 3.14159274, 1.57079633],
                [0.05, 2.3561945, 1.57079633],
                [0.05, 1.57079637, 1.57079633],
                [0.05, 0.78539819, 1.57079633],
            ])
        elif mic_type == "alimeeting":
            self.mic_positions_spherical = np.array([
                [0.05, 0., 1.57079633],
                [0.05, 0.78539819, 1.57079633],
                [0.05, 1.57079637, 1.57079633],
                [0.05, 2.3561945, 1.57079633],
                [0.05, 3.14159274, 1.57079633],
                [0.05, -2.3561945, 1.57079633],
                [0.05, -1.57079637, 1.57079633],
                [0.05, -0.78539819, 1.57079633],
            ])
        elif mic_type == "circular":
            self.mic_positions_spherical = np.array([
                [ 0.05,  0.        ,  1.57079633],
                [ 0.05,  0.78539816,  1.57079633],
                [ 0.05,  1.57079633,  1.57079633],
                [ 0.05,  2.35619449,  1.57079633],
                [ 0.05,  3.14159265,  1.57079633],
                [ 0.05, -2.35619449,  1.57079633],
                [ 0.05, -1.57079633,  1.57079633],
                [ 0.05, -0.78539816,  1.57079633]
            ])
        elif mic_type == "linear":
            self.mic_positions_spherical = np.array([
                [0.0385    , 3.14159265, 1.57079633],
                [0.0275    , 3.14159265, 1.57079633],
                [0.0165    , 3.14159265, 1.57079633],
                [0.0055    , 3.14159265, 1.57079633],
                [0.0055    , 0.        , 1.57079633],
                [0.0165    , 0.        , 1.57079633],
                [0.0275    , 0.        , 1.57079633],
                [0.0385    , 0.        , 1.57079633]
            ])
        elif mic_type == "square":
            self.mic_positions_spherical = np.array([
                [ 0.01739253, -2.8198421 ,  1.57079633],
                [ 0.00777817, -2.35619449,  1.57079633],
                [ 0.00777817, -0.78539816,  1.57079633],
                [ 0.01739253, -0.32175055,  1.57079633],
                [ 0.01739253,  2.8198421 ,  1.57079633],
                [ 0.00777817,  2.35619449,  1.57079633],
                [ 0.00777817,  0.78539816,  1.57079633],
                [ 0.01739253,  0.32175055,  1.57079633]
            ])
        elif mic_type == "706":
            self.mic_positions_spherical = np.array([
                [ 0.03794733,  1.89254688,  1.57079633], #2
                [ 0.05091169,  2.35619449,  1.57079633], #3
                [ 0.03794733,  0.32175055,  1.57079633], #4
                [ 0.03794733,  2.8198421 ,  1.57079633], #7
                [ 0.03794733, -0.32175055,  1.57079633], #8
                [ 0.01697056, -2.35619449,  1.57079633], #10
                [ 0.03794733, -2.8198421 ,  1.57079633], #11
                [ 0.05091169, -0.78539816,  1.57079633], #12
                [ 0.03794733, -1.24904577,  1.57079633], #13
                [ 0.05091169, -2.35619449,  1.57079633]  #15
            ])
        else:
            raise TypeError("Unsupported mic_type")
        
    def forward(self, sig, sh_type='real', Random_Flag = False):
        if self.requires_grad:
            return self._forward_with_grad(sig, sh_type, Random_Flag)
        else:
            with torch.no_grad():  # Disable gradient calculation
                return self._forward_without_grad(sig, sh_type, Random_Flag)

    def _forward_with_grad(self, sig, sh_type='real', Random_Flag = False):
        # Check for input dimensions
        if sig.ndim == 3:
            # Randomly select channels, ensuring at least two channels, but keep original order
            num_channels = sig.shape[1]
            if self.rand_selected_channels and Random_Flag:
                num_selected = max(2, np.random.randint(2, num_channels+1))  # Randomly select at least 2 channels
                selected_channels = np.random.choice(num_channels, num_selected, replace=False)

                # Keep the selected channels in the original order
                sig = sig[:, sorted(selected_channels), :]

                # Spherical Harmonics Transformation
                azi = self.mic_positions_spherical[selected_channels, 1]
                colat = self.mic_positions_spherical[selected_channels, 2]
            elif self.selected_channels is not None:
                # Keep the selected channels in the original order
                sig = sig[:, self.selected_channels, :]

                # Spherical Harmonics Transformation
                azi = self.mic_positions_spherical[self.selected_channels, 1]
                colat = self.mic_positions_spherical[self.selected_channels, 2]
            else:
                # Spherical Harmonics Transformation
                azi = self.mic_positions_spherical[:, 1]
                colat = self.mic_positions_spherical[:, 2]
            
            Y_nm = torch.tensor(sh_matrix(self.sph_order, azi, colat, sh_type=sh_type)).to(sig.device)
            Y_nm_conj_T = Y_nm.conj().T
            
            # Apply transformation
            Y_nm_conj_T_batch = Y_nm_conj_T.unsqueeze(0).repeat(sig.shape[0], 1, 1).float()
            transformed_sig = torch.bmm(Y_nm_conj_T_batch.to(sig.device), sig)
            
            if self.need_stft:
                return torch.cat([transformed_sig, sig], dim=1)
            else:
                return transformed_sig
            
        else:
            raise TypeError("Input signal must have 3 dimensions (batch, channels, time)")

    def _forward_without_grad(self, sig, sh_type='real', Random_Flag = False):
        # The implementation is identical to _forward_with_grad but without gradients
        return self._forward_with_grad(sig, sh_type, Random_Flag)

    def inverse(self, p_mn, target_direction):
        """
        Inverse SHT to reconstruct the signal in a specific direction.
        target_direction: tuple (azimuth, colatitude) in radians, e.g. (phi0, theta0)
        Returns:
          - Reconstructed signal in the specified direction
        """
        phi0, theta0 = target_direction
        Y_dir = sh_matrix(self.sph_order,
                           azi=np.array([phi0]),
                           zen=np.array([theta0]),
                           sh_type='real').squeeze()  # shape (M,)
        Y_dir = torch.tensor(Y_dir, dtype=torch.float32).to(p_mn.device)
        #w = torch.tensor(Y_dir.conj(), dtype=p_mn.dtype, device=p_mn.device)  # (M,)
        w = Y_dir.conj().to(p_mn.device).type_as(p_mn)
        beamformed = torch.einsum('bmt,m->bt', p_mn, w)
        return beamformed
        

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)



if __name__ == "__main__":
    import soundfile as sf
    import os
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    #from your_module import SphericalHarmonicsProcessor  # 请根据实际路径修改

    # 参数设置
    wav_path = "/home/XiangzhuKong/workspace/CAT/1_simu/simulated_multi_speaker.wav"
    output_dir = "/home/XiangzhuKong/workspace/CAT/1_simu/plot"
    os.makedirs(output_dir, exist_ok=True)

    # 窗口参数（秒）
    window_sec = 2  # 每个窗口 0.5 秒
    hop_sec = 0.4    # 每次滑动 0.25 秒

    # 读取多通道信号
    sig_np, sr = sf.read(wav_path)  # sig_np shape: (n_samples, n_channels)
    print(f"Loaded audio: {wav_path} | Samplerate: {sr} Hz | Shape: {sig_np.shape}")

    # 将参数转换为样本数
    window_size = int(window_sec * sr)
    hop_size = int(hop_sec * sr)
    n_samples = sig_np.shape[0]

    # 初始化 SHT 处理器
    sph_order = 6
    sh_processor = SphericalHarmonicsProcessor(
        sph_order=sph_order,
        mic_type="aishell4",
        selected_channels=None,
        requires_grad=False,
        need_stft=False
    )

    # 极坐标的角度位置（0–360°）
    az_degs = np.arange(0, 360, 1)
    theta = np.pi / 2

    # 为所有窗口预分配功率列表
    all_powers = []  # 每个元素是一个长度为360的功率数组
    window_idx = 0
    time_stamps = []  # 保存每个窗口对应的时间秒数

    # 滑动窗口分块
    for start in range(0, n_samples - window_size + 1, hop_size):
        end = start + window_size
        seg = sig_np[start:end, :]  # (window_size, n_channels)
        time_stamps.append(start / sr)  # 保存窗口起始时间
        print(f"Window {window_idx}: samples [{start}:{end}]")

        # 转换为 tensor (1, channels, time)
        sig = torch.from_numpy(seg.T).unsqueeze(0).float()

        # 计算 SHT
        p_mn = sh_processor.forward(sig)  # (1, M, T)

        # 计算每个方向的功率
        powers = []
        for deg in az_degs:
            phi = np.deg2rad(deg)
            beam = sh_processor.inverse(p_mn, (phi, theta))  # (1, T)
            beam_np = beam.squeeze(0).cpu().numpy()
            powers.append(np.mean(beam_np**2))
        all_powers.append(powers)
        window_idx += 1

    # 绘制每个窗口的 DOA 极坐标图
    n_windows = len(all_powers)
    n_cols = 4  # 每行的子图数量
    n_rows = (n_windows + n_cols - 1) // n_cols  # 计算行数

    fig, axs = plt.subplots(n_rows, n_cols, subplot_kw={'projection': 'polar'}, figsize=(15, 5 * n_rows))

    # 扁平化子图数组方便访问
    axs = axs.flatten()

    for idx, (powers, start_time) in enumerate(zip(all_powers, time_stamps)):
        ax = axs[idx]
        ax.plot(np.deg2rad(az_degs), powers, label='Power Spectrum')
        
        # 标注最大值对应角度
        max_idx = np.argmax(powers)
        max_angle = az_degs[max_idx]
        ax.plot(
            [np.deg2rad(max_angle), np.deg2rad(max_angle)],
            [0, powers[max_idx]],
            color='red',
            linewidth=2,
            label=f'Max: {max_angle}°'
        )

        # 添加标题
        ax.set_title(f"Window {idx} | Start: {start_time:.2f}s")

        ax.legend(loc='upper right')

    # 隐藏多余的子图
    for idx in range(n_windows, len(axs)):
        axs[idx].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'sliding_window_doa_subplots_with_max5.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved sliding-window DOA subplots to: {plot_path}")







# if __name__ == "__main__":
#     # sph_order = 1
#     # sh_processor = SphericalHarmonicsProcessor(sph_order=sph_order, mic_type="aishell4", selected_channels=None)

#     # # Example signal
#     # # /mnt/nas4_workspace/kongxz/data/AISHELL-4/new_gen/simu_data/test_simu/mix/00002.wav
#     # sig = torch.randn(10, 8, 1000)  # Replace with your signal

#     # # Compute forward transformation
#     # transformed_sig = sh_processor.forward(sig)
#     # print("Transformed Signal:", transformed_sig.shape)
#     # #print("selected_channels:", sorted(sh_processor.selected_channels))
#     # Path to your multichannel mixture .wav file
#     import soundfile as sf
#     import os
#     import matplotlib.pyplot as plt
#     wav_path = "/mnt/nas4_workspace/kongxz/data/AISHELL-4/new_gen/simu_data/test_simu/mix/00002.wav"
#     #wav_path = "/mnt/nas4_workspace/kongxz/data/workspace_data/alimeeting/Eval_Ali_far/data/format.1/Eval-far-R8001_M8004_MS801-N_SPK8013-0000690-0001829.wav"
#     #wav_path = "/mnt/nas4_workspace/kongxz/data/workspace_data/alimeeting/Eval_Ali_far/data/format.1/Eval-far-R8003_M8001_MS801-N_SPK8001-0206494-0206511.wav"
#     # Directory to save output beams
#     output_dir = "/home/XiangzhuKong/workspace/CAT/egs/aishell4/data/test_sph"
#     os.makedirs(output_dir, exist_ok=True)

#     # Read waveform: shape (n_samples, n_channels)
#     sig_np, sr = sf.read(wav_path)
#     print(f"Loaded audio: {wav_path} | Samplerate: {sr} Hz | Shape: {sig_np.shape}")
#     # sig_np =  sig_np[16000*8:16000*8+8000, :]
#     # print(f"Loaded audio: {wav_path} | Samplerate: {sr} Hz | Shape: {sig_np.shape}")

#     # Convert to torch tensor of shape (batch=1, channels, time)
#     sig = torch.from_numpy(sig_np.T).unsqueeze(0).float()

#     # Initialize SHT processor
#     sph_order = 6  # or your desired spherical order
#     sh_processor = SphericalHarmonicsProcessor(
#         sph_order=sph_order,
#         mic_type="aishell4",
#         selected_channels=None,
#         requires_grad=False,
#         need_stft=False
#     )

#     # Forward spherical harmonics transform
#     p_mn = sh_processor.forward(sig)
#     print("Forward transform output shape:", p_mn.shape)

#     # # Inverse SH transform for given directions (azimuth, colatitude)
#     # # Azimuth angles: 0, 30, 60, 90, 120, 180, 270 degrees
#     # az_degs = [0, 30, 60, 90, 120, 180, 270]
#     # theta = np.pi / 2  # colatitude fixed at pi/2 (horizontal plane)
#     # for deg in az_degs:
#     #     phi = np.deg2rad(deg)
#     #     beam = sh_processor.inverse(p_mn, (phi, theta))  # shape (batch, time)
#     #     beam_np = beam.squeeze(0).cpu().numpy()
#     #     out_path = os.path.join(output_dir, f"beam_{deg}deg.wav")
#     #     sf.write(out_path, beam_np, sr)
#     #     print(f"Saved beam at {deg}° to: {out_path}")

#     # print("All beams have been saved.")
#     # 定义扫描角度
#     az_degs = np.arange(0, 360, 1)
#     theta = np.pi / 2  # 水平面

#     # 存储功率谱
#     powers = []

#     for deg in az_degs:
#         phi = np.deg2rad(deg)
#         beam = sh_processor.inverse(p_mn, (phi, theta))  # (batch, time)
#         beam_np = beam.squeeze(0).cpu().numpy()
#         # 计算输出信号能量
#         power = np.mean(beam_np**2)
#         powers.append(power)
#         # 保存每个方向的波束信号
#         #out_wav = os.path.join(output_dir, f"beam_{deg}deg.wav")
#         #sf.write(out_wav, beam_np, sr)

#     # 绘制 DOA 功率谱
#     plt.figure()
#     plt.polar(np.deg2rad(az_degs), powers)
#     plt.title('DOA Steered Response Power')
#     plt.tight_layout()
#     plot_path = os.path.join(output_dir, '2spk_6_2.png')
#     plt.savefig(plot_path)
#     plt.close()

#     print(f"DOA power spectrum saved to: {plot_path}")