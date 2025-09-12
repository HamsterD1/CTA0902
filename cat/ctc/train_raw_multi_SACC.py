# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Xiangzhu Kong(kongxiangzhu99@gmail.com), Keyu An, Huahuan Zheng

"""Top interface of CTC training.
"""

__all__ = ["AMTrainer", "build_model", "_parser", "main"]

import torchaudio
from ..shared.manager_wo import Manager
from ..shared import coreutils
from ..shared import encoder as model_zoo
from ..shared.data import KaldiSpeechDataset, sortedPadCollateASR

from ..front.stft import Stft
from ..front.log_mel import LogMel
from ..front.beamformer_net import BeamformerNet
from ..front.kaldifbank import Feature_Trans

from ..front.SHT_np import SphericalHarmonicsProcessor
from ..front.cbam import CSA_Beamformer
from ..front.c_cbam import cCSA_Beamformer
from ..front.tfgridnet import TFGridNet
from ..front.SHT_IGCRN import IGCRN
from ..front.ChannelAug import ChannelAugment
from ..front.multi2mono import ChannelSelector
from ..front.EaBNet import EaBNet


import os
import argparse
import Levenshtein
from typing import *
from ctcdecode import CTCBeamDecoder
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast

import math
import numpy as np

import matplotlib.pyplot as plt

# NOTE (huahuan):
#   1/4 subsampling is used for Conformer model defaultly
#   for other sampling ratios, you need to modify the value.
#   Commonly, you can use a relatively larger value for allowing some margin.
SUBSAMPLING = 4


def check_label_len_for_ctc(
    tupled_mat_label: Tuple[torch.FloatTensor, torch.LongTensor]
):
    """filter the short seqs for CTC/CRF"""
    return tupled_mat_label[0].shape[0] // SUBSAMPLING > tupled_mat_label[1].shape[0]


def filter_hook(dataset):
    return dataset.select(check_label_len_for_ctc)


def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace, **mkwargs):
    coreutils.set_random_seed(args.seed)
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    if "T_dataset" not in mkwargs:
        mkwargs["T_dataset"] = KaldiSpeechDataset

    if "collate_fn" not in mkwargs:
        mkwargs["collate_fn"] = sortedPadCollateASR(flatten_target=True)

    if "func_build_model" not in mkwargs:
        mkwargs["func_build_model"] = build_model

    if "_wds_hook" not in mkwargs:
        mkwargs["_wds_hook"] = filter_hook

    if (
        "func_eval" not in mkwargs
        and hasattr(args, "eval_error_rate")
        and args.eval_error_rate
    ):
        mkwargs["func_eval"] = custom_evaluate

    mkwargs["args"] = args
    manager = Manager(**mkwargs)

    # NOTE: for CTC training, the input feat len must be longer than the label len
    #       ... when using webdataset (--ld) to load the data, we deal with
    #       ... the issue by `_wds_hook`; if not, we filter the unqualified utterances
    #       ... before training start.
    if args.ld is None:
        coreutils.distprint(f"> filter seqs by ctc restriction", args.gpu)
        tr_dataset = manager.trainloader.dl.dataset
        orilen = len(tr_dataset)
        tr_dataset.filt_by_len(lambda x, y: x // SUBSAMPLING > y)
        coreutils.distprint(
            f"  filtered {orilen-len(tr_dataset)} utterances.", args.gpu
        )

    # training
    manager.run(args)


class AMTrainer(nn.Module):
    def __init__(
        self,
        encoder: model_zoo.AbsEncoder,
        use_crf: bool = False,
        den_lm: Optional[str] = None,
        lamb: Optional[float] = 0.01,
        decoder: CTCBeamDecoder = None,
        # front
        use_kaldi: bool = False,
        fs: int = 16000,
        n_fft: int = 512,
        win_length:int = 400,
        hop_length:int = 160,
        idim: int = 80,
        beamforming: str = "mvdr",
        wpe: bool = False,
        noSE: bool = False,
        use_TFGrid: bool = False,
        use_IGCRN: bool = False,
        use_EaBNet: bool = False,
        use_SACC: bool = False,
        SACC_type: str = "TC",
        use_WSJCA: bool = False,
        sph_order: int = 4,
        ChannelAug: bool = False
    ):
        super().__init__()
        
        self.frame_length = win_length // (fs * 0.001)
        self.frame_shift = hop_length // (fs * 0.001)
        self.sample_frequency = fs
        self.use_kaldi = use_kaldi
        if self.use_kaldi:
            self.trans = Feature_Trans(
                num_mel_bins=idim,
                window_size=win_length,
                window_shift=hop_length,
                dither=0.0,
                # librosa
                # preemphasis_coefficient=0.0,
                # remove_dc_offset=False,
                # window_type= "hanning",
                )
            print("use kaldi features")
        
        
        

        self.stft = Stft(n_fft, win_length, hop_length)
        self.logmel = LogMel(n_mels=idim)
        self.use_SACC = use_SACC
        self.use_WSJCA = use_WSJCA
        
        if sph_order is not None:
            self.sph_order = sph_order
            self.SHT = SphericalHarmonicsProcessor(
                mic_type = "aishell4", 
                #mic_type = "alimeeting",
                #mic_type = "706", 
                #mic_type = "circular",
                #mic_type = "linear",
                #mic_type = "square",
                sph_order = self.sph_order,
                #selected_channels = [0, 1, 3, 4, 8, 9, 15],
                requires_grad=False,
                #selected_channels=None
                #selected_channels = [1,3,5,7],
                need_stft=use_IGCRN,
                rand_selected_channels=False
            )
        elif ChannelAug:
            self.SHT = ChannelAugment(
                min_channels=2, 
                max_channels=8,
                #selected_channels=None
                #selected_channels=[1,3,5,7]
                )
            print("Channel Aug")
        else:
            self.SHT = None
            print("no SHT")
            
        
        if noSE:
            self.beamformer = ChannelSelector(total_channels=8,chosen_channel=0)
        
        
        
        elif use_TFGrid:
            self.beamformer = TFGridNet(
                            n_layers=1,
                            lstm_hidden_units=128,
                            attn_n_head=4,
                            attn_approx_qk_dim=256,
                            emb_dim=16
                                )
            print("Use TFGridNet")
        
        elif use_IGCRN:
            self.beamformer = IGCRN(in_ch_sph= (self.sph_order + 1)**2,
                                    in_ch_stft= 8,
                                    channels=20
                                    )
            print("use Proposed-parallel IGCRN")
        
        elif use_EaBNet:
            self.beamformer = EaBNet(
                                d_feat = 448, 
                                M = 8,
                                topo_type = "mimo",)
            print("Use EaBNet")
            
        
        elif use_SACC:

            if SACC_type == "CSA":
                if sph_order is not None:
                    self.beamformer = CSA_Beamformer(
                                                channel=(self.sph_order+1)**2,
                                                n_fft=n_fft,
                                                #kernel_size=3,
                                                reduction = (self.sph_order + 1)
                                                )
                else:
                    self.beamformer = CSA_Beamformer(
                                                channel=8,
                                                n_fft=n_fft,
                                                #kernel_size=3
                                                reduction = 5
                                                )
                print("use CSA")
            
            elif SACC_type == "Complex_CSA":
                if sph_order is not None:
                    self.beamformer = cCSA_Beamformer(
                                                channel=(self.sph_order+1)**2,
                                                n_fft=n_fft,
                                                #kernel_size=3,
                                                reduction = (self.sph_order + 1)
                                                )
                else:
                    self.beamformer = cCSA_Beamformer(
                                                channel=8,
                                                n_fft=n_fft,
                                                #kernel_size=3
                                                reduction = 5
                                                )
                print("use Complex_CSA")


        else:    
            if wpe:
                self.beamformer = BeamformerNet(beamformer_type=beamforming,use_wpe=True)
            else:
                self.beamformer = BeamformerNet(beamformer_type=beamforming)
                print("use MB BF")
        
        self.encoder = encoder
        self.is_crf = use_crf
        if use_crf:
            self.den_lm = den_lm
            assert den_lm is not None and os.path.isfile(den_lm)

            from ctc_crf import CTC_CRF_LOSS as CRFLoss

            self.criterion = CRFLoss(lamb=lamb)
            self._crf_ctx = None
        else:
            self.den_lm = None
            self.criterion = nn.CTCLoss()
        
        self.attach = {"decoder": decoder}

    def clean_unpickable_objs(self):
        # CTCBeamDecoder is unpickable,
        # So, this is required for inference.
        self.attach["decoder"] = None

    def register_crf_ctx(self, den_lm: Optional[str] = None):
        """Register the CRF context on model device."""
        assert self.is_crf

        from ctc_crf import CRFContext

        self._crf_ctx = CRFContext(
            den_lm, next(iter(self.encoder.parameters())).device.index
        )

    @torch.no_grad()
    def get_wer(
        self, xs: torch.Tensor, ys: torch.Tensor, lx: torch.Tensor, ly: torch.Tensor
    ):
        if self.attach["decoder"] is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: self.attach['decoder'] is not initialized."
            )

        bs = xs.size(0)
        logits, lx = self.encoder(xs, lx)

        # y_samples: (N, k, L), ly_samples: (N, k)
        y_samples, _, _, ly_samples = self.attach["decoder"].decode(
            logits.float().cpu(), lx.cpu()
        )

        """NOTE:
            for CTC training, we flatten the label seqs to 1-dim,
            so here we need to deal with that
        """
        if ys.dim() == 1:
            ground_truth = [t.cpu().tolist() for t in torch.split(ys, ly.tolist())]
        else:
            ground_truth = [ys[i, : ly[i]] for i in range(ys.size(0))]
        hypos = [y_samples[n, 0, : ly_samples[n, 0]].tolist() for n in range(bs)]

        return cal_wer(ground_truth, hypos)

    def wave2fbank(self,audio, lx):
        audio = audio[...,0]
        samples, flens = self.stft(audio, lx)
        input_power = samples[..., 0] ** 2 + samples[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        feats, _ = self.logmel(input_amp, flens)
        
        return feats,flens
    
    # def get_model_flops(self, model = None, 
    #                     time = 10, 
    #                     time_steps = 120, 
    #                     channels = None, 
    #                     freq_bins = 257
    #                     ):
    #     # 假设 10s 的语音片段， 1000帧，对应chunk大小为80 + 40，一共有9个 chunk
        
    #     # 输入形状 (batch_size, time_steps, channels, freq_bins, _)
    #     #batch_size, time_steps, channels, freq_bins, _ = 1, 100, 25, 257, 2
    #     if self.SHT is not None and channels is None:
    #         channels=(self.sph_order+1)**2
        
    #     max_input_length = int(
    #         time_steps * (math.ceil(float(time*100) / time_steps))
    #     )
        
    #     batch_size = max_input_length // time_steps
        

    #     input_shape = (batch_size, time_steps, channels, freq_bins, 2)  
        
    #     if model is None:
    #         model = self.beamformer

    #     # 转换为张量并指定设备
    #     from thop import profile
    #     dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
    #     flens = torch.full([dummy_input.size(0)], dummy_input.size(1)).to(next(model.parameters()).device)
    #     flops, params = profile(model, inputs=(dummy_input, flens), verbose=False)
        
    #     print(f"Model FLOPs: {flops / 1e9:.2f} GFLOPs")
    #     print(f"Model Parameters: {params / 1e6:.2f} M")
        
    #     return 
     
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

        from thop import profile
        # 转换为张量并指定设备
        dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
        flens = torch.full([dummy_input.size(0)], dummy_input.size(1)).to(next(model.parameters()).device)
        flops, params = profile(model, inputs=(dummy_input, flens), verbose=verbose)
        
        print(f"Model FLOPs: {flops / 1e9:.2f} GFLOPs")
        print(f"Model Parameters: {params / 1e6:.2f} M")
        
        return flops, params
    
    def beamforming(self,audio, lx):
        audio = audio.permute(0,2,1)
        if self.SHT is not None:
            audio = self.SHT(audio, Random_Flag = self.training)
        else:
            # Optionally handle the case where SHT is None
            #print("SHT is None, skipping spherical harmonics processing")
            #pass
            _, C, _ = audio.shape
            if not C == 8:
                 audio = audio[:,:8, :]
        
        
        if self.use_kaldi:
            samples, flens = self.trans.cal_stft(audio, lx)
            # (B, C, T, F, 2) --> (B, T, C, F, 2)
            samples = samples.permute(0,2,1,3,4)
            samples, flens1 , _ =self.beamformer(samples,flens)
            assert (flens == flens1).all()
            feats, _ = self.trans.spectrum_to_fbank(samples,flens1)
        else:
            audio = audio.permute(0,2,1)
            samples, flens = self.stft(audio, lx)
            samples, flens1 , weights =self.beamformer(samples,flens)
            assert (flens == flens1).all()
            
            # cal fbank
            if self.use_SACC:
                input_power = samples
            else:
                input_power = samples[..., 0] ** 2 + samples[..., 1] ** 2
            
            input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
            feats, _ = self.logmel(input_amp, flens)
        
        return feats, flens
    
    def forward(self, audio, lx, labels, ly):
        
        feats, flens = self.beamforming(audio, lx)
        

        
        if torch.isnan(feats).any():
            print("feats array contains NaN values!")

        
        logits, lx = self.encoder(feats, flens)
        logits = torch.log_softmax(logits, dim=-1)

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
        
        if float('inf') == loss:
            print("loss is inf!")
        
        if math.isnan(loss):
            print("loss is NaN!")    
        
        
        return loss,loss,loss


def cal_wer(gt: List[List[int]], hy: List[List[int]]) -> Tuple[int, int]:
    """compute error count for list of tokens"""
    assert len(gt) == len(hy)
    err = 0
    cnt = 0
    for i in range(len(gt)):
        err += Levenshtein.distance(
            "".join(chr(n) for n in hy[i]), "".join(chr(n) for n in gt[i])
        )
        cnt += len(gt[i])
    return (err, cnt)


@torch.no_grad()
def custom_evaluate(testloader, args: argparse.Namespace, manager: Manager) -> float:
    model = manager.model
    cnt_tokens = 0
    cnt_err = 0
    n_proc = dist.get_world_size()

    for minibatch in tqdm(
        testloader,
        desc=f"Epoch: {manager.epoch} | eval",
        unit="batch",
        disable=(args.gpu != 0),
        leave=False,
    ):
        feats, ilens, labels, olens = minibatch[:4]
        feats = feats.cuda(args.gpu, non_blocking=True)
        ilens = ilens.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)
        olens = olens.cuda(args.gpu, non_blocking=True)

        part_cnt_err, part_cnt_sum = model.module.get_wer(feats, labels, ilens, olens)
        cnt_err += part_cnt_err
        cnt_tokens += part_cnt_sum

    gather_obj = [None for _ in range(n_proc)]
    dist.gather_object(
        (cnt_err, cnt_tokens), gather_obj if args.rank == 0 else None, dst=0
    )
    dist.barrier()
    if args.rank == 0:
        l_err, l_sum = list(zip(*gather_obj))
        wer = sum(l_err) / sum(l_sum)
        manager.writer.add_scalar("loss/dev-token-error-rate", wer, manager.step)
        scatter_list = [wer]
    else:
        scatter_list = [None]

    dist.broadcast_object_list(scatter_list, src=0)
    return scatter_list[0]


def build_beamdecoder(cfg: dict) -> CTCBeamDecoder:
    """
    beam_size:
    num_classes:
    kenlm:
    alpha:
    beta:
    ...
    """

    assert "num_classes" in cfg, "number of vocab size is required."

    if "kenlm" in cfg:
        labels = [str(i) for i in range(cfg["num_classes"])]
        labels[0] = "<s>"
        labels[1] = "<unk>"
    else:
        labels = [""] * cfg["num_classes"]

    return CTCBeamDecoder(
        labels=labels,
        model_path=cfg.get("kenlm", None),
        beam_width=cfg.get("beam_size", 16),
        alpha=cfg.get("alpha", 1.0),
        beta=cfg.get("beta", 0.0),
        num_processes=cfg.get("num_processes", 6),
        log_probs_input=True,
        is_token_based=("kenlm" in cfg),
    )


def build_model(
    cfg: dict,
    args: Optional[Union[argparse.Namespace, dict]] = None,
    dist: bool = True,
    wrapper: bool = True,
) -> Union[nn.parallel.DistributedDataParallel, AMTrainer, model_zoo.AbsEncoder]:
    """
    for ctc-crf training, you need to add extra settings in
    cfg:
        trainer:
            use_crf: true/false,
            lamb: 0.01,
            den_lm: xxx

            decoder:
                beam_size:
                num_classes:
                kenlm:
                alpha:
                beta:
                ...
        ...
    """
    if "trainer" not in cfg:
        cfg["trainer"] = {}

    assert "encoder" in cfg
    netconfigs = cfg["encoder"]
    net_kwargs = netconfigs["kwargs"]  # type:dict

    # when immigrate configure from RNN-T to CTC,
    # one usually forget to set the `with_head=True` and 'num_classes'
    if "with_head" in net_kwargs and not net_kwargs["with_head"]:
        print(
            "WARNING: 'with_head' in field:encoder:kwargs is False, "
            "If you don't know what this means, set it to True."
        )

    if "num_classes" not in net_kwargs:
        raise Exception(
            "error: 'num_classes' in field:encoder:kwargs is not set. "
            "You should specify it according to your vocab size."
        )

    am_model = getattr(model_zoo, netconfigs["type"])(
        **net_kwargs
    )  # type: model_zoo.AbsEncoder
    if not wrapper:
        return am_model

    # initialize beam searcher
    if "decoder" in cfg["trainer"]:
        cfg["trainer"]["decoder"] = build_beamdecoder(cfg["trainer"]["decoder"])

    model = AMTrainer(am_model, **cfg["trainer"])
    if not dist:
        return model

    assert args is not None, f"You must tell the GPU id to build a DDP model."
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    elif not isinstance(args, dict):
        raise ValueError(f"unsupport type of args: {type(args)}")

    # make batchnorm synced across all processes
    model = coreutils.convert_syncBatchNorm(model)

    model.cuda(args["gpu"])
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[args["gpu"]],
        find_unused_parameters=False
        )
    return model


def _parser():
    parser = coreutils.basic_trainer_parser("CTC trainer.")
    parser.add_argument(
        "--eval-error-rate",
        action="store_true",
        help="Use token error rate for evaluation instead of CTC loss (default). "
        "If specified, you should setup 'decoder' in 'trainer' configuration.",
    )
    return parser


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
    
    if not ori:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    if logs:
        # Move tensor to CPU and normalize it
        device, dtype = tensor.device, tensor.dtype
        from torchaudio.compliance.kaldi import _get_epsilon
        epsilon = _get_epsilon(device, dtype)
        tensor = torch.max(tensor.abs().pow(2.0), epsilon).log()
    
    tensor = tensor.cpu()
    
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
    print(
        "NOTE:\n"
        "    since we import the build_model() function in cat.ctc,\n"
        "    we should avoid calling `python -m cat.ctc.train`, instead\n"
        "    running `python -m cat.ctc`"
    )
