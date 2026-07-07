import torch
import torch.nn as nn
import numpy as np
from src.mp_senet.model.transformer import TransformerBlock
from src.mp_senet.utils import LearnableSigmoid2d
from pesq import pesq
from joblib import Parallel, delayed


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super().__init__()

        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * r,
            kernel_size=kernel_size,
            stride=(1, 1),
        )
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv(x)

        batch_size, n_channels, height, width = x.shape

        x = x.view(
            batch_size,
            self.r,
            n_channels // self.r,
            height,
            width,
        )
        x = x.permute(0, 2, 3, 4, 1)
        x = x.contiguous().view(
            batch_size,
            n_channels // self.r,
            height,
            -1,
        )

        return x


class DenseBlock(nn.Module):
    def __init__(self, h, kernel_size=(2, 3), depth=4):
        super().__init__()

        self.h = h
        self.depth = depth
        self.dense_block = nn.ModuleList(
            [
                self._make_dense_layer(h, kernel_size, i)
                for i in range(depth)
            ]
        )

    @staticmethod
    def _make_dense_layer(h, kernel_size, layer_idx):
        dilation = 2 ** layer_idx
        pad_length = dilation

        return nn.Sequential(
            nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            nn.Conv2d(
                h.dense_channel * (layer_idx + 1),
                h.dense_channel,
                kernel_size,
                dilation=(dilation, 1),
            ),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

    def forward(self, x):
        skip = x

        for dense_layer in self.dense_block:
            x = dense_layer(skip)
            skip = torch.cat([x, skip], dim=1)

        return x


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel):
        super().__init__()

        self.h = h

        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

        self.dense_block = DenseBlock(h, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(
                h.dense_channel,
                h.dense_channel,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

    def forward(self, x):
        x = self.dense_conv_1(x)  # [B, C, T, F]
        x = self.dense_block(x)   # [B, C, T, F]
        x = self.dense_conv_2(x)  # [B, C, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()

        self.dense_block = DenseBlock(h, depth=4)

        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
            nn.Conv2d(h.dense_channel, out_channel, (1, 2)),
        )

        self.lsigmoid = LearnableSigmoid2d(
            h.n_fft // 2 + 1,
            beta=h.beta,
        )

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)

        # [B, 1, T, F] -> [B, F, T]
        x = x.permute(0, 3, 2, 1).squeeze(-1)

        x = self.lsigmoid(x)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()

        self.dense_block = DenseBlock(h, depth=4)

        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

        self.phase_conv_r = nn.Conv2d(
            h.dense_channel,
            out_channel,
            kernel_size=(1, 2),
        )
        self.phase_conv_i = nn.Conv2d(
            h.dense_channel,
            out_channel,
            kernel_size=(1, 2),
        )

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)

        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)

        x = torch.atan2(x_i, x_r)

        # [B, 1, T, F] -> [B, F, T]
        x = x.permute(0, 3, 2, 1).squeeze(-1)

        return x


class TSTransformerBlock(nn.Module):
    def __init__(self, h):
        super().__init__()

        self.h = h
        self.time_transformer = TransformerBlock(
            d_model=h.dense_channel,
            n_heads=4,
        )
        self.freq_transformer = TransformerBlock(
            d_model=h.dense_channel,
            n_heads=4,
        )

    def forward(self, x):
        batch_size, channels, time_steps, freq_bins = x.size()

        # Time transformer
        # [B, C, T, F] -> [B * F, T, C]
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch_size * freq_bins, time_steps, channels)
        x = self.time_transformer(x) + x

        # Frequency transformer
        # [B * F, T, C] -> [B * T, F, C]
        x = x.view(batch_size, freq_bins, time_steps, channels)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * time_steps, freq_bins, channels)
        x = self.freq_transformer(x) + x

        # [B * T, F, C] -> [B, C, T, F]
        x = x.view(batch_size, time_steps, freq_bins, channels)
        x = x.permute(0, 3, 1, 2)

        return x


class MPNet(nn.Module):
    def __init__(self, h, num_tsblocks=4):
        super().__init__()

        self.h = h

        self.num_tscblocks = num_tsblocks

        self.dense_encoder = DenseEncoder(h, in_channel=2)

        self.TSTransformer = nn.ModuleList(
            [TSTransformerBlock(h) for _ in range(num_tsblocks)]
        )

        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)

    def forward(self, noisy_amp, noisy_pha):
        # noisy_amp, noisy_pha: [B, F, T]

        # [B, F, T], [B, F, T] -> [B, 2, T, F]
        x = torch.stack((noisy_amp, noisy_pha), dim=-1)
        x = x.permute(0, 3, 2, 1)

        x = self.dense_encoder(x)

        for ts_block in self.TSTransformer:
            x = ts_block(x)

        denoised_amp = noisy_amp * self.mask_decoder(x)
        denoised_pha = self.phase_decoder(x)

        denoised_com = torch.stack(
            (
                denoised_amp * torch.cos(denoised_pha),
                denoised_amp * torch.sin(denoised_pha),
            ),
            dim=-1,
        )

        return denoised_amp, denoised_pha, denoised_com


def anti_wrapping_function(x):
    return torch.abs(
        x - torch.round(x / (2 * np.pi)) * 2 * np.pi
    )


def phase_losses(phase_r, phase_g):
    ip_loss = torch.mean(
        anti_wrapping_function(phase_r - phase_g)
    )

    gd_loss = torch.mean(
        anti_wrapping_function(
            torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)
        )
    )

    iaf_loss = torch.mean(
        anti_wrapping_function(
            torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)
        )
    )

    return ip_loss, gd_loss, iaf_loss


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        pesq_score = -1

    return pesq_score


def pesq_score(utts_r, utts_g, h):
    scores = Parallel(n_jobs=30)(
        delayed(eval_pesq)(
            utts_r[i].squeeze().cpu().numpy(),
            utts_g[i].squeeze().cpu().numpy(),
            h.sampling_rate,
        )
        for i in range(len(utts_r))
    )

    return np.mean(scores)
