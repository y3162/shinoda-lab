import torch
import torch.nn as nn
from einops import rearrange

from src.se.se_mamba.model.lsigmoid import LearnableSigmoid2D


def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (
        int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
        int((kernel_size[1] * dilation[1] - dilation[1]) / 2),
    )


class DenseBlock(nn.Module):
    def __init__(self, cfg, kernel_size=(3, 3), depth=4):
        super().__init__()
        self.depth = depth
        self.dense_block = nn.ModuleList()
        self.hid_feature = cfg.hid_feature
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(
                    self.hid_feature * (i + 1),
                    self.hid_feature,
                    kernel_size,
                    dilation=(dil, 1),
                    padding=get_padding_2d(kernel_size, (dil, 1)),
                ),
                nn.InstanceNorm2d(self.hid_feature, affine=True),
                nn.PReLU(self.hid_feature),
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_channel = cfg.input_channel
        self.hid_feature = cfg.hid_feature
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(self.input_channel, self.hid_feature, (1, 1)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature),
        )
        self.dense_block = DenseBlock(cfg, depth=4)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature),
        )

    def forward(self, x):
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class MagDecoder(nn.Module):
    def __init__(self, cfg, n_fft):
        super().__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg.hid_feature
        self.output_channel = cfg.output_channel
        self.beta = cfg.beta
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.Conv2d(self.hid_feature, self.output_channel, (1, 1)),
            nn.InstanceNorm2d(self.output_channel, affine=True),
            nn.PReLU(self.output_channel),
            nn.Conv2d(self.output_channel, self.output_channel, (1, 1)),
        )
        self.lsigmoid = LearnableSigmoid2D(n_fft // 2 + 1, beta=self.beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = rearrange(x, 'b c t f -> b f t c').squeeze(-1)
        x = self.lsigmoid(x)
        x = rearrange(x, 'b f t -> b t f').unsqueeze(1)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg.hid_feature
        self.output_channel = cfg.output_channel
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature),
        )
        self.phase_conv_r = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        return torch.atan2(x_i, x_r)
