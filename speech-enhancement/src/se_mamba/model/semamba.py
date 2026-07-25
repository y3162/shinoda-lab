import torch
import torch.nn as nn
from einops import rearrange

from src.se_mamba.model.codec_module import DenseEncoder, MagDecoder, PhaseDecoder
from src.se_mamba.model.mamba_block import TFMambaBlock


class SEMamba(nn.Module):
    def __init__(self, cfg, n_fft):
        super().__init__()
        self.num_tscblocks = (
            cfg.num_tfmamba if cfg.num_tfmamba is not None else 4
        )
        self.dense_encoder = DenseEncoder(cfg)
        self.TSMamba = nn.ModuleList(
            [TFMambaBlock(cfg) for _ in range(self.num_tscblocks)]
        )
        self.mask_decoder = MagDecoder(cfg, n_fft)
        self.phase_decoder = PhaseDecoder(cfg)

    def forward(self, noisy_mag, noisy_pha):
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)
        x = torch.cat((noisy_mag, noisy_pha), dim=1)
        x = self.dense_encoder(x)
        for block in self.TSMamba:
            x = block(x)
        denoised_mag = rearrange(
            self.mask_decoder(x) * noisy_mag, 'b c t f -> b f t c',
        ).squeeze(-1)
        denoised_pha = rearrange(
            self.phase_decoder(x), 'b c t f -> b f t c',
        ).squeeze(-1)
        denoised_com = torch.stack(
            (
                denoised_mag * torch.cos(denoised_pha),
                denoised_mag * torch.sin(denoised_pha),
            ),
            dim=-1,
        )
        return denoised_mag, denoised_pha, denoised_com
