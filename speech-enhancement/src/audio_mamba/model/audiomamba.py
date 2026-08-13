from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from mamba_ssm.ops.triton.layer_norm import RMSNorm

from src.audio_mamba.model.mamba_block import AuMBlock


@dataclass
class AudioMambaOutput:
    noise_type_logits: torch.Tensor
    snr_db: torch.Tensor


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=(16, 16),
        strides=(16, 16),
        in_chans=1,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.strides = tuple(strides)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.strides,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T] -> [B, N, D]
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class AudioMamba(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.num_classes = cfg.num_classes

        spectrogram_size = tuple(cfg.spectrogram_size)
        patch_size = tuple(cfg.patch_size)
        strides = tuple(cfg.strides)

        freq_patches = (
            spectrogram_size[0] - patch_size[0]
        ) // strides[0] + 1
        time_patches = (
            spectrogram_size[1] - patch_size[1]
        ) // strides[1] + 1
        self.num_patches = freq_patches * time_patches
        self.num_tokens = self.num_patches + 1

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            strides=strides,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, cfg.embed_dim),
        )
        self.pos_drop = nn.Dropout(p=cfg.drop_rate)

        rms_norm = bool(getattr(cfg, 'rms_norm', True))
        self.layers = nn.ModuleList(
            [
                AuMBlock(cfg, layer_idx=i)
                for i in range(cfg.depth)
            ]
        )
        self.norm_f = (
            RMSNorm if rms_norm else nn.LayerNorm
        )(cfg.embed_dim, eps=cfg.norm_epsilon)

        self.noise_type_head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self.snr_head = nn.Linear(cfg.embed_dim, 1)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.noise_type_head.apply(self._init_head)
        self.snr_head.apply(self._init_head)

    @staticmethod
    def _init_head(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, F, T]
        x = self.patch_embed(x)
        batch_size, num_patches, _ = x.shape

        if num_patches != self.num_patches:
            raise ValueError(
                f'Expected {self.num_patches} patches from spectrogram_size='
                f'{tuple(self.cfg.spectrogram_size)}, got {num_patches}. '
                f'Input spatial shape must match the configured spectrogram size.'
            )

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        token_position = num_patches // 2
        x = torch.cat(
            (x[:, :token_position, :], cls_token, x[:, token_position:, :]),
            dim=1,
        )
        x = x + self.pos_embed
        x = self.pos_drop(x)

        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        if residual is None:
            residual = hidden_states
        else:
            residual = residual + hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        return hidden_states[:, token_position, :]

    def forward(self, mel: torch.Tensor) -> AudioMambaOutput:
        features = self.forward_features(mel)
        return AudioMambaOutput(
            noise_type_logits=self.noise_type_head(features),
            snr_db=self.snr_head(features),
        )
