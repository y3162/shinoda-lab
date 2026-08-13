from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.audio_mamba.model.audiomamba import AudioMambaOutput


class AudioMambaLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.snr_loss_weight = float(cfg.snr_loss_weight)

    def forward(
        self,
        output: AudioMambaOutput,
        noise_type: torch.Tensor,
        snr_db: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        noise_type_loss = F.cross_entropy(
            output.noise_type_logits,
            noise_type,
        )
        snr_target = snr_db.view_as(output.snr_db).to(
            dtype=output.snr_db.dtype,
        )
        snr_loss = F.mse_loss(output.snr_db, snr_target)
        total = noise_type_loss + self.snr_loss_weight * snr_loss
        return {
            'loss': total,
            'noise_type_loss': noise_type_loss,
            'snr_loss': snr_loss,
        }
