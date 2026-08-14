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
        snr_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        noise_type_loss = F.cross_entropy(
            output.noise_type_logits,
            noise_type,
        )
        snr_pred = output.snr_db.view(-1)
        snr_target = snr_db.view(-1).to(dtype=snr_pred.dtype)

        if snr_valid is None:
            snr_loss = F.mse_loss(snr_pred, snr_target)
        else:
            mask = snr_valid.view(-1).bool()
            if mask.any():
                snr_loss = F.mse_loss(snr_pred[mask], snr_target[mask])
            else:
                snr_loss = snr_pred.new_zeros(())

        total = noise_type_loss + self.snr_loss_weight * snr_loss
        return {
            'loss': total,
            'noise_type_loss': noise_type_loss,
            'snr_loss': snr_loss,
        }
