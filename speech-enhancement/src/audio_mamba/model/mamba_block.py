from functools import partial

import torch
import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm


def create_block(
    d_model,
    cfg,
    layer_idx=0,
    rms_norm=True,
    fused_add_norm=False,
    residual_in_fp32=False,
):
    mixer_cls = partial(
        Mamba,
        layer_idx=layer_idx,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm,
        eps=cfg.norm_epsilon,
    )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls=nn.Identity,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class AuMBlock(nn.Module):
    """Fo-Bi Audio Mamba block: shared prenorm path scanned forward and backward."""

    def __init__(self, cfg, layer_idx=0):
        super().__init__()
        rms_norm = bool(getattr(cfg, 'rms_norm', True))
        self.forward_block = create_block(
            cfg.embed_dim,
            cfg,
            layer_idx=layer_idx,
            rms_norm=rms_norm,
        )
        self.backward_block = create_block(
            cfg.embed_dim,
            cfg,
            layer_idx=layer_idx,
            rms_norm=rms_norm,
        )
        self.apply(partial(_init_weights, n_layer=1))

    def forward(self, hidden_states, residual=None):
        hidden_forward, residual_forward = self.forward_block(
            hidden_states,
            residual,
        )
        hidden_backward, residual_backward = self.backward_block(
            hidden_states.flip([1]),
            None if residual is None else residual.flip([1]),
        )
        hidden_states = hidden_forward + hidden_backward.flip([1])
        residual = residual_forward + residual_backward.flip([1])
        return hidden_states, residual
