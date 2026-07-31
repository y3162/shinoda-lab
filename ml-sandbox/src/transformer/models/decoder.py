from pathlib import Path

import torch
import torch.nn as nn

from src.transformer.models.attention import (
    Attention,
)
from src.utils.print import (
    print_log,
)
from src.utils.config import (
    AttrDict,
    json2attr,
)


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        config: AttrDict,
    ) -> None:

        super().__init__()

        self.d_model = config.d_model
        self.d_ff = config.get('d_ff', 4 * self.d_model)
        self.pre_norm = config.get('pre_norm', False)

        self.cross_attention_block = Attention(config)
        self.layer_norm_cross_attn = nn.LayerNorm(self.d_model)

        self.self_attention_block = Attention(config)
        self.layer_norm_self_attn = nn.LayerNorm(self.d_model)

        self.feed_forward_network = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.d_model),
        )
        self.layer_norm_ffn = nn.LayerNorm(self.d_model)

    def forward(
        self,
        x_dec: torch.Tensor,
        x_enc: torch.Tensor,
    ) -> torch.Tensor:

        if self.pre_norm:
            x_dec_norm = self.layer_norm_self_attn(x_dec)
            x_dec = x_dec + self.self_attention_block(x_dec_norm, x_dec_norm)

            x_dec_norm = self.layer_norm_cross_attn(x_dec)
            x_dec = x_dec + self.cross_attention_block(x_dec_norm, x_enc)

            x_dec_norm = self.layer_norm_ffn(x_dec)
            x_dec = x_dec + self.feed_forward_network(x_dec_norm)

        else:
            x_dec = x_dec + self.self_attention_block(x_dec, x_dec)
            x_dec = self.layer_norm_self_attn(x_dec)

            x_dec = x_dec + self.cross_attention_block(x_dec, x_enc)
            x_dec = self.layer_norm_cross_attn(x_dec)

            x_dec = x_dec + self.feed_forward_network(x_dec)
            x_dec = self.layer_norm_ffn(x_dec)

        return x_dec
