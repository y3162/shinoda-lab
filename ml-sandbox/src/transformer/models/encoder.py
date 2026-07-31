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


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        config: AttrDict,
    ) -> None:

        super().__init__()

        self.d_model = config.d_model
        self.d_ff = config.get('d_ff', 4 * self.d_model)
        self.pre_norm = config.get('pre_norm', False)

        self.attention_block = Attention(config)
        self.layer_norm_attn = nn.LayerNorm(self.d_model)

        self.feed_forward_network = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.d_model),
        )
        self.layer_norm_ffn = nn.LayerNorm(self.d_model)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if self.pre_norm:
            x_norm = self.layer_norm_attn(x)
            x = x + self.attention_block(x_norm, x_norm)

            x_norm = self.layer_norm_ffn(x)
            x = x + self.feed_forward_network(x)

        else:
            x = x + self.attention_block(x, x)
            x = self.layer_norm_attn(x)

            x = x + self.feed_forward_network(x)
            x = self.layer_norm_ffn(x)

        return x
