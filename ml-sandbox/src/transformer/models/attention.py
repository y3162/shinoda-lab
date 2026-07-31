import math

import torch
import torch.nn as nn

from src.utils.print import (
    print_log,
)
from src.utils.config import (
    AttrDict,
)


class Attention(nn.Module):
    def __init__(
        self,
        config: AttrDict,
    ) -> None:

        super().__init__()

        print_log(config)
        self.d_model = config.d_model
        self.n_heads = config.get('n_heads', 1)
        self.d_head = config.get('d_head', self.d_model // self.n_heads)
        self.d_v = config.get('d_v', self.d_model)

        self.w_q = nn.Linear(self.d_model, self.d_head * self.n_heads)
        self.w_k = nn.Linear(self.d_model, self.d_head * self.n_heads)
        self.w_v = nn.Linear(self.d_model, self.d_v * self.n_heads)

        self.w_o = nn.Linear(self.d_v * self.n_heads, self.d_model)

    def _split_head(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return x.view(*x.shape[:-1], self.n_heads, -1).movedim(-2, -3)

    def _concat_head(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return x.movedim(-3, -2).flatten(-2)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
    ) -> torch.Tensor:

        query = self.w_q(x_q)
        key = self.w_k(x_kv)
        value = self.w_v(x_kv)

        query = self._split_head(query)
        key = self._split_head(key)
        value = self._split_head(value)

        attention_score = query @ key.transpose(-2, -1)
        attention_score = attention_score / math.sqrt(self.d_head)
        attention_score = attention_score.softmax(dim=-1)

        weighted_value = attention_score @ value
        weighted_value = self._concat_head(weighted_value)

        output = self.w_o(weighted_value)

        return output
