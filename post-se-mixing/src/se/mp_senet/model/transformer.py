import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    MultiheadAttention,
    GRU,
    Linear,
    LayerNorm,
    Dropout,
)


class FFN(nn.Module):
    def __init__(self, d_model, bidirectional=True, dropout=0):
        super().__init__()

        hidden_size = d_model * 2

        self.gru = GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True,
        )

        linear_in_dim = hidden_size * 2 if bidirectional else hidden_size

        self.linear = Linear(linear_in_dim, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        self.gru.flatten_parameters()

        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, bidirectional=True, dropout=0):
        super().__init__()

        self.norm1 = LayerNorm(d_model)
        self.attention = MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = Dropout(dropout)

        self.norm2 = LayerNorm(d_model)

        # 元コードと同じ挙動を維持するため、dropout は渡さない
        self.ffn = FFN(
            d_model=d_model,
            bidirectional=bidirectional,
        )
        self.dropout2 = Dropout(dropout)

        self.norm3 = LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        xt = self.norm1(x)

        xt, _ = self.attention(
            xt,
            xt,
            xt,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout1(xt)

        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)

        x = self.norm3(x)

        return x


def main():
    x = torch.randn(4, 64, 401, 201)

    batch_size, channels, time_steps, freq_bins = x.size()

    # [B, C, T, F] -> [B, F * T, C]
    x = x.permute(0, 3, 2, 1).contiguous()
    x = x.view(batch_size, freq_bins * time_steps, channels)

    transformer = TransformerBlock(
        d_model=64,
        n_heads=4,
    )

    x = transformer(x)

    # [B, F * T, C] -> [B, C, T, F]
    x = x.view(batch_size, freq_bins, time_steps, channels)
    x = x.permute(0, 3, 2, 1)

    print(x.size())
