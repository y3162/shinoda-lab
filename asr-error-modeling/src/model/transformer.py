import math

import torch
import torch.nn as nn


def sinusoidal_positional_encoding(
    seq_len: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        d_model: int,
        n_heads: int,
        context_length: int,
        pad_id: int,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.context_length = context_length
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )

        # weight tying
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def add_global_prefix(
        self,
        src_ids: torch.Tensor,
        global_prefix_ids: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([global_prefix_ids, src_ids], dim=1)

    def _apply_src_positional_encoding(
        self,
        src_emb: torch.Tensor,
        global_prefix_len: int,
    ) -> torch.Tensor:
        if global_prefix_len <= 0:
            return src_emb + sinusoidal_positional_encoding(
                src_emb.size(1),
                self.d_model,
                device=src_emb.device,
                dtype=src_emb.dtype,
            )

        prefix_emb = src_emb[:, :global_prefix_len]
        text_emb = src_emb[:, global_prefix_len:]
        text_emb = text_emb + sinusoidal_positional_encoding(
            text_emb.size(1),
            self.d_model,
            device=text_emb.device,
            dtype=text_emb.dtype,
        )
        return torch.cat([prefix_emb, text_emb], dim=1)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        global_prefix_len: int = 0,
    ) -> torch.Tensor:
        decoder_input = tgt_ids[:, :-1]

        src_emb = self.token_embedding(src_ids)
        tgt_emb = self.token_embedding(decoder_input)

        src_emb = self._apply_src_positional_encoding(src_emb, global_prefix_len)
        tgt_emb = tgt_emb + sinusoidal_positional_encoding(
            tgt_emb.size(1),
            self.d_model,
            device=tgt_emb.device,
            dtype=tgt_emb.dtype,
        )

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            decoder_input.size(1),
            device=decoder_input.device,
        )

        src_key_padding_mask = src_ids.eq(self.pad_id)
        tgt_key_padding_mask = decoder_input.eq(self.pad_id)

        hidden = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        logits = self.lm_head(hidden)
        return logits

    def cross_entropy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        labels = targets[:, 1:].contiguous()
        return self.criterion(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )


if __name__ == '__main__':
    model = Seq2SeqTransformer(
        vocab_size=100,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_model=128,
        n_heads=8,
        context_length=10,
        pad_id=0,
    )
    print(model)

    src_ids = torch.randint(0, 100, (2, 10))
    tgt_ids = torch.randint(0, 100, (2, 10))
    logits = model(src_ids, tgt_ids)
    loss = model.cross_entropy_loss(logits, tgt_ids)
    print(logits.shape)
    print(loss)
