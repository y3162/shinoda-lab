## Train *MP-SENet* (Conformer, VoiceBank+DEMAND)

Defaults are loaded from the config JSON.
`--dataset` defaults to `voicebank`.

```bash
python -m src.mp_senet.train \
    --config src/mp_senet/configs/conformer.json
```

Checkpoints are written under `data/checkpoints/mp_senet/<YYYYMMDD_HHMMSS>/`.

---

## Train *MP-SENet* (Transformer, VoiceBank+DEMAND)

```bash
python -m src.mp_senet.train \
    --config src/mp_senet/configs/transformer.json
```

---

## Train *MP-SENet* (LibriSpeech)

`--train_splits` and `--validation_splits` are required for `--dataset librispeech`.
If `--noise_config_ids` is omitted, all non-clean noise configs are used.

```bash
python -m src.mp_senet.train \
    --dataset librispeech \
    --config src/mp_senet/configs/conformer.json \
    --train_splits train-clean-360 \
    --validation_splits dev-clean
```

---

## Train *SEMamba++* (VoiceBank+DEMAND)

```bash
python -m src.se_mamba_pp.train \
    --config src/se_mamba_pp/configs/default.json
```

Checkpoints are written under `data/checkpoints/se_mamba_pp/<YYYYMMDD_HHMMSS>/`.

---

## Train *SEMamba++* (LibriSpeech)

```bash
python -m src.se_mamba_pp.train \
    --dataset librispeech \
    --config src/se_mamba_pp/configs/default.json \
    --train_splits train-clean-360 \
    --validation_splits dev-clean
```

---

## Multi-GPU

Both trainers support DDP via `torchrun`.

```bash
torchrun --nproc_per_node=2 -m src.mp_senet.train \
    --config src/mp_senet/configs/conformer.json
```

```bash
torchrun --nproc_per_node=2 -m src.se_mamba_pp.train \
    --config src/se_mamba_pp/configs/default.json
```
