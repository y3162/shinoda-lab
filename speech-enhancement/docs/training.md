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

Defaults are loaded from `src/se_mamba_pp/configs/default.json`.
Nested keys can be overridden with dotted CLI flags (for example `--train.env.batch_size 4`).

```bash
python -m src.se_mamba_pp.train \
    --config src/se_mamba_pp/configs/default.json
```

Checkpoints, `config.json`, and TensorBoard logs are written under
`data/checkpoints/se_mamba_pp/<YYYYMMDD_HHMMSS>/`.
Stdout progress uses `print_log` and tqdm; TensorBoard is under `<run>/logs/`.

Optional: `--train.env.max_steps N` stops after `N` optimizer steps (useful for smoke tests).

### Resume

Resume creates a **new** run directory (non-destructive), copies
`g_latest` / `do_latest` / `g_best` / `logs/` from the previous run, and continues.
CLI overrides are allowed.

```bash
python -m src.se_mamba_pp.train \
    --resume data/checkpoints/se_mamba_pp/<YYYYMMDD_HHMMSS>
```

### LibriSpeech

Set dataset and splits via config or CLI:

```bash
python -m src.se_mamba_pp.train \
    --config src/se_mamba_pp/configs/default.json \
    --data.dataset librispeech \
    --data.librispeech.train_splits train-clean-360 \
    --data.librispeech.validation_splits dev-clean
```

If `data.librispeech.sql_root` is null, `SQL_ROOT` from `src.config` is used.

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
