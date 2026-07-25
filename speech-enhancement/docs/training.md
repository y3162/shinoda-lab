## Train *MP-SENet* (Conformer, VoiceBank+DEMAND)

Defaults are loaded from `src/mp_senet/configs/conformer.json`.
Nested keys can be overridden with dotted CLI flags (for example `--train.env.batch_size 4`).

```bash
python -m src.mp_senet.train \
    --config src/mp_senet/configs/conformer.json
```

Checkpoints, `config.json`, and TensorBoard logs are written under
`data/checkpoints/mp_senet/<YYYYMMDD_HHMMSS>/`.
Stdout progress uses `print_log` and tqdm; TensorBoard is under `<run>/logs/`.

Optional: `--train.env.max_steps N` stops after `N` optimizer steps (useful for smoke tests).

### Resume

Resume creates a **new** run directory (non-destructive), copies
`g_latest` / `do_latest` / `g_best` / `logs/` from the previous run, and continues.
CLI overrides are allowed.

```bash
python -m src.mp_senet.train \
    --resume data/checkpoints/mp_senet/<YYYYMMDD_HHMMSS>
```

---

## Train *MP-SENet* (Transformer, VoiceBank+DEMAND)

```bash
python -m src.mp_senet.train \
    --config src/mp_senet/configs/transformer.json
```

---

## Train *MP-SENet* (LibriSpeech)

Set dataset and splits via config or CLI:

```bash
python -m src.mp_senet.train \
    --config src/mp_senet/configs/conformer.json \
    --data.dataset librispeech \
    --data.librispeech.train_splits train-clean-360 \
    --data.librispeech.validation_splits dev-clean
```

If `data.librispeech.sql_root` is null, `SQL_ROOT` from `src.config` is used.
If `data.librispeech.noise_config_ids` is null, all non-clean noise configs are used.

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

## Train *SEMamba* (advanced, VoiceBank+DEMAND)

Activate `se_mamba_pp_env` (shared with SEMamba++). Defaults are in `src/se_mamba/configs/advanced.json`.
PCS training target: `src/se_mamba/configs/advanced_pcs.json` (`train.use_pcs400=true`).

```bash
source .venv/se_mamba_pp_env/bin/activate
source commands/export.sh

torchrun --standalone --nproc_per_node=1 -m src.se_mamba.train \
    --config src/se_mamba/configs/advanced.json
```

Checkpoints under `data/checkpoints/se_mamba/<YYYYMMDD_HHMMSS>/`.
Uses synchronous sequential `batch_pesq` for the MetricDiscriminator (same loss form as upstream SEMamba).

### Resume

```bash
torchrun --standalone --nproc_per_node=1 -m src.se_mamba.train \
    --resume data/checkpoints/se_mamba/<YYYYMMDD_HHMMSS>
```

### Inference

```bash
python -m src.se_mamba.infer \
    --checkpoint data/checkpoints/se_mamba/<YYYYMMDD_HHMMSS> \
    --input_folder path/to/noisy_wavs \
    --output_folder path/to/enhanced \
    --post_processing_pcs false
```

`--checkpoint` may be a run directory (`g_best` preferred, else `g_latest`) or a checkpoint file.
Omit `--config` to load `<run>/config.json`.

---

## Multi-GPU

Trainers support DDP via `torchrun`.

```bash
torchrun --nproc_per_node=2 -m src.mp_senet.train \
    --config src/mp_senet/configs/conformer.json
```

```bash
torchrun --nproc_per_node=2 -m src.se_mamba.train \
    --config src/se_mamba/configs/advanced.json
```

```bash
torchrun --nproc_per_node=2 -m src.se_mamba_pp.train \
    --config src/se_mamba_pp/configs/default.json
```
