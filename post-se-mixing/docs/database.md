## (optional) Clip *DEMAND* Dataset

Loading a long noise audio file takes a lot of time.
To speed up the loading process, we can clip the noise audio file into shorter segments.

```bash
python -m src.database.00_clip_demand --duration 20
```

---

## Create Utterances Table

```bash
python -m src.database.01_1_create_table_utterances
```

### Insert Utterances from *LibriSpeech* Dataset

```bash
python -m src.database.01_2_insert_utterances_from_librispeech
```

---

## Create Noises Table

```bash
python -m src.database.02_1_create_table_noises
```

### Insert Noises from *DEMAND* Dataset

```bash
python -m src.database.02_2_insert_noises_from_demand
```

---

## Create Noise Configs Table

```bash
python -m src.database.03_1_create_table_noise_configs
```

### Insert Noise Configs from *DEMAND* Dataset

```bash
python -m src.database.03_2_insert_noise_configs_from_demand
```

---

## Create Observation Eval Runs Table

One row is one evaluation run: SE model / checkpoint, ASR model, noise config, noise seed, and utterance split.
`checkpoint_dir` must be stored relative to `PROJECT_ROOT`.

```bash
python -m src.database.04_1_create_table_observation_eval_runs
```

---

## Create Observation ASR Results Table

Filled rows only (no empty placeholders).
Each row is one ASR result for `(run, utterance, mixture_family, mixture_coeff)`.

- `mixture_family`: `oa` (`obs = enhanced + coeff * noisy`) or `linear` (`obs = coeff * noisy + (1 - coeff) * enhanced`)
- Training / analysis should use existing rows in this table (computed results only).

```bash
python -m src.database.05_1_create_table_observation_asr_results
```
