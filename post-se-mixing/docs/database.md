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

### Insert Run IDs per Noise Config

```bash
python -m src.database.04_2_insert_run_id_per_noise_config \
    --noise_config_ids 26225,26241,26257 \
    --splits test-clean \
    --se_model_name semamba_pp \
    --checkpoint_dir data/checkpoints/YYYYMMDD_HHMMSS \
    --checkpoint_name g_best \
    --asr_model_name parakeet-tdt-0.6b-v2 \
    --noise_seed 0
```

---

## Create Observation ASR Results Table

Filled rows only (no empty placeholders).
Each row is one ASR result for `(run, utterance, mixture_family, mixture_coeff)`.

- `mixture_family`: `oa` (`obs = enhanced + coeff * noisy`) or `linear` (`obs = coeff * noisy + (1 - coeff) * enhanced`)
- Parquet fill writes results to disk first; import into this table is not implemented yet.

```bash
python -m src.database.05_1_create_table_observation_asr_results
```

---

## Write Observation ASR Results to Parquet for Run ID(s)

Processes all utterances in each run's `split`. SE and ASR are mini-batched.
Existing `(run, utterance, linear, coeff)` rows in Parquet are skipped.
Linear coeffs: `-0.5, -0.4, ..., 1.5`.

Output layout:

```text
data/parquet/observation_asr_results/run_id={id}/part-{batch:06d}.parquet
```

Columns (same as `observation_asr_results` without `id`):

`run_id, utterance_id, mixture_family, mixture_coeff, hypothesis, wer, n_errors, n_ref_words`

Multiple fill processes may run concurrently when their `run_id` sets do not overlap.
Do not run two processes on the same `run_id`.

```bash
# all run ids in observation_eval_runs (default; test → dev → train)
CUDA_VISIBLE_DEVICES=0 bash ./commands/fill_observation_asr_results.sh

# test-clean runs only (run_id 1–18)
CUDA_VISIBLE_DEVICES=0 bash ./commands/fill_observation_asr_results_test_clean.sh

# dev-clean runs only (run_id 19–36)
CUDA_VISIBLE_DEVICES=0 bash ./commands/fill_observation_asr_results_dev_clean.sh

# train-clean-100 (4 shards; run_id 37–54 → 5+5+5+3)
SHARD=0 CUDA_VISIBLE_DEVICES=0 bash ./commands/fill_observation_asr_results_train_clean_100.sh
SHARD=1 CUDA_VISIBLE_DEVICES=1 bash ./commands/fill_observation_asr_results_train_clean_100.sh
SHARD=2 CUDA_VISIBLE_DEVICES=2 bash ./commands/fill_observation_asr_results_train_clean_100.sh
SHARD=3 CUDA_VISIBLE_DEVICES=3 bash ./commands/fill_observation_asr_results_train_clean_100.sh

# train-clean-360 runs only (run_id 55–72)
CUDA_VISIBLE_DEVICES=0 bash ./commands/fill_observation_asr_results_train_clean_360.sh

# or explicitly
RUN_ID=all BATCH_SIZE=4 CUDA_VISIBLE_DEVICES=0 \
    bash ./commands/fill_observation_asr_results.sh

# selected run ids
RUN_ID=1,2,3 BATCH_SIZE=4 CUDA_VISIBLE_DEVICES=0 \
    bash ./commands/fill_observation_asr_results.sh
```

Query Parquet with DuckDB:

```sql
SELECT run_id, COUNT(*)
FROM read_parquet('data/parquet/observation_asr_results/run_id=1/part-*.parquet')
GROUP BY run_id;
```

Or:

```bash
source ./commands/export.sh
source ./.venv/base_env/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m src.database.05_2_write_observation_asr_results_parquet \
    --run_id 1 \
    --batch_size 4 \
    --device cuda:0
```
