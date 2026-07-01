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
python -m src.database.03_2_insert_noise_confgs_from_demand
```

---

## Create ASR Results Table

```bash
python -m src.database.04_1_create_table_asr_results
```
