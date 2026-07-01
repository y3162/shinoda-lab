## Virtual Environment Setup

### Environment Activation
```bash
uv venv ./.venv/nemo_asr_env --python 3.10
source ./.venv/nemo_asr_env/bin/activate
```

### Install Dependencies
```bash
uv pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    "nemo_toolkit[asr]" \
    evaluate \
    jiwer \
    num2words \
    cmudict \
    g2p_en \
    duckdb \
    duckdb-cli \
    "setuptools<82" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple
```

---

## Dataset Preparation

### Speech Dataset

- [ ] [LibriSpeech](https://www.openslr.org/12).

```bash
ln -s /path/to/librispeech data/raw/
```

### Noise Dataset

- [ ] [DEMAND](https://dcase-repo.github.io/dcase_datalist/datasets/scenes/demand.html)

```bash
ln -s /path/to/demand data/raw/
```
