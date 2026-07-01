## Virtual Environment Setup

### Parakeet TDT 0.6B v2

#### Environment Activation

```bash
uv venv ./.venv/nemo_asr_env --python 3.10
source ./.venv/nemo_asr_env/bin/activate
```

#### Install Dependencies

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

### Whisper Large V3

#### Environment Activation

```bash
uv venv ./.venv/whisper_env --python 3.10
source ./.venv/whisper_env/bin/activate
```

#### Install Dependencies

```bash
uv pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    evaluate \
    jiwer \
    num2words \
    cmudict \
    g2p_en \
    duckdb \
    duckdb-cli \
    transformers \
    accelerate \
    safetensors \
    pandas \
    pyarrow \
    tqdm \
    packaging \
    psutil \
    ninja \
    wheel \
    setuptools \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
uv pip install protobuf
```
---

## Dataset Preparation

### Speech Dataset

- [ ] [LibriSpeech](https://www.openslr.org/12)

```bash
ln -s /path/to/librispeech data/raw/
```

### Noise Dataset

- [ ] [DEMAND](https://dcase-repo.github.io/dcase_datalist/datasets/scenes/demand.html)

```bash
ln -s /path/to/demand data/raw/
```
