## Virtual Environment Setup

### MP-SENet

#### Environment Activation

```bash
uv venv ./.venv/mp_senet_env --python 3.10
source ./.venv/mp_senet_env/bin/activate
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
    "setuptools<82" \
      numpy \
    librosa \
    scipy \
    tensorboard \
    matplotlib \
    SoundFile \
    einops \
    joblib \
    natsort \
    pesq \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple
```

### SEMamba++

#### Environment Activation

```bash
uv venv ./.venv/se_mamba_pp_env --python 3.10
source ./.venv/se_mamba_pp_env/bin/activate
```

#### Install Dependencies

```bash
uv pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    triton==3.2.0 \
    "setuptools<82" \
    packaging \
    ninja \
    wheel \
    numpy==2.2.6 \
    scipy==1.15.3 \
    librosa==0.11.0 \
    matplotlib==3.10.9 \
    SoundFile==0.13.1 \
    einops==0.8.2 \
    joblib \
    natsort \
    PyYAML \
    tqdm \
    easydict \
    auraloss==0.4.0 \
    nnAudio==0.3.4 \
    pyroomacoustics \
    torchmetrics \
    transformers \
    accelerate \
    wandb \
    tensorboard \
    pesq==0.0.4 \
    evaluate \
    jiwer \
    num2words \
    cmudict \
    g2p_en \
    duckdb \
    duckdb-cli \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple
uv pip install --no-deps \
    "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
```

## Dataset Preparation

### Speech Dataset

- [ ] [LibriSpeech](https://www.openslr.org/12)

```bash
ln -s /path/to/librispeech data/raw/
```

### Noisy Speech Dataset

- [ ] [VoiceBank + DEMAND](https://datashare.ed.ac.uk/items/6ed35425-bf14-4d2b-93a1-0a4984952757)

```bash
ln -s /path/to/voicebank+demand data/raw/
```

### Noise Dataset

- [ ] [DEMAND](https://dcase-repo.github.io/dcase_datalist/datasets/scenes/demand.html)

```bash
ln -s /path/to/demand data/raw/
```
