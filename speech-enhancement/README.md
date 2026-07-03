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
