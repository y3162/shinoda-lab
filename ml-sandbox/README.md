## Virtual Environment Setup

### Default Environment

#### Environment Activation

```bash
uv venv ./.venv/default_env --python 3.10
source ./.venv/default_env/bin/activate
```

#### Install Dependencies

```bash
uv pip install \
    torch==2.6.0 \
    numpy==2.2.6 \
    matplotlib==3.10.9 \
    tqdm \
    PyYAML \
    packaging \
    tensorboard \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple
```
