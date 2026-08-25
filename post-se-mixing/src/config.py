import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ['PROJECT_ROOT'])


def resolve_project_path(path: str | Path) -> Path:
    """Resolve database/config paths relative to the project root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


LIBRISPEECH_ROOT    = PROJECT_ROOT / './data/raw/LibriSpeech'
DEMAND_CLIPPED_ROOT = PROJECT_ROOT / './data/processed/DEMAND'
DEMAND_ROOT         = PROJECT_ROOT / './data/raw/DEMAND' if not DEMAND_CLIPPED_ROOT.exists() else DEMAND_CLIPPED_ROOT
SQL_ROOT            = PROJECT_ROOT / './data/sql/database.duckdb'
PARQUET_ROOT        = PROJECT_ROOT / './data/parquet'

DEFAULT_SAMPLE_RATE = 16000
