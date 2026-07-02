import os
from pathlib import Path

LIBRISPEECH_ROOT = Path(os.environ['LIBRISPEECH_ROOT'])
DEMAND_CLIPPED_ROOT = Path(os.environ['DEMAND_CLIPPED_ROOT'])
DEMAND_ROOT = DEMAND_CLIPPED_ROOT if DEMAND_CLIPPED_ROOT.exists() else Path(os.environ['DEMAND_ROOT'])
SQL_ROOT = Path(os.environ['SQL_ROOT'])
PARQUET_ROOT = Path(os.environ['PARQUET_ROOT'])

DEFAULT_SAMPLE_RATE = 16000
