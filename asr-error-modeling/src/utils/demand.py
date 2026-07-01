"""
DEMAND
└── <noise_type>
    ├── ch01.wav # microphone 1
    ├── ch02.wav # microphone 2
    ...
"""

from pathlib import Path
from typing import Iterator
from src.config import DEMAND_ROOT


def find_all_audio_files() -> Iterator[Path]:
    yield from DEMAND_ROOT.rglob("*.wav")


def get_noise_type(
    audio_file: Path,
) -> str:
    return audio_file.parent.name
