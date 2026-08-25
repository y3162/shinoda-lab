"""
DEMAND
└── <noise_type>
    ├── ch01.wav # microphone 1 (raw)
    ├── ch01_train_000.wav # clipped
    ...
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from src.config import DEMAND_ROOT

_CLIP_NAME_RE = re.compile(
    r'^ch01_(?P<split>train|dev|test)_(?P<index>\d+)\.wav$'
)


def find_all_audio_files() -> Iterator[Path]:
    yield from DEMAND_ROOT.rglob('*.wav')


def find_clipped_audio_files() -> Iterator[Path]:
    """Yield only split-clipped wavs (ch01_{train|dev|test}_*.wav)."""
    for path in find_all_audio_files():
        if parse_clip_split(path) is not None:
            yield path


def get_noise_type(
    audio_file: Path,
) -> str:
    return audio_file.parent.name


def parse_clip_split(audio_file: Path) -> str | None:
    match = _CLIP_NAME_RE.match(audio_file.name)
    if not match:
        return None
    return match.group('split')
