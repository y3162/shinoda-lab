"""
LibriSpeech
└── <subset_name>
    └── <speaker_id>
        └── <section_id>
            ├── <speaker_id>-<section_id>-0000.flac
            ├── <speaker_id>-<section_id>-0001.flac
            ...
            └── <speaker_id>-<section_id>.trans.txt
"""

from pathlib import Path
from typing import (
    List,
    Tuple,
    Iterator,
)
from src.config import LIBRISPEECH_ROOT


def find_all_transcript_files() -> Iterator[Path]:
    yield from LIBRISPEECH_ROOT.rglob("*.trans.txt")


def parse_transcript_file(
    transcript_file: Path,
) -> List[Tuple[Path, str]]:
    results = []
    with open(transcript_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            audio_path = transcript_file.parent / (parts[0] + '.flac')
            transcript = ' '.join(parts[1:])
            results.append((audio_path, transcript))
    return results


def get_section_id(
    audio_path: Path,
) -> str:
    section_dir = audio_path.parent
    section_id = section_dir.name
    return section_id


def get_speaker_id(
    audio_path: Path,
) -> str:
    speaker_dir = audio_path.parent.parent
    speaker_id = speaker_dir.name
    return speaker_id


def get_subset_name(
    audio_path: Path,
) -> str:
    subset_dir = audio_path.parent.parent.parent
    subset_name = subset_dir.name
    return subset_name
