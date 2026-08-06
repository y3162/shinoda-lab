"""Clip DEMAND ch01 into train/dev/test 20s segments at 16 kHz.

Regions (seconds on raw ch01.wav):
  train: 0-260 (random starts, default 90 clips / noise type)
  dev:   260-280 (single clip at region start)
  test:  280-300 (single clip at region start)

Output under DEMAND_CLIPPED_ROOT (always DEFAULT_SAMPLE_RATE):
  <noise_type>/ch01_{split}_{idx:03d}.wav
  <noise_type>/ch01_{split}_{idx:03d}.json  (metadata)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from src.config import PROJECT_ROOT, DEMAND_ROOT, DEMAND_CLIPPED_ROOT, DEFAULT_SAMPLE_RATE
from src.utils.demand import get_noise_type
from src.utils.print import print_error, print_log

SPLIT_REGIONS: dict[str, tuple[float, float]] = {
    'train': (0.0, 260.0),
    'dev': (260.0, 280.0),
    'test': (280.0, 300.0),
}


def _iter_raw_ch01() -> list[Path]:
    files = sorted(
        p for p in DEMAND_ROOT.rglob('ch01.wav') if p.is_file()
    )
    return files


def _save_clip(
    clip: torch.Tensor,
    sr: int,
    out_wav: Path,
    meta: dict,
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_wav), clip, sr)
    out_json = out_wav.with_suffix('.json')
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')


def _resample_if_needed(waveform: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return waveform
    return torchaudio.transforms.Resample(sr, target_sr)(waveform)


def clip_one_file(
    audio_file: Path,
    duration_sec: float,
    seed: int,
    train_n: int,
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> int:
    audio, sr = torchaudio.load(str(audio_file))
    noise_type = get_noise_type(audio_file)
    n_written = 0
    duration_frames = int(round(duration_sec * sr))
    total_frames = audio.shape[-1]
    out_duration_frames = int(round(duration_sec * target_sr))

    for split, (region_start_s, region_end_s) in SPLIT_REGIONS.items():
        region_start = int(round(region_start_s * sr))
        region_end = int(round(region_end_s * sr))
        region_end = min(region_end, total_frames)
        region_len = region_end - region_start
        if region_len < duration_frames:
            print_error(
                f'{noise_type}/{split}: region too short '
                f'({region_len / sr:.3f}s < {duration_sec}s)'
            )
            raise SystemExit(1)

        max_start_offset = region_len - duration_frames
        if split == 'train':
            rng = random.Random(seed + hash(noise_type) % 10_000_000)
            starts = [
                region_start + rng.randint(0, max_start_offset)
                for _ in range(train_n)
            ]
        else:
            # Region length == clip length => only the region start is valid.
            starts = [region_start]

        for idx, start in enumerate(starts):
            end = start + duration_frames
            clip = audio[..., start:end]
            clip = _resample_if_needed(clip, sr, target_sr)
            if clip.shape[-1] != out_duration_frames:
                # Guard against rounding drift after resample.
                if clip.shape[-1] > out_duration_frames:
                    clip = clip[..., :out_duration_frames]
                else:
                    clip = torch.nn.functional.pad(
                        clip, (0, out_duration_frames - clip.shape[-1]),
                    )
            name = f'ch01_{split}_{idx:03d}'
            out_wav = DEMAND_CLIPPED_ROOT / noise_type / f'{name}.wav'
            meta = {
                'noise_type': noise_type,
                'split': split,
                'index': idx,
                'start_sec': start / sr,
                'end_sec': end / sr,
                'duration_sec': duration_sec,
                'sample_rate': target_sr,
                'source_sample_rate': sr,
                'seed': seed,
                'source': str(audio_file.relative_to(PROJECT_ROOT)),
            }
            _save_clip(clip, target_sr, out_wav, meta)
            n_written += 1
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=20.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_n', type=int, default=90)
    parser.add_argument(
        '--target_sr',
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f'Output sample rate (default {DEFAULT_SAMPLE_RATE})',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Delete existing DEMAND_CLIPPED_ROOT before writing',
    )
    args = parser.parse_args()

    if not DEMAND_ROOT.is_dir():
        print_error(f'RAW DEMAND_ROOT missing: {DEMAND_ROOT}')
        raise SystemExit(1)

    if DEMAND_CLIPPED_ROOT.exists():
        if not args.force:
            print_error(
                f'DEMAND_CLIPPED_ROOT {DEMAND_CLIPPED_ROOT} already exists '
                '(pass --force to recreate)'
            )
            raise SystemExit(1)
        print_log(f'Removing existing DEMAND_CLIPPED_ROOT {DEMAND_CLIPPED_ROOT}')
        shutil.rmtree(DEMAND_CLIPPED_ROOT)

    print_log(f'Creating DEMAND_CLIPPED_ROOT {DEMAND_CLIPPED_ROOT}')
    print_log(
        f'duration={args.duration}s seed={args.seed} train_n={args.train_n} '
        f'target_sr={args.target_sr} regions={SPLIT_REGIONS}'
    )
    DEMAND_CLIPPED_ROOT.mkdir(parents=True, exist_ok=True)

    sources = _iter_raw_ch01()
    if not sources:
        print_error(f'No ch01.wav under {DEMAND_ROOT}')
        raise SystemExit(1)

    total = 0
    for audio_file in tqdm(sources, desc='Clipping DEMAND ch01'):
        total += clip_one_file(
            audio_file,
            duration_sec=float(args.duration),
            seed=int(args.seed),
            train_n=int(args.train_n),
            target_sr=int(args.target_sr),
        )
    print_log(f'Done. wrote {total} clips under {DEMAND_CLIPPED_ROOT}')


if __name__ == '__main__':
    main()
