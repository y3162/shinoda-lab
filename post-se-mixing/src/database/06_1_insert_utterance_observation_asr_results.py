from __future__ import annotations

import argparse
import gc
from pathlib import Path

import duckdb as db
import torch
import torchaudio

from src.config import (
    DEFAULT_SAMPLE_RATE,
    PROJECT_ROOT,
    SQL_ROOT,
    resolve_project_path,
)
from src.utils.mixture import mix_linear
from src.utils.noise import NoiseGenerator, get_noise_option
from src.utils.print import print_error, print_log
from src.utils.wer import norm, utterance_errors


MIXTURE_FAMILY = 'linear'
LINEAR_COEFFS = [round(x * 0.1, 1) for x in range(-5, 16)]

INSERT_SQL = """
    INSERT INTO observation_asr_results (
        run_id,
        utterance_id,
        mixture_family,
        mixture_coeff,
        hypothesis,
        wer,
        n_errors,
        n_ref_words
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""


def checkpoint_dir_for_db(checkpoint_dir: str | Path) -> str:
    resolved = resolve_project_path(checkpoint_dir).resolve()
    return str(resolved.relative_to(PROJECT_ROOT.resolve()))


def load_mono_16k(audio_path: Path) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(str(audio_path))
    if sample_rate != DEFAULT_SAMPLE_RATE:
        audio = torchaudio.functional.resample(
            audio,
            sample_rate,
            DEFAULT_SAMPLE_RATE,
        )
    if audio.dim() == 2:
        audio = audio.mean(dim=0)
    elif audio.dim() != 1:
        raise ValueError(
            f'Audio tensor must have 1 or 2 dimensions, but got {audio.dim()}',
        )
    return audio


def get_utterance(
    con: db.DuckDBPyConnection,
    utterance_id: int,
    split: str,
) -> tuple[str, str]:
    row = con.execute(
        """
        SELECT split, audio_path, transcript
        FROM utterances
        WHERE id = ?
        """,
        [utterance_id],
    ).fetchone()
    if row is None:
        print_error(f'utterance_id not found: {utterance_id}')
        raise SystemExit(1)
    utterance_split, audio_path, transcript = row
    if utterance_split != split:
        print_error(
            f'utterance split {utterance_split!r} does not match run split {split!r}',
        )
        raise SystemExit(1)
    if norm(transcript) == '':
        print_error(f'reference transcript is empty after normalization: utterance_id={utterance_id}')
        raise SystemExit(1)
    return str(audio_path), str(transcript)


def get_or_create_run(
    con: db.DuckDBPyConnection,
    *,
    se_model_name: str,
    checkpoint_dir: str,
    checkpoint_name: str,
    asr_model_name: str,
    noise_config_id: int,
    noise_seed: int,
    split: str,
) -> int:
    row = con.execute(
        """
        SELECT id
        FROM observation_eval_runs
        WHERE se_model_name = ?
          AND checkpoint_dir = ?
          AND checkpoint_name = ?
          AND asr_model_name = ?
          AND noise_config_id = ?
          AND noise_seed = ?
          AND split = ?
        """,
        [
            se_model_name,
            checkpoint_dir,
            checkpoint_name,
            asr_model_name,
            noise_config_id,
            noise_seed,
            split,
        ],
    ).fetchone()
    if row is not None:
        return int(row[0])

    inserted = con.execute(
        """
        INSERT INTO observation_eval_runs (
            se_model_name,
            checkpoint_dir,
            checkpoint_name,
            asr_model_name,
            noise_config_id,
            noise_seed,
            split
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        [
            se_model_name,
            checkpoint_dir,
            checkpoint_name,
            asr_model_name,
            noise_config_id,
            noise_seed,
            split,
        ],
    ).fetchone()
    return int(inserted[0])


def existing_linear_coeffs(
    con: db.DuckDBPyConnection,
    run_id: int,
    utterance_id: int,
) -> set[float]:
    rows = con.execute(
        """
        SELECT mixture_coeff
        FROM observation_asr_results
        WHERE run_id = ?
          AND utterance_id = ?
          AND mixture_family = ?
        """,
        [run_id, utterance_id, MIXTURE_FAMILY],
    ).fetchall()
    return {round(float(coeff), 1) for (coeff,) in rows}


def transcribe_mixes(
    asr_model,
    mixes: torch.Tensor,
) -> list[str]:
    try:
        texts = asr_model.transcribe(mixes)
        if isinstance(texts, str):
            return [texts]
        return texts
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print_log('ASR batch ran out of memory; falling back to one mix at a time')
        texts = []
        for i in range(mixes.size(0)):
            texts.append(asr_model.transcribe(mixes[i]))
        return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--utterance_id', type=int, required=True)
    parser.add_argument('--se_model_name', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--asr_model_name', type=str, required=True)
    parser.add_argument('--noise_config_id', type=int, required=True)
    parser.add_argument('--noise_seed', type=int, required=True)
    parser.add_argument('--split', type=str, required=True)
    args = parser.parse_args()

    if not SQL_ROOT.exists():
        print_error(f'SQL database does not exist at {SQL_ROOT}')
        raise SystemExit(1)

    checkpoint_dir_abs = resolve_project_path(args.checkpoint_dir)
    try:
        checkpoint_dir_rel = checkpoint_dir_for_db(checkpoint_dir_abs)
    except ValueError:
        print_error(
            f'checkpoint_dir must be under PROJECT_ROOT ({PROJECT_ROOT}): {checkpoint_dir_abs}',
        )
        raise SystemExit(1)

    con = db.connect(SQL_ROOT)

    audio_path, transcript = get_utterance(con, args.utterance_id, args.split)
    noise_exists = con.execute(
        """
        SELECT 1
        FROM noise_configs
        WHERE id = ?
        """,
        [args.noise_config_id],
    ).fetchone()
    if noise_exists is None:
        print_error(f'noise_config_id not found: {args.noise_config_id}')
        raise SystemExit(1)
    noise_option = get_noise_option(con, args.noise_config_id)
    run_id = get_or_create_run(
        con,
        se_model_name=args.se_model_name,
        checkpoint_dir=checkpoint_dir_rel,
        checkpoint_name=args.checkpoint_name,
        asr_model_name=args.asr_model_name,
        noise_config_id=args.noise_config_id,
        noise_seed=args.noise_seed,
        split=args.split,
    )
    missing = [
        coeff
        for coeff in LINEAR_COEFFS
        if coeff not in existing_linear_coeffs(con, run_id, args.utterance_id)
    ]
    if not missing:
        print_log(
            f'run_id={run_id} utterance_id={args.utterance_id}: '
            'all linear coefficients already exist; skip',
        )
        con.close()
        return

    print_log(
        f'run_id={run_id} utterance_id={args.utterance_id} '
        f'missing={len(missing)}/{len(LINEAR_COEFFS)}',
    )

    clean = load_mono_16k(resolve_project_path(audio_path))
    rng = torch.Generator()
    rng.manual_seed((args.noise_seed + args.utterance_id) & 0xFFFFFFFFFFFFFFFF)
    noisy = NoiseGenerator(noise_option).generate(
        clean.clone(),
        DEFAULT_SAMPLE_RATE,
        rng=rng,
    ).audio

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from src.se.api import SEModel
    se_model = SEModel(
        args.se_model_name,
        checkpoint_dir_abs,
        args.checkpoint_name,
        device=device,
    )
    enhanced = se_model.enhance(noisy).detach().cpu()
    del se_model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    mixes = []
    for coeff in missing:
        mixed = mix_linear(noisy, enhanced, coeff)
        if coeff == 0.0:
            length = min(noisy.shape[-1], enhanced.shape[-1])
            if not torch.equal(mixed, enhanced[..., :length]):
                print_error('coeff=0 mix does not equal enhanced')
                raise SystemExit(1)
        if coeff == 1.0:
            length = min(noisy.shape[-1], enhanced.shape[-1])
            if not torch.equal(mixed, noisy[..., :length]):
                print_error('coeff=1 mix does not equal noisy')
                raise SystemExit(1)
        mixes.append(mixed)
    batch = torch.stack(mixes, dim=0)

    from src.asr.api import ASRModel
    asr_model = ASRModel(args.asr_model_name, device=device)
    hypotheses = transcribe_mixes(asr_model, batch)
    del asr_model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    if len(hypotheses) != len(missing):
        print_error(
            f'ASR returned {len(hypotheses)} hypotheses for {len(missing)} mixes',
        )
        raise SystemExit(1)

    rows = []
    for coeff, hypothesis in zip(missing, hypotheses):
        n_errors, n_ref_words = utterance_errors(transcript, hypothesis)
        if n_ref_words == 0:
            print_error(
                f'reference has 0 words after normalization: utterance_id={args.utterance_id}',
            )
            raise SystemExit(1)
        rows.append((
            run_id,
            args.utterance_id,
            MIXTURE_FAMILY,
            coeff,
            hypothesis,
            n_errors / n_ref_words,
            n_errors,
            n_ref_words,
        ))

    con.executemany(INSERT_SQL, rows)
    print_log(f'inserted {len(rows)} observation_asr_results rows')
    con.close()


if __name__ == '__main__':
    main()
