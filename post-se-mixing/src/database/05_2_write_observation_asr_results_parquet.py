from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path


def _limit_cpu_threads() -> None:
    for key, value in (
        ('OMP_NUM_THREADS', '1'),
        ('MKL_NUM_THREADS', '1'),
        ('OPENBLAS_NUM_THREADS', '1'),
        ('NUMEXPR_NUM_THREADS', '1'),
        ('TORCH_NUM_THREADS', '1'),
        ('TORCH_NUM_INTEROP_THREADS', '1'),
        ('TORCHINDUCTOR_COMPILE_THREADS', '1'),
        ('TORCHDYNAMO_DISABLE', '1'),
        ('TOKENIZERS_PARALLELISM', 'false'),
    ):
        os.environ.setdefault(key, value)


_limit_cpu_threads()

import duckdb as db
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError as exc:
    # Already initialized elsewhere; keep going but surface the failure.
    import sys
    print(
        f'[WARNING] torch.set_num_interop_threads(1) failed: {exc}',
        file=sys.stderr,
    )

import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from src.asr.api import ASRModel
from src.config import (
    DEFAULT_SAMPLE_RATE,
    SQL_ROOT,
    resolve_project_path,
)
from src.se.api import SEModel
from src.utils.mixture import mix_linear
from src.utils.noise import NoiseGenerator, get_noise_option
from src.utils.observation_asr_results_parquet import (
    load_existing_coeffs,
    next_batch_index,
    next_part_path,
    run_dir,
    write_batch,
)
from src.utils.print import print_error, print_log, print_warning
from src.utils.wer import norm, utterance_errors


MIXTURE_FAMILY = 'linear'
LINEAR_COEFFS = [round(x * 0.1, 1) for x in range(-5, 16)]


def parse_int_list(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(',') if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError('expected at least one integer')
    return [int(value) for value in values]


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


def pad_waveforms(waveforms: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.stack(
        [F.pad(w, (0, max_len - w.shape[-1])) for w in waveforms],
        dim=0,
    )
    return padded, lengths


def get_run(con: db.DuckDBPyConnection, run_id: int) -> dict:
    row = con.execute(
        """
        SELECT
            id,
            se_model_name,
            checkpoint_dir,
            checkpoint_name,
            asr_model_name,
            noise_config_id,
            noise_seed,
            split
        FROM observation_eval_runs
        WHERE id = ?
        """,
        [run_id],
    ).fetchone()
    if row is None:
        print_error(f'run_id not found: {run_id}')
        raise SystemExit(1)
    return {
        'id': int(row[0]),
        'se_model_name': str(row[1]),
        'checkpoint_dir': str(row[2]),
        'checkpoint_name': str(row[3]),
        'asr_model_name': str(row[4]),
        'noise_config_id': int(row[5]),
        'noise_seed': int(row[6]),
        'split': str(row[7]),
    }


def model_cache_key(run: dict) -> tuple[str, str, str, str]:
    return (
        run['se_model_name'],
        run['checkpoint_dir'],
        run['checkpoint_name'],
        run['asr_model_name'],
    )


def list_pending_utterances(
    con: db.DuckDBPyConnection,
    run_id: int,
    split: str,
) -> list[tuple[int, str, str, list[float]]]:
    rows = con.execute(
        """
        SELECT id, audio_path, transcript
        FROM utterances
        WHERE split = ?
        ORDER BY id
        """,
        [split],
    ).fetchall()

    existing_by_utterance = load_existing_coeffs(run_id, MIXTURE_FAMILY)

    pending: list[tuple[int, str, str, list[float]]] = []
    for utterance_id, audio_path, transcript in rows:
        utterance_id = int(utterance_id)
        transcript = str(transcript)
        if norm(transcript) == '':
            print_warning(
                f'skip empty reference after normalization: utterance_id={utterance_id}',
            )
            continue
        existing = existing_by_utterance.get(utterance_id, set())
        missing = [coeff for coeff in LINEAR_COEFFS if coeff not in existing]
        if not missing:
            continue
        pending.append((utterance_id, str(audio_path), transcript, missing))
    return pending


def generate_noisy(
    clean: torch.Tensor,
    noise_option: dict,
    noise_seed: int,
    utterance_id: int,
) -> torch.Tensor:
    rng = torch.Generator()
    rng.manual_seed((noise_seed + utterance_id) & 0xFFFFFFFFFFFFFFFF)
    return NoiseGenerator(noise_option).generate(
        clean.clone(),
        DEFAULT_SAMPLE_RATE,
        rng=rng,
    ).audio


def transcribe_padded(
    asr_model: ASRModel,
    mixes: list[torch.Tensor],
) -> list[str]:
    if not mixes:
        return []
    batch, lengths = pad_waveforms(mixes)
    try:
        texts = asr_model.transcribe(batch, lengths=lengths)
        if isinstance(texts, str):
            return [texts]
        return texts
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print_log('ASR batch OOM; falling back to one mix at a time')
        texts = []
        for mix in mixes:
            texts.append(asr_model.transcribe(mix))
        return texts


def process_batch(
    *,
    run: dict,
    batch_items: list[tuple[int, str, str, list[float]]],
    noise_option: dict,
    se_model: SEModel,
    asr_model: ASRModel,
    part_path: Path,
) -> int:
    device = se_model.device
    noisys: list[torch.Tensor] = []
    for utterance_id, audio_path, _transcript, _missing in batch_items:
        clean = load_mono_16k(resolve_project_path(audio_path))
        noisy = generate_noisy(
            clean,
            noise_option,
            run['noise_seed'],
            utterance_id,
        )
        noisys.append(noisy)

    noisy_batch, noisy_lengths = pad_waveforms(noisys)
    try:
        enhanced_batch = se_model.enhance(noisy_batch, lengths=noisy_lengths).detach()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print_log('SE batch OOM; falling back to one utterance at a time')
        enhanced_unpadded = [
            se_model.enhance(noisy).detach()
            for noisy in noisys
        ]
    else:
        enhanced_unpadded = [
            enhanced_batch[i, : int(noisy_lengths[i].item())]
            for i in range(len(batch_items))
        ]

    all_mixes: list[torch.Tensor] = []
    mix_index: list[tuple[int, float]] = []
    for i, (_utterance_id, _audio_path, _transcript, missing) in enumerate(batch_items):
        noisy = noisys[i].to(device, non_blocking=True)
        enhanced = enhanced_unpadded[i]
        if enhanced.device != device:
            enhanced = enhanced.to(device, non_blocking=True)
        for coeff in missing:
            all_mixes.append(mix_linear(noisy, enhanced, coeff))
            mix_index.append((i, coeff))

    hypotheses = transcribe_padded(asr_model, all_mixes)
    if len(hypotheses) != len(mix_index):
        print_error(
            f'ASR returned {len(hypotheses)} hypotheses for {len(mix_index)} mixes',
        )
        raise SystemExit(1)

    rows = []
    hyp_offset = 0
    for utterance_id, _audio_path, transcript, missing in batch_items:
        for coeff in missing:
            hypothesis = hypotheses[hyp_offset]
            hyp_offset += 1
            n_errors, n_ref_words = utterance_errors(transcript, hypothesis)
            if n_ref_words == 0:
                print_warning(
                    f'skip write: n_ref_words=0 utterance_id={utterance_id}',
                )
                continue
            rows.append((
                run['id'],
                utterance_id,
                MIXTURE_FAMILY,
                coeff,
                hypothesis,
                n_errors / n_ref_words,
                n_errors,
                n_ref_words,
            ))

    write_batch(part_path, rows)
    return len(rows)


def fill_run(
    *,
    con: db.DuckDBPyConnection,
    run: dict,
    batch_size: int,
    se_model: SEModel,
    asr_model: ASRModel,
) -> None:
    run_id = run['id']
    noise_option = get_noise_option(con, run['noise_config_id'])
    pending = list_pending_utterances(con, run_id, run['split'])
    if not pending:
        print_log(f'run_id={run_id}: nothing to do')
        return

    print_log(
        f'run_id={run_id} split={run["split"]} '
        f'pending_utterances={len(pending)} batch_size={batch_size}',
    )

    run_dir_path = run_dir(run_id)
    wrote_total = 0
    batch_index = next_batch_index(run_id)
    for start in tqdm(range(0, len(pending), batch_size), desc=f'run {run_id}'):
        batch_items = pending[start:start + batch_size]
        part_path = next_part_path(run_dir_path, batch_index)
        wrote_total += process_batch(
            run=run,
            batch_items=batch_items,
            noise_option=noise_option,
            se_model=se_model,
            asr_model=asr_model,
            part_path=part_path,
        )
        batch_index += 1

    print_log(f'run_id={run_id}: wrote {wrote_total} rows to {run_dir_path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Write observation ASR results to Parquet for registered run_id(s). '
            'SE/ASR are batched over utterances; existing linear coeffs in '
            'Parquet are skipped.'
        ),
    )
    parser.add_argument(
        '--run_id',
        type=parse_int_list,
        required=True,
        help='Comma-separated observation_eval_runs.id values',
    )
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        print_error(f'batch_size must be >= 1, got {args.batch_size}')
        raise SystemExit(1)

    if not SQL_ROOT.exists():
        print_error(f'SQL database does not exist at {SQL_ROOT}')
        raise SystemExit(1)

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print_log(
        f'torch threads={torch.get_num_threads()} '
        f'interop={torch.get_num_interop_threads()}',
    )

    con = db.connect(str(SQL_ROOT), read_only=True)
    se_model: SEModel | None = None
    asr_model: ASRModel | None = None
    loaded_key: tuple[str, str, str, str] | None = None
    try:
        for run_id in args.run_id:
            run = get_run(con, run_id)
            checkpoint_dir = resolve_project_path(run['checkpoint_dir'])
            if not (checkpoint_dir / 'config.json').is_file():
                print_error(
                    f'config not found under checkpoint_dir: {checkpoint_dir}',
                )
                raise SystemExit(1)

            key = model_cache_key(run)
            if key != loaded_key:
                if se_model is not None:
                    del se_model
                    del asr_model
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                print_log(
                    f'loading models: se={run["se_model_name"]} '
                    f'asr={run["asr_model_name"]} ckpt={run["checkpoint_name"]}',
                )
                se_model = SEModel(
                    run['se_model_name'],
                    checkpoint_dir,
                    run['checkpoint_name'],
                    device=device,
                )
                asr_model = ASRModel(run['asr_model_name'], device=device)
                loaded_key = key

            assert se_model is not None and asr_model is not None
            fill_run(
                con=con,
                run=run,
                batch_size=args.batch_size,
                se_model=se_model,
                asr_model=asr_model,
            )
    finally:
        if se_model is not None:
            del se_model
        if asr_model is not None:
            del asr_model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        con.close()


if __name__ == '__main__':
    main()
