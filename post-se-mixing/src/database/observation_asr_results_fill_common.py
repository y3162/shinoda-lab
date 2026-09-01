from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import duckdb as db
import torch
import torch.nn.functional as F
import torchaudio

from src.asr.api import ASRModel
from src.config import DEFAULT_SAMPLE_RATE, SQL_ROOT, resolve_project_path
from src.se.api import SEModel
from src.utils.mixture import mix_linear
from src.utils.noise import NoiseGenerator, get_noise_option
from src.utils.observation_asr_results_parquet import (
    DEFAULT_LINEAR_COEFFS,
    load_existing_coeffs,
    next_batch_index,
    next_part_path,
    run_dir,
    write_batch,
)
from src.utils.print import print_error, print_log, print_warning
from src.utils.wer import norm, utterance_errors


MIXTURE_FAMILY = 'linear'
LINEAR_COEFFS = DEFAULT_LINEAR_COEFFS

BatchItem = tuple[int, str, str, list[float]]


def limit_cpu_threads() -> None:
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


def init_torch_threads() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        import sys
        print(
            f'[WARNING] torch.set_num_interop_threads(1) failed: {exc}',
            file=sys.stderr,
        )


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
) -> list[BatchItem]:
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

    pending: list[BatchItem] = []
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


def transcribe_one_mix(
    asr_model: ASRModel,
    mix: torch.Tensor,
    *,
    utterance_id: int,
    coeff: float,
) -> str | None:
    try:
        return asr_model.transcribe(mix)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print_warning(
            f'ASR OOM for utterance_id={utterance_id} coeff={coeff}; skipping coeff',
        )
        return None


def transcribe_padded(
    asr_model: ASRModel,
    mixes: list[torch.Tensor],
    *,
    mix_meta: list[tuple[int, float]] | None = None,
) -> list[str | None]:
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
        texts: list[str | None] = []
        for idx, mix in enumerate(mixes):
            if mix_meta is not None:
                utterance_id, coeff = mix_meta[idx]
            else:
                utterance_id, coeff = -1, float('nan')
            texts.append(
                transcribe_one_mix(
                    asr_model,
                    mix,
                    utterance_id=utterance_id,
                    coeff=coeff,
                ),
            )
        return texts


def enhance_noisys(
    se_model: SEModel,
    noisys: list[torch.Tensor],
) -> list[torch.Tensor]:
    noisy_batch, noisy_lengths = pad_waveforms(noisys)
    try:
        enhanced_batch = se_model.enhance(noisy_batch, lengths=noisy_lengths).detach()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print_log('SE batch OOM; falling back to one utterance at a time')
        return [se_model.enhance(noisy).detach() for noisy in noisys]
    return [
        enhanced_batch[i, : int(noisy_lengths[i].item())]
        for i in range(len(noisys))
    ]


def prepare_noisys_and_enhanced(
    *,
    run: dict,
    batch_items: list[BatchItem],
    noise_option: dict,
    se_model: SEModel,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
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
    enhanced_unpadded = enhance_noisys(se_model, noisys)
    return noisys, enhanced_unpadded


def build_mixes(
    *,
    batch_items: list[BatchItem],
    noisys: list[torch.Tensor],
    enhanced_unpadded: list[torch.Tensor],
    se_model: SEModel,
) -> tuple[list[torch.Tensor], list[tuple[int, float]]]:
    device = se_model.device
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
    return all_mixes, mix_index


def rows_from_hypotheses(
    *,
    run: dict,
    batch_items: list[BatchItem],
    hypotheses: list[str | None],
    mix_index: list[tuple[int, float]],
) -> list[tuple]:
    if len(hypotheses) != len(mix_index):
        print_error(
            f'ASR returned {len(hypotheses)} hypotheses for {len(mix_index)} mixes',
        )
        raise SystemExit(1)

    rows = []
    for mix_pos, (batch_idx, coeff) in enumerate(mix_index):
        hypothesis = hypotheses[mix_pos]
        if hypothesis is None:
            continue
        utterance_id, _audio_path, transcript, _missing = batch_items[batch_idx]
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
    return rows


def compute_batch_rows_once(
    *,
    run: dict,
    batch_items: list[BatchItem],
    noise_option: dict,
    se_model: SEModel,
    asr_model: ASRModel,
) -> list[tuple]:
    noisys, enhanced_unpadded = prepare_noisys_and_enhanced(
        run=run,
        batch_items=batch_items,
        noise_option=noise_option,
        se_model=se_model,
    )
    all_mixes, mix_index = build_mixes(
        batch_items=batch_items,
        noisys=noisys,
        enhanced_unpadded=enhanced_unpadded,
        se_model=se_model,
    )
    mix_meta = [
        (batch_items[batch_idx][0], coeff)
        for batch_idx, coeff in mix_index
    ]
    hypotheses = transcribe_padded(
        asr_model,
        all_mixes,
        mix_meta=mix_meta,
    )
    return rows_from_hypotheses(
        run=run,
        batch_items=batch_items,
        hypotheses=hypotheses,
        mix_index=mix_index,
    )


def compute_batch_rows_mix_fallback(
    *,
    run: dict,
    batch_items: list[BatchItem],
    noise_option: dict,
    se_model: SEModel,
    asr_model: ASRModel,
    retry_mode: bool,
) -> list[tuple]:
    noisys, enhanced_unpadded = prepare_noisys_and_enhanced(
        run=run,
        batch_items=batch_items,
        noise_option=noise_option,
        se_model=se_model,
    )
    all_mixes, mix_index = build_mixes(
        batch_items=batch_items,
        noisys=noisys,
        enhanced_unpadded=enhanced_unpadded,
        se_model=se_model,
    )

    hypotheses: list[str | None] = []
    deferred: list[tuple[int, int, float]] = []
    for mix_pos, (batch_idx, coeff) in enumerate(mix_index):
        utterance_id = batch_items[batch_idx][0]
        hypothesis = transcribe_one_mix(
            asr_model,
            all_mixes[mix_pos],
            utterance_id=utterance_id,
            coeff=coeff,
        )
        if hypothesis is None and not retry_mode:
            deferred.append((batch_idx, utterance_id, coeff))
        hypotheses.append(hypothesis)

    rows = rows_from_hypotheses(
        run=run,
        batch_items=batch_items,
        hypotheses=hypotheses,
        mix_index=mix_index,
    )

    if deferred and not retry_mode:
        rows.extend(
            process_batch_per_coeff_items(
                run=run,
                batch_items=batch_items,
                noise_option=noise_option,
                se_model=se_model,
                asr_model=asr_model,
                coeff_items=deferred,
            ),
        )
    return rows


def compute_batch_rows_normal(
    *,
    run: dict,
    batch_items: list[BatchItem],
    noise_option: dict,
    se_model: SEModel,
    asr_model: ASRModel,
) -> list[tuple]:
    try:
        return compute_batch_rows_once(
            run=run,
            batch_items=batch_items,
            noise_option=noise_option,
            se_model=se_model,
            asr_model=asr_model,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch_items) > 1:
            print_log('batch OOM; falling back to one utterance at a time')
            rows: list[tuple] = []
            for item in batch_items:
                rows.extend(
                    compute_batch_rows_normal(
                        run=run,
                        batch_items=[item],
                        noise_option=noise_option,
                        se_model=se_model,
                        asr_model=asr_model,
                    ),
                )
            return rows
        print_log('single-utterance batch OOM; falling back to mix-by-mix ASR')
        return compute_batch_rows_mix_fallback(
            run=run,
            batch_items=batch_items,
            noise_option=noise_option,
            se_model=se_model,
            asr_model=asr_model,
            retry_mode=False,
        )


def compute_batch_rows_retry(
    *,
    run: dict,
    batch_items: list[BatchItem],
    noise_option: dict,
    se_model: SEModel,
    asr_model: ASRModel,
) -> list[tuple]:
    if len(batch_items) != 1:
        raise ValueError('retry batch must contain exactly one utterance')
    try:
        return compute_batch_rows_once(
            run=run,
            batch_items=batch_items,
            noise_option=noise_option,
            se_model=se_model,
            asr_model=asr_model,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print_log('retry: utterance batch OOM; falling back to mix-by-mix ASR')
        return compute_batch_rows_mix_fallback(
            run=run,
            batch_items=batch_items,
            noise_option=noise_option,
            se_model=se_model,
            asr_model=asr_model,
            retry_mode=True,
        )


def process_batch_normal(
    *,
    run: dict,
    batch_items: list[BatchItem],
    noise_option: dict,
    se_model: SEModel,
    asr_model: ASRModel,
    part_path: Path,
) -> int:
    rows = compute_batch_rows_normal(
        run=run,
        batch_items=batch_items,
        noise_option=noise_option,
        se_model=se_model,
        asr_model=asr_model,
    )
    write_batch(part_path, rows)
    return len(rows)


def process_batch_retry(
    *,
    run: dict,
    batch_items: list[BatchItem],
    noise_option: dict,
    se_model: SEModel,
    asr_model: ASRModel,
    part_path: Path,
) -> int:
    rows = compute_batch_rows_retry(
        run=run,
        batch_items=batch_items,
        noise_option=noise_option,
        se_model=se_model,
        asr_model=asr_model,
    )
    write_batch(part_path, rows)
    return len(rows)


def process_batch_per_coeff_items(
    *,
    run: dict,
    batch_items: list[BatchItem],
    noise_option: dict,
    se_model: SEModel,
    asr_model: ASRModel,
    coeff_items: list[tuple[int, int, float]],
) -> list[tuple]:
    print_log(
        f'falling back to one coeff at a time for {len(coeff_items)} mix(es)',
    )
    device = se_model.device
    rows: list[tuple] = []
    for batch_idx, utterance_id, coeff in coeff_items:
        _uid, audio_path, transcript, _missing = batch_items[batch_idx]
        clean = load_mono_16k(resolve_project_path(audio_path))
        noisy = generate_noisy(
            clean,
            noise_option,
            run['noise_seed'],
            utterance_id,
        )
        try:
            enhanced = se_model.enhance(noisy).detach()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print_warning(
                f'SE OOM for utterance_id={utterance_id} coeff={coeff}; skipping coeff',
            )
            continue
        noisy_dev = noisy.to(device, non_blocking=True)
        if enhanced.device != device:
            enhanced = enhanced.to(device, non_blocking=True)
        mix = mix_linear(noisy_dev, enhanced, coeff)
        hypothesis = transcribe_one_mix(
            asr_model,
            mix,
            utterance_id=utterance_id,
            coeff=coeff,
        )
        if hypothesis is None:
            continue
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
    return rows


class ModelCache:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.se_model: SEModel | None = None
        self.asr_model: ASRModel | None = None
        self.loaded_key: tuple[str, str, str, str] | None = None

    def ensure_loaded(self, run: dict) -> tuple[SEModel, ASRModel]:
        checkpoint_dir = resolve_project_path(run['checkpoint_dir'])
        if not (checkpoint_dir / 'config.json').is_file():
            print_error(f'config not found under checkpoint_dir: {checkpoint_dir}')
            raise SystemExit(1)

        key = model_cache_key(run)
        if key != self.loaded_key:
            self.release()
            print_log(
                f'loading models: se={run["se_model_name"]} '
                f'asr={run["asr_model_name"]} ckpt={run["checkpoint_name"]}',
            )
            self.se_model = SEModel(
                run['se_model_name'],
                checkpoint_dir,
                run['checkpoint_name'],
                device=self.device,
            )
            self.asr_model = ASRModel(run['asr_model_name'], device=self.device)
            self.loaded_key = key

        assert self.se_model is not None and self.asr_model is not None
        return self.se_model, self.asr_model

    def release(self) -> None:
        if self.se_model is not None:
            del self.se_model
            self.se_model = None
        if self.asr_model is not None:
            del self.asr_model
            self.asr_model = None
        self.loaded_key = None
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


def validate_sql_root() -> None:
    if not SQL_ROOT.exists():
        print_error(f'SQL database does not exist at {SQL_ROOT}')
        raise SystemExit(1)
