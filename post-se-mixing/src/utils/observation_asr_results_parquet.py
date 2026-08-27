from __future__ import annotations

from pathlib import Path

import duckdb as db

from src.config import PARQUET_ROOT


RESULT_COLUMNS = (
    'run_id',
    'utterance_id',
    'mixture_family',
    'mixture_coeff',
    'hypothesis',
    'wer',
    'n_errors',
    'n_ref_words',
)

ObservationAsrResultRow = tuple[
    int,
    int,
    str,
    float,
    str,
    float,
    int,
    int,
]


def results_root() -> Path:
    return PARQUET_ROOT / 'observation_asr_results'


def run_dir(run_id: int) -> Path:
    return results_root() / f'run_id={run_id}'


def next_part_path(run_dir_path: Path, batch_index: int) -> Path:
    return run_dir_path / f'part-{batch_index:06d}.parquet'


def next_batch_index(run_id: int) -> int:
    run_dir_path = run_dir(run_id)
    max_index = 0
    for path in run_dir_path.glob('part-*.parquet'):
        stem = path.stem
        if stem.startswith('part-'):
            try:
                max_index = max(max_index, int(stem.removeprefix('part-')))
            except ValueError:
                continue
    return max_index + 1


def _part_glob(run_id: int) -> str:
    return str(run_dir(run_id) / 'part-*.parquet')


def has_parts(run_id: int) -> bool:
    return any(run_dir(run_id).glob('part-*.parquet'))


def load_existing_coeffs(
    run_id: int,
    mixture_family: str = 'linear',
) -> dict[int, set[float]]:
    if not has_parts(run_id):
        return {}

    con = db.connect()
    try:
        rows = con.execute(
            """
            SELECT utterance_id, mixture_coeff
            FROM read_parquet(?)
            WHERE mixture_family = ?
            """,
            [_part_glob(run_id), mixture_family],
        ).fetchall()
    finally:
        con.close()

    existing: dict[int, set[float]] = {}
    for utterance_id, mixture_coeff in rows:
        utterance_id = int(utterance_id)
        existing.setdefault(utterance_id, set()).add(round(float(mixture_coeff), 1))
    return existing


def write_batch(path: Path, rows: list[ObservationAsrResultRow]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    value_groups = ', '.join(['(?, ?, ?, ?, ?, ?, ?, ?)'] * len(rows))
    params: list[object] = [item for row in rows for item in row]
    path_sql = str(path).replace("'", "''")

    con = db.connect()
    try:
        con.execute(
            f"""
            COPY (
                SELECT
                    column0::INTEGER AS run_id,
                    column1::INTEGER AS utterance_id,
                    column2::VARCHAR AS mixture_family,
                    column3::DOUBLE AS mixture_coeff,
                    column4::VARCHAR AS hypothesis,
                    column5::DOUBLE AS wer,
                    column6::INTEGER AS n_errors,
                    column7::INTEGER AS n_ref_words
                FROM (VALUES {value_groups}) AS batch_rows(
                    column0,
                    column1,
                    column2,
                    column3,
                    column4,
                    column5,
                    column6,
                    column7
                )
            ) TO '{path_sql}' (FORMAT PARQUET)
            """,
            params,
        )
    finally:
        con.close()
