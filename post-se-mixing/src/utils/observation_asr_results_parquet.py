from __future__ import annotations

from pathlib import Path

import duckdb as db

from src.config import PARQUET_ROOT
from src.utils.wer import norm


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

DEFAULT_LINEAR_COEFFS = [round(x * 0.1, 1) for x in range(-5, 16)]


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


def part_glob(run_id: int) -> str:
    return _part_glob(run_id)


def count_parquet_rows(run_id: int) -> int:
    if not has_parts(run_id):
        return 0
    con = db.connect()
    try:
        row = con.execute(
            """
            SELECT COUNT(*)
            FROM read_parquet(?)
            """,
            [part_glob(run_id)],
        ).fetchone()
    finally:
        con.close()
    return int(row[0])


def count_parquet_distinct_keys(run_id: int) -> int:
    if not has_parts(run_id):
        return 0
    con = db.connect()
    try:
        row = con.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT
                    run_id,
                    utterance_id,
                    mixture_family,
                    mixture_coeff
                FROM read_parquet(?)
            )
            """,
            [part_glob(run_id)],
        ).fetchone()
    finally:
        con.close()
    return int(row[0])


def list_run_ids_with_parquet(con: db.DuckDBPyConnection) -> list[int]:
    rows = con.execute(
        """
        SELECT id
        FROM observation_eval_runs
        ORDER BY
            CASE split
                WHEN 'test-clean' THEN 0
                WHEN 'dev-clean' THEN 1
                WHEN 'train-clean-100' THEN 2
                WHEN 'train-clean-360' THEN 3
                ELSE 9
            END,
            id
        """,
    ).fetchall()
    return [int(row[0]) for row in rows if has_parts(int(row[0]))]


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


def list_run_ids_with_pending(
    con: db.DuckDBPyConnection,
    *,
    mixture_family: str = 'linear',
    linear_coeffs: list[float] | None = None,
) -> list[int]:
    coeffs = linear_coeffs if linear_coeffs is not None else DEFAULT_LINEAR_COEFFS
    runs = con.execute(
        """
        SELECT id, split
        FROM observation_eval_runs
        ORDER BY id
        """,
    ).fetchall()

    pending_run_ids: list[int] = []
    for run_id, split in runs:
        run_id = int(run_id)
        utterances = con.execute(
            """
            SELECT id, transcript
            FROM utterances
            WHERE split = ?
            """,
            [str(split)],
        ).fetchall()
        existing_by_utterance = load_existing_coeffs(run_id, mixture_family)
        for utterance_id, transcript in utterances:
            if norm(str(transcript)) == '':
                continue
            existing = existing_by_utterance.get(int(utterance_id), set())
            if any(coeff not in existing for coeff in coeffs):
                pending_run_ids.append(run_id)
                break
    return pending_run_ids


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
