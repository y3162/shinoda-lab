from __future__ import annotations

import argparse

import duckdb as db
from tqdm import tqdm

from src.config import SQL_ROOT
from src.database.observation_asr_results_fill_common import parse_int_list
from src.utils.observation_asr_results_parquet import (
    count_parquet_distinct_keys,
    count_parquet_rows,
    has_parts,
    list_run_ids_with_parquet,
    part_glob,
)
from src.utils.print import print_error, print_log, print_warning


def resolve_run_ids(con: db.DuckDBPyConnection, run_ids: list[int] | None) -> list[int]:
    if run_ids is not None:
        return run_ids
    return list_run_ids_with_parquet(con)


def verify_run_registered(con: db.DuckDBPyConnection, run_id: int) -> None:
    row = con.execute(
        """
        SELECT 1
        FROM observation_eval_runs
        WHERE id = ?
        """,
        [run_id],
    ).fetchone()
    if row is None:
        print_error(f'run_id not found in observation_eval_runs: {run_id}')
        raise SystemExit(1)


def verify_parquet_unique_keys(run_id: int, parquet_rows: int) -> None:
    distinct_keys = count_parquet_distinct_keys(run_id)
    if distinct_keys != parquet_rows:
        print_error(
            f'run_id={run_id}: duplicate Parquet keys detected '
            f'rows={parquet_rows} distinct_keys={distinct_keys}',
        )
        raise SystemExit(1)


def count_db_rows(con: db.DuckDBPyConnection, run_id: int) -> int:
    row = con.execute(
        """
        SELECT COUNT(*)
        FROM observation_asr_results
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()
    return int(row[0])


def import_run(con: db.DuckDBPyConnection, run_id: int) -> int:
    verify_run_registered(con, run_id)
    if not has_parts(run_id):
        print_warning(f'run_id={run_id}: no Parquet parts; skip')
        return 0

    parquet_rows = count_parquet_rows(run_id)
    if parquet_rows == 0:
        print_warning(f'run_id={run_id}: Parquet is empty; skip')
        return 0

    verify_parquet_unique_keys(run_id, parquet_rows)
    db_rows_before = count_db_rows(con, run_id)
    if db_rows_before == parquet_rows:
        print_log(f'run_id={run_id}: already complete ({parquet_rows} rows); skip')
        return 0

    con.execute(
        """
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
        SELECT
            p.run_id,
            p.utterance_id,
            p.mixture_family,
            p.mixture_coeff,
            p.hypothesis,
            p.wer,
            p.n_errors,
            p.n_ref_words
        FROM read_parquet(?) AS p
        WHERE NOT EXISTS (
            SELECT 1
            FROM observation_asr_results AS o
            WHERE o.run_id = p.run_id
              AND o.utterance_id = p.utterance_id
              AND o.mixture_family = p.mixture_family
              AND o.mixture_coeff = p.mixture_coeff
        )
        """,
        [part_glob(run_id)],
    )
    db_rows_after = count_db_rows(con, run_id)
    inserted = db_rows_after - db_rows_before
    if db_rows_after != parquet_rows:
        print_error(
            f'run_id={run_id}: row count mismatch after import '
            f'parquet={parquet_rows} db={db_rows_after}',
        )
        raise SystemExit(1)
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Import observation ASR results from Parquet into DuckDB. '
            'Parquet files are read-only and never deleted. '
            'Existing DB rows are kept; only missing keys are inserted.'
        ),
    )
    parser.add_argument(
        '--run_id',
        type=parse_int_list,
        default=None,
        help=(
            'Comma-separated observation_eval_runs.id values. '
            'If omitted, all runs with Parquet data are processed.'
        ),
    )
    args = parser.parse_args()

    if not SQL_ROOT.exists():
        print_error(f'SQL database does not exist at {SQL_ROOT}')
        raise SystemExit(1)

    con = db.connect(str(SQL_ROOT))
    try:
        run_ids = resolve_run_ids(con, args.run_id)
        if not run_ids:
            print_log('no runs with Parquet data')
            return

        for run_id in run_ids:
            verify_run_registered(con, run_id)

        inserted_total = 0
        for run_id in tqdm(run_ids, desc='import'):
            inserted = import_run(con, run_id)
            inserted_total += inserted
            if inserted:
                print_log(f'run_id={run_id}: inserted {inserted} rows')
            else:
                print_log(f'run_id={run_id}: nothing to insert')

        db_total = con.execute(
            """
            SELECT COUNT(*)
            FROM observation_asr_results
            """,
        ).fetchone()[0]
        print_log(
            f'import complete: runs={len(run_ids)} '
            f'inserted_rows={inserted_total} db_total={int(db_total)}',
        )
    finally:
        con.close()


if __name__ == '__main__':
    main()
