from __future__ import annotations

import argparse
from pathlib import Path

import duckdb as db

from src.config import PROJECT_ROOT, SQL_ROOT, resolve_project_path
from src.utils.print import print_error, print_log


def checkpoint_dir_for_db(checkpoint_dir: str | Path) -> str:
    resolved = resolve_project_path(checkpoint_dir).resolve()
    return str(resolved.relative_to(PROJECT_ROOT.resolve()))


def parse_int_list(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(',') if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError('expected at least one integer')
    return [int(value) for value in values]


def parse_str_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(',') if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError('expected at least one value')
    return values


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Insert observation_eval_runs for each (noise_config_id, split) pair. '
            'Existing unique rows are reused (get-or-create).'
        ),
    )
    parser.add_argument(
        '--noise_config_ids',
        type=parse_int_list,
        required=True,
        help='Comma-separated noise_config ids, e.g. 26225,26241',
    )
    parser.add_argument(
        '--splits',
        type=parse_str_list,
        required=True,
        help='Comma-separated utterance splits, e.g. test-clean',
    )
    parser.add_argument('--se_model_name', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--asr_model_name', type=str, required=True)
    parser.add_argument('--noise_seed', type=int, required=True)
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

    for noise_config_id in args.noise_config_ids:
        exists = con.execute(
            """
            SELECT 1
            FROM noise_configs
            WHERE id = ?
            """,
            [noise_config_id],
        ).fetchone()
        if exists is None:
            print_error(f'noise_config_id not found: {noise_config_id}')
            raise SystemExit(1)

    run_ids: list[int] = []
    for split in args.splits:
        for noise_config_id in args.noise_config_ids:
            run_id = get_or_create_run(
                con,
                se_model_name=args.se_model_name,
                checkpoint_dir=checkpoint_dir_rel,
                checkpoint_name=args.checkpoint_name,
                asr_model_name=args.asr_model_name,
                noise_config_id=noise_config_id,
                noise_seed=args.noise_seed,
                split=split,
            )
            run_ids.append(run_id)
            print_log(
                f'run_id={run_id} noise_config_id={noise_config_id} split={split}',
            )

    con.close()
    print_log(f'registered {len(run_ids)} runs: {run_ids}')


if __name__ == '__main__':
    main()
