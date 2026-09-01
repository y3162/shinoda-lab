from __future__ import annotations

import argparse

import duckdb as db
import torch
from tqdm import tqdm

from src.config import SQL_ROOT
from src.database.observation_asr_results_fill_common import (
    ModelCache,
    get_run,
    init_torch_threads,
    limit_cpu_threads,
    list_pending_utterances,
    parse_int_list,
    process_batch_retry,
    validate_sql_root,
)
from src.utils.noise import get_noise_option
from src.utils.observation_asr_results_parquet import (
    list_run_ids_with_pending,
    next_batch_index,
    next_part_path,
    run_dir,
)
from src.utils.print import print_error, print_log, print_warning

RETRY_BATCH_SIZE = 1

limit_cpu_threads()
init_torch_threads()


def fill_run_retry(
    *,
    con: db.DuckDBPyConnection,
    run: dict,
    model_cache: ModelCache,
) -> None:
    run_id = run['id']
    noise_option = get_noise_option(con, run['noise_config_id'])
    pending = list_pending_utterances(con, run_id, run['split'])
    if not pending:
        print_log(f'run_id={run_id}: nothing to do')
        return

    print_log(
        f'run_id={run_id} split={run["split"]} '
        f'pending_utterances={len(pending)} batch_size={RETRY_BATCH_SIZE}',
    )

    se_model, asr_model = model_cache.ensure_loaded(run)
    run_dir_path = run_dir(run_id)
    wrote_total = 0
    batch_index = next_batch_index(run_id)
    for batch_items in tqdm(pending, desc=f'retry run {run_id}'):
        part_path = next_part_path(run_dir_path, batch_index)
        try:
            wrote_total += process_batch_retry(
                run=run,
                batch_items=[batch_items],
                noise_option=noise_option,
                se_model=se_model,
                asr_model=asr_model,
                part_path=part_path,
            )
            batch_index += 1
        except Exception as exc:
            print_warning(
                f'retry batch failed run_id={run_id} batch_index={batch_index} '
                f'utterance_id={batch_items[0]}: {exc}',
            )
            if model_cache.device.type == 'cuda':
                torch.cuda.empty_cache()
            continue

    print_log(f'run_id={run_id}: wrote {wrote_total} rows to {run_dir_path}')


def resolve_run_ids(con: db.DuckDBPyConnection, run_ids: list[int] | None) -> list[int]:
    if run_ids is not None:
        return run_ids
    pending = list_run_ids_with_pending(con)
    if not pending:
        print_log('no pending runs')
        return []
    print_log(f'pending run_ids: {",".join(str(run_id) for run_id in pending)}')
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Retry observation ASR Parquet fill for pending (run, utterance, coeff) '
            'rows only. Always uses batch_size=1; OOM falls back to mix-by-mix ASR '
            'then skips individual coeffs.'
        ),
    )
    parser.add_argument(
        '--run_id',
        type=parse_int_list,
        default=None,
        help=(
            'Comma-separated observation_eval_runs.id values. '
            'If omitted, all runs with pending Parquet gaps are processed.'
        ),
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
    )
    args = parser.parse_args()

    validate_sql_root()

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print_log(
        f'torch threads={torch.get_num_threads()} '
        f'interop={torch.get_num_interop_threads()}',
    )

    con = db.connect(str(SQL_ROOT), read_only=True)
    model_cache = ModelCache(device)
    try:
        run_ids = resolve_run_ids(con, args.run_id)
        if not run_ids:
            return
        for run_id in run_ids:
            run = get_run(con, run_id)
            fill_run_retry(
                con=con,
                run=run,
                model_cache=model_cache,
            )
    finally:
        model_cache.release()
        con.close()


if __name__ == '__main__':
    main()
