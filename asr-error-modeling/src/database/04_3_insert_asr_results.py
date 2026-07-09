import argparse
from pathlib import Path
from typing import List

import duckdb as db
from tqdm import tqdm

from src.config import (
    SQL_ROOT,
    PARQUET_ROOT,
)
from src.utils.print import (
    print_error,
    print_log,
    print_warning,
)

INSERT_SQL = """
    INSERT INTO asr_results (
        utterance_id,
        noise_config_id,
        transcript,
        model_name
    )
    SELECT
        p.utterance_id,
        p.noise_config_id,
        p.transcript,
        ? AS model_name
    FROM read_parquet(?) AS p
    WHERE NOT EXISTS (
        SELECT 1
        FROM asr_results AS ar
        WHERE ar.utterance_id = p.utterance_id
            AND ar.noise_config_id = ?
            AND ar.model_name = ?
    );
"""


def get_target_noise_config_ids(
    con: db.DuckDBPyConnection,
    model_name: str,
) -> List[int]:
    noise_config_ids = con.execute(
        """
        SELECT nc.id
        FROM noise_configs AS nc
        WHERE (
            SELECT COUNT(DISTINCT ar.utterance_id)
            FROM asr_results AS ar
            WHERE ar.noise_config_id = nc.id
                AND ar.model_name = ?
        ) < (
            SELECT COUNT(*)
            FROM utterances
        );
        """,
        [model_name],
    ).fetchall()
    return [int(noise_config_id[0]) for noise_config_id in noise_config_ids]


def insert_asr_results_from_parquet(
    con: db.DuckDBPyConnection,
    parquet_file: Path,
    noise_config_id: int,
    model_name: str,
) -> None:
    con.execute(
        INSERT_SQL,
        [model_name, str(parquet_file), noise_config_id, model_name],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    if not SQL_ROOT.exists():
        print_error(f'SQL database does not exist at {SQL_ROOT}')
        return

    con = db.connect(SQL_ROOT)

    target_noise_config_ids = get_target_noise_config_ids(con, args.model_name)
    print_log(
        f'Inserting ASR results for model {args.model_name} '
        f'into {SQL_ROOT} ({len(target_noise_config_ids)} noise configs)'
    )

    for noise_config_id in tqdm(target_noise_config_ids, desc='Inserting ASR results'):
        parquet_file = PARQUET_ROOT / f'results_noise_config_id_{noise_config_id}.parquet'
        if not parquet_file.exists():
            print_warning(f'Parquet file not found: {parquet_file}. Skipping.')
            continue

        insert_asr_results_from_parquet(con, parquet_file, noise_config_id, args.model_name)

    con.commit()
    con.close()


if __name__ == '__main__':
    main()
