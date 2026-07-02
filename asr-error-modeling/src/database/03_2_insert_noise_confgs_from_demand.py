from dataclasses import dataclass
@dataclass
class NoiseConfigRow:
    noise_id: int
    snr: int

import duckdb as db
from tqdm import tqdm
import json

from src.utils.print import (
    print_warning,
    print_log,
)
from src.config import SQL_ROOT


def insert_noise_config(
    con: db.DuckDBPyConnection,
) -> None:
    insert_sql = """
        INSERT OR IGNORE INTO noise_configs (
            config_json
        )
        VALUES (?)
    """

    all_noise_ids = con.execute(
        """
        SELECT id, audio_path
        FROM noises
        WHERE audio_path LIKE '%ch01_0-20.wav';
        """,
    ).fetchall()

    con.execute('BEGIN TRANSACTION')
    with tqdm(desc='Inserted noise_configs', unit='rows') as pbar:
        con.execute(insert_sql, [json.dumps(
            {
                'generator_type': 'additive',
                'args': [],
            }
        )])
        for noise_id, audio_path in all_noise_ids:
            for snr in range(-10, 5+1):
                con.execute(insert_sql, [json.dumps(
                    {
                        'generator_type': 'additive',
                        'args': [
                            {
                                'type': 'audiofile',
                                'noise_id': noise_id,
                                'snr_db': snr,
                            }
                        ],
                    }
                )])
                pbar.update(1)
    con.execute('COMMIT')


def main() -> None:
    if SQL_ROOT.exists():
        print_warning(f'SQL database already exists at {SQL_ROOT}')
    else:
        print_log(f'Creating SQL database at {SQL_ROOT}')
        SQL_ROOT.parent.mkdir(parents=True, exist_ok=True)

    con = db.connect(SQL_ROOT)

    insert_noise_config(con)

    con.close()


if __name__ == '__main__':
    main()