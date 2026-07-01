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
    print_error,
)
from src.config import SQL_ROOT
from src.utils.sql import check_table_exists

def create_asr_results_table_if_needed(
    con: db.DuckDBPyConnection,
) -> None:
    if check_table_exists(con, 'asr_results'):
        print_warning("asr_results table already exists. Skip creating 'asr_results' table.")
        return

    con.execute(
        """
        CREATE SEQUENCE asr_results_id_seq START 1;
        """
    )

    con.execute(
        """
        CREATE TABLE asr_results (
            id                 INTEGER PRIMARY KEY DEFAULT nextval('asr_results_id_seq'),
            utterance_id       INTEGER NOT NULL,
            noise_config_id    INTEGER NOT NULL,
            model_name         VARCHAR NOT NULL,
            transcript         VARCHAR NOT NULL,

            UNIQUE (
                utterance_id,
                noise_config_id,
                model_name
            )
        );
        """
    )


def main() -> None:
    if SQL_ROOT.exists():
        print_warning(f'SQL database already exists at {SQL_ROOT}')
    else:
        print_error(f'SQL database does not exist at {SQL_ROOT}')
        return

    con = db.connect(SQL_ROOT)

    create_asr_results_table_if_needed(con)

    con.close()


if __name__ == '__main__':
    main()