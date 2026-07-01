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


def create_noise_config_table_if_needed(
    con: db.DuckDBPyConnection,
) -> None:
    if check_table_exists(con, 'noise_config'):
        print_warning("noise_config table already exists. Skip creating 'noise_config' table.")
        return

    con.execute(
        """
        CREATE SEQUENCE noise_config_id_seq START 1;
        """
    )

    con.execute(
        """
        CREATE TABLE noise_config (
            id             INTEGER PRIMARY KEY DEFAULT nextval('noise_config_id_seq'),
            config_json    JSON NOT NULL,

            UNIQUE (
                config_json
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

    create_noise_config_table_if_needed(con)

    con.close()


if __name__ == '__main__':
    main()