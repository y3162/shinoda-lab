import duckdb as db

from src.utils.print import (
    print_warning,
    print_error,
)
from src.config import SQL_ROOT
from src.utils.sql import check_table_exists


def create_observation_eval_runs_table_if_needed(
    con: db.DuckDBPyConnection,
) -> None:
    if check_table_exists(con, 'observation_eval_runs'):
        print_warning(
            "observation_eval_runs table already exists. "
            "Skip creating 'observation_eval_runs' table."
        )
        return

    con.execute(
        """
        CREATE SEQUENCE observation_eval_runs_id_seq START 1;
        """
    )

    con.execute(
        """
        CREATE TABLE observation_eval_runs (
            id                 INTEGER PRIMARY KEY
                               DEFAULT nextval('observation_eval_runs_id_seq'),
            se_model_name      VARCHAR NOT NULL,
            checkpoint_dir     VARCHAR NOT NULL,
            checkpoint_name    VARCHAR NOT NULL,
            asr_model_name     VARCHAR NOT NULL,
            noise_config_id    INTEGER NOT NULL,
            noise_seed         INTEGER NOT NULL,
            split              VARCHAR NOT NULL,
            created_at         TIMESTAMP DEFAULT current_timestamp,

            UNIQUE (
                se_model_name,
                checkpoint_dir,
                checkpoint_name,
                asr_model_name,
                noise_config_id,
                noise_seed,
                split
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

    create_observation_eval_runs_table_if_needed(con)

    con.close()


if __name__ == '__main__':
    main()
