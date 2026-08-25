import duckdb as db

from src.utils.print import (
    print_warning,
    print_error,
)
from src.config import SQL_ROOT
from src.utils.sql import check_table_exists


def create_observation_asr_results_table_if_needed(
    con: db.DuckDBPyConnection,
) -> None:
    if check_table_exists(con, 'observation_asr_results'):
        print_warning(
            "observation_asr_results table already exists. "
            "Skip creating 'observation_asr_results' table."
        )
        return

    con.execute(
        """
        CREATE SEQUENCE observation_asr_results_id_seq START 1;
        """
    )

    con.execute(
        """
        CREATE TABLE observation_asr_results (
            id                 INTEGER PRIMARY KEY
                               DEFAULT nextval('observation_asr_results_id_seq'),
            run_id             INTEGER NOT NULL,
            utterance_id       INTEGER NOT NULL,
            mixture_family     VARCHAR NOT NULL,
            mixture_coeff      DOUBLE NOT NULL,
            hypothesis         VARCHAR NOT NULL,
            wer                DOUBLE NOT NULL,
            n_errors           INTEGER NOT NULL,
            n_ref_words        INTEGER NOT NULL,

            CHECK (mixture_family IN ('oa', 'linear')),
            UNIQUE (
                run_id,
                utterance_id,
                mixture_family,
                mixture_coeff
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

    create_observation_asr_results_table_if_needed(con)

    con.close()


if __name__ == '__main__':
    main()
