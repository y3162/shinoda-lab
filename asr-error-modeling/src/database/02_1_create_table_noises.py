import duckdb as db

from src.utils.print import (
    print_error,
    print_warning,
    print_log,
)
from src.config import SQL_ROOT
from src.utils.sql import check_table_exists


def create_noises_table_if_needed(
    con: db.DuckDBPyConnection,
) -> None:
    if check_table_exists(con, 'noises'):
        print_error("noises table already exists. Skip creating 'noises' table.")
        return

    con.execute(
        """
        CREATE SEQUENCE noises_id_seq START 1;
        """
    )

    con.execute(
        """
        CREATE TABLE noises (
            id                  INTEGER PRIMARY KEY DEFAULT nextval('noises_id_seq'),
            dataset_name        VARCHAR NOT NULL,
            audio_path          VARCHAR NOT NULL,
            noise_type          VARCHAR NOT NULL,
            sample_rate         INTEGER NOT NULL,
            channels            INTEGER NOT NULL,
            frame_count         INTEGER NOT NULL,

            UNIQUE (
                dataset_name,
                audio_path,
                noise_type
            )
        );
        """
    )


def main() -> None:
    if SQL_ROOT.exists():
        print_warning(f'SQL database already exists at {SQL_ROOT}')
    else:
        print_log(f'Creating SQL database at {SQL_ROOT}')
        SQL_ROOT.parent.mkdir(parents=True, exist_ok=True)

    con = db.connect(SQL_ROOT)

    create_noises_table_if_needed(con)

    con.close()


if __name__ == '__main__':
    main()
