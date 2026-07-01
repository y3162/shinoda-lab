import duckdb as db

from src.utils.print import (
    print_log,
    print_warning,
    print_error,
)
from src.config import SQL_ROOT
from src.utils.sql import check_table_exists


def create_utterances_table_if_needed(
    con: db.DuckDBPyConnection,
) -> None:
    if check_table_exists(con, 'utterances'):
        print_error("Utterances table already exists. Skip creating \'utterances\' table.")
        return

    con.execute(
        """
        CREATE SEQUENCE utterances_id_seq START 1;
        """
    )

    con.execute(
        """
        CREATE TABLE utterances (
            id           INTEGER PRIMARY KEY DEFAULT nextval('utterances_id_seq'),
            dataset_name VARCHAR NOT NULL,
            split        VARCHAR NOT NULL,
            audio_path   VARCHAR NOT NULL,
            speaker_id   VARCHAR NOT NULL,
            section_id   VARCHAR NOT NULL,
            transcript   VARCHAR NOT NULL,
            sample_rate  INTEGER NOT NULL,
            channels     INTEGER NOT NULL,
            frame_count  INTEGER NOT NULL,

            UNIQUE (
                dataset_name,
                split,
                audio_path,
                speaker_id,
                section_id
            )
        );
        """
    )


def main() -> None:
    if SQL_ROOT.exists():
        print_warning(f'SQL database already exists at {SQL_ROOT}')
        return
    else:
        print_log(f'Creating SQL database at {SQL_ROOT}')
        SQL_ROOT.parent.mkdir(parents=True, exist_ok=True)

    con = db.connect(SQL_ROOT)

    create_utterances_table_if_needed(con)

    con.close()


if __name__ == '__main__':
    main()
