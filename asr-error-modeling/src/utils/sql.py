import duckdb as db


def check_table_exists(
    con: db.DuckDBPyConnection,
    table_name: str,
) -> bool:
    exists = con.execute(
        """
        SELECT COUNT(*) > 0
        FROM information_schema.tables
        WHERE table_name = ?
        """,
        [table_name],
    ).fetchone()[0]

    return exists
