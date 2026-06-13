"""Optional PostgreSQL helper.

Verifies connectivity and optionally loads the existing CSV into a table. It does
not generate the dataset. Configured via .env: PG_HOST, PG_PORT, PG_DATABASE,
PG_USER, PG_PASSWORD, PG_TABLE.

Usage:
    python db.py --check          # check connection
    python db.py --load-dataset   # load the existing CSV into the table
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

DATA_PATH = os.path.join(HERE, "data", "simulated_dataset.csv")


def _conn_params():
    return dict(
        host=os.environ.get("PG_HOST", "localhost"),
        port=os.environ.get("PG_PORT", "5432"),
        dbname=os.environ.get("PG_DATABASE", "material_engagement"),
        user=os.environ.get("PG_USER", "postgres"),
        password=os.environ.get("PG_PASSWORD", ""),
    )


def get_connection():
    import psycopg2

    return psycopg2.connect(**_conn_params())


def test_connection() -> bool:
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
        conn.close()
        print(f"Connection OK -> {version}")
        return True
    except Exception as exc:
        print(f"[ERROR] Could not connect to PostgreSQL: {exc}")
        return False


def load_csv_to_table():
    """Load the existing dataset into PG_TABLE, creating the table if needed."""
    import csv

    table = os.environ.get("PG_TABLE", "ml_dataset")
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found: {DATA_PATH}")
        return

    with open(DATA_PATH, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))

    # Known text columns map to TEXT, the rest to NUMERIC.
    text_cols = {
        "options_selection", "preferred_content_type", "navigation_sequence",
        "academic_grade", "target_vak_label",
    }
    cols_def = ", ".join(
        f'"{c}" {"TEXT" if c in text_cols else "NUMERIC"}' for c in header
    )

    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({cols_def});')
        cur.execute(f'TRUNCATE "{table}";')
        with open(DATA_PATH, encoding="utf-8") as fh:
            cur.copy_expert(
                f'COPY "{table}" FROM STDIN WITH CSV HEADER', fh
            )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM "{table}";')
        n = cur.fetchone()[0]
    conn.close()
    print(f"Loaded {n} rows into table '{table}'.")


def main():
    parser = argparse.ArgumentParser(description="PostgreSQL helper for the ML module.")
    parser.add_argument("--check", action="store_true", help="Check the connection.")
    parser.add_argument("--load-dataset", action="store_true", help="Load the CSV into the table.")
    args = parser.parse_args()

    if not (args.check or args.load_dataset):
        parser.print_help()
        sys.exit(0)

    if args.check and not test_connection():
        sys.exit(1)
    if args.load_dataset:
        load_csv_to_table()


if __name__ == "__main__":
    main()
