"""
db.py
=====
Utilidades de conexion a PostgreSQL (Notion - Fase 1: "Verificar conexion a
PostgreSQL desde el script Python" y HU-39: almacenar resultados en BD).

NO genera el dataset (ya existe en data/dataset_simulado.csv). Provee:
  - test_connection(): verifica conectividad.
  - load_csv_to_table(): carga el CSV existente en la tabla ml_dataset (opcional).

Configuracion via .env: PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD, PG_TABLE.

Uso:
    python db.py --check          # verifica conexion
    python db.py --load-dataset   # carga el CSV existente en la tabla
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

DATA_PATH = os.path.join(HERE, "data", "dataset_simulado.csv")


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
        print(f"Conexion OK -> {version}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] No se pudo conectar a PostgreSQL: {exc}")
        return False


def load_csv_to_table():
    """Carga el dataset existente en la tabla PG_TABLE (crea la tabla si no existe)."""
    import csv

    table = os.environ.get("PG_TABLE", "ml_dataset")
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] No existe el dataset: {DATA_PATH}")
        return

    with open(DATA_PATH, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))

    # Tipos: las columnas de texto conocidas como TEXT, el resto NUMERIC.
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
    print(f"Cargados {n} registros en la tabla '{table}'.")


def main():
    parser = argparse.ArgumentParser(description="Utilidades PostgreSQL del modulo ML.")
    parser.add_argument("--check", action="store_true", help="Verifica la conexion.")
    parser.add_argument("--load-dataset", action="store_true", help="Carga el CSV en la tabla.")
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
