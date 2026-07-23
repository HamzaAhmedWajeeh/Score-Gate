"""Load the raw Home Credit CSVs into a local DuckDB database.

Each CSV becomes a raw_* table, replaced on every run so ingestion is
idempotent. Feature engineering happens later in versioned SQL, not here.
"""

import argparse
from pathlib import Path

import duckdb

RAW_TABLES = {
    "raw_application_train": "application_train.csv",
    "raw_application_test": "application_test.csv",
    "raw_bureau": "bureau.csv",
    "raw_bureau_balance": "bureau_balance.csv",
    "raw_previous_application": "previous_application.csv",
    "raw_installments_payments": "installments_payments.csv",
    "raw_credit_card_balance": "credit_card_balance.csv",
    "raw_pos_cash_balance": "POS_CASH_balance.csv",
}


def ingest(raw_dir: Path, db_path: Path) -> dict[str, int]:
    """Load every raw CSV into db_path and return row counts per table."""
    missing = [name for name in RAW_TABLES.values() if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw files in {raw_dir}: {missing}. Run scoregate.download_data first."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with duckdb.connect(str(db_path)) as con:
        for table, filename in RAW_TABLES.items():
            con.execute(
                f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto(?, header=true)",
                [str(raw_dir / filename)],
            )
            row = con.execute(f"SELECT count(*) FROM {table}").fetchone()
            counts[table] = int(row[0]) if row else 0
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--db", type=Path, default=Path("data/scoregate.duckdb"))
    args = parser.parse_args()

    counts = ingest(args.raw_dir, args.db)
    for table, n in counts.items():
        print(f"{table}: {n:,} rows")


if __name__ == "__main__":
    main()
