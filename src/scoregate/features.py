"""Build the feature layer in DuckDB by running the versioned SQL in order.

Each sql/NN_*.sql file creates one or more feat_* tables, and the final file
assembles them into the `features` table plus a separate `fairness_metadata`
table. Running the files in filename order rebuilds the whole feature layer from
the raw_* tables ingested earlier, so the step stays reproducible.
"""

import argparse
from pathlib import Path

import duckdb


def build_features(db_path: Path, sql_dir: Path) -> int:
    """Run every SQL file in sql_dir against db_path; return the features row count."""
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found. Run scoregate.ingest first.")
    files = sorted(sql_dir.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"No .sql files found in {sql_dir}.")

    with duckdb.connect(str(db_path)) as con:
        for path in files:
            con.execute(path.read_text(encoding="utf-8"))
        row = con.execute("SELECT count(*) FROM features").fetchone()
    return int(row[0]) if row else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/scoregate.duckdb"))
    parser.add_argument("--sql-dir", type=Path, default=Path("sql"))
    args = parser.parse_args()

    n = build_features(args.db, args.sql_dir)
    print(f"features: {n:,} rows")


if __name__ == "__main__":
    main()
