"""Row-count regression test for the ingested raw tables.

Counts are frozen from the first successful ingestion of the Home Credit
dataset. The test skips cleanly when the DuckDB file is absent, so CI runners
and cloners who never download the data don't see failures.
"""

from pathlib import Path

import duckdb
import pytest

from scoregate.ingest import RAW_TABLES

DB_PATH = Path("data/scoregate.duckdb")

# Frozen after the first successful ingestion. Every raw_* table in
# RAW_TABLES must have an entry here.
EXPECTED_COUNTS = {
    "raw_application_train": 307_511,
    "raw_application_test": 48_744,
    "raw_bureau": 1_716_428,
    "raw_bureau_balance": 27_299_925,
    "raw_previous_application": 1_670_214,
    "raw_installments_payments": 13_605_401,
    "raw_credit_card_balance": 3_840_312,
    "raw_pos_cash_balance": 10_001_358,
}


def test_expected_counts_cover_every_table() -> None:
    assert set(EXPECTED_COUNTS) == set(RAW_TABLES)


@pytest.mark.skipif(not DB_PATH.exists(), reason="ingested DuckDB not present")
@pytest.mark.parametrize("table", sorted(EXPECTED_COUNTS))
def test_row_count(table: str) -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        row = con.execute(f"SELECT count(*) FROM {table}").fetchone()
    actual = int(row[0]) if row else 0
    assert actual == EXPECTED_COUNTS[table]
