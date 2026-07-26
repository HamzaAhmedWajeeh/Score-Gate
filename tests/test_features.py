"""Tests for the SQL feature layer.

Two kinds. The window-construct tests run the exact OVER clauses used in the
feature SQL against tiny hand-computed frames, so they always run and pin the
semantics of the three window functions. The real-data test mirrors the counts
test: it skips cleanly unless the built DuckDB and its `features` table are
present.
"""

from pathlib import Path

import duckdb
import pytest

DB_PATH = Path("data/scoregate.duckdb")
APPLICATION_TRAIN_ROWS = 307_511


# --- window-construct tests, always run ------------------------------------


def test_lag_delay_trend() -> None:
    """LAG within a credit, matching 05_installments_features."""
    con = duckdb.connect()
    con.execute("CREATE TABLE inst(sk_id_prev INT, num INT, pay_delay DOUBLE)")
    con.execute(
        "INSERT INTO inst VALUES "
        "(1, 1, 2.0), (1, 2, 5.0), (1, 3, 4.0), "  # trend: NULL, +3, -1
        "(2, 1, 10.0), (2, 2, 7.0)"  # trend: NULL, -3 (partition resets LAG)
    )
    rows = con.execute(
        """
        SELECT sk_id_prev, num,
               pay_delay - LAG(pay_delay) OVER (
                   PARTITION BY sk_id_prev ORDER BY num
               ) AS delay_trend
        FROM inst ORDER BY sk_id_prev, num
        """
    ).fetchall()
    con.close()
    assert rows == [
        (1, 1, None),
        (1, 2, 3.0),
        (1, 3, -1.0),
        (2, 1, None),
        (2, 2, -3.0),
    ]


def test_row_number_latest_status() -> None:
    """ROW_NUMBER dedup keeping the most recent month, matching 03_bureau_balance."""
    con = duckdb.connect()
    con.execute("CREATE TABLE bb(sk_id_bureau INT, months_balance INT, status VARCHAR)")
    con.execute(
        "INSERT INTO bb VALUES "
        "(10, -3, '2'), (10, -1, '0'), (10, -2, '1'), "  # latest is months -1 -> '0'
        "(20, -5, '4'), (20, -4, '3')"  # latest is months -4 -> '3'
    )
    rows = con.execute(
        """
        WITH ranked AS (
            SELECT sk_id_bureau, months_balance, status,
                   ROW_NUMBER() OVER (
                       PARTITION BY sk_id_bureau ORDER BY months_balance DESC
                   ) AS rn
            FROM bb
        )
        SELECT sk_id_bureau, status FROM ranked WHERE rn = 1
        ORDER BY sk_id_bureau
        """
    ).fetchall()
    con.close()
    assert rows == [(10, "0"), (20, "3")]


def test_rolling_six_month_average() -> None:
    """Frame-based rolling AVG over 6 months, matching 06_credit_card_features."""
    con = duckdb.connect()
    con.execute("CREATE TABLE cc(sk_id_prev INT, months_balance INT, utilization DOUBLE)")
    con.execute(
        "INSERT INTO cc VALUES "
        "(1, -6, 1.0), (1, -5, 2.0), (1, -4, 3.0), (1, -3, 4.0), "
        "(1, -2, 5.0), (1, -1, 6.0), (1, 0, 7.0)"
    )
    rows = con.execute(
        """
        SELECT months_balance,
               AVG(utilization) OVER (
                   PARTITION BY sk_id_prev ORDER BY months_balance
                   ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
               ) AS rolling_util_6m
        FROM cc ORDER BY months_balance
        """
    ).fetchall()
    con.close()
    # first row is its own value; the last row averages months -5..0 = 2..7 -> 4.5
    assert rows[0] == (-6, 1.0)
    assert rows[3] == (-3, 2.5)  # months -6..-3 = 1,2,3,4
    assert rows[-1] == (0, 4.5)


# --- real-data assembly test, skip when absent -----------------------------


@pytest.mark.skipif(not DB_PATH.exists(), reason="ingested DuckDB not present")
def test_feature_table_shape() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "features" not in tables:
            pytest.skip("features table not built; run scoregate.features")

        total, distinct = con.execute(
            "SELECT count(*), count(DISTINCT SK_ID_CURR) FROM features"
        ).fetchone()
        columns = {r[0] for r in con.execute("DESCRIBE features").fetchall()}

    # exactly one row per applicant, matching application_train
    assert total == APPLICATION_TRAIN_ROWS
    assert distinct == APPLICATION_TRAIN_ROWS
    # gender never leaks into the model feature table
    assert "CODE_GENDER" not in columns


@pytest.mark.skipif(not DB_PATH.exists(), reason="ingested DuckDB not present")
def test_fairness_metadata_holds_gender() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "fairness_metadata" not in tables:
            pytest.skip("fairness_metadata table not built; run scoregate.features")

        total = con.execute("SELECT count(*) FROM fairness_metadata").fetchone()[0]
        columns = {r[0] for r in con.execute("DESCRIBE fairness_metadata").fetchall()}

    assert total == APPLICATION_TRAIN_ROWS
    assert {"SK_ID_CURR", "CODE_GENDER", "age_band"} <= columns
