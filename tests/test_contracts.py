"""Tests for the feature-table data contract.

The negative tests are the point: a gate that has never been seen to close is
unproven. Each builds a synthetic frame that passes the contract cleanly, mutates
exactly one thing, and asserts the gate closes on that fault. They always run.
The positive test on the real table skips cleanly when the built DuckDB is absent.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from scoregate.contracts import (
    NULL_BANDS,
    validate_fairness_metadata,
    validate_features,
)

DB_PATH = Path("data/scoregate.duckdb")
APPLICATION_TRAIN_ROWS = 307_511
N = 1000

# Columns that carry no NULLs, filled with an in-range value.
NONNULL_FILL = {
    "employment_recorded": 1,
    "ext_source_count": 2,
    "AMT_INCOME_TOTAL": 100_000.0,
    "AMT_CREDIT": 500_000.0,
    "DAYS_BIRTH": -15_000,
    "CNT_CHILDREN": 0,
    "credit_to_income": 5.0,
}

# Nullable columns, each with an in-range fill value; NULLs are injected below at
# the midpoint of the column's frozen band so the baseline passes cleanly.
NULLABLE_FILL = {
    "annuity_to_income": 0.2,
    "credit_to_goods": 1.0,
    "employed_share": 0.3,
    "ext_source_mean": 0.5,
    "ext_source_min": 0.4,
    "AMT_ANNUITY": 25_000.0,
    "DAYS_EMPLOYED": -2_000.0,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.5,
    "EXT_SOURCE_3": 0.5,
    "bureau_credit_count": 3.0,
    "bureau_active_count": 1.0,
    "bureau_debt_to_credit": 0.5,
    "bureau_overdue_max": 0.0,
    "bureau_prolong_sum": 0.0,
    "bureau_recency": -500.0,
    "bureau_current_dpd_count": 0.0,
    "bureau_ever_dpd3_share": 0.0,
    "prev_app_count": 2.0,
    "prev_refused_share": 0.0,
    "prev_approved_credit_mean": 100_000.0,
    "inst_late_share": 0.1,
    "inst_shortfall_share": 0.1,
    "inst_mean_delay": 5.0,
    "inst_delay_trend": 0.0,
    "card_count": 1.0,
    "card_rolling_util_6m": 0.3,
}


def _valid_features_frame(n: int = N) -> pd.DataFrame:
    """A synthetic feature frame built to pass the contract cleanly."""
    data: dict[str, np.ndarray] = {
        "SK_ID_CURR": np.arange(n, dtype="int64"),
    }
    target = np.zeros(n, dtype="int64")
    target[: round(0.08 * n)] = 1  # 8% base rate, inside [0.06, 0.10]
    data["TARGET"] = target

    for col, val in NONNULL_FILL.items():
        data[col] = np.full(n, val)

    for col, val in NULLABLE_FILL.items():
        arr = np.full(n, float(val), dtype="float64")
        low, high = NULL_BANDS[col]
        arr[: round((low + high) / 2 * n)] = np.nan  # null rate at band midpoint
        data[col] = arr

    return pd.DataFrame(data)


def _valid_fairness_frame(n: int = N) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n, dtype="int64"),
            "CODE_GENDER": ["M", "F"] * (n // 2),
            "age_band": ["35-44"] * n,
        }
    )


# --- positive controls on the synthetic frames -----------------------------


def test_valid_frame_passes() -> None:
    validate_features(_valid_features_frame())
    validate_fairness_metadata(_valid_fairness_frame())


# --- negative tests: the gate must close on each fault ----------------------


def test_reintroduced_sentinel_fails() -> None:
    df = _valid_features_frame()
    df.loc[0, "DAYS_EMPLOYED"] = 365243
    with pytest.raises(SchemaErrors) as exc:
        validate_features(df)
    assert "sentinel" in exc.value.failure_cases.to_string()


def test_duplicate_sk_id_curr_fails() -> None:
    df = _valid_features_frame()
    df.loc[1, "SK_ID_CURR"] = df.loc[0, "SK_ID_CURR"]
    with pytest.raises(SchemaErrors) as exc:
        validate_features(df)
    assert (exc.value.failure_cases["column"] == "SK_ID_CURR").any()


def test_bureau_nulls_coalesced_trips_floor() -> None:
    df = _valid_features_frame()
    # A stray COALESCE(...,0) would erase the thin-file NULLs and crash the rate.
    df["bureau_credit_count"] = df["bureau_credit_count"].fillna(0)
    with pytest.raises(SchemaErrors) as exc:
        validate_features(df)
    assert "bureau_credit_count null rate outside" in exc.value.failure_cases.to_string()


def test_target_mean_out_of_band_fails() -> None:
    df = _valid_features_frame()
    df["TARGET"] = 0  # base rate collapses to 0, outside [0.06, 0.10]
    with pytest.raises(SchemaErrors) as exc:
        validate_features(df)
    assert "TARGET mean outside" in exc.value.failure_cases.to_string()


def test_unexpected_column_fails() -> None:
    df = _valid_features_frame()
    df["surprise_col"] = 1.0
    with pytest.raises(SchemaErrors) as exc:
        validate_features(df)
    assert "surprise_col" in exc.value.failure_cases.to_string()


# --- positive test on the real built table ---------------------------------


@pytest.mark.skipif(not DB_PATH.exists(), reason="ingested DuckDB not present")
def test_real_features_pass_contract() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "features" not in tables or "fairness_metadata" not in tables:
            pytest.skip("feature tables not built; run scoregate.features")
        features = con.execute("SELECT * FROM features").df()
        fairness = con.execute("SELECT * FROM fairness_metadata").df()

    validated = validate_features(features)
    assert len(validated) == APPLICATION_TRAIN_ROWS
    validate_fairness_metadata(fairness)
