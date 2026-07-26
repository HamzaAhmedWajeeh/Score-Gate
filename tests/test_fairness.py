"""Tests for the fairness snapshot.

Synthetic tests always run: the approval threshold hits the target rate and the
group comparison computes the rates and gaps correctly. The real-data test skips
when the calibrated artifacts are absent.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from scoregate.fairness import AGE_BANDS, approval_threshold, build_fairness, compare_groups

DB_PATH = Path("data/scoregate.duckdb")
CAL_SCORECARD = Path("artifacts/calibrated_scorecard.pkl")
CAL_CHALLENGER = Path("artifacts/calibrated_challenger.pkl")


# --- always run -------------------------------------------------------------


def test_approval_threshold_hits_target_rate() -> None:
    pd_hat = np.arange(1000) / 1000.0
    threshold = approval_threshold(pd_hat, 0.8)
    approved = pd_hat <= threshold
    assert abs(approved.mean() - 0.8) < 0.02


def test_compare_groups_rates_and_gaps() -> None:
    labels = pd.Series(["F", "F", "M", "M"])
    approved = np.array([True, False, True, True])
    target = np.array([0, 1, 0, 1])
    comparison = compare_groups(labels, approved, target, ["F", "M"])

    # F: approval 1/2; approve|repay = [T] -> 1.0; approve|default = [F] -> 0.0
    assert comparison.groups["F"].approval_rate == 0.5
    assert comparison.groups["F"].tpr == 1.0
    assert comparison.groups["F"].fpr == 0.0
    # M: approval 2/2; approve|repay = [T] -> 1.0; approve|default = [T] -> 1.0
    assert comparison.groups["M"].approval_rate == 1.0
    assert comparison.groups["M"].fpr == 1.0

    assert comparison.approval_rate_gap == 0.5
    assert comparison.fpr_gap == 1.0


def test_compare_groups_excludes_absent_labels() -> None:
    labels = pd.Series(["F", "M", "XNA"])
    approved = np.array([True, True, True])
    target = np.array([0, 1, 0])
    comparison = compare_groups(labels, approved, target, ["F", "M"])
    assert set(comparison.groups) == {"F", "M"}  # XNA never enters the gender slice


# --- real data, skip when absent -------------------------------------------


@pytest.mark.skipif(not DB_PATH.exists(), reason="ingested DuckDB not present")
def test_real_fairness_snapshot(tmp_path: Path) -> None:
    if not (CAL_SCORECARD.exists() and CAL_CHALLENGER.exists()):
        pytest.skip("calibrated artifacts not built")
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "fairness_metadata" not in tables:
        pytest.skip("fairness_metadata not built")

    record = build_fairness(
        DB_PATH, CAL_SCORECARD, CAL_CHALLENGER, tmp_path / "fairness.json", target_rate=0.8
    )
    for name in ("scorecard", "challenger"):
        snapshot = record["models"][name]
        assert set(snapshot["gender"]["groups"]) <= {"F", "M"}  # XNA excluded
        assert set(snapshot["age_band"]["groups"]) <= set(AGE_BANDS)
        assert 0.75 < snapshot["overall_approval_rate"] < 0.85
        assert snapshot["gender"]["approval_rate_gap"] >= 0.0
        assert snapshot["age_band"]["tpr_gap"] >= 0.0
