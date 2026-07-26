"""Tests for the deterministic stratified split.

The determinism tests run on synthetic frames and always run. The real-data test
skips cleanly when the built DuckDB is absent, and proves the rebuilt split still
matches the committed manifest, so holdout membership stays auditable.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from scoregate.config import HOLDOUT_FRACTION
from scoregate.split import assign_splits, build_manifest

DB_PATH = Path("data/scoregate.duckdb")
MANIFEST_PATH = Path("split_manifest.json")


def _synthetic_ids_targets(n: int = 5000) -> pd.DataFrame:
    ids = np.arange(100_000, 100_000 + n, dtype="int64")
    targets = np.zeros(n, dtype="int64")
    targets[::12] = 1  # roughly an 8% base rate
    return pd.DataFrame({"SK_ID_CURR": ids, "TARGET": targets})


# --- determinism, always run -----------------------------------------------


def test_split_is_deterministic() -> None:
    frame = _synthetic_ids_targets()
    first = assign_splits(frame)
    second = assign_splits(frame)
    pd.testing.assert_frame_equal(first, second)


def test_split_is_order_invariant() -> None:
    frame = _synthetic_ids_targets()
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    from_ordered = assign_splits(frame).reset_index(drop=True)
    from_shuffled = assign_splits(shuffled).reset_index(drop=True)
    pd.testing.assert_frame_equal(from_ordered, from_shuffled)


def test_split_ratio_and_stratification() -> None:
    frame = _synthetic_ids_targets(10_000)
    merged = assign_splits(frame).merge(frame, on="SK_ID_CURR")
    holdout = merged[merged["split"] == "holdout"]
    train = merged[merged["split"] == "train"]
    assert abs(len(holdout) / len(merged) - HOLDOUT_FRACTION) < 0.01
    assert abs(train["TARGET"].mean() - holdout["TARGET"].mean()) < 0.01


# --- real data, skip when absent -------------------------------------------


@pytest.mark.skipif(not DB_PATH.exists(), reason="ingested DuckDB not present")
def test_real_split_matches_manifest() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "features" not in tables:
            pytest.skip("features table not built; run scoregate.features")
        ids_targets = con.execute(
            "SELECT SK_ID_CURR, TARGET FROM features ORDER BY SK_ID_CURR"
        ).df()

    assignment = assign_splits(ids_targets)
    manifest = build_manifest(assignment.merge(ids_targets, on="SK_ID_CURR"))
    committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    # rebuilt membership still hashes to the committed artifact
    for name in ("train", "holdout"):
        assert manifest.splits[name].id_sha256 == committed["splits"][name]["id_sha256"]

    # splits are disjoint and exhaustive
    all_ids = set(ids_targets["SK_ID_CURR"])
    train_ids = set(assignment.loc[assignment["split"] == "train", "SK_ID_CURR"])
    holdout_ids = set(assignment.loc[assignment["split"] == "holdout", "SK_ID_CURR"])
    assert train_ids.isdisjoint(holdout_ids)
    assert train_ids | holdout_ids == all_ids

    # holdout default rate stays within 0.5pp of train
    train_rate = manifest.splits["train"].default_rate
    holdout_rate = manifest.splits["holdout"].default_rate
    assert abs(train_rate - holdout_rate) <= 0.005
