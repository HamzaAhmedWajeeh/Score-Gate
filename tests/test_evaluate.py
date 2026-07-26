"""Tests for the train/holdout evaluation and its committed record.

The synthetic tests always run. The real-data test skips when the built model
artifacts are absent, and checks the recomputed evaluation against the committed
evaluation.json plus the expected overfitting ordering.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from scoregate.evaluate import build_evaluation, evaluate_model

DB_PATH = Path("data/scoregate.duckdb")
SCORECARD_ARTIFACT = Path("artifacts/scorecard.pkl")
CHALLENGER_ARTIFACT = Path("artifacts/challenger.pkl")
EVALUATION_PATH = Path("evaluation.json")


# --- always run -------------------------------------------------------------


def test_evaluate_perfect_separation() -> None:
    train = pd.DataFrame({"TARGET": [0, 0, 1, 1], "x": [0.1, 0.2, 0.8, 0.9]})
    holdout = pd.DataFrame({"TARGET": [0, 1], "x": [0.2, 0.7]})
    ev = evaluate_model(lambda df: df["x"], train, holdout)
    assert ev.train.gini == 1.0
    assert ev.holdout.gini == 1.0
    assert ev.train_minus_holdout_gini == 0.0


def test_overfit_gap_is_train_minus_holdout() -> None:
    train = pd.DataFrame({"TARGET": [0, 0, 1, 1], "x": [0.1, 0.2, 0.8, 0.9]})  # perfect
    holdout = pd.DataFrame({"TARGET": [0, 1, 0, 1], "x": [0.5, 0.4, 0.3, 0.9]})  # imperfect
    ev = evaluate_model(lambda df: df["x"], train, holdout)
    assert ev.holdout.gini < ev.train.gini
    assert ev.train_minus_holdout_gini == round(ev.train.gini - ev.holdout.gini, 6)


# --- real data, skip when absent -------------------------------------------


@pytest.mark.skipif(not DB_PATH.exists(), reason="ingested DuckDB not present")
def test_real_evaluation_matches_record(tmp_path: Path) -> None:
    if not (SCORECARD_ARTIFACT.exists() and CHALLENGER_ARTIFACT.exists()):
        pytest.skip("model artifacts not built")
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "split_assignment" not in tables:
        pytest.skip("split not built")

    record = build_evaluation(
        DB_PATH, SCORECARD_ARTIFACT, CHALLENGER_ARTIFACT, tmp_path / "evaluation.json"
    )
    committed = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    models = record["models"]

    for name in ("scorecard", "challenger"):
        holdout = models[name]["holdout"]
        # recomputed holdout metrics reproduce the committed record
        assert abs(holdout["auc"] - committed["models"][name]["holdout"]["auc"]) < 1e-3
        # holdout discrimination is real, and generalisation gap is non-negative
        assert holdout["auc"] > 0.70
        assert models[name]["train_minus_holdout_gini"] >= 0.0

    # the challenger overfits more than the scorecard, as expected
    assert (
        models["challenger"]["train_minus_holdout_gini"]
        > models["scorecard"]["train_minus_holdout_gini"]
    )
