"""Tests for isotonic probability calibration.

The synthetic tests always run: calibration lowers the Brier score, keeps
probabilities in range, and preserves ranking (isotonic is monotonic). The
real-data test skips when the calibrated artifacts and record are absent.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from scoregate.calibration import CalibratedModel, calibrate
from scoregate.challenger import Challenger
from scoregate.metrics import brier_score, roc_auc
from scoregate.scorecard import Scorecard

DB_PATH = Path("data/scoregate.duckdb")
CALIBRATION_PATH = Path("calibration.json")
CAL_SCORECARD = Path("artifacts/calibrated_scorecard.pkl")
CAL_CHALLENGER = Path("artifacts/calibrated_challenger.pkl")
RAW_SCORECARD = Path("artifacts/scorecard.pkl")
RAW_CHALLENGER = Path("artifacts/challenger.pkl")


def _imbalanced() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 4000
    frame = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    logit = 1.3 * frame["a"] - 0.8 * frame["b"] - 1.5  # skewed toward the negative class
    target = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int))
    return frame, target


def _fit_calibrated(frame: pd.DataFrame, target: pd.Series) -> CalibratedModel:
    estimator = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    return CalibratedModel(calibrate(estimator, frame, target), list(frame.columns))


# --- always run -------------------------------------------------------------


def test_calibration_lowers_brier_and_preserves_ranking() -> None:
    frame, target = _imbalanced()
    calibrated = _fit_calibrated(frame, target)

    raw = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    raw.fit(frame, target)
    raw_pd = np.asarray(raw.predict_proba(frame))[:, 1]
    calibrated_pd = calibrated.predict_pd(frame).to_numpy()

    assert ((calibrated_pd >= 0.0) & (calibrated_pd <= 1.0)).all()
    assert brier_score(target, calibrated_pd) < brier_score(target, raw_pd)
    # isotonic is monotonic, so ranking is essentially unchanged
    assert abs(roc_auc(target, calibrated_pd) - roc_auc(target, raw_pd)) < 0.02


def test_calibrated_model_roundtrip(tmp_path: Path) -> None:
    frame, target = _imbalanced()
    calibrated = _fit_calibrated(frame, target)
    path = tmp_path / "cal.pkl"
    calibrated.save(path)
    reloaded = CalibratedModel.load(path)
    pd.testing.assert_series_equal(calibrated.predict_pd(frame), reloaded.predict_pd(frame))


# --- real data, skip when absent -------------------------------------------


@pytest.mark.skipif(not DB_PATH.exists(), reason="ingested DuckDB not present")
def test_real_calibration_improves_brier() -> None:
    artifacts = [CAL_SCORECARD, CAL_CHALLENGER, RAW_SCORECARD, RAW_CHALLENGER]
    if not all(p.exists() for p in artifacts):
        pytest.skip("calibrated or raw artifacts not built")

    record = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    for name in ("scorecard", "challenger"):
        model = record["models"][name]
        assert model["after"]["brier"] < model["before"]["brier"]

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        holdout = con.execute(
            """
            SELECT f.*
            FROM features f
            JOIN split_assignment s USING (SK_ID_CURR)
            WHERE s.split = 'holdout'
            ORDER BY f.SK_ID_CURR
            """
        ).df()

    pairs = [
        (Scorecard.load(RAW_SCORECARD), CalibratedModel.load(CAL_SCORECARD)),
        (Challenger.load(RAW_CHALLENGER), CalibratedModel.load(CAL_CHALLENGER)),
    ]
    for raw, calibrated in pairs:
        raw_auc = roc_auc(holdout["TARGET"], raw.predict_pd(holdout))
        calibrated_auc = roc_auc(holdout["TARGET"], calibrated.predict_pd(holdout))
        assert abs(calibrated_auc - raw_auc) < 0.01  # ranking preserved
