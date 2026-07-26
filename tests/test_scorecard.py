"""Tests for the logistic scorecard and its points scaling.

The synthetic tests always run: the scaling constants, the direction convention
(higher score means lower risk), and the identity that per-applicant points sum to
the scaled score. The real-data test skips when the built artifact is absent.
"""

import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from optbinning import BinningProcess

from scoregate.metrics import roc_auc
from scoregate.scorecard import BASE_ODDS, BASE_SCORE, PDO, Scorecard, ScorecardScaling

DB_PATH = Path("data/scoregate.duckdb")
ARTIFACT_PATH = Path("artifacts/scorecard.pkl")


def _fit_synthetic(n: int = 3000) -> tuple[Scorecard, pd.DataFrame]:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(0.0, 1.0, n),
            "x2": rng.normal(0.0, 1.0, n),
            "x3": rng.normal(0.0, 1.0, n),
        }
    )
    logit = 0.9 * frame["x1"] - 0.6 * frame["x2"] + 0.3 * frame["x3"]
    target = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int))

    binning = BinningProcess(variable_names=["x1", "x2", "x3"])
    binning.fit(frame, target)
    scorecard = Scorecard(binning=binning, features=["x1", "x2", "x3"])
    scorecard.fit(frame, target)
    return scorecard, frame


# --- always run -------------------------------------------------------------


def test_scaling_constants() -> None:
    scaling = ScorecardScaling()
    assert (scaling.base_score, scaling.base_odds, scaling.pdo) == (BASE_SCORE, BASE_ODDS, PDO)
    assert abs(scaling.factor - PDO / math.log(2.0)) < 1e-12
    assert abs(scaling.offset - (BASE_SCORE - scaling.factor * math.log(BASE_ODDS))) < 1e-9


def test_higher_score_means_lower_risk() -> None:
    scorecard, frame = _fit_synthetic()
    score = scorecard.score(frame).to_numpy()
    pd_hat = scorecard.predict_pd(frame).to_numpy()
    # walking up the score, PD must never increase
    pd_along_score = pd_hat[np.argsort(score)]
    assert np.all(np.diff(pd_along_score) <= 1e-9)


def test_points_sum_to_score() -> None:
    scorecard, frame = _fit_synthetic()
    score = scorecard.score(frame).to_numpy()

    woe = scorecard._woe(frame).to_numpy()
    n = len(scorecard.features)
    intercept = float(scorecard.model.intercept_[0])
    base_points = (scorecard.scaling.offset - scorecard.scaling.factor * intercept) / n
    points_sum = n * base_points - scorecard.scaling.factor * (woe @ scorecard.model.coef_[0])
    assert np.allclose(points_sum, score, atol=1e-6)


def test_points_table_covers_every_feature() -> None:
    scorecard, _ = _fit_synthetic()
    table = scorecard.points_table()
    assert set(table["feature"]) == set(scorecard.features)
    assert np.isfinite(table["points"].to_numpy()).all()


# --- real data, skip when absent -------------------------------------------


@pytest.mark.skipif(not DB_PATH.exists(), reason="ingested DuckDB not present")
def test_real_scorecard_discriminates() -> None:
    if not ARTIFACT_PATH.exists():
        pytest.skip("scorecard artifact not built; run scoregate.scorecard")

    scorecard = Scorecard.load(ARTIFACT_PATH)
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "features" not in tables or "split_assignment" not in tables:
            pytest.skip("feature tables or split not built")
        train = con.execute(
            """
            SELECT f.*
            FROM features f
            JOIN split_assignment s USING (SK_ID_CURR)
            WHERE s.split = 'train'
            """
        ).df()

    pd_hat = scorecard.predict_pd(train)
    assert roc_auc(train["TARGET"], pd_hat) > 0.70
