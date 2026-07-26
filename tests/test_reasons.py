"""Tests for adverse-action reason codes.

Synthetic tests always run: both models return top-k adverse reasons, all pushing
toward default and ranked by strength. The real-data test skips when the model
artifacts are absent.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from optbinning import BinningProcess

from scoregate.challenger import Challenger
from scoregate.reasons import ReasonCode, reason_codes
from scoregate.scorecard import Scorecard

DB_PATH = Path("data/scoregate.duckdb")
SCORECARD_ARTIFACT = Path("artifacts/scorecard.pkl")
CHALLENGER_ARTIFACT = Path("artifacts/challenger.pkl")

FAST_PARAMS = {"n_estimators": 40, "num_leaves": 15, "learning_rate": 0.1}


def _synthetic() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 2000
    frame = pd.DataFrame(
        {"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n), "x3": rng.normal(0, 1, n)}
    )
    logit = 1.1 * frame["x1"] - 0.9 * frame["x2"] + 0.4 * frame["x3"]
    target = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int))
    return frame, target


def _assert_adverse_and_ranked(reasons: list[ReasonCode], k: int) -> None:
    assert len(reasons) <= k
    assert all(reason.contribution > 0.0 for reason in reasons)  # only adverse pushes
    contributions = [reason.contribution for reason in reasons]
    assert contributions == sorted(contributions, reverse=True)  # ranked by strength


# --- always run -------------------------------------------------------------


def test_scorecard_reason_codes() -> None:
    frame, target = _synthetic()
    binning = BinningProcess(variable_names=["x1", "x2", "x3"])
    binning.fit(frame, target)
    scorecard = Scorecard(binning=binning, features=["x1", "x2", "x3"]).fit(frame, target)

    reasons = reason_codes(scorecard, frame.iloc[[0]], k=3)
    _assert_adverse_and_ranked(reasons, 3)
    assert all(reason.feature in scorecard.features for reason in reasons)


def test_challenger_reason_codes() -> None:
    frame, target = _synthetic()
    challenger = Challenger(list(frame.columns), FAST_PARAMS).fit(frame, target)

    reasons = reason_codes(challenger, frame.iloc[[0]], k=3)
    _assert_adverse_and_ranked(reasons, 3)


def test_unsupported_model_raises() -> None:
    with pytest.raises(TypeError):
        reason_codes(object(), pd.DataFrame({"x": [1.0]}), k=5)


# --- real data, skip when absent -------------------------------------------


@pytest.mark.skipif(not DB_PATH.exists(), reason="ingested DuckDB not present")
def test_real_reason_codes() -> None:
    if not (SCORECARD_ARTIFACT.exists() and CHALLENGER_ARTIFACT.exists()):
        pytest.skip("model artifacts not built")
    scorecard = Scorecard.load(SCORECARD_ARTIFACT)
    challenger = Challenger.load(CHALLENGER_ARTIFACT)

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

    # a high-risk applicant, who will have adverse reasons under both models
    riskiest = holdout.iloc[[int(np.argmax(challenger.predict_pd(holdout).to_numpy()))]]
    for model in (scorecard, challenger):
        reasons = reason_codes(model, riskiest, k=5)
        assert 1 <= len(reasons) <= 5
        _assert_adverse_and_ranked(reasons, 5)
