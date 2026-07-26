"""Tests for the LightGBM challenger.

Synthetic tests always run: the frozen params load, the fit is deterministic, and
the model learns signal while reading NaN natively. The real-data test skips when the
built artifact is absent.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from scoregate.challenger import Challenger, load_params
from scoregate.metrics import roc_auc

DB_PATH = Path("data/scoregate.duckdb")
ARTIFACT_PATH = Path("artifacts/challenger.pkl")

# small params keep the synthetic tests fast; the frozen config is tested separately
FAST_PARAMS = {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 15}


def _synthetic(n: int = 2000) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "a": rng.normal(0.0, 1.0, n),
            "b": rng.normal(0.0, 1.0, n),
            "c": rng.normal(0.0, 1.0, n),
        }
    )
    logit = 1.2 * frame["a"] - 0.8 * frame["b"]
    target = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int))
    frame.loc[frame.index[:150], "c"] = np.nan  # LightGBM must read NaN natively
    return frame, target


def test_load_params() -> None:
    params = load_params()
    assert params["n_estimators"] == 300
    assert params["subsample"] == 0.8
    assert params["subsample_freq"] > 0  # otherwise subsample is silently ignored


def test_challenger_is_deterministic() -> None:
    frame, target = _synthetic()
    features = list(frame.columns)
    first = Challenger(features, FAST_PARAMS).fit(frame, target).predict_pd(frame)
    second = Challenger(features, FAST_PARAMS).fit(frame, target).predict_pd(frame)
    pd.testing.assert_series_equal(first, second)


def test_challenger_learns_signal_with_nan() -> None:
    frame, target = _synthetic()
    challenger = Challenger(list(frame.columns), FAST_PARAMS).fit(frame, target)
    pd_hat = challenger.predict_pd(frame)
    assert pd_hat.between(0.0, 1.0).all()
    assert roc_auc(target, pd_hat) > 0.75


@pytest.mark.skipif(not DB_PATH.exists(), reason="ingested DuckDB not present")
def test_real_challenger_discriminates() -> None:
    if not ARTIFACT_PATH.exists():
        pytest.skip("challenger artifact not built; run scoregate.challenger")

    challenger = Challenger.load(ARTIFACT_PATH)
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

    assert roc_auc(train["TARGET"], challenger.predict_pd(train)) > 0.70
