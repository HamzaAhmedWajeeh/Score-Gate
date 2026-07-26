"""Tests for the sweep's cross-validation core and its config.

The CV core is exercised on synthetic data and always runs; the W&B agent entry
point is not unit-tested since it needs a sweep context.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scoregate.sweep import FIXED_PARAMS, cross_val_auc

SWEEP_PATH = Path("configs/sweep.yaml")

FAST_PARAMS = {
    **FIXED_PARAMS,
    "n_estimators": 40,
    "num_leaves": 15,
    "max_depth": 4,
    "learning_rate": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 20,
}


def _synthetic(n: int = 1500) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n), "c": rng.normal(0, 1, n)}
    )
    logit = 1.3 * frame["a"] - 0.7 * frame["b"]
    target = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int))
    return frame, target


def test_cross_val_auc_learns_signal() -> None:
    frame, target = _synthetic()
    auc = cross_val_auc(FAST_PARAMS, frame, target)
    assert 0.5 < auc <= 1.0
    assert auc > 0.7


def test_cross_val_auc_is_deterministic() -> None:
    frame, target = _synthetic()
    assert cross_val_auc(FAST_PARAMS, frame, target) == cross_val_auc(FAST_PARAMS, frame, target)


def test_sweep_config_is_valid() -> None:
    config = yaml.safe_load(SWEEP_PATH.read_text(encoding="utf-8"))
    assert config["method"] == "bayes"
    assert config["metric"] == {"name": "cv_auc", "goal": "maximize"}
    assert set(config["parameters"]) == {
        "num_leaves",
        "max_depth",
        "learning_rate",
        "reg_lambda",
        "min_child_samples",
    }
    # tuned knobs stay out of the fixed set
    assert set(config["parameters"]).isdisjoint(FIXED_PARAMS)
