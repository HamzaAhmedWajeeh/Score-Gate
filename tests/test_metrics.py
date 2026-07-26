"""Unit tests for the hand-rolled metrics.

Binning monotonicity is covered in test_binning.py, so it is not repeated here.
"""

import math

import numpy as np
from sklearn.metrics import roc_auc_score

from scoregate.metrics import (
    brier_score,
    gini,
    ks_statistic,
    psi,
    psi_categorical,
    psi_severity,
    roc_auc,
)

# --- PSI --------------------------------------------------------------------


def test_psi_zero_on_identical() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0.0, 1.0, 10_000)
    assert psi(ref, ref.copy()) == 0.0


def test_psi_rises_monotonically_with_shift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0.0, 1.0, 20_000)
    values = [psi(ref, rng.normal(shift, 1.0, 20_000)) for shift in (0.0, 0.25, 0.5, 1.0, 2.0)]
    assert all(earlier < later for earlier, later in zip(values, values[1:], strict=False))


def test_psi_two_bin_hand_computed() -> None:
    # Two bins split at 0.5. Expected is 50/50, actual is 30/70.
    # PSI = (0.3-0.5)*ln(0.3/0.5) + (0.7-0.5)*ln(0.7/0.5) = 0.16945957...
    expected = [0.0] * 50 + [1.0] * 50
    actual = [0.0] * 30 + [1.0] * 70
    hand = (0.3 - 0.5) * math.log(0.3 / 0.5) + (0.7 - 0.5) * math.log(0.7 / 0.5)
    assert abs(psi(expected, actual, n_bins=2) - hand) < 1e-9


def test_psi_empty_actual_bin_stays_finite() -> None:
    expected = np.arange(1000, dtype=float)
    actual = np.zeros(500, dtype=float)  # populates only the lowest bin
    value = psi(expected, actual)
    assert math.isfinite(value)
    assert value > 0.0


def test_psi_categorical_identical_and_new_category() -> None:
    assert psi_categorical(["M", "F", "M", "F"], ["M", "F", "M", "F"]) == 0.0
    appeared = psi_categorical(["M", "F"] * 100, ["M", "F", "X"] * 100)
    assert math.isfinite(appeared)
    assert appeared > 0.0


def test_psi_severity_bands() -> None:
    assert psi_severity(0.05) == "none"
    assert psi_severity(0.15) == "moderate"
    assert psi_severity(0.30) == "significant"
    # boundaries: 0.25 reads as significant (the Phase 3 ticket trigger)
    assert psi_severity(0.25) == "significant"
    assert psi_severity(0.10) == "moderate"


# --- KS and Gini ------------------------------------------------------------


def test_ks_and_gini_perfect_separation() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ks_statistic(y, score) == 1.0
    assert gini(y, score) == 1.0


def test_gini_near_zero_on_random() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 20_000)
    score = rng.random(20_000)
    assert abs(gini(y, score)) < 0.05


def test_brier_score() -> None:
    y = np.array([0, 0, 1, 1])
    assert brier_score(y, y.astype(float)) == 0.0  # perfect predictions
    assert brier_score(y, np.full(4, 0.5)) == 0.25  # uninformative 0.5


def test_gini_matches_sklearn_oracle() -> None:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 500)
    score = rng.random(500)
    assert abs(roc_auc(y, score) - roc_auc_score(y, score)) < 1e-9
    assert abs(gini(y, score) - (2.0 * roc_auc_score(y, score) - 1.0)) < 1e-9

    # ties in the score must still match the oracle's tie handling
    y_tied = np.array([0, 1, 0, 1, 1, 0])
    score_tied = np.array([0.5, 0.5, 0.2, 0.9, 0.5, 0.2])
    assert abs(roc_auc(y_tied, score_tied) - roc_auc_score(y_tied, score_tied)) < 1e-9
