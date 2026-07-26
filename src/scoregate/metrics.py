"""KS, Gini, and PSI, hand-rolled.

KS and Gini are computed here by hand rather than pulled from sklearn: the point is
to own the arithmetic behind the scorecard's headline numbers. PSI lives here too
because Phase 3's drift monitor reuses exactly this function.

PSI has three design choices frozen below, because each one silently changes the
number if you get it wrong:

1. Binning. Ten quantile bins are cut from the REFERENCE (expected) distribution,
   and those exact edges are frozen and reused to bucket the actual distribution.
   The actual is never re-quantiled on its own: independent quantiles put the same
   fraction in every bin by construction, which hides the very shift PSI exists to
   catch. Edge computation therefore lives with the reference, and psi(expected,
   actual) takes expected first for that reason.

2. Empty-bin epsilon. Any zero proportion is replaced with 1e-6 before the log, so
   a bin the actual never populates contributes a large but finite amount instead
   of infinity. This keeps a total-population shift detectable as a big PSI rather
   than a NaN that quietly drops out of comparisons.

3. Categorical vs numeric. Numeric features use the quantile edges above.
   Categorical features (Phase 3 monitors CODE_GENDER-style fields) bucket on
   category membership over the union of categories seen in either distribution, so
   a category that appears or vanishes is caught, with the same epsilon.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

# Credit-standard PSI reading. 0.25 is the threshold Phase 3 wires to auto
# retraining tickets, so the constant has exactly one home, here.
PSI_NO_SHIFT = 0.10  # below this: no material shift
PSI_SIGNIFICANT = 0.25  # at or above this: significant shift, the Phase 3 ticket trigger
# between the two: moderate shift, worth watching
PSI_EPSILON = 1e-6
PSI_DEFAULT_BINS = 10


def psi_severity(value: float) -> str:
    """Map a PSI value to the credit-standard band label."""
    if value < PSI_NO_SHIFT:
        return "none"
    if value < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def _drop_nan(values: ArrayLike) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[~np.isnan(arr)]


def _psi_from_counts(expected_counts: NDArray[np.intp], actual_counts: NDArray[np.intp]) -> float:
    expected_prop = expected_counts.astype(np.float64) / float(expected_counts.sum())
    actual_prop = actual_counts.astype(np.float64) / float(actual_counts.sum())
    expected_prop = np.where(expected_prop == 0.0, PSI_EPSILON, expected_prop)
    actual_prop = np.where(actual_prop == 0.0, PSI_EPSILON, actual_prop)
    return float(np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop)))


def psi(expected: ArrayLike, actual: ArrayLike, n_bins: int = PSI_DEFAULT_BINS) -> float:
    """Population Stability Index for a numeric feature, bins frozen from expected."""
    expected_arr = _drop_nan(expected)
    actual_arr = _drop_nan(actual)
    edges = np.unique(np.quantile(expected_arr, np.linspace(0.0, 1.0, n_bins + 1)))
    # Drop the outer edges so the tail bins are open-ended and every actual value,
    # including any beyond the reference range, lands in a bin.
    interior = edges[1:-1]
    n_bins_actual = len(interior) + 1
    expected_counts = np.bincount(np.digitize(expected_arr, interior), minlength=n_bins_actual)
    actual_counts = np.bincount(np.digitize(actual_arr, interior), minlength=n_bins_actual)
    return _psi_from_counts(expected_counts, actual_counts)


def psi_categorical(expected: ArrayLike, actual: ArrayLike) -> float:
    """PSI for a categorical feature, keyed on the union of observed categories."""
    expected_arr = np.asarray(expected)
    actual_arr = np.asarray(actual)
    categories = np.unique(np.concatenate([expected_arr, actual_arr]))
    expected_counts = np.array([int((expected_arr == c).sum()) for c in categories], dtype=np.intp)
    actual_counts = np.array([int((actual_arr == c).sum()) for c in categories], dtype=np.intp)
    return _psi_from_counts(expected_counts, actual_counts)


def _average_ranks(values: FloatArray) -> FloatArray:
    """1-based ranks with ties resolved to their average, for the AUC identity."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    n = len(values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """AUC via the Mann-Whitney rank identity: mean rank of positives, normalised."""
    y = np.asarray(y_true)
    ranks = _average_ranks(np.asarray(y_score, dtype=np.float64))
    positives = y == 1
    n_pos = int(positives.sum())
    n_neg = int(len(y)) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes present.")
    rank_sum_pos = float(ranks[positives].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def gini(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Gini coefficient, 2*AUC - 1."""
    return 2.0 * roc_auc(y_true, y_score) - 1.0


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Mean squared error between predicted probabilities and outcomes; lower is better."""
    y = np.asarray(y_true, dtype=np.float64)
    prob = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((prob - y) ** 2))


def ks_statistic(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """KS: the largest gap between the cumulative bad and good rates over sorted scores."""
    y = np.asarray(y_true, dtype=np.float64)
    order = np.argsort(np.asarray(y_score, dtype=np.float64), kind="mergesort")
    y_sorted = y[order]
    n_pos = float(y_sorted.sum())
    n_neg = float(len(y_sorted)) - n_pos
    if n_pos == 0.0 or n_neg == 0.0:
        raise ValueError("KS needs both classes present.")
    cum_pos = np.cumsum(y_sorted) / n_pos
    cum_neg = np.cumsum(1.0 - y_sorted) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))
