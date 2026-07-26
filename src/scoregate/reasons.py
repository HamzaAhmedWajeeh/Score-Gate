"""Adverse-action reason codes for a single decision.

reason_codes(model, x, k) returns the top-k features that pushed this applicant
toward default, the regulatory adverse-action framing: reasons the score was worse,
not better. Each model is explained by its most faithful method.

Challenger: SHAP TreeExplainer contributions, keeping only the features with a
positive contribution (toward the default class). Scorecard: points lost against the
best attainable points for each characteristic, the standard scorecard reason-code
method, keeping only features that actually cost points. Both return the same shape,
so a caller does not care which model produced the decision.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from scoregate.challenger import Challenger
from scoregate.scorecard import Scorecard


@dataclass(frozen=True)
class ReasonCode:
    feature: str
    contribution: float  # strength of the adverse push; always positive
    value: float  # the applicant's raw feature value, for the notice


def reason_codes(model: Any, x: pd.DataFrame, k: int = 5) -> list[ReasonCode]:
    """Top-k adverse-action reasons for the single applicant in the one-row frame x."""
    if isinstance(model, Scorecard):
        return _scorecard_reasons(model, x, k)
    if isinstance(model, Challenger):
        return _challenger_reasons(model, x, k)
    raise TypeError(f"unsupported model type: {type(model).__name__}")


def _top_k(reasons: list[ReasonCode], k: int) -> list[ReasonCode]:
    reasons.sort(key=lambda reason: reason.contribution, reverse=True)
    return reasons[:k]


def _raw_value(row: pd.DataFrame, feature: str) -> float:
    """The applicant's feature value as a float, with thin-file NULLs kept as NaN."""
    value = row.iloc[0][feature]
    return float(value) if pd.notna(value) else float("nan")


def _challenger_reasons(model: Challenger, x: pd.DataFrame, k: int) -> list[ReasonCode]:
    row = x[model.features]
    explainer = shap.TreeExplainer(model.model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # SHAP is noisy about its binary output shape
        raw = explainer.shap_values(row)
    # normalise across SHAP versions: a positive-class array or a per-class list
    values = np.asarray(raw[-1] if isinstance(raw, list) else raw)
    if values.ndim == 3:  # (rows, features, classes)
        values = values[..., -1]
    contributions = values[0]  # (n_features,)
    reasons = [
        ReasonCode(feature=feature, contribution=float(value), value=_raw_value(row, feature))
        for feature, value in zip(model.features, contributions, strict=True)
        if value > 0.0  # only features pushing toward default
    ]
    return _top_k(reasons, k)


def _scorecard_reasons(model: Scorecard, x: pd.DataFrame, k: int) -> list[ReasonCode]:
    row = x[model.features]
    woe = model.binning.transform(row, metric="woe")
    n = len(model.features)
    intercept = float(model.model.intercept_[0])
    factor = model.scaling.factor
    base_points = (model.scaling.offset - factor * intercept) / n
    best_points = model.points_table().groupby("feature")["points"].max()

    reasons = []
    for j, feature in enumerate(model.features):
        coef = float(model.model.coef_[0][j])
        woe_value = float(woe.iloc[0][feature])
        actual_points = base_points - factor * coef * woe_value
        points_lost = float(best_points[feature] - actual_points)
        if points_lost > 1e-9:  # only characteristics that cost points
            reasons.append(
                ReasonCode(
                    feature=feature,
                    contribution=points_lost,
                    value=_raw_value(row, feature),
                )
            )
    return _top_k(reasons, k)
