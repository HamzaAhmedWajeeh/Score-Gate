"""Isotonic probability calibration for both models.

class_weight='balanced' distorts raw probabilities: predicted PDs run well above
observed default rates. Both models are calibrated with isotonic regression via
CalibratedClassifierCV(cv=5) on the training portion only. ensemble=False keeps the
base model the one fit on all of train and fits a single isotonic map on the
cross-validated predictions, so the scorecard's points and scaling are untouched;
calibration wraps the probability output only.

The holdout is used only to measure the after-calibration reliability curve and Brier
score, never to fit the calibrator. Isotonic is monotonic, so ranking metrics
(AUC/KS/Gini) are unchanged and the champion decision in evaluation.json still stands.
"""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from scoregate.challenger import Challenger, load_params
from scoregate.config import SEED
from scoregate.evaluate import reliability_curve
from scoregate.metrics import brier_score
from scoregate.scorecard import Scorecard


class _Scorer(Protocol):
    def predict_pd(self, features: pd.DataFrame) -> pd.Series: ...


@dataclass
class CalibratedModel:
    estimator: Any  # a fitted CalibratedClassifierCV
    features: list[str]
    binning: Any = None  # optional fixed WOE transform, used by the scorecard

    def predict_pd(self, features: pd.DataFrame) -> pd.Series:
        data = features[self.features]
        if self.binning is not None:
            data = self.binning.transform(data, metric="woe")
        proba = np.asarray(self.estimator.predict_proba(data))
        return pd.Series(proba[:, 1], index=features.index)

    def save(self, path: Path) -> None:
        # only library objects and a list of names, so the artifact loads anywhere
        with path.open("wb") as handle:
            pickle.dump(
                {"estimator": self.estimator, "features": self.features, "binning": self.binning},
                handle,
            )

    @classmethod
    def load(cls, path: Path) -> "CalibratedModel":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return cls(
            estimator=payload["estimator"],
            features=payload["features"],
            binning=payload.get("binning"),
        )


def calibrate(estimator: Any, features: pd.DataFrame, target: pd.Series) -> CalibratedClassifierCV:
    """Isotonic calibration via 5-fold CV, base refit on all of train (ensemble=False)."""
    calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv=5, ensemble=False)
    calibrated.fit(features, target)
    return calibrated


def _load_split(con: duckdb.DuckDBPyConnection, split: str) -> pd.DataFrame:
    return con.execute(
        """
        SELECT f.*
        FROM features f
        JOIN split_assignment s USING (SK_ID_CURR)
        WHERE s.split = ?
        ORDER BY f.SK_ID_CURR
        """,
        [split],
    ).df()


def build_calibration(
    db_path: Path,
    selection_path: Path,
    scorecard_artifact: Path,
    challenger_artifact: Path,
    params_path: Path,
    calibrated_scorecard_path: Path,
    calibrated_challenger_path: Path,
    calibration_path: Path,
) -> dict[str, object]:
    """Calibrate both models on train, persist them, and record before/after on holdout."""
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = [f["name"] for f in selection["features"] if f["selected"]]
    params = load_params(params_path)

    raw_scorecard = Scorecard.load(scorecard_artifact)
    raw_challenger = Challenger.load(challenger_artifact)
    challenger_features = raw_challenger.features

    with duckdb.connect(str(db_path), read_only=True) as con:
        train = _load_split(con, "train")
        holdout = _load_split(con, "holdout")
    y_train = train["TARGET"]
    y_holdout = holdout["TARGET"]

    # scorecard: transform to WOE once with the scorecard's own fixed binning, then
    # calibrate only the logistic layer. Refitting binning per CV fold is both
    # non-deterministic (optbinning's optimiser) and unnecessary, since the binning is
    # frozen; this keeps the calibrated scorecard on the exact points-scorecard binning.
    woe_train = raw_scorecard.binning.transform(train[selected], metric="woe")
    scorecard_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    calibrated_scorecard = CalibratedModel(
        estimator=calibrate(scorecard_lr, woe_train, y_train),
        features=selected,
        binning=raw_scorecard.binning,
    )

    # challenger: the same LightGBM configuration as the raw challenger
    challenger_estimator = LGBMClassifier(
        **params,
        class_weight="balanced",
        random_state=SEED,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    calibrated_challenger = CalibratedModel(
        estimator=calibrate(challenger_estimator, train[challenger_features], y_train),
        features=challenger_features,
    )

    calibrated_scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    calibrated_scorecard.save(calibrated_scorecard_path)
    calibrated_challenger.save(calibrated_challenger_path)

    pairs: list[tuple[str, _Scorer, _Scorer]] = [
        ("scorecard", raw_scorecard, calibrated_scorecard),
        ("challenger", raw_challenger, calibrated_challenger),
    ]
    models: dict[str, object] = {}
    for name, raw, calibrated in pairs:
        before = raw.predict_pd(holdout)
        after = calibrated.predict_pd(holdout)
        models[name] = {
            "before": {
                "brier": round(brier_score(y_holdout, before), 6),
                "reliability": reliability_curve(y_holdout, before),
            },
            "after": {
                "brier": round(brier_score(y_holdout, after), 6),
                "reliability": reliability_curve(y_holdout, after),
            },
        }

    record: dict[str, object] = {"seed": SEED, "method": "isotonic", "models": models}
    calibration_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    for name, model in models.items():
        before_brier = model["before"]["brier"]  # type: ignore[index]
        after_brier = model["after"]["brier"]  # type: ignore[index]
        print(f"{name:<11} holdout Brier {before_brier:.4f} -> {after_brier:.4f} (isotonic)")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/scoregate.duckdb"))
    parser.add_argument("--selection", type=Path, default=Path("feature_selection.json"))
    parser.add_argument("--scorecard", type=Path, default=Path("artifacts/scorecard.pkl"))
    parser.add_argument("--challenger", type=Path, default=Path("artifacts/challenger.pkl"))
    parser.add_argument("--params", type=Path, default=Path("configs/lightgbm.yaml"))
    parser.add_argument(
        "--calibrated-scorecard", type=Path, default=Path("artifacts/calibrated_scorecard.pkl")
    )
    parser.add_argument(
        "--calibrated-challenger", type=Path, default=Path("artifacts/calibrated_challenger.pkl")
    )
    parser.add_argument("--calibration", type=Path, default=Path("calibration.json"))
    args = parser.parse_args()

    build_calibration(
        args.db,
        args.selection,
        args.scorecard,
        args.challenger,
        args.params,
        args.calibrated_scorecard,
        args.calibrated_challenger,
        args.calibration,
    )


if __name__ == "__main__":
    main()
