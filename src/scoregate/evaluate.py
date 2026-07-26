"""Evaluate the scorecard and challenger on train and the frozen holdout.

The holdout is touched here for reporting only: nothing is fit, tuned, selected, or
calibrated on it. evaluation.json is the committed comparison record and the artifact
MLflow reads at registration to choose champion versus challenger.

train_minus_holdout_gini is the overfitting gap, small for the scorecard and larger
for the challenger, and it is the number that flags a sweep that has begun to overfit.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pandas as pd

from scoregate.challenger import Challenger
from scoregate.config import SEED
from scoregate.metrics import gini, ks_statistic, roc_auc
from scoregate.scorecard import Scorecard

PredictPD = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class SplitMetrics:
    auc: float
    ks: float
    gini: float


@dataclass(frozen=True)
class ModelEvaluation:
    train: SplitMetrics
    holdout: SplitMetrics
    train_minus_holdout_gini: float


def _metrics(target: pd.Series, pd_hat: pd.Series) -> SplitMetrics:
    return SplitMetrics(
        auc=round(roc_auc(target, pd_hat), 6),
        ks=round(ks_statistic(target, pd_hat), 6),
        gini=round(gini(target, pd_hat), 6),
    )


def evaluate_model(
    predict_pd: PredictPD,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    target_col: str = "TARGET",
) -> ModelEvaluation:
    """Score one model on both splits and record the train-minus-holdout Gini gap."""
    train_metrics = _metrics(train[target_col], predict_pd(train))
    holdout_metrics = _metrics(holdout[target_col], predict_pd(holdout))
    return ModelEvaluation(
        train=train_metrics,
        holdout=holdout_metrics,
        train_minus_holdout_gini=round(train_metrics.gini - holdout_metrics.gini, 6),
    )


def _load_split(con: duckdb.DuckDBPyConnection, split: str) -> pd.DataFrame:
    return con.execute(
        """
        SELECT f.*
        FROM features f
        JOIN split_assignment s USING (SK_ID_CURR)
        WHERE s.split = ?
        """,
        [split],
    ).df()


def build_evaluation(
    db_path: Path,
    scorecard_artifact: Path,
    challenger_artifact: Path,
    evaluation_path: Path,
) -> dict[str, object]:
    """Evaluate both models on both splits and write the committed evaluation record."""
    scorecard = Scorecard.load(scorecard_artifact)
    challenger = Challenger.load(challenger_artifact)

    with duckdb.connect(str(db_path), read_only=True) as con:
        train = _load_split(con, "train")
        holdout = _load_split(con, "holdout")

    evaluations = {
        "scorecard": evaluate_model(scorecard.predict_pd, train, holdout),
        "challenger": evaluate_model(challenger.predict_pd, train, holdout),
    }

    record: dict[str, object] = {
        "seed": SEED,
        "models": {name: asdict(ev) for name, ev in evaluations.items()},
    }
    evaluation_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    for name, ev in evaluations.items():
        print(
            f"{name:<11} holdout: AUC {ev.holdout.auc:.4f}  KS {ev.holdout.ks:.4f}  "
            f"Gini {ev.holdout.gini:.4f}   "
            f"(train Gini {ev.train.gini:.4f}, overfit gap {ev.train_minus_holdout_gini:.4f})"
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/scoregate.duckdb"))
    parser.add_argument("--scorecard", type=Path, default=Path("artifacts/scorecard.pkl"))
    parser.add_argument("--challenger", type=Path, default=Path("artifacts/challenger.pkl"))
    parser.add_argument("--evaluation", type=Path, default=Path("evaluation.json"))
    args = parser.parse_args()

    build_evaluation(args.db, args.scorecard, args.challenger, args.evaluation)


if __name__ == "__main__":
    main()
