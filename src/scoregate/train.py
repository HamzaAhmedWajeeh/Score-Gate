"""End-to-end training orchestrator, and the W&B mirror.

make train runs the whole reproducible pipeline in order: fit both models, evaluate
them on the frozen holdout, calibrate, snapshot fairness, and register both in MLflow
with champion/challenger aliases. One command reproduces the run and the registry
state.

Each step persists its own outputs; this orchestrator then reads those persisted
records and mirrors them into one W&B run per model. It computes and re-evaluates
nothing: evaluation.json, calibration.json, and fairness.json are the single sources
of truth, and W&B only reflects them. W&B is optional and offline-safe (see tracking);
--no-wandb disables it entirely and the pipeline produces identical artifacts either
way.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scoregate.calibration import build_calibration
from scoregate.challenger import DEFAULT_PARAMS_PATH, build_challenger
from scoregate.config import HOLDOUT_FRACTION, SEED
from scoregate.evaluate import build_evaluation
from scoregate.fairness import build_fairness
from scoregate.registry import register
from scoregate.scorecard import BASE_ODDS, BASE_SCORE, PDO, build_scorecard
from scoregate.tracking import RunTracker, make_tracker

WANDB_PROJECT = "scoregate"
RUN_GROUP = "phase1-baseline"

ARTIFACTS = Path("artifacts")
BINNING_ARTIFACT = ARTIFACTS / "binning_process.pkl"
SCORECARD_ARTIFACT = ARTIFACTS / "scorecard.pkl"
CHALLENGER_ARTIFACT = ARTIFACTS / "challenger.pkl"
CALIBRATED_SCORECARD = ARTIFACTS / "calibrated_scorecard.pkl"
CALIBRATED_CHALLENGER = ARTIFACTS / "calibrated_challenger.pkl"
SELECTION_PATH = Path("feature_selection.json")
EVALUATION_PATH = Path("evaluation.json")
CALIBRATION_PATH = Path("calibration.json")
FAIRNESS_PATH = Path("fairness.json")


def _flat_metrics(model_record: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split in ("train", "holdout"):
        for name, value in model_record[split].items():
            metrics[f"{split}/{name}"] = value
    metrics["overfit_gap_gini"] = model_record["train_minus_holdout_gini"]
    return metrics


def _fairness_metrics(model_fairness: dict[str, Any]) -> dict[str, float]:
    return {
        "fairness/gender_approval_gap": model_fairness["gender"]["approval_rate_gap"],
        "fairness/gender_tpr_gap": model_fairness["gender"]["tpr_gap"],
        "fairness/age_approval_gap": model_fairness["age_band"]["approval_rate_gap"],
        "fairness/age_tpr_gap": model_fairness["age_band"]["tpr_gap"],
    }


def _reliability_figure(reliability: list[dict[str, Any]], title: str) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    predicted = [row["mean_predicted"] for row in reliability]
    observed = [row["observed_rate"] for row in reliability]
    figure, axes = plt.subplots(figsize=(5, 5))
    axes.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    axes.plot(predicted, observed, "o-", label="model")
    axes.set_xlabel("mean predicted PD")
    axes.set_ylabel("observed default rate")
    axes.set_title(title)
    axes.legend()
    return figure


def _feature_iv_table() -> pd.DataFrame:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    return pd.DataFrame(
        [
            {"feature": f["name"], "iv": f["iv"], "disposition": f["disposition"]}
            for f in selection["features"]
        ]
    )


def _log_curves(tracker: RunTracker, model_key: str, calibration: dict[str, Any]) -> None:
    if not tracker.enabled:
        return
    reliability = calibration["models"][model_key]
    before = _reliability_figure(reliability["before"]["reliability"], f"{model_key} uncalibrated")
    after = _reliability_figure(reliability["after"]["reliability"], f"{model_key} calibrated")
    tracker.log_figure("reliability_before", before)
    tracker.log_figure("reliability_after", after)


def train(db_path: Path, *, disabled: bool) -> None:
    # Run the whole pipeline; each step persists its own outputs.
    scorecard = build_scorecard(db_path, BINNING_ARTIFACT, SELECTION_PATH, SCORECARD_ARTIFACT)
    challenger = build_challenger(db_path, DEFAULT_PARAMS_PATH, CHALLENGER_ARTIFACT)
    build_evaluation(db_path, SCORECARD_ARTIFACT, CHALLENGER_ARTIFACT, EVALUATION_PATH)
    build_calibration(
        db_path,
        SELECTION_PATH,
        SCORECARD_ARTIFACT,
        CHALLENGER_ARTIFACT,
        DEFAULT_PARAMS_PATH,
        CALIBRATED_SCORECARD,
        CALIBRATED_CHALLENGER,
        CALIBRATION_PATH,
    )
    build_fairness(db_path, CALIBRATED_SCORECARD, CALIBRATED_CHALLENGER, FAIRNESS_PATH)
    register(
        db_path,
        EVALUATION_PATH,
        CALIBRATION_PATH,
        FAIRNESS_PATH,
        CALIBRATED_SCORECARD,
        CALIBRATED_CHALLENGER,
    )

    # Read back only what the steps persisted, and mirror it into W&B.
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    fairness = json.loads(FAIRNESS_PATH.read_text(encoding="utf-8"))

    scorecard_tracker = make_tracker(
        project=WANDB_PROJECT,
        name="scorecard",
        group=RUN_GROUP,
        tags=["scorecard", "logistic"],
        config={
            "seed": SEED,
            "model": "scorecard",
            "base_score": BASE_SCORE,
            "base_odds": BASE_ODDS,
            "pdo": PDO,
            "n_features": len(scorecard.features),
            "holdout_fraction": HOLDOUT_FRACTION,
        },
        disabled=disabled,
    )
    scorecard_tracker.log(_flat_metrics(evaluation["models"]["scorecard"]))
    scorecard_tracker.log(_fairness_metrics(fairness["models"]["scorecard"]))
    scorecard_tracker.log_table("points_table", scorecard.points_table())
    scorecard_tracker.log_table("feature_ivs", _feature_iv_table())
    _log_curves(scorecard_tracker, "scorecard", calibration)
    scorecard_tracker.finish()

    challenger_tracker = make_tracker(
        project=WANDB_PROJECT,
        name="challenger",
        group=RUN_GROUP,
        tags=["challenger", "lightgbm"],
        config={
            "seed": SEED,
            "model": "challenger",
            "n_features": len(challenger.features),
            **challenger.params,
        },
        disabled=disabled,
    )
    challenger_tracker.log(_flat_metrics(evaluation["models"]["challenger"]))
    challenger_tracker.log(_fairness_metrics(fairness["models"]["challenger"]))
    importances = pd.DataFrame(
        {"feature": challenger.features, "importance": challenger.model.feature_importances_}
    ).sort_values("importance", ascending=False)
    challenger_tracker.log_table("feature_importances", importances)
    _log_curves(challenger_tracker, "challenger", calibration)
    challenger_tracker.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/scoregate.duckdb"))
    parser.add_argument("--no-wandb", action="store_true", help="disable W&B entirely")
    args = parser.parse_args()

    train(args.db, disabled=args.no_wandb)


if __name__ == "__main__":
    main()
