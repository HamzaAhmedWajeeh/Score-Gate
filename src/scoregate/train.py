"""Train orchestrator: run the model steps, then mirror their results into W&B.

This is the make train skeleton. It runs the existing build and evaluate steps and
reads the numbers they persist, then reflects them into one W&B run per model. It
computes and re-evaluates nothing: evaluation.json is the single source of truth for
metrics and reliability, and W&B only mirrors it. A value the orchestrator needs but
cannot find persisted is a signal the upstream step should persist it, not that the
orchestrator should compute it.

W&B is optional and offline-safe (see tracking). --no-wandb disables it entirely, and
the pipeline produces identical artifacts either way.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scoregate.challenger import DEFAULT_PARAMS_PATH, build_challenger
from scoregate.config import HOLDOUT_FRACTION, SEED
from scoregate.evaluate import build_evaluation
from scoregate.scorecard import BASE_ODDS, BASE_SCORE, PDO, build_scorecard
from scoregate.tracking import make_tracker

WANDB_PROJECT = "scoregate"
RUN_GROUP = "phase1-baseline"


def _flat_metrics(model_record: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split in ("train", "holdout"):
        for name, value in model_record[split].items():
            metrics[f"{split}/{name}"] = value
    metrics["overfit_gap_gini"] = model_record["train_minus_holdout_gini"]
    return metrics


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


def _feature_iv_table(selection_path: Path) -> pd.DataFrame:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    return pd.DataFrame(
        [
            {"feature": f["name"], "iv": f["iv"], "disposition": f["disposition"]}
            for f in selection["features"]
        ]
    )


def train(
    db_path: Path,
    *,
    disabled: bool,
    binning_artifact: Path = Path("artifacts/binning_process.pkl"),
    selection_path: Path = Path("feature_selection.json"),
    scorecard_artifact: Path = Path("artifacts/scorecard.pkl"),
    params_path: Path = DEFAULT_PARAMS_PATH,
    challenger_artifact: Path = Path("artifacts/challenger.pkl"),
    evaluation_path: Path = Path("evaluation.json"),
) -> None:
    # Run the pipeline steps; each one persists its own outputs.
    scorecard = build_scorecard(db_path, binning_artifact, selection_path, scorecard_artifact)
    challenger = build_challenger(db_path, params_path, challenger_artifact)
    build_evaluation(db_path, scorecard_artifact, challenger_artifact, evaluation_path)

    # Read back only what the steps already produced, and mirror it.
    record = json.loads(evaluation_path.read_text(encoding="utf-8"))
    models = record["models"]

    scorecard_run = models["scorecard"]
    tracker = make_tracker(
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
    tracker.log(_flat_metrics(scorecard_run))
    tracker.log_table("points_table", scorecard.points_table())
    tracker.log_table("feature_ivs", _feature_iv_table(selection_path))
    if tracker.enabled:
        figure = _reliability_figure(scorecard_run["holdout_reliability"], "scorecard reliability")
        tracker.log_figure("reliability_holdout", figure)
    tracker.finish()

    challenger_run = models["challenger"]
    tracker = make_tracker(
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
    tracker.log(_flat_metrics(challenger_run))
    importances = pd.DataFrame(
        {"feature": challenger.features, "importance": challenger.model.feature_importances_}
    ).sort_values("importance", ascending=False)
    tracker.log_table("feature_importances", importances)
    if tracker.enabled:
        reliability = challenger_run["holdout_reliability"]
        tracker.log_figure("reliability_holdout", _reliability_figure(reliability, "challenger"))
    tracker.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/scoregate.duckdb"))
    parser.add_argument("--no-wandb", action="store_true", help="disable W&B entirely")
    args = parser.parse_args()

    train(args.db, disabled=args.no_wandb)


if __name__ == "__main__":
    main()
