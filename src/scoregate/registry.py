"""Register both calibrated models in MLflow with cards and champion/challenger aliases.

MLflow runs against a local sqlite backend under mlruns/, so the repo stays
self-contained and needs no credentials. Both calibrated models are logged as pyfunc
models under one registered model, scoregate, each with a signature, an input example,
and a markdown model card attached as an artifact.

The champion is chosen from evaluation.json by holdout Gini, never hardcoded:
@champion points to the better version and @challenger to the other. Modern MLflow
deprecated stages in favour of aliases, and these names make the registry speak the
champion/challenger vocabulary directly.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import mlflow
import pandas as pd
from mlflow.client import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel  # type: ignore[attr-defined]  # runtime export only

from scoregate.calibration import CalibratedModel

MODEL_NAME = "scoregate"
DEFAULT_TRACKING_DB = Path("mlruns/mlflow.db")
MODEL_KEYS = ("scorecard", "challenger")


class ScoregateModel(PythonModel):
    def load_context(self, context: Any) -> None:
        from scoregate.calibration import CalibratedModel as _CalibratedModel

        self._model = _CalibratedModel.load(Path(context.artifacts["calibrated_model"]))

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> Any:
        return self._model.predict_pd(model_input).to_numpy()


def pick_champion(evaluation: dict[str, Any]) -> str:
    """The model with the higher holdout Gini; ties go to the scorecard (simpler)."""
    gini = {key: evaluation["models"][key]["holdout"]["gini"] for key in MODEL_KEYS}
    return "challenger" if gini["challenger"] > gini["scorecard"] else "scorecard"


def _model_card(
    model_key: str,
    alias: str,
    evaluation: dict[str, Any],
    calibration: dict[str, Any],
    fairness: dict[str, Any],
) -> str:
    holdout = evaluation["models"][model_key]["holdout"]
    gap = evaluation["models"][model_key]["train_minus_holdout_gini"]
    cal = calibration["models"][model_key]
    gender = fairness["models"][model_key]["gender"]
    age_band = fairness["models"][model_key]["age_band"]
    return "\n".join(
        [
            f"# scoregate model card: {model_key} (@{alias})",
            "",
            "## Purpose",
            "Credit default-risk scoring on Home Credit Default Risk data. Educational",
            "reference implementation of model governance, not financial advice.",
            "",
            "## Data",
            "application_train only, 80/20 stratified split on TARGET. Metrics are",
            "reported on the frozen holdout, touched once for evaluation.",
            "",
            "## Metrics (frozen holdout)",
            f"- AUC {holdout['auc']}, KS {holdout['ks']}, Gini {holdout['gini']}",
            f"- Overfitting gap (train minus holdout Gini): {gap}",
            f"- Brier, before to after isotonic calibration: {cal['before']['brier']} "
            f"-> {cal['after']['brier']}",
            "",
            "## Fairness snapshot (holdout, at the approval-rate cutoff)",
            f"- Gender approval-rate gap {gender['approval_rate_gap']}, "
            f"TPR gap {gender['tpr_gap']}, FPR gap {gender['fpr_gap']}",
            f"- Age-band approval-rate gap {age_band['approval_rate_gap']}, "
            f"TPR gap {age_band['tpr_gap']}",
            "",
            "## Limitations",
            "- Educational, public data. CODE_GENDER is excluded as a feature, but proxy",
            "  effects can remain, which is why outcome parity is measured.",
            "- Age is a feature; age-in-scoring is jurisdiction-dependent.",
            "- Reject inference: outcomes are observed only for approved applicants, a",
            "  known limitation of scorecards built on historical decisions.",
        ]
    )


def _feature_sample(db_path: Path, n: int = 5) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        frame = con.execute(
            """
            SELECT f.*
            FROM features f
            JOIN split_assignment s USING (SK_ID_CURR)
            WHERE s.split = 'holdout'
            ORDER BY f.SK_ID_CURR
            LIMIT ?
            """,
            [n],
        ).df()
    # cast to float so the MLflow signature is uniformly double; nullable integer
    # columns otherwise reject the thin-file NaNs. Binning and LightGBM are
    # value-based, so predictions are unchanged.
    return frame.drop(columns=["SK_ID_CURR", "TARGET"]).astype("float64")


def register(
    db_path: Path,
    evaluation_path: Path,
    calibration_path: Path,
    fairness_path: Path,
    calibrated_scorecard_path: Path,
    calibrated_challenger_path: Path,
    tracking_db: Path = DEFAULT_TRACKING_DB,
) -> dict[str, int]:
    """Log and register both calibrated models, attach cards, and set the aliases."""
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    fairness = json.loads(fairness_path.read_text(encoding="utf-8"))
    champion = pick_champion(evaluation)

    tracking_db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")

    sample = _feature_sample(db_path)
    paths = {
        "scorecard": calibrated_scorecard_path,
        "challenger": calibrated_challenger_path,
    }
    versions: dict[str, int] = {}
    for key in MODEL_KEYS:
        model = CalibratedModel.load(paths[key])
        signature = infer_signature(sample, model.predict_pd(sample).to_numpy())
        alias = "champion" if key == champion else "challenger"
        holdout = evaluation["models"][key]["holdout"]
        with mlflow.start_run(run_name=key):
            mlflow.log_metrics({f"holdout_{name}": value for name, value in holdout.items()})
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=ScoregateModel(),
                artifacts={"calibrated_model": str(paths[key])},
                signature=signature,
                input_example=sample,
                registered_model_name=MODEL_NAME,
            )
            mlflow.log_text(
                _model_card(key, alias, evaluation, calibration, fairness), "model_card.md"
            )
        versions[key] = int(info.registered_model_version)

    client = MlflowClient()
    for key in MODEL_KEYS:
        alias = "champion" if key == champion else "challenger"
        client.set_registered_model_alias(MODEL_NAME, alias, str(versions[key]))

    print(f"registered {MODEL_NAME}: " + ", ".join(f"{k} v{versions[k]}" for k in MODEL_KEYS))
    print(f"@champion -> {champion} (v{versions[champion]})")
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/scoregate.duckdb"))
    parser.add_argument("--evaluation", type=Path, default=Path("evaluation.json"))
    parser.add_argument("--calibration", type=Path, default=Path("calibration.json"))
    parser.add_argument("--fairness", type=Path, default=Path("fairness.json"))
    parser.add_argument(
        "--calibrated-scorecard", type=Path, default=Path("artifacts/calibrated_scorecard.pkl")
    )
    parser.add_argument(
        "--calibrated-challenger", type=Path, default=Path("artifacts/calibrated_challenger.pkl")
    )
    parser.add_argument("--tracking-db", type=Path, default=DEFAULT_TRACKING_DB)
    args = parser.parse_args()

    register(
        args.db,
        args.evaluation,
        args.calibration,
        args.fairness,
        args.calibrated_scorecard,
        args.calibrated_challenger,
        args.tracking_db,
    )


if __name__ == "__main__":
    main()
