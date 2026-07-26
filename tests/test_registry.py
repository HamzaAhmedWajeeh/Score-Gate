"""Tests for MLflow registration.

The champion selection and model card are pure and always run. The real test
registers both models into a temporary sqlite backend and loads @champion, so it
skips when the calibrated artifacts are absent.
"""

from pathlib import Path

import duckdb
import mlflow
import pytest
from mlflow.client import MlflowClient

from scoregate.registry import _model_card, pick_champion, register

DB_PATH = Path("data/scoregate.duckdb")
EVALUATION_PATH = Path("evaluation.json")
CALIBRATION_PATH = Path("calibration.json")
FAIRNESS_PATH = Path("fairness.json")
CAL_SCORECARD = Path("artifacts/calibrated_scorecard.pkl")
CAL_CHALLENGER = Path("artifacts/calibrated_challenger.pkl")


# --- always run -------------------------------------------------------------


def test_pick_champion_by_holdout_gini() -> None:
    higher_challenger = {
        "models": {
            "scorecard": {"holdout": {"gini": 0.50}},
            "challenger": {"holdout": {"gini": 0.53}},
        }
    }
    assert pick_champion(higher_challenger) == "challenger"

    higher_scorecard = {
        "models": {
            "scorecard": {"holdout": {"gini": 0.60}},
            "challenger": {"holdout": {"gini": 0.53}},
        }
    }
    assert pick_champion(higher_scorecard) == "scorecard"


def test_model_card_has_governance_sections() -> None:
    evaluation = {
        "models": {
            "scorecard": {
                "holdout": {"auc": 0.75, "ks": 0.37, "gini": 0.50},
                "train_minus_holdout_gini": 0.002,
            }
        }
    }
    calibration = {"models": {"scorecard": {"before": {"brier": 0.20}, "after": {"brier": 0.07}}}}
    fairness = {
        "models": {
            "scorecard": {
                "gender": {"approval_rate_gap": 0.05, "tpr_gap": 0.04, "fpr_gap": 0.05},
                "age_band": {"approval_rate_gap": 0.40, "tpr_gap": 0.30},
            }
        }
    }
    card = _model_card("scorecard", "champion", evaluation, calibration, fairness)
    assert "# scoregate model card: scorecard (@champion)" in card
    assert "## Metrics (frozen holdout)" in card
    assert "## Fairness snapshot" in card
    assert "## Limitations" in card
    assert "Reject inference" in card


# --- real data, skip when absent -------------------------------------------


@pytest.mark.skipif(not DB_PATH.exists(), reason="ingested DuckDB not present")
def test_register_and_load_champion(tmp_path: Path) -> None:
    if not (CAL_SCORECARD.exists() and CAL_CHALLENGER.exists()):
        pytest.skip("calibrated artifacts not built")

    tracking_db = tmp_path / "mlflow.db"
    versions = register(
        DB_PATH,
        EVALUATION_PATH,
        CALIBRATION_PATH,
        FAIRNESS_PATH,
        CAL_SCORECARD,
        CAL_CHALLENGER,
        tracking_db,
    )
    assert set(versions) == {"scorecard", "challenger"}

    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    client = MlflowClient()
    champion = client.get_model_version_by_alias("scoregate", "champion")
    challenger = client.get_model_version_by_alias("scoregate", "challenger")
    assert champion.version != challenger.version

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        sample = (
            con.execute(
                """
                SELECT f.*
                FROM features f
                JOIN split_assignment s USING (SK_ID_CURR)
                WHERE s.split = 'holdout'
                ORDER BY f.SK_ID_CURR
                LIMIT 5
                """
            )
            .df()
            .drop(columns=["SK_ID_CURR", "TARGET"])
            .astype("float64")
        )
    model = mlflow.pyfunc.load_model("models:/scoregate@champion")
    predictions = model.predict(sample)
    assert all(0.0 <= float(p) <= 1.0 for p in predictions)
