"""Fairness snapshot at a fixed approval rate, off the frozen holdout.

At an approval cutoff set to a target approval rate (approve the lowest-risk share
of applicants by calibrated PD), we measure three quantities across CODE_GENDER and
age bands: approval-rate parity, the true-positive-rate gap (approval among applicants
who would repay, equal-opportunity), and the false-positive-rate gap (approval among
applicants who would default). The gap is max minus min across groups.

This is educational and illustrative. CODE_GENDER is never a model feature; it lives
only in the fairness metadata and is used here solely to measure outcomes. Age stays a
feature, and its parity is reported because proxy effects can remain even when a
protected attribute is excluded. Metrics come off the frozen holdout, scored with the
calibrated probabilities a deployed system would actually threshold on. The gender
slice is restricted to M/F; XNA rows are kept in the data but excluded from the gender
comparison.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from scoregate.calibration import CalibratedModel

APPROVAL_RATE = 0.80
AGE_BANDS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
GENDERS = ["F", "M"]


@dataclass(frozen=True)
class GroupRates:
    n: int
    approval_rate: float
    tpr: float  # approval rate among applicants who repay (TARGET == 0)
    fpr: float  # approval rate among applicants who default (TARGET == 1)


@dataclass(frozen=True)
class GroupComparison:
    groups: dict[str, GroupRates]
    approval_rate_gap: float
    tpr_gap: float
    fpr_gap: float


def approval_threshold(pd_hat: np.ndarray, target_rate: float) -> float:
    """The PD cutoff that approves the lowest-risk target_rate share of applicants."""
    return float(np.quantile(pd_hat, target_rate))


def _group_rates(approved: np.ndarray, target: np.ndarray) -> GroupRates:
    repays = target == 0
    defaults = target == 1
    return GroupRates(
        n=int(len(target)),
        approval_rate=round(float(approved.mean()), 6),
        tpr=round(float(approved[repays].mean()), 6) if repays.any() else float("nan"),
        fpr=round(float(approved[defaults].mean()), 6) if defaults.any() else float("nan"),
    )


def compare_groups(
    labels: pd.Series, approved: np.ndarray, target: np.ndarray, order: list[str]
) -> GroupComparison:
    present = set(labels)
    rates: dict[str, GroupRates] = {}
    for group in order:
        if group not in present:
            continue
        mask = (labels == group).to_numpy()
        rates[group] = _group_rates(approved[mask], target[mask])

    def gap(attribute: str) -> float:
        values = [float(getattr(rate, attribute)) for rate in rates.values()]
        return round(max(values) - min(values), 6)

    return GroupComparison(rates, gap("approval_rate"), gap("tpr"), gap("fpr"))


def _load_holdout(db_path: Path) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(
            """
            SELECT f.*, m.CODE_GENDER, m.age_band
            FROM features f
            JOIN split_assignment s USING (SK_ID_CURR)
            JOIN fairness_metadata m USING (SK_ID_CURR)
            WHERE s.split = 'holdout'
            ORDER BY f.SK_ID_CURR
            """
        ).df()


def build_fairness(
    db_path: Path,
    calibrated_scorecard_path: Path,
    calibrated_challenger_path: Path,
    fairness_path: Path,
    target_rate: float = APPROVAL_RATE,
) -> dict[str, object]:
    """Snapshot fairness for both calibrated models and write the committed record."""
    holdout = _load_holdout(db_path)
    target = holdout["TARGET"].to_numpy()
    gender = holdout["CODE_GENDER"]
    age_band = holdout["age_band"]

    models: dict[str, object] = {}
    for name, path in [
        ("scorecard", calibrated_scorecard_path),
        ("challenger", calibrated_challenger_path),
    ]:
        model = CalibratedModel.load(path)
        pd_hat = model.predict_pd(holdout).to_numpy()
        threshold = approval_threshold(pd_hat, target_rate)
        approved = pd_hat <= threshold  # approve the lowest-risk applicants
        models[name] = {
            "approval_pd_threshold": round(threshold, 6),
            "overall_approval_rate": round(float(approved.mean()), 6),
            "gender": asdict(compare_groups(gender, approved, target, GENDERS)),
            "age_band": asdict(compare_groups(age_band, approved, target, AGE_BANDS)),
        }

    record: dict[str, object] = {
        "approval_rate_target": target_rate,
        "framing": (
            "Educational and illustrative. CODE_GENDER is not a model feature; it is used "
            "only to measure outcomes. Metrics are computed on the frozen holdout with "
            "calibrated probabilities."
        ),
        "models": models,
    }
    fairness_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    for name, snapshot in models.items():
        gender_cmp = snapshot["gender"]  # type: ignore[index]
        age_cmp = snapshot["age_band"]  # type: ignore[index]
        print(
            f"{name:<11} approval-rate gap  gender {gender_cmp['approval_rate_gap']:.4f}  "
            f"age {age_cmp['approval_rate_gap']:.4f}   |   TPR gap  gender "
            f"{gender_cmp['tpr_gap']:.4f}  age {age_cmp['tpr_gap']:.4f}"
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/scoregate.duckdb"))
    parser.add_argument(
        "--calibrated-scorecard", type=Path, default=Path("artifacts/calibrated_scorecard.pkl")
    )
    parser.add_argument(
        "--calibrated-challenger", type=Path, default=Path("artifacts/calibrated_challenger.pkl")
    )
    parser.add_argument("--fairness", type=Path, default=Path("fairness.json"))
    parser.add_argument("--approval-rate", type=float, default=APPROVAL_RATE)
    args = parser.parse_args()

    build_fairness(
        args.db,
        args.calibrated_scorecard,
        args.calibrated_challenger,
        args.fairness,
        args.approval_rate,
    )


if __name__ == "__main__":
    main()
