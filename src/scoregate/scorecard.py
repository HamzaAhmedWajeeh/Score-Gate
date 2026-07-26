"""Logistic scorecard on WOE features with points scaling.

The scorecard fits a logistic regression on the WOE-transformed selected features
(fit on the training split only, reusing the binning persisted upstream) and scales
the model into credit points.

Scaling convention: score = offset + factor * ln(good:bad odds), with factor and
offset chosen so a score of 600 sits at 50:1 good:bad odds and every extra 20 points
(the PDO) doubles the odds. A higher score therefore means lower risk. Since the
logistic regression predicts P(bad), ln(good:bad odds) = -logit(P(bad)) = minus the
model's decision function, which is where the minus sign in score() comes from.

class_weight='balanced' pulls the minority default class into the coefficients. That
distorts the raw probabilities, which is deliberate and handled later by calibration;
the scorecard's own job is ranking and points, and both survive the reweighting.

Points decompose per feature: each (feature, bin) contributes
base_points - factor * coef * WOE(bin), where base_points spreads the offset and the
intercept evenly across features. The per-applicant points therefore sum exactly to
the scaled score.
"""

import argparse
import json
import math
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb
import pandas as pd
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression

from scoregate.config import SEED
from scoregate.metrics import gini, ks_statistic, roc_auc

BASE_SCORE = 600.0
BASE_ODDS = 50.0  # good:bad odds at the base score
PDO = 20.0  # points to double the odds


@dataclass(frozen=True)
class ScorecardScaling:
    base_score: float = BASE_SCORE
    base_odds: float = BASE_ODDS
    pdo: float = PDO

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2.0)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)


@dataclass
class Scorecard:
    binning: BinningProcess
    features: list[str]
    scaling: ScorecardScaling = field(default_factory=ScorecardScaling)
    model: LogisticRegression = field(
        default_factory=lambda: LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        )
    )

    def _woe(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.binning.transform(features[self.features], metric="woe")

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "Scorecard":
        self.model.fit(self._woe(features), target)
        return self

    def predict_pd(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict_proba(self._woe(features))[:, 1], index=features.index)

    def score(self, features: pd.DataFrame) -> pd.Series:
        logit_bad = self.model.decision_function(self._woe(features))
        return pd.Series(
            self.scaling.offset - self.scaling.factor * logit_bad, index=features.index
        )

    def points_table(self) -> pd.DataFrame:
        n = len(self.features)
        intercept = float(self.model.intercept_[0])
        base_points = (self.scaling.offset - self.scaling.factor * intercept) / n
        rows: list[dict[str, object]] = []
        for j, feature in enumerate(self.features):
            coef = float(self.model.coef_[0][j])
            table = self.binning.get_binned_variable(feature).binning_table.build()
            for _, entry in table.iterrows():
                woe = pd.to_numeric(entry["WoE"], errors="coerce")
                if pd.isna(woe):  # skip the Totals row
                    continue
                rows.append(
                    {
                        "feature": feature,
                        "bin": str(entry["Bin"]),
                        "woe": float(woe),
                        "points": base_points - self.scaling.factor * coef * float(woe),
                    }
                )
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        # Persist only library objects and plain values, never this module's own
        # classes, so the artifact loads regardless of how the trainer was invoked.
        payload = {
            "model": self.model,
            "binning": self.binning,
            "features": self.features,
            "scaling": asdict(self.scaling),
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: Path) -> "Scorecard":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        scorecard = cls(
            binning=payload["binning"],
            features=payload["features"],
            scaling=ScorecardScaling(**payload["scaling"]),
        )
        scorecard.model = payload["model"]
        return scorecard


def _load_selected(selection_path: Path) -> list[str]:
    record = json.loads(selection_path.read_text(encoding="utf-8"))
    return [f["name"] for f in record["features"] if f["selected"]]


def build_scorecard(
    db_path: Path,
    binning_artifact: Path,
    selection_path: Path,
    scorecard_artifact: Path,
) -> Scorecard:
    """Fit the scorecard on train rows, persist it, and report train discrimination."""
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found. Run scoregate.split first.")

    selected = _load_selected(selection_path)
    binning = BinningProcess.load(str(binning_artifact))

    with duckdb.connect(str(db_path), read_only=True) as con:
        train = con.execute(
            """
            SELECT f.*
            FROM features f
            JOIN split_assignment s USING (SK_ID_CURR)
            WHERE s.split = 'train'
            ORDER BY f.SK_ID_CURR
            """
        ).df()

    scorecard = Scorecard(binning=binning, features=selected)
    scorecard.fit(train, train["TARGET"])

    scorecard_artifact.parent.mkdir(parents=True, exist_ok=True)
    scorecard.save(scorecard_artifact)

    pd_hat = scorecard.predict_pd(train)
    print(f"scorecard fit on {len(train):,} train rows, {len(selected)} features")
    print(f"train AUC:  {roc_auc(train['TARGET'], pd_hat):.4f}")
    print(f"train Gini: {gini(train['TARGET'], pd_hat):.4f}")
    print(f"train KS:   {ks_statistic(train['TARGET'], pd_hat):.4f}")
    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/scoregate.duckdb"))
    parser.add_argument("--binning", type=Path, default=Path("artifacts/binning_process.pkl"))
    parser.add_argument("--selection", type=Path, default=Path("feature_selection.json"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/scorecard.pkl"))
    args = parser.parse_args()

    build_scorecard(args.db, args.binning, args.selection, args.artifact)


if __name__ == "__main__":
    main()
