"""LightGBM challenger on raw features with class weights.

The challenger trains on the full raw feature set, not the WOE/IV-selected scorecard
set: gradient boosting handles nonlinearities, interactions, and missing values
directly, so the IV drops that shape the scorecard do not apply here. LightGBM reads
NaN natively, so the thin-file NULLs go in as-is and the model learns them.

Hyperparameters are frozen in configs/lightgbm.yaml so the baseline fit is
deterministic and reproducible. make sweep tunes from that file and writes the winning
params back into it as its own commit; nothing is tuned inline here. The determinism
flags (fixed seed, deterministic, force_col_wise) are set in code, not the tunable
config, so a reproducible fit survives any param change.

class_weight='balanced' mirrors the scorecard, so champion and challenger treat the
8% default imbalance the same way. As with the scorecard, this distorts raw
probabilities and calibration corrects them downstream.
"""

import argparse
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier

from scoregate.config import SEED
from scoregate.metrics import gini, ks_statistic, roc_auc

DEFAULT_PARAMS_PATH = Path("configs/lightgbm.yaml")


def load_params(path: Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass
class Challenger:
    features: list[str]
    params: dict[str, Any]
    model: LGBMClassifier = field(init=False)

    def __post_init__(self) -> None:
        self.model = LGBMClassifier(
            **self.params,
            class_weight="balanced",
            random_state=SEED,
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
        )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "Challenger":
        self.model.fit(features[self.features], target)
        return self

    def predict_pd(self, features: pd.DataFrame) -> pd.Series:
        proba = np.asarray(self.model.predict_proba(features[self.features]))
        return pd.Series(proba[:, 1], index=features.index)

    def save(self, path: Path) -> None:
        payload = {"model": self.model, "features": self.features, "params": self.params}
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: Path) -> "Challenger":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        challenger = cls(features=payload["features"], params=payload["params"])
        challenger.model = payload["model"]
        return challenger


def build_challenger(db_path: Path, params_path: Path, artifact_path: Path) -> Challenger:
    """Fit the challenger on train rows, persist it, and report train discrimination."""
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found. Run scoregate.split first.")

    params = load_params(params_path)
    with duckdb.connect(str(db_path), read_only=True) as con:
        train = con.execute(
            """
            SELECT f.*
            FROM features f
            JOIN split_assignment s USING (SK_ID_CURR)
            WHERE s.split = 'train'
            """
        ).df()

    features = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    challenger = Challenger(features=features, params=params).fit(train, train["TARGET"])

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    challenger.save(artifact_path)

    pd_hat = challenger.predict_pd(train)
    print(f"challenger fit on {len(train):,} train rows, {len(features)} raw features")
    print(f"train AUC:  {roc_auc(train['TARGET'], pd_hat):.4f}")
    print(f"train Gini: {gini(train['TARGET'], pd_hat):.4f}")
    print(f"train KS:   {ks_statistic(train['TARGET'], pd_hat):.4f}")
    return challenger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/scoregate.duckdb"))
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/challenger.pkl"))
    args = parser.parse_args()

    build_challenger(args.db, args.params, args.artifact)


if __name__ == "__main__":
    main()
