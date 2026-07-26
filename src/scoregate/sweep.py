"""One W&B sweep trial: 5-fold cross-validated AUC of the LightGBM challenger.

The sweep tunes the challenger's hyperparameters with the objective of mean 5-fold CV
AUC computed on the training split only. The frozen holdout is never touched during
tuning. This is deliberately separate from make train, which uses the frozen
configs/lightgbm.yaml; the winning params are written back into that file as their own
commit, after review.

The cross-validation splits inside train, so this never leaks the holdout, and the
folds are seeded for reproducibility. wandb is imported only in main(), so the CV core
stays importable and testable without a tracking dependency.

main() is exposed as the scoregate-sweep console entry point, which the W&B agent runs
per trial. That shim points at the venv interpreter directly, which is how the agent
finds the package on Windows where a bare "python" resolves to the Store alias.
"""

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from scoregate.config import SEED
from scoregate.metrics import roc_auc

# Held fixed during the sweep; only the five tuned params vary.
FIXED_PARAMS = {
    "n_estimators": 300,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
}


def cross_val_auc(params: dict[str, Any], features: pd.DataFrame, target: pd.Series) -> float:
    """Mean AUC over 5 stratified folds of the training data."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs: list[float] = []
    for train_idx, val_idx in folds.split(features, target):
        model = LGBMClassifier(
            **params,
            class_weight="balanced",
            random_state=SEED,
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
        )
        model.fit(features.iloc[train_idx], target.iloc[train_idx])
        proba = np.asarray(model.predict_proba(features.iloc[val_idx]))[:, 1]
        aucs.append(roc_auc(target.iloc[val_idx], proba))
    return float(np.mean(aucs))


def _load_train(db_path: Path) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(
            """
            SELECT f.*
            FROM features f
            JOIN split_assignment s USING (SK_ID_CURR)
            WHERE s.split = 'train'
            ORDER BY f.SK_ID_CURR
            """
        ).df()


def main() -> None:
    import wandb

    run = wandb.init()
    params = {**FIXED_PARAMS, **dict(wandb.config)}

    train = _load_train(Path("data/scoregate.duckdb"))
    feature_cols = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    auc = cross_val_auc(params, train[feature_cols], train["TARGET"])

    wandb.log({"cv_auc": auc})
    run.finish()


if __name__ == "__main__":
    main()
