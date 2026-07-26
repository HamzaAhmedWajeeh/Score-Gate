"""Tests for the train orchestrator's read-and-mirror helpers.

These cover the pure helpers that reshape already-persisted records for W&B. The
full online/offline pipeline is verified end to end outside the unit suite.
"""

from pathlib import Path

from scoregate.train import _feature_iv_table, _flat_metrics, _reliability_figure

SELECTION_PATH = Path("feature_selection.json")


def test_flat_metrics_flattens_record() -> None:
    record = {
        "train": {"auc": 0.80, "ks": 0.40, "gini": 0.60},
        "holdout": {"auc": 0.75, "ks": 0.37, "gini": 0.50},
        "train_minus_holdout_gini": 0.10,
    }
    metrics = _flat_metrics(record)
    assert metrics["train/auc"] == 0.80
    assert metrics["holdout/gini"] == 0.50
    assert metrics["overfit_gap_gini"] == 0.10


def test_reliability_figure_builds() -> None:
    reliability = [
        {"mean_predicted": 0.1, "observed_rate": 0.05, "count": 10},
        {"mean_predicted": 0.5, "observed_rate": 0.30, "count": 10},
    ]
    figure = _reliability_figure(reliability, "test")
    assert figure is not None
    assert len(figure.axes) == 1


def test_feature_iv_table_from_committed_selection() -> None:
    table = _feature_iv_table(SELECTION_PATH)
    assert set(table.columns) == {"feature", "iv", "disposition"}
    assert len(table) == 34  # every candidate feature, selected or dropped
