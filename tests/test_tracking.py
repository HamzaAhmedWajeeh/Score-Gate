"""Tests for the offline-safe W&B tracking wrapper.

These never touch the network: they exercise the no-op tracker and the mode
resolution that keeps W&B optional.
"""

import pandas as pd

from scoregate.tracking import NullTracker, RunTracker, _resolve_mode, make_tracker


def test_null_tracker_is_noop() -> None:
    tracker: RunTracker = NullTracker()
    assert tracker.enabled is False
    # every method is callable and does nothing
    tracker.log({"metric": 1.0})
    tracker.log_table("table", pd.DataFrame({"a": [1, 2]}))
    tracker.log_figure("figure", object())
    tracker.finish()


def test_resolve_mode_disabled_wins() -> None:
    assert _resolve_mode(disabled=True) == "disabled"


def test_resolve_mode_respects_explicit_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert _resolve_mode(disabled=False) == "offline"


def test_make_tracker_disabled_returns_null() -> None:
    tracker = make_tracker(
        project="scoregate", name="x", group="g", tags=[], config={}, disabled=True
    )
    assert tracker.enabled is False
    tracker.log({"metric": 1.0})  # no network, no error
    tracker.finish()
