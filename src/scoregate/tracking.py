"""W&B tracking as an optional sink, never a pipeline dependency.

The pipeline computes and persists every artifact, metric, and record regardless of
whether tracking is on; this module only mirrors those already-produced numbers into
W&B. The RunTracker interface has no read method, so nothing can ever be pulled back
from W&B into a decision.

W&B is off unless a key is present: no key means offline (local files only, no
network), and --no-wandb or CI means disabled (a complete no-op). Any failure in
wandb.init falls back to the no-op tracker, so a run with no account and no network
still completes normally.
"""

import os
import sys
from typing import Any, Protocol

import pandas as pd


class RunTracker(Protocol):
    enabled: bool

    def log(self, metrics: dict[str, float]) -> None: ...
    def log_table(self, name: str, frame: pd.DataFrame) -> None: ...
    def log_figure(self, name: str, figure: Any) -> None: ...
    def finish(self) -> None: ...


class NullTracker:
    """No-op tracker used whenever W&B is disabled or unavailable."""

    enabled = False

    def log(self, metrics: dict[str, float]) -> None: ...
    def log_table(self, name: str, frame: pd.DataFrame) -> None: ...
    def log_figure(self, name: str, figure: Any) -> None: ...
    def finish(self) -> None: ...


class WandbTracker:
    enabled = True

    def __init__(self, run: Any) -> None:
        self._run = run

    def log(self, metrics: dict[str, float]) -> None:
        self._run.log(metrics)

    def log_table(self, name: str, frame: pd.DataFrame) -> None:
        import wandb

        self._run.log({name: wandb.Table(dataframe=frame)})

    def log_figure(self, name: str, figure: Any) -> None:
        import wandb

        self._run.log({name: wandb.Image(figure)})

    def finish(self) -> None:
        self._run.finish()


def _resolve_mode(disabled: bool) -> str:
    if disabled:
        return "disabled"
    env_mode = os.environ.get("WANDB_MODE")
    if env_mode:  # an explicit choice always wins
        return env_mode
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    try:
        import wandb

        if wandb.api.api_key:  # a key from `wandb login` netrc
            return "online"
    except Exception:
        pass
    return "offline"  # no key: local only, no network, never prompts


def make_tracker(
    *,
    project: str,
    name: str,
    group: str,
    tags: list[str],
    config: dict[str, Any],
    disabled: bool = False,
) -> RunTracker:
    """Return a live W&B tracker, or the no-op tracker when W&B is off or fails."""
    mode = _resolve_mode(disabled)
    if mode == "disabled":
        return NullTracker()
    try:
        import wandb

        run = wandb.init(
            project=project,
            name=name,
            group=group,
            tags=tags,
            config=config,
            mode=mode,  # type: ignore[arg-type]  # wandb validates the mode string at runtime
            reinit=True,
            settings=wandb.Settings(silent=True),
        )
        url = getattr(run, "url", None)
        if url:
            print(f"W&B run '{name}': {url}", file=sys.stderr)
        return WandbTracker(run)
    except Exception as exc:  # any failure at all must fall back cleanly
        print(f"W&B unavailable ({exc}); continuing without tracking", file=sys.stderr)
        return NullTracker()
