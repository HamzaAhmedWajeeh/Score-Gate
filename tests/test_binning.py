"""Tests for WOE/IV binning and feature selection.

Two synthetic always-run tests: a designed-monotonic predictor yields monotonic
WOE, and the documented override keeps a sub-threshold feature. The real-data test
skips when the built DuckDB is absent and checks the fitted binning reproduces the
committed selection dispositions.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from scoregate.binning import (
    KEEP_OVERRIDES,
    fit_and_select,
    select_features,
)

DB_PATH = Path("data/scoregate.duckdb")
SELECTION_PATH = Path("feature_selection.json")


# --- synthetic, always run --------------------------------------------------


def test_monotonic_predictor_yields_monotonic_woe() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    mono = np.linspace(0.0, 1.0, n)
    target = pd.Series((rng.random(n) < mono).astype(int))  # default risk rises with mono
    frame = pd.DataFrame({"mono": mono, "filler": rng.random(n)})

    scorecard_binning, dispositions = fit_and_select(frame, target, ["mono", "filler"])
    assert any(d.name == "mono" and d.selected for d in dispositions)

    table = scorecard_binning.get_binned_variable("mono").binning_table.build()
    is_bin = ~table["Bin"].astype(str).isin(["Special", "Missing", ""])
    woe = pd.to_numeric(table.loc[is_bin, "WoE"], errors="coerce").dropna().to_numpy()
    diffs = np.diff(woe)
    assert len(woe) >= 3
    assert np.all(diffs <= 1e-9) or np.all(diffs >= -1e-9)


def test_override_keeps_sub_threshold_feature() -> None:
    ivs = {"strong": 0.30, "annuity_to_income": 0.005, "weak_noise": 0.004}
    disposition = {d.name: d for d in select_features(ivs)}

    # the override key is policy data, not a name baked into the branching logic
    assert "annuity_to_income" in KEEP_OVERRIDES
    assert disposition["annuity_to_income"].disposition == "kept-override"
    assert disposition["annuity_to_income"].selected is True
    # an equally weak feature without an override still drops
    assert disposition["weak_noise"].disposition == "dropped-low-iv"
    assert disposition["weak_noise"].selected is False
    assert disposition["strong"].selected is True


# --- real data, skip when absent -------------------------------------------


@pytest.mark.skipif(not DB_PATH.exists(), reason="ingested DuckDB not present")
def test_real_binning_reproduces_selection() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "features" not in tables or "split_assignment" not in tables:
            pytest.skip("feature tables or split not built")
        train = con.execute(
            """
            SELECT f.*
            FROM features f
            JOIN split_assignment s USING (SK_ID_CURR)
            WHERE s.split = 'train'
            """
        ).df()

    variables = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    _, dispositions = fit_and_select(train, train["TARGET"], variables)

    committed = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    committed_by_name = {f["name"]: f for f in committed["features"]}

    for d in dispositions:
        assert d.disposition == committed_by_name[d.name]["disposition"]
        assert d.selected == committed_by_name[d.name]["selected"]
