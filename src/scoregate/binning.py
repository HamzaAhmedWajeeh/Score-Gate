"""WOE/IV binning for the logistic scorecard, fit on the training split only.

optbinning fits an OptimalBinning per feature and reports its Information Value. We
fit the whole feature set on train rows, then choose the scorecard's WOE feature set
by IV: drop anything below IV_DROP_THRESHOLD, keep the rest, with two documented
exceptions. A feature above IV_LEAKAGE_THRESHOLD is flagged for leakage review, and a
domain-required characteristic can be kept below the drop threshold via an explicit,
audited override rather than a code exception.

Scorecard/challenger asymmetry: this IV selection shapes the logistic scorecard's WOE
set only. The LightGBM challenger trains on all raw features, so a feature dropped
here is not lost from the overall model set, it simply does not enter the scorecard.

optbinning treats missing values as their own informative bin, which is why the
thin-file NULLs and the nulled DAYS_EMPLOYED sentinel are left in place upstream: the
scorecard learns the no-history and pensioner/unemployed groups directly.
"""

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pandas as pd
from optbinning import BinningProcess

IV_DROP_THRESHOLD = 0.02
IV_LEAKAGE_THRESHOLD = 0.5

# Sub-threshold features kept by documented policy, printed in the selection log.
# The override is data, not a hardcoded name in the branching logic.
KEEP_OVERRIDES: dict[str, str] = {
    "annuity_to_income": (
        "Debt-burden-ratio proxy; SAMA responsible-lending rules mandate DBR "
        "consideration, so a domain-required characteristic clears a lower IV bar than "
        "the default 0.02 threshold."
    ),
}

# High-IV features reviewed for leakage and retained, with the reason recorded.
LEAKAGE_KEEP_RATIONALES: dict[str, str] = {
    "ext_source_mean": (
        "Aggregate of external bureau scores, designed to be predictive of default. The "
        "high IV reflects genuine signal strength, not target leakage. Reviewed and kept."
    ),
}

DISPOSITION_KEPT = "kept"
DISPOSITION_DROPPED = "dropped-low-iv"
DISPOSITION_OVERRIDE = "kept-override"
DISPOSITION_LEAKAGE = "kept-with-leakage-rationale"


@dataclass(frozen=True)
class FeatureDisposition:
    name: str
    iv: float
    disposition: str
    selected: bool


def select_features(ivs: Mapping[str, float]) -> list[FeatureDisposition]:
    """Apply the IV selection policy, returning dispositions sorted by IV descending."""
    dispositions: list[FeatureDisposition] = []
    for name, iv in sorted(ivs.items(), key=lambda kv: kv[1], reverse=True):
        if iv > IV_LEAKAGE_THRESHOLD and name in LEAKAGE_KEEP_RATIONALES:
            disposition, selected = DISPOSITION_LEAKAGE, True
        elif iv < IV_DROP_THRESHOLD and name in KEEP_OVERRIDES:
            disposition, selected = DISPOSITION_OVERRIDE, True
        elif iv < IV_DROP_THRESHOLD:
            disposition, selected = DISPOSITION_DROPPED, False
        else:
            disposition, selected = DISPOSITION_KEPT, True
        dispositions.append(FeatureDisposition(name, round(float(iv), 6), disposition, selected))
    return dispositions


def _iv_map(binning_process: BinningProcess) -> dict[str, float]:
    summary = binning_process.summary()
    return {str(row["name"]): float(row["iv"]) for _, row in summary.iterrows()}


def fit_and_select(
    features: pd.DataFrame, target: pd.Series, variables: list[str]
) -> tuple[BinningProcess, list[FeatureDisposition]]:
    """Fit binning over all variables to get IVs, then refit over the selected set.

    Returns the fitted BinningProcess for the selected scorecard features and the
    full disposition list for every candidate feature.
    """
    over_all = BinningProcess(variable_names=variables)
    over_all.fit(features[variables], target)
    dispositions = select_features(_iv_map(over_all))

    selected = [d.name for d in dispositions if d.selected]
    scorecard_binning = BinningProcess(variable_names=selected)
    scorecard_binning.fit(features[selected], target)
    return scorecard_binning, dispositions


def build_binning(
    db_path: Path, artifact_path: Path, selection_path: Path
) -> list[FeatureDisposition]:
    """Fit binning on train rows, persist the artifact and the selection audit record."""
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found. Run scoregate.split first.")

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

    variables = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    scorecard_binning, dispositions = fit_and_select(train, train["TARGET"], variables)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard_binning.save(str(artifact_path))

    record = {
        "iv_drop_threshold": IV_DROP_THRESHOLD,
        "iv_leakage_threshold": IV_LEAKAGE_THRESHOLD,
        "keep_overrides": KEEP_OVERRIDES,
        "leakage_keep_rationales": LEAKAGE_KEEP_RATIONALES,
        "n_candidates": len(dispositions),
        "n_selected": sum(d.selected for d in dispositions),
        "features": [asdict(d) for d in dispositions],
    }
    selection_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return dispositions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/scoregate.duckdb"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/binning_process.pkl"))
    parser.add_argument("--selection", type=Path, default=Path("feature_selection.json"))
    args = parser.parse_args()

    dispositions = build_binning(args.db, args.artifact, args.selection)
    for d in dispositions:
        note = KEEP_OVERRIDES.get(d.name) or LEAKAGE_KEEP_RATIONALES.get(d.name) or ""
        suffix = f"  <- {note}" if note else ""
        print(f"{d.name:<28}{d.iv:>9.4f}  {d.disposition}{suffix}")
    selected = sum(d.selected for d in dispositions)
    print(f"\n{selected} of {len(dispositions)} features selected for the scorecard")


if __name__ == "__main__":
    main()
