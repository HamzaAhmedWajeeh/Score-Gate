"""Deterministic stratified train/holdout split with a persisted assignment.

The split is two-way only: 80% train, 20% holdout, stratified on TARGET. There is
no separate validation set; sweeps and calibration both run cross-validation
inside the train portion, so the holdout is touched exactly once, at final
evaluation.

Determinism has two legs. Construction: the IDs are pulled ordered by SK_ID_CURR
and sorted again here, because a seeded split is only reproducible on a stable row
order and DuckDB gives no order without an explicit ORDER BY. Persistence: the
assignment is written to the split_assignment table so downstream stages JOIN
membership rather than re-deriving it. The committed split_manifest.json records
the seed, ratio, per-split counts and default rates, and a sha256 of each split's
sorted ID list, which makes holdout membership an auditable artifact.
"""

import argparse
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split

from scoregate.config import HOLDOUT_FRACTION, SEED


@dataclass(frozen=True)
class SplitStats:
    rows: int
    default_rate: float
    id_sha256: str


@dataclass(frozen=True)
class SplitManifest:
    seed: int
    holdout_fraction: float
    splits: dict[str, SplitStats]


def assign_splits(ids_targets: pd.DataFrame) -> pd.DataFrame:
    """Assign each SK_ID_CURR to 'train' or 'holdout', stratified on TARGET.

    Sorting by SK_ID_CURR here makes the seeded split reproducible regardless of
    the input row order.
    """
    frame = ids_targets.sort_values("SK_ID_CURR").reset_index(drop=True)
    train_idx, _ = train_test_split(
        frame.index,
        test_size=HOLDOUT_FRACTION,
        stratify=frame["TARGET"],
        random_state=SEED,
    )
    split = pd.Series("holdout", index=frame.index)
    split.loc[train_idx] = "train"
    return pd.DataFrame({"SK_ID_CURR": frame["SK_ID_CURR"], "split": split})


def _sha256_ids(ids: Iterable[int]) -> str:
    """Hash a split's membership: sorted integer IDs, newline-joined, utf-8."""
    payload = "\n".join(str(i) for i in sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_manifest(assignment_with_target: pd.DataFrame) -> SplitManifest:
    """Summarise the assignment into the auditable manifest structure."""
    stats: dict[str, SplitStats] = {}
    for name in ("train", "holdout"):
        sub = assignment_with_target[assignment_with_target["split"] == name]
        stats[name] = SplitStats(
            rows=int(len(sub)),
            default_rate=round(float(sub["TARGET"].mean()), 6),
            id_sha256=_sha256_ids(sub["SK_ID_CURR"].tolist()),
        )
    return SplitManifest(seed=SEED, holdout_fraction=HOLDOUT_FRACTION, splits=stats)


def build_split(db_path: Path, manifest_path: Path) -> SplitManifest:
    """Build and persist the split, then write the manifest. Returns the manifest."""
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found. Run scoregate.features first.")

    with duckdb.connect(str(db_path)) as con:
        ids_targets = con.execute(
            "SELECT SK_ID_CURR, TARGET FROM features ORDER BY SK_ID_CURR"
        ).df()
        assignment = assign_splits(ids_targets)
        con.register("split_df", assignment)
        con.execute(
            "CREATE OR REPLACE TABLE split_assignment AS SELECT SK_ID_CURR, split FROM split_df"
        )
        con.unregister("split_df")

    merged = assignment.merge(ids_targets, on="SK_ID_CURR")
    manifest = build_manifest(merged)
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/scoregate.duckdb"))
    parser.add_argument("--manifest", type=Path, default=Path("split_manifest.json"))
    args = parser.parse_args()

    manifest = build_split(args.db, args.manifest)
    for name, stats in manifest.splits.items():
        print(f"{name}: {stats.rows:,} rows, default rate {stats.default_rate:.5f}")
    print(f"manifest written to {args.manifest}")


if __name__ == "__main__":
    main()
