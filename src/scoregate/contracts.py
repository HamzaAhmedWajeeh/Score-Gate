"""Pandera data contract gating the feature table before any training runs.

The gate is fatal by design: if the built `features` or `fairness_metadata`
table drifts from the frozen expectations, validation raises and the pipeline
stops. Pipeline order is features -> contract -> training.

Two kinds of expectation live here:

- A priori checks that need no profiling: key uniqueness, the TARGET domain and
  base rate, the DAYS_* sign and ranges, the [0, 1] external scores and *_share
  features, non-negative counts, and sentinel eradication.
- Profile-frozen checks read off the first clean build of the feature table:
  two-sided null-rate bands (observed +/- 3 pp) and wide value bounds for the
  unbounded ratios. The bounds exist to catch impossible or regressed values;
  binning absorbs ordinary outliers, so they sit far outside the observed range.

Column-level Checks only see non-null values, so null-rate bands are enforced as
dataframe-level checks. Their floors are as load-bearing as their ceilings: a
floor protects the thin-file NULL semantics from a future accidental COALESCE.
"""

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

# The DAYS_EMPLOYED "not employed" sentinel. It is nulled during feature build,
# and its continued absence is a contract clause rather than a hope.
DAYS_EMPLOYED_SENTINEL = 365243

# Home Credit application_train base rate; a mean outside this band signals label
# corruption or a wrong join.
TARGET_MEAN_LOW = 0.06
TARGET_MEAN_HIGH = 0.10

# Two-sided null-rate bands (observed +/- 3 pp), frozen from the first clean build.
# Enforced at dataframe level because column checks never see the NULLs.
NULL_BANDS: dict[str, tuple[float, float]] = {
    "annuity_to_income": (0.000, 0.030),
    "AMT_ANNUITY": (0.000, 0.030),
    "credit_to_goods": (0.000, 0.031),
    "ext_source_mean": (0.000, 0.031),
    "ext_source_min": (0.000, 0.031),
    "EXT_SOURCE_2": (0.000, 0.032),
    "employed_share": (0.150, 0.210),
    "DAYS_EMPLOYED": (0.150, 0.210),
    "EXT_SOURCE_3": (0.168, 0.228),
    "EXT_SOURCE_1": (0.534, 0.594),
    "bureau_credit_count": (0.113, 0.173),
    "bureau_active_count": (0.113, 0.173),
    "bureau_overdue_max": (0.113, 0.173),
    "bureau_prolong_sum": (0.113, 0.173),
    "bureau_recency": (0.113, 0.173),
    "bureau_debt_to_credit": (0.140, 0.200),
    "bureau_current_dpd_count": (0.670, 0.730),
    "bureau_ever_dpd3_share": (0.670, 0.730),
    "prev_app_count": (0.024, 0.084),
    "prev_refused_share": (0.024, 0.084),
    "prev_approved_credit_mean": (0.027, 0.087),
    "inst_late_share": (0.022, 0.082),
    "inst_shortfall_share": (0.022, 0.082),
    "inst_mean_delay": (0.022, 0.082),
    "inst_delay_trend": (0.024, 0.084),
    "card_count": (0.687, 0.747),
    "card_rolling_util_6m": (0.748, 0.808),
}


def _null_band_check(column: str, low: float, high: float) -> pa.Check:
    """A dataframe-level check that a column's null rate stays inside [low, high]."""
    return pa.Check(
        lambda df, _c=column, _lo=low, _hi=high: _lo <= df[_c].isna().mean() <= _hi,
        name=f"null_rate::{column}",
        error=f"{column} null rate outside [{low}, {high}]",
    )


# Value checks apply only to non-null values, which is what we want; nullability is
# handled separately by the null-rate bands above.
FEATURES_SCHEMA = pa.DataFrameSchema(
    columns={
        # --- structural non-null: guaranteed present by construction ---
        "SK_ID_CURR": pa.Column(nullable=False, unique=True),
        "TARGET": pa.Column(nullable=False, checks=pa.Check.isin([0, 1])),
        "employment_recorded": pa.Column(nullable=False, checks=pa.Check.isin([0, 1])),
        "ext_source_count": pa.Column(nullable=False, checks=pa.Check.in_range(0, 3)),
        # --- empirical non-null: observed zero nulls in this dataset version ---
        "AMT_INCOME_TOTAL": pa.Column(nullable=False, checks=pa.Check.in_range(0, 500_000_000)),
        "AMT_CREDIT": pa.Column(nullable=False, checks=pa.Check.in_range(0, 20_000_000)),
        "DAYS_BIRTH": pa.Column(nullable=False, checks=pa.Check.in_range(-32850, -6570)),
        "CNT_CHILDREN": pa.Column(nullable=False, checks=pa.Check.in_range(0, 50)),
        "credit_to_income": pa.Column(nullable=False, checks=pa.Check.in_range(0, 250)),
        # --- nullable ratios and scores ---
        "annuity_to_income": pa.Column(nullable=True, checks=pa.Check.in_range(0, 10)),
        "credit_to_goods": pa.Column(nullable=True, checks=pa.Check.in_range(0, 20)),
        "employed_share": pa.Column(nullable=True, checks=pa.Check.in_range(0, 1)),
        "ext_source_mean": pa.Column(nullable=True, checks=pa.Check.in_range(0, 1)),
        "ext_source_min": pa.Column(nullable=True, checks=pa.Check.in_range(0, 1)),
        "AMT_ANNUITY": pa.Column(nullable=True, checks=pa.Check.in_range(0, 2_000_000)),
        # DAYS_EMPLOYED is <= 0 with a generous floor; the sentinel is nulled upstream.
        "DAYS_EMPLOYED": pa.Column(nullable=True, checks=pa.Check.in_range(-30000, 0)),
        "EXT_SOURCE_1": pa.Column(nullable=True, checks=pa.Check.in_range(0, 1)),
        "EXT_SOURCE_2": pa.Column(nullable=True, checks=pa.Check.in_range(0, 1)),
        "EXT_SOURCE_3": pa.Column(nullable=True, checks=pa.Check.in_range(0, 1)),
        # --- bureau: three-depth NULL semantics. A NULL bureau_credit_count means no
        # bureau history at all; a NULL bureau_current_dpd_count with a non-null
        # bureau_credit_count means bureau history exists but no monthly archive was
        # reported; a real 0 means the archive is present and clean. None of these are
        # ever COALESCEd, and the null-rate floors above guard exactly that. ---
        "bureau_credit_count": pa.Column(nullable=True, checks=pa.Check.in_range(0, 500)),
        "bureau_active_count": pa.Column(nullable=True, checks=pa.Check.in_range(0, 500)),
        "bureau_debt_to_credit": pa.Column(nullable=True, checks=pa.Check.in_range(-2000, 2000)),
        "bureau_overdue_max": pa.Column(nullable=True, checks=pa.Check.in_range(0, 100_000_000)),
        "bureau_prolong_sum": pa.Column(nullable=True, checks=pa.Check.in_range(0, 100)),
        "bureau_recency": pa.Column(nullable=True, checks=pa.Check.in_range(-30000, 0)),
        "bureau_current_dpd_count": pa.Column(nullable=True, checks=pa.Check.in_range(0, 500)),
        "bureau_ever_dpd3_share": pa.Column(nullable=True, checks=pa.Check.in_range(0, 1)),
        # --- previous applications ---
        "prev_app_count": pa.Column(nullable=True, checks=pa.Check.in_range(0, 500)),
        "prev_refused_share": pa.Column(nullable=True, checks=pa.Check.in_range(0, 1)),
        "prev_approved_credit_mean": pa.Column(
            nullable=True, checks=pa.Check.in_range(0, 20_000_000)
        ),
        # --- installments ---
        "inst_late_share": pa.Column(nullable=True, checks=pa.Check.in_range(0, 1)),
        "inst_shortfall_share": pa.Column(nullable=True, checks=pa.Check.in_range(0, 1)),
        "inst_mean_delay": pa.Column(nullable=True, checks=pa.Check.in_range(-10000, 10000)),
        "inst_delay_trend": pa.Column(nullable=True, checks=pa.Check.in_range(-10000, 10000)),
        # --- credit card ---
        "card_count": pa.Column(nullable=True, checks=pa.Check.in_range(0, 500)),
        "card_rolling_util_6m": pa.Column(nullable=True, checks=pa.Check.in_range(0, 50)),
    },
    checks=[
        # TARGET base rate stays in band; catches label corruption or a wrong join.
        pa.Check(
            lambda df: TARGET_MEAN_LOW <= df["TARGET"].mean() <= TARGET_MEAN_HIGH,
            name="target_mean_band",
            error=f"TARGET mean outside [{TARGET_MEAN_LOW}, {TARGET_MEAN_HIGH}]",
        ),
        # Sentinel eradication. 365243 is checked in DAYS_EMPLOYED only: it is a valid
        # SK_ID_CURR, and in DAYS_BIRTH it is already impossible under the range check.
        pa.Check(
            lambda df: not (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).any(),
            name="days_employed_sentinel_eradicated",
            error=f"{DAYS_EMPLOYED_SENTINEL} sentinel present in DAYS_EMPLOYED",
        ),
        *[_null_band_check(col, lo, hi) for col, (lo, hi) in NULL_BANDS.items()],
    ],
    strict=True,  # any unexpected column is a violation
    ordered=False,
    name="features",
)

FAIRNESS_SCHEMA = pa.DataFrameSchema(
    columns={
        "SK_ID_CURR": pa.Column(nullable=False, unique=True),
        "CODE_GENDER": pa.Column(nullable=False, checks=pa.Check.isin(["M", "F", "XNA"])),
        "age_band": pa.Column(
            nullable=False,
            checks=pa.Check.isin(["<25", "25-34", "35-44", "45-54", "55-64", "65+"]),
        ),
    },
    strict=True,
    ordered=False,
    name="fairness_metadata",
)


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the feature frame against the contract; raise SchemaErrors on failure."""
    return FEATURES_SCHEMA.validate(df, lazy=True)


def validate_fairness_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the fairness metadata frame; raise SchemaErrors on failure."""
    return FAIRNESS_SCHEMA.validate(df, lazy=True)


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(f"SELECT * FROM {table}").df()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/scoregate.duckdb"))
    args = parser.parse_args()

    try:
        features = _load_table(args.db, "features")
        fairness = _load_table(args.db, "fairness_metadata")
        validate_features(features)
        validate_fairness_metadata(fairness)
    except SchemaErrors as err:
        print("Contract FAILED. Violations:", file=sys.stderr)
        print(err.failure_cases.to_string(), file=sys.stderr)
        sys.exit(1)

    print(f"Contract passed: features {len(features):,} rows, fairness {len(fairness):,} rows")


if __name__ == "__main__":
    main()
