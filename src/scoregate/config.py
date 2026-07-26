"""Shared pipeline configuration.

Downstream stages import SEED from here so every seeded operation, the split
first and later the sweeps and calibration folds, stays reproducible across runs.
"""

SEED = 42
HOLDOUT_FRACTION = 0.20
