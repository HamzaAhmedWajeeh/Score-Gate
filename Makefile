ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
VENV := py -3.11 -m venv .venv
else
PY := .venv/bin/python
VENV := python3.11 -m venv .venv
endif

.PHONY: install lint test download ingest features contract split binning scorecard challenger evaluate calibrate fairness register train sweep

install:
	$(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests
	$(PY) -m mypy

test:
	$(PY) -m pytest

# Data acquisition is expensive and one-time, so download and ingest stay
# independent. The training pipeline order is features, then contract, then
# training, expressed as prerequisites so the gate always runs on freshly built
# features and training will only ever run behind a passing contract.
download:
	$(PY) -m scoregate.download_data

ingest:
	$(PY) -m scoregate.ingest

features:
	$(PY) -m scoregate.features

contract: features
	$(PY) -m scoregate.contracts

split: contract
	$(PY) -m scoregate.split

binning: split
	$(PY) -m scoregate.binning

scorecard: binning
	$(PY) -m scoregate.scorecard

challenger: split
	$(PY) -m scoregate.challenger

evaluate: scorecard challenger
	$(PY) -m scoregate.evaluate

calibrate: scorecard challenger
	$(PY) -m scoregate.calibration

fairness: calibrate
	$(PY) -m scoregate.fairness

register: evaluate fairness
	$(PY) -m scoregate.registry

# Full model training with W&B tracking. Depends on binning so the data chain
# (features, contract, split, binning) is in place; train.py builds both models,
# evaluates them, and mirrors the results into W&B. Runs offline with no key.
train: binning
	$(PY) -m scoregate.train

# Register the W&B sweep and print the agent command. Build the pipeline first
# (make binning). Launch trials with: wandb agent <sweep-id> --count N. Runs online
# and is separate from make train, which stays on the frozen params.
sweep:
	$(PY) -m wandb sweep configs/sweep.yaml
