ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
VENV := py -3.11 -m venv .venv
else
PY := .venv/bin/python
VENV := python3.11 -m venv .venv
endif

.PHONY: install lint test download ingest features contract

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
