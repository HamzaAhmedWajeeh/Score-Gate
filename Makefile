ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
VENV := py -3.11 -m venv .venv
else
PY := .venv/bin/python
VENV := python3.11 -m venv .venv
endif

.PHONY: install lint test

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
