.PHONY: test test-unit test-e2e lint typecheck venv

VENV := .venv/bin
PYTHON := $(VENV)/python
PYTEST := $(VENV)/pytest

venv:
	uv venv --python 3.13 .venv
	uv pip install pytest faker pandas --python $(PYTHON)

test:
	$(PYTEST) tests/ -v --tb=short

test-unit:
	$(PYTEST) tests/ -v --tb=short -m "unit"

test-e2e:
	$(PYTEST) tests/ -v --tb=short -m "e2e"

lint:
	$(VENV)/ruff check src/ tests/

typecheck:
	$(VENV)/mypy src/ --ignore-missing-imports
