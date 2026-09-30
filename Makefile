.PHONY: install pipeline train api mlflow lint format test audit check docker-build docker-up clean

PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/pip install -e '.[dev]'

pipeline:
	$(BIN)/python scripts/run_pipeline.py

train:
	$(BIN)/gelato-forecast train

api:
	$(BIN)/gelato-forecast serve

mlflow:
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

lint:
	$(BIN)/ruff format --check src tests scripts
	$(BIN)/ruff check src tests scripts

format:
	$(BIN)/ruff format src tests scripts
	$(BIN)/ruff check src tests scripts --fix

test:
	$(BIN)/pytest -q --cov=gelato_forecast --cov-report=term-missing --cov-report=xml

audit:
	$(BIN)/pip-audit --skip-editable
	$(BIN)/pip check

check: lint test audit

docker-build:
	docker build --tag gelato-demand-forecast:local .

docker-up:
	docker compose up --build

clean:
	rm -rf .coverage coverage.xml htmlcov .pytest_cache .ruff_cache
