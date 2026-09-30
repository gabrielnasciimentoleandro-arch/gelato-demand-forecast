from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from gelato_forecast.data import generate_sales_data
from gelato_forecast.modeling import TrainingResult, train_and_evaluate


@pytest.fixture(scope="session")
def sample_data():
    return generate_sales_data(rows=180, seed=7)


@pytest.fixture(scope="session")
def training_result(sample_data) -> TrainingResult:
    return train_and_evaluate(sample_data, seed=7, test_size=0.2)


@pytest.fixture
def model_artifacts(tmp_path: Path, training_result: TrainingResult) -> tuple[Path, Path]:
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump(training_result.model, model_path)
    metadata_path.write_text(
        json.dumps(training_result.metadata(), ensure_ascii=False),
        encoding="utf-8",
    )
    return model_path, metadata_path
