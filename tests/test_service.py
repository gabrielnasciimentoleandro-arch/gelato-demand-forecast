from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from gelato_forecast.service import (
    ForecastService,
    InvalidTemperatureError,
    ModelNotReadyError,
)


def test_service_requires_model_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ModelNotReadyError, match="not found"):
        ForecastService(tmp_path / "model.joblib", tmp_path / "metadata.json")


def test_service_predicts_with_interval(model_artifacts) -> None:
    service = ForecastService(*model_artifacts)
    forecast = service.predict(30)

    assert forecast.temperature_c == 30
    assert forecast.predicted_sales > 0
    assert forecast.lower_bound <= forecast.predicted_sales <= forecast.upper_bound
    assert forecast.warning is None
    assert service.metadata["model_name"] == forecast.model_name


def test_service_warns_about_extrapolation(model_artifacts) -> None:
    forecast = ForecastService(*model_artifacts).predict(-5)
    assert forecast.warning is not None


@pytest.mark.parametrize("value", [True, "30", math.nan, math.inf, -21, 61])
def test_service_rejects_invalid_temperature(model_artifacts, value) -> None:
    service = ForecastService(*model_artifacts)
    with pytest.raises(InvalidTemperatureError):
        service.predict(value)


def test_service_rejects_missing_metadata_fields(model_artifacts) -> None:
    model_path, metadata_path = model_artifacts
    metadata_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ModelNotReadyError, match="missing"):
        ForecastService(model_path, metadata_path)


def test_service_rejects_invalid_temperature_range_metadata(model_artifacts) -> None:
    model_path, metadata_path = model_artifacts
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_temperature_range_c"] = [10]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelNotReadyError, match="temperature range"):
        ForecastService(model_path, metadata_path)
