from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from gelato_forecast.api import app, get_forecast_service, get_settings
from gelato_forecast.config import Settings
from gelato_forecast.service import ForecastService, ModelNotReadyError


def test_root_returns_service_metadata() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "Gelato Demand Forecast API"
    assert response.json()["documentation"] == "/docs"


def test_health_reports_model_state(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        data_path=tmp_path / "data.csv",
        model_path=tmp_path / "model.joblib",
        metadata_path=tmp_path / "metadata.json",
        reports_dir=tmp_path / "reports",
        tracking_uri="sqlite:///test.db",
        experiment_name="test",
        registered_model_name="test-model",
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "model_ready": False}


def test_predict_returns_forecast(model_artifacts) -> None:
    service = ForecastService(*model_artifacts)
    app.dependency_overrides[get_forecast_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json={"temperature_c": 30.0})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_sales"] > 0
    assert len(payload["prediction_interval_95"]) == 2


def test_predict_rejects_invalid_payload() -> None:
    with TestClient(app) as client:
        out_of_range = client.post("/predict", json={"temperature_c": 80.0})
        unknown = client.post("/predict", json={"temperature_c": 30.0, "extra": True})
        coerced = client.post("/predict", json={"temperature_c": "30"})

    assert out_of_range.status_code == 422
    assert unknown.status_code == 422
    assert coerced.status_code == 422


def test_predict_returns_problem_details_when_model_is_missing() -> None:
    def unavailable_service() -> ForecastService:
        raise ModelNotReadyError("Train the model first")

    app.dependency_overrides[get_forecast_service] = unavailable_service
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json={"temperature_c": 30.0})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["detail"] == "Train the model first"
