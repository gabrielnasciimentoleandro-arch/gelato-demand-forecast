from __future__ import annotations

from pathlib import Path

from gelato_forecast.config import Settings


def test_settings_resolves_defaults(tmp_path: Path, monkeypatch) -> None:
    for variable in (
        "GELATO_PROJECT_ROOT",
        "GELATO_DATA_PATH",
        "GELATO_MODEL_PATH",
        "GELATO_METADATA_PATH",
        "GELATO_REPORTS_DIR",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_MODEL_NAME",
    ):
        monkeypatch.delenv(variable, raising=False)

    settings = Settings.from_environment(tmp_path)
    assert settings.data_path == (tmp_path / "data/ice_cream_sales.csv").resolve()
    assert settings.tracking_uri.endswith("mlflow.db")
    assert settings.registered_model_name == "gelato-demand-model"


def test_settings_accepts_environment_overrides(tmp_path: Path, monkeypatch) -> None:
    absolute = tmp_path / "external.csv"
    monkeypatch.setenv("GELATO_DATA_PATH", str(absolute))
    monkeypatch.setenv("GELATO_MODEL_PATH", "custom/model.joblib")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "custom-experiment")

    settings = Settings.from_environment(tmp_path)
    assert settings.data_path == absolute.resolve()
    assert settings.model_path == (tmp_path / "custom/model.joblib").resolve()
    assert settings.experiment_name == "custom-experiment"


def test_settings_accepts_project_root_from_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GELATO_PROJECT_ROOT", str(tmp_path))
    settings = Settings.from_environment()
    assert settings.project_root == tmp_path.resolve()
