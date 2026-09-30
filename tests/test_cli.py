from __future__ import annotations

import json
from pathlib import Path

import pytest

from gelato_forecast.cli import (
    _forecast_dict,
    _model_paths,
    _read_temperatures,
    build_parser,
    main,
)
from gelato_forecast.config import Settings
from gelato_forecast.data import generate_sales_data
from gelato_forecast.service import ForecastService
from gelato_forecast.tracking import TrackingResult


def test_build_parser_requires_and_parses_commands() -> None:
    parser = build_parser()
    arguments = parser.parse_args(["predict", "27.5"])
    assert arguments.command == "predict"
    assert arguments.temperature == 27.5


def test_read_temperatures_reads_non_empty_lines(tmp_path: Path) -> None:
    path = tmp_path / "values.txt"
    path.write_text("18\n\n22.5\n", encoding="utf-8")
    assert _read_temperatures(path) == [18.0, 22.5]


def test_read_temperatures_reports_invalid_line(tmp_path: Path) -> None:
    path = tmp_path / "values.txt"
    path.write_text("18\nhot\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        _read_temperatures(path)


def test_read_temperatures_rejects_missing_or_empty_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _read_temperatures(tmp_path / "missing.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain"):
        _read_temperatures(empty)


def test_forecast_dict_has_api_shape(model_artifacts) -> None:
    service = ForecastService(*model_artifacts)
    payload = _forecast_dict(service, 25)
    assert set(payload) == {
        "temperature_c",
        "predicted_sales",
        "prediction_interval_95",
        "model_name",
        "warning",
    }


def test_main_generates_data(tmp_path: Path, capsys) -> None:
    output = tmp_path / "generated.csv"
    main(["generate-data", "--rows", "50", "--output", str(output)])
    assert output.is_file()
    assert "Dataset generated" in capsys.readouterr().out


def test_main_predicts_using_custom_model_directory(
    tmp_path: Path, model_artifacts, capsys
) -> None:
    model_path, metadata_path = model_artifacts
    model_dir = tmp_path / "artifacts"
    model_dir.mkdir()
    model_path.replace(model_dir / "model.joblib")
    metadata_path.replace(model_dir / "metadata.json")

    main(["predict", "30", "--model-dir", str(model_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["temperature_c"] == 30


def test_main_batch_predicts_using_text_file(tmp_path: Path, model_artifacts, capsys) -> None:
    model_path, metadata_path = model_artifacts
    model_dir = tmp_path / "artifacts"
    model_dir.mkdir()
    model_path.replace(model_dir / "model.joblib")
    metadata_path.replace(model_dir / "metadata.json")
    input_path = tmp_path / "temperatures.txt"
    input_path.write_text("20\n30\n", encoding="utf-8")

    main(["batch-predict", str(input_path), "--model-dir", str(model_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert [item["temperature_c"] for item in payload] == [20, 30]


def test_main_trains_without_mlflow(tmp_path: Path, capsys) -> None:
    main(
        [
            "train",
            "--data",
            str(tmp_path / "sales.csv"),
            "--model-dir",
            str(tmp_path / "models"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--seed",
            "9",
            "--no-mlflow",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mlflow"] == {"enabled": False}
    assert Path(payload["model_path"]).is_file()


def test_main_tracks_training_without_registering(tmp_path: Path, monkeypatch, capsys) -> None:
    data_path = tmp_path / "sales.csv"
    generate_sales_data(rows=60).to_csv(data_path, index=False)
    tracked = TrackingResult(
        run_id="run-1",
        experiment_id="experiment-1",
        model_uri="models:/model-1",
        registered_model_version=None,
    )
    monkeypatch.setattr("gelato_forecast.cli.log_training_run", lambda *args, **kwargs: tracked)

    main(
        [
            "train",
            "--data",
            str(data_path),
            "--model-dir",
            str(tmp_path / "models"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--no-register",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mlflow"]["run_id"] == "run-1"
    assert payload["mlflow"]["registered_model_version"] is None


def test_main_starts_server(monkeypatch) -> None:
    captured = {}

    def fake_run(application: str, *, host: str, port: int) -> None:
        captured.update({"application": application, "host": host, "port": port})

    monkeypatch.setattr("gelato_forecast.cli.uvicorn.run", fake_run)
    main(["serve", "--host", "127.0.0.1", "--port", "9000"])
    assert captured == {
        "application": "gelato_forecast.api:app",
        "host": "127.0.0.1",
        "port": 9000,
    }


def test_model_paths_uses_settings_defaults(tmp_path: Path) -> None:
    settings = Settings.from_environment(tmp_path)
    assert _model_paths(None, settings) == (settings.model_path, settings.metadata_path)
