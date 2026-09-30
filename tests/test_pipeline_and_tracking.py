from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from gelato_forecast.pipeline import run_local_pipeline
from gelato_forecast.tracking import log_training_run


def test_local_pipeline_creates_reproducible_outputs(tmp_path: Path) -> None:
    output = run_local_pipeline(
        data_path=tmp_path / "data" / "sales.csv",
        model_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        rows=100,
        seed=11,
    )

    assert output["model_path"].is_file()
    assert output["metadata_path"].is_file()
    assert output["metrics_path"].is_file()
    assert output["model_card_path"].is_file()
    assert len(output["images"]) == 3
    metrics = json.loads(output["metrics_path"].read_text(encoding="utf-8"))
    assert metrics["metrics"]["r2"] > 0.7

    repeated = run_local_pipeline(
        data_path=tmp_path / "data" / "sales.csv",
        model_dir=tmp_path / "models-repeat",
        reports_dir=tmp_path / "reports-repeat",
        rows=100,
        seed=11,
    )
    assert repeated["model_path"].is_file()


def test_mlflow_logs_and_registers_training_run(
    tmp_path: Path, monkeypatch, sample_data, training_result
) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "sales.csv"
    sample_data.to_csv(data_path, index=False)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(training_result.metadata(), ensure_ascii=False), encoding="utf-8"
    )
    report_path = tmp_path / "report.txt"
    report_path.write_text("test report", encoding="utf-8")
    tracking_uri = f"sqlite:///{tmp_path / 'tracking.db'}"

    tracked = log_training_run(
        training_result,
        tracking_uri=tracking_uri,
        experiment_name="test-experiment",
        registered_model_name="test-gelato-model",
        data_path=data_path,
        metadata_path=metadata_path,
        report_paths=[report_path],
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    run = client.get_run(tracked.run_id)
    versions = client.search_model_versions("name='test-gelato-model'")
    assert run.data.metrics["test_r2"] > 0.7
    assert tracked.registered_model_version in {"1", 1}
    assert len(versions) == 1
