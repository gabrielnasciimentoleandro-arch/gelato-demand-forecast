from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from gelato_forecast.modeling import TrainingResult


@dataclass(frozen=True, slots=True)
class TrackingResult:
    run_id: str
    experiment_id: str
    model_uri: str
    registered_model_version: str | None


def log_training_run(  # noqa: PLR0913
    result: TrainingResult,
    *,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str | None,
    data_path: Path,
    metadata_path: Path,
    report_paths: list[Path],
) -> TrackingResult:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    input_example = pd.DataFrame({"temperature_c": [18.0, 25.0, 32.0]})
    output_example = result.model.predict(input_example)
    signature = infer_signature(input_example, output_example)

    with mlflow.start_run(run_name=f"train-{result.model_name}") as run:
        mlflow.set_tags(
            {
                "project": "gelato-demand-forecast",
                "data_origin": "synthetic",
                "task": "regression",
            }
        )
        mlflow.log_params(
            {
                "selected_model": result.model_name,
                "random_seed": result.seed,
                "test_size": result.test_size,
                "training_rows": result.training_rows,
                "test_rows": result.test_rows,
                "data_sha256": _sha256(data_path),
            }
        )
        mlflow.log_metrics(
            {
                "test_mae": result.metrics.mae,
                "test_rmse": result.metrics.rmse,
                "test_r2": result.metrics.r2,
                **{f"cv_rmse_{score.name}": score.cv_rmse for score in result.candidates},
            }
        )
        mlflow.log_artifact(str(data_path), artifact_path="data")
        mlflow.log_artifact(str(metadata_path), artifact_path="reports")
        for report_path in report_paths:
            mlflow.log_artifact(str(report_path), artifact_path="reports")

        model_info = mlflow.sklearn.log_model(
            sk_model=result.model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

    return TrackingResult(
        run_id=run.info.run_id,
        experiment_id=run.info.experiment_id,
        model_uri=model_info.model_uri,
        registered_model_version=getattr(model_info, "registered_model_version", None),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()
