from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import uvicorn

from gelato_forecast.config import Settings
from gelato_forecast.data import generate_sales_data, load_dataset, save_dataset
from gelato_forecast.modeling import save_training_result, train_and_evaluate
from gelato_forecast.plotting import create_reports
from gelato_forecast.service import ForecastService
from gelato_forecast.tracking import log_training_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gelato-forecast",
        description="Train and serve an ice cream demand forecasting model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate-data", help="Generate deterministic synthetic data")
    generate.add_argument("--rows", type=int, default=730)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output", type=Path)
    generate.add_argument("--force", action="store_true")

    train = commands.add_parser("train", help="Compare, train and persist regression models")
    train.add_argument("--data", type=Path)
    train.add_argument("--model-dir", type=Path)
    train.add_argument("--reports-dir", type=Path)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--no-mlflow", action="store_true")
    train.add_argument("--no-register", action="store_true")

    predict = commands.add_parser("predict", help="Predict demand for one temperature")
    predict.add_argument("temperature", type=float)
    predict.add_argument("--model-dir", type=Path)

    batch = commands.add_parser("batch-predict", help="Predict temperatures from a text file")
    batch.add_argument("input", type=Path)
    batch.add_argument("--model-dir", type=Path)

    serve = commands.add_parser("serve", help="Start the HTTP API")
    serve.add_argument("--host", default="0.0.0.0")  # noqa: S104
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    settings = Settings.from_environment()

    if arguments.command == "generate-data":
        output = (arguments.output or settings.data_path).resolve()
        path = save_dataset(
            generate_sales_data(arguments.rows, arguments.seed),
            output,
            overwrite=arguments.force,
        )
        print(f"Dataset generated: {path}")
        return

    if arguments.command == "train":
        summary = _train(arguments, settings)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if arguments.command in {"predict", "batch-predict"}:
        model_path, metadata_path = _model_paths(arguments.model_dir, settings)
        service = ForecastService(model_path, metadata_path)
        if arguments.command == "predict":
            print(json.dumps(_forecast_dict(service, arguments.temperature), ensure_ascii=False))
            return
        temperatures = _read_temperatures(arguments.input)
        print(
            json.dumps(
                [_forecast_dict(service, temperature) for temperature in temperatures],
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    uvicorn.run("gelato_forecast.api:app", host=arguments.host, port=arguments.port)


def _train(arguments: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    data_path = (arguments.data or settings.data_path).resolve()
    if not data_path.exists():
        save_dataset(generate_sales_data(seed=arguments.seed), data_path)
    data = load_dataset(data_path)
    result = train_and_evaluate(data, seed=arguments.seed, test_size=arguments.test_size)

    model_path, metadata_path = _model_paths(arguments.model_dir, settings)
    save_training_result(result, model_path, metadata_path)
    reports_dir = (arguments.reports_dir or settings.reports_dir).resolve()
    report_paths = create_reports(data, result, reports_dir / "images")

    tracking_summary: dict[str, Any] = {"enabled": False}
    if not arguments.no_mlflow:
        tracking = log_training_run(
            result,
            tracking_uri=settings.tracking_uri,
            experiment_name=settings.experiment_name,
            registered_model_name=(
                None if arguments.no_register else settings.registered_model_name
            ),
            data_path=data_path,
            metadata_path=metadata_path,
            report_paths=report_paths,
        )
        tracking_summary = {
            "enabled": True,
            "run_id": tracking.run_id,
            "experiment_id": tracking.experiment_id,
            "model_uri": tracking.model_uri,
            "registered_model_version": tracking.registered_model_version,
        }
        save_training_result(
            result,
            model_path,
            metadata_path,
            extra_metadata={"mlflow": tracking_summary},
        )

    return {
        "selected_model": result.model_name,
        "metrics": result.metadata()["metrics"],
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "reports": [str(path) for path in report_paths],
        "mlflow": tracking_summary,
    }


def _model_paths(model_dir: Path | None, settings: Settings) -> tuple[Path, Path]:
    if model_dir is None:
        return settings.model_path, settings.metadata_path
    directory = model_dir.resolve()
    return directory / "model.joblib", directory / "metadata.json"


def _read_temperatures(path: Path) -> list[float]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    temperatures: list[float] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            temperatures.append(float(line))
        except ValueError as error:
            raise ValueError(f"Invalid temperature at line {line_number}: {line}") from error
    if not temperatures:
        raise ValueError("Input file does not contain temperatures")
    return temperatures


def _forecast_dict(service: ForecastService, temperature: float) -> dict[str, Any]:
    forecast = service.predict(temperature)
    return {
        "temperature_c": forecast.temperature_c,
        "predicted_sales": forecast.predicted_sales,
        "prediction_interval_95": [forecast.lower_bound, forecast.upper_bound],
        "model_name": forecast.model_name,
        "warning": forecast.warning,
    }
