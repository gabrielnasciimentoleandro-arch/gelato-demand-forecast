from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gelato_forecast.data import generate_sales_data, load_dataset, save_dataset
from gelato_forecast.model_card import render_model_card
from gelato_forecast.modeling import save_training_result, train_and_evaluate
from gelato_forecast.plotting import create_reports


def run_local_pipeline(  # noqa: PLR0913
    *,
    data_path: Path,
    model_dir: Path,
    reports_dir: Path,
    seed: int = 42,
    rows: int = 730,
    test_size: float = 0.2,
) -> dict[str, Any]:
    if not data_path.exists():
        save_dataset(generate_sales_data(rows=rows, seed=seed), data_path)
    data = load_dataset(data_path)
    result = train_and_evaluate(data, seed=seed, test_size=test_size)

    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"
    save_training_result(result, model_path, metadata_path)
    images = create_reports(data, result, reports_dir / "images")

    metadata = result.metadata()
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    model_card_path = reports_dir / "MODEL_CARD.md"
    model_card_path.write_text(render_model_card(metadata), encoding="utf-8")

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metrics_path": metrics_path,
        "model_card_path": model_card_path,
        "images": images,
        "result": result,
    }
