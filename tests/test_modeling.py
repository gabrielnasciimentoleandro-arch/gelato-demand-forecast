from __future__ import annotations

import json
from pathlib import Path

import pytest

from gelato_forecast.model_card import render_model_card
from gelato_forecast.modeling import (
    build_candidates,
    save_training_result,
    train_and_evaluate,
)


def test_build_candidates_returns_expected_algorithms() -> None:
    assert set(build_candidates(42)) == {
        "linear_regression",
        "polynomial_ridge",
        "random_forest",
    }


def test_train_and_evaluate_selects_quality_model(sample_data) -> None:
    result = train_and_evaluate(sample_data, seed=42, test_size=0.25)

    assert result.model_name in {score.name for score in result.candidates}
    assert result.candidates == sorted(result.candidates, key=lambda score: score.cv_rmse)
    assert result.metrics.mae > 0
    assert result.metrics.rmse >= result.metrics.mae
    assert result.metrics.r2 > 0.75
    assert result.training_rows + result.test_rows == len(sample_data)
    assert len(result.predictions) == result.test_rows


@pytest.mark.parametrize("test_size", [0.09, 0.41])
def test_train_and_evaluate_rejects_invalid_test_size(sample_data, test_size: float) -> None:
    with pytest.raises(ValueError, match=r"between 0\.1 and 0\.4"):
        train_and_evaluate(sample_data, test_size=test_size)


def test_save_training_result_writes_model_and_metadata(tmp_path: Path, training_result) -> None:
    model_path = tmp_path / "model" / "model.joblib"
    metadata_path = tmp_path / "model" / "metadata.json"

    saved = save_training_result(
        training_result,
        model_path,
        metadata_path,
        extra_metadata={"test_marker": True},
    )

    assert saved == (model_path, metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["test_marker"] is True
    assert metadata["data_origin"].startswith("synthetic")
    assert metadata["metrics"]["r2"] > 0.75


def test_render_model_card_documents_metrics(training_result) -> None:
    card = render_model_card(training_result.metadata())

    assert "Model Card" in card
    assert training_result.model_name in card
    assert "dados sintéticos" in card
    assert "Limitações" in card
