from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class ModelNotReadyError(RuntimeError):
    """Raised when model artifacts have not been generated yet."""


class InvalidTemperatureError(ValueError):
    """Raised when a temperature cannot be used for inference."""


@dataclass(frozen=True, slots=True)
class Forecast:
    temperature_c: float
    predicted_sales: int
    lower_bound: int
    upper_bound: int
    model_name: str
    warning: str | None


class ForecastService:
    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        if not model_path.is_file() or not metadata_path.is_file():
            raise ModelNotReadyError(
                "Model artifacts were not found. Run the training pipeline first."
            )
        self._model = joblib.load(model_path)
        self._metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._validate_metadata()

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def predict(self, temperature_c: float) -> Forecast:
        if isinstance(temperature_c, bool) or not isinstance(temperature_c, int | float):
            raise InvalidTemperatureError("Temperature must be numeric")
        temperature = float(temperature_c)
        if not math.isfinite(temperature):
            raise InvalidTemperatureError("Temperature must be finite")
        if not -20 <= temperature <= 60:
            raise InvalidTemperatureError("Temperature must be between -20 and 60 °C")

        raw_prediction = float(
            self._model.predict(pd.DataFrame({"temperature_c": [temperature]}))[0]
        )
        prediction = max(0, round(raw_prediction))
        residual_std = float(self._metadata["residual_std"])
        margin = 1.96 * residual_std
        lower_bound = max(0, round(raw_prediction - margin))
        upper_bound = max(lower_bound, round(raw_prediction + margin))

        minimum, maximum = self._metadata["training_temperature_range_c"]
        warning = None
        if not float(minimum) <= temperature <= float(maximum):
            warning = "Temperature is outside the range observed during training."

        return Forecast(
            temperature_c=temperature,
            predicted_sales=prediction,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            model_name=str(self._metadata["model_name"]),
            warning=warning,
        )

    def _validate_metadata(self) -> None:
        required = {
            "schema_version",
            "model_name",
            "residual_std",
            "training_temperature_range_c",
        }
        missing = sorted(required - set(self._metadata))
        if missing:
            raise ModelNotReadyError(f"Invalid model metadata; missing: {', '.join(missing)}")
        limits = self._metadata["training_temperature_range_c"]
        if not isinstance(limits, list) or len(limits) != 2:
            raise ModelNotReadyError("Invalid training temperature range in metadata")
