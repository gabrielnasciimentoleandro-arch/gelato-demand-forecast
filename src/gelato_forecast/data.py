from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("date", "temperature_c", "ice_creams_sold")


class DatasetValidationError(ValueError):
    """Raised when the training dataset does not satisfy the expected contract."""


def generate_sales_data(rows: int = 730, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic synthetic dataset with a documented demand relationship."""
    if rows < 50:
        raise ValueError("rows must be at least 50")

    generator = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    day = np.arange(rows)
    seasonal_temperature = 26 + 7.5 * np.sin(2 * np.pi * (day - 35) / 365.25)
    temperature = np.clip(
        seasonal_temperature + generator.normal(loc=0, scale=2.6, size=rows),
        10,
        42,
    )

    weekend_bonus = np.where(dates.dayofweek >= 5, 5.0, 0.0)
    unobserved_effects = generator.normal(loc=0, scale=7.5, size=rows)
    demand = 20 + 1.65 * temperature + 0.068 * temperature**2
    sales = np.maximum(0, np.rint(demand + weekend_bonus + unobserved_effects)).astype(int)

    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "temperature_c": np.round(temperature, 2),
            "ice_creams_sold": sales,
        }
    )


def save_dataset(data: pd.DataFrame, output_path: Path, *, overwrite: bool = False) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Dataset already exists: {output_path}")
    validated = validate_dataset(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output_path, index=False)
    return output_path


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return validate_dataset(pd.read_csv(path))


def validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise DatasetValidationError(f"Missing required columns: {', '.join(missing)}")
    if len(data) < 50:
        raise DatasetValidationError("Dataset must contain at least 50 rows")

    validated = data.loc[:, list(REQUIRED_COLUMNS)].copy()
    if validated.isna().any().any():
        raise DatasetValidationError("Dataset cannot contain null values")

    parsed_dates = pd.to_datetime(validated["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise DatasetValidationError("Column 'date' contains invalid dates")
    if parsed_dates.duplicated().any():
        raise DatasetValidationError("Column 'date' must contain unique values")

    validated["temperature_c"] = pd.to_numeric(validated["temperature_c"], errors="coerce")
    validated["ice_creams_sold"] = pd.to_numeric(validated["ice_creams_sold"], errors="coerce")
    if validated[["temperature_c", "ice_creams_sold"]].isna().any().any():
        raise DatasetValidationError("Temperature and sales columns must be numeric")
    if not validated["temperature_c"].between(-10, 55).all():
        raise DatasetValidationError("Temperatures must be between -10 and 55 °C")
    if not validated["ice_creams_sold"].between(0, 1000).all():
        raise DatasetValidationError("Sales must be between 0 and 1000 units")

    validated["date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    validated["ice_creams_sold"] = validated["ice_creams_sold"].round().astype(int)
    return validated.sort_values("date", ignore_index=True)
