from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gelato_forecast.data import (
    DatasetValidationError,
    generate_sales_data,
    load_dataset,
    save_dataset,
    validate_dataset,
)


def test_generate_sales_data_is_deterministic_and_well_formed() -> None:
    first = generate_sales_data(rows=60, seed=12)
    second = generate_sales_data(rows=60, seed=12)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["date", "temperature_c", "ice_creams_sold"]
    assert first["ice_creams_sold"].min() >= 0


def test_generate_sales_data_rejects_small_dataset() -> None:
    with pytest.raises(ValueError, match="at least 50"):
        generate_sales_data(rows=49)


def test_save_and_load_dataset(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "sales.csv"
    source = generate_sales_data(rows=60)

    assert save_dataset(source, output) == output
    loaded = load_dataset(output)
    assert len(loaded) == 60
    assert loaded["date"].is_monotonic_increasing

    with pytest.raises(FileExistsError):
        save_dataset(source, output)


def test_load_dataset_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.csv")


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.drop(columns=["temperature_c"]), "Missing required columns"),
        (lambda frame: frame.head(20), "at least 50"),
        (lambda frame: frame.assign(temperature_c=None), "null values"),
        (lambda frame: frame.assign(date="invalid"), "invalid dates"),
        (
            lambda frame: frame.assign(date=["2024-01-01"] * len(frame)),
            "unique values",
        ),
        (lambda frame: frame.assign(temperature_c="hot"), "must be numeric"),
        (lambda frame: frame.assign(temperature_c=80), "between -10 and 55"),
        (lambda frame: frame.assign(ice_creams_sold=-1), "between 0 and 1000"),
    ],
)
def test_validate_dataset_rejects_invalid_data(mutator, message: str) -> None:
    data = generate_sales_data(rows=60)
    with pytest.raises(DatasetValidationError, match=message):
        validate_dataset(mutator(data))
