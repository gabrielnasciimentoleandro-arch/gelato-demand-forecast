from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from gelato_forecast.modeling import FEATURES, TARGET, TrainingResult

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def create_reports(data: pd.DataFrame, result: TrainingResult, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _plot_demand_curve(data, result, output_dir / "demand-vs-temperature.png"),
        _plot_candidate_scores(result, output_dir / "model-comparison.png"),
        _plot_residuals(result, output_dir / "residuals.png"),
    ]
    return paths


def _plot_demand_curve(data: pd.DataFrame, result: TrainingResult, path: Path) -> Path:
    temperature_grid = np.linspace(
        float(data[FEATURES[0]].min()), float(data[FEATURES[0]].max()), 200
    )
    grid_frame = pd.DataFrame({FEATURES[0]: temperature_grid})
    curve = result.model.predict(grid_frame)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.scatter(
        data[FEATURES[0]],
        data[TARGET],
        alpha=0.28,
        color="#5B3FD0",
        edgecolors="none",
        label="Dados sintéticos",
    )
    axis.plot(temperature_grid, curve, color="#E85D04", linewidth=2.5, label="Modelo selecionado")
    axis.set(
        title="Demanda de sorvetes por temperatura",
        xlabel="Temperatura (°C)",
        ylabel="Sorvetes vendidos",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _plot_candidate_scores(result: TrainingResult, path: Path) -> Path:
    labels = [score.name.replace("_", " ").title() for score in result.candidates]
    values = [score.cv_rmse for score in result.candidates]
    errors = [score.cv_std for score in result.candidates]

    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.barh(labels, values, xerr=errors, color=["#2A9D8F", "#457B9D", "#F4A261"])
    axis.invert_yaxis()
    axis.set(title="Comparação por validação cruzada", xlabel="RMSE médio (menor é melhor)")
    axis.bar_label(bars, fmt="%.2f", padding=4)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _plot_residuals(result: TrainingResult, path: Path) -> Path:
    residuals = result.y_test.to_numpy(dtype=float) - result.predictions
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.scatter(result.predictions, residuals, alpha=0.65, color="#6A4C93")
    axis.axhline(0, color="#D62828", linewidth=1.5, linestyle="--")
    axis.set(
        title="Resíduos no conjunto de teste",
        xlabel="Vendas previstas",
        ylabel="Erro observado - previsto",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
