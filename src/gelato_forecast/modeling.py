from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATURES = ["temperature_c"]
TARGET = "ice_creams_sold"


@dataclass(frozen=True, slots=True)
class CandidateScore:
    name: str
    cv_rmse: float
    cv_std: float


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    mae: float
    rmse: float
    r2: float


@dataclass(slots=True)
class TrainingResult:
    model: RegressorMixin
    model_name: str
    metrics: EvaluationMetrics
    candidates: list[CandidateScore]
    residual_std: float
    training_temperature_min: float
    training_temperature_max: float
    training_rows: int
    test_rows: int
    x_test: pd.DataFrame
    y_test: pd.Series
    predictions: np.ndarray
    seed: int
    test_size: float

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "model_name": self.model_name,
            "features": FEATURES,
            "target": TARGET,
            "metrics": {
                "mae": round(self.metrics.mae, 4),
                "rmse": round(self.metrics.rmse, 4),
                "r2": round(self.metrics.r2, 6),
            },
            "candidate_cv_results": [
                {
                    "name": score.name,
                    "cv_rmse": round(score.cv_rmse, 4),
                    "cv_std": round(score.cv_std, 4),
                }
                for score in self.candidates
            ],
            "residual_std": round(self.residual_std, 4),
            "training_temperature_range_c": [
                round(self.training_temperature_min, 2),
                round(self.training_temperature_max, 2),
            ],
            "training_rows": self.training_rows,
            "test_rows": self.test_rows,
            "random_seed": self.seed,
            "test_size": self.test_size,
            "data_origin": "synthetic and deterministic; not real business data",
            "runtime": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
            },
        }


def build_candidates(seed: int) -> dict[str, RegressorMixin]:
    return {
        "linear_regression": LinearRegression(),
        "polynomial_ridge": Pipeline(
            steps=[
                ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale", StandardScaler()),
                ("regression", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=1,
        ),
    }


def train_and_evaluate(
    data: pd.DataFrame,
    *,
    seed: int = 42,
    test_size: float = 0.2,
) -> TrainingResult:
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4")

    features = data.loc[:, FEATURES]
    target = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
    )

    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = build_candidates(seed)
    scores: list[CandidateScore] = []

    for name, candidate in candidates.items():
        fold_scores = -cross_val_score(
            candidate,
            x_train,
            y_train,
            cv=folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        scores.append(
            CandidateScore(
                name=name,
                cv_rmse=float(np.mean(fold_scores)),
                cv_std=float(np.std(fold_scores)),
            )
        )

    scores.sort(key=lambda item: item.cv_rmse)
    best_name = scores[0].name
    best_model = candidates[best_name]
    best_model.fit(x_train, y_train)
    predictions = np.asarray(best_model.predict(x_test), dtype=float)
    residuals = y_test.to_numpy(dtype=float) - predictions

    metrics = EvaluationMetrics(
        mae=float(mean_absolute_error(y_test, predictions)),
        rmse=float(math.sqrt(mean_squared_error(y_test, predictions))),
        r2=float(r2_score(y_test, predictions)),
    )
    return TrainingResult(
        model=best_model,
        model_name=best_name,
        metrics=metrics,
        candidates=scores,
        residual_std=float(np.std(residuals, ddof=1)),
        training_temperature_min=float(x_train["temperature_c"].min()),
        training_temperature_max=float(x_train["temperature_c"].max()),
        training_rows=len(x_train),
        test_rows=len(x_test),
        x_test=x_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        predictions=predictions,
        seed=seed,
        test_size=test_size,
    )


def save_training_result(
    result: TrainingResult,
    model_path: Path,
    metadata_path: Path,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, model_path)

    metadata = result.metadata()
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return model_path, metadata_path
