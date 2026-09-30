from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime paths and MLflow configuration."""

    project_root: Path
    data_path: Path
    model_path: Path
    metadata_path: Path
    reports_dir: Path
    tracking_uri: str
    experiment_name: str
    registered_model_name: str

    @classmethod
    def from_environment(cls, project_root: Path | None = None) -> Settings:
        configured_root = os.getenv("GELATO_PROJECT_ROOT")
        root = project_root or (Path(configured_root) if configured_root else Path.cwd())
        root = root.resolve()
        return cls(
            project_root=root,
            data_path=_resolve_path(
                root, os.getenv("GELATO_DATA_PATH", "data/ice_cream_sales.csv")
            ),
            model_path=_resolve_path(root, os.getenv("GELATO_MODEL_PATH", "models/model.joblib")),
            metadata_path=_resolve_path(
                root, os.getenv("GELATO_METADATA_PATH", "models/metadata.json")
            ),
            reports_dir=_resolve_path(root, os.getenv("GELATO_REPORTS_DIR", "reports")),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{root / 'mlflow.db'}"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "gelato-demand-forecast"),
            registered_model_name=os.getenv("MLFLOW_MODEL_NAME", "gelato-demand-model"),
        )


def _resolve_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()
