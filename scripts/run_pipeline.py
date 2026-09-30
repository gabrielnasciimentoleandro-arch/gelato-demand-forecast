#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gelato_forecast.config import Settings
from gelato_forecast.pipeline import run_local_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the reproducible Gelato ML pipeline")
    parser.add_argument("--data", type=Path, help="Dataset CSV path")
    parser.add_argument("--model-dir", type=Path, help="Model output directory")
    parser.add_argument("--reports-dir", type=Path, help="Report output directory")
    parser.add_argument("--rows", type=int, default=730)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    settings = Settings.from_environment()
    output = run_local_pipeline(
        data_path=(arguments.data or settings.data_path).resolve(),
        model_dir=(arguments.model_dir or settings.model_path.parent).resolve(),
        reports_dir=(arguments.reports_dir or settings.reports_dir).resolve(),
        rows=arguments.rows,
        seed=arguments.seed,
        test_size=arguments.test_size,
    )
    result = output["result"]
    print(
        json.dumps(
            {
                "selected_model": result.model_name,
                "metrics": result.metadata()["metrics"],
                "model_path": str(output["model_path"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
