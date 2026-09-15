"""Load project configuration and resolve paths independently of the shell."""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "model.yaml"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with project_path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    features = feature_names(config)
    if len(features) < 8 or len(features) != len(set(features)):
        raise ValueError("Configure at least eight distinct predictors.")
    if config["data"]["target"] in features:
        raise ValueError("The target cannot be included among predictors.")
    test_size = config["split"]["test_size"]
    val_size = config["split"]["validation_size"]
    if not (0 < test_size < 1 and 0 < val_size < 1 - test_size):
        raise ValueError("Training, validation, and test splits must all be nonempty.")
    if config["model"]["type"] != "RandomForestRegressor":
        raise ValueError("This project supports RandomForestRegressor only.")
    return config


def feature_names(config: dict) -> list[str]:
    return config["data"]["numeric_features"] + config["data"]["categorical_features"]
