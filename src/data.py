"""Read, validate, and reproducibly split the DVC-managed dataset."""

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.config import feature_names, project_path


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def dataset_hash(config: dict) -> str:
    path = project_path(config["data"]["raw_path"])
    if not path.is_file():
        raise FileNotFoundError("Dataset missing. Run the data setup command and dvc pull.")
    actual = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    pointer = project_path(config["data"]["pointer_path"])
    if not pointer.is_file():
        raise FileNotFoundError("DVC pointer missing; initialize data versioning first.")
    expected = yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]["md5"]
    if actual != expected:
        raise ValueError("Dataset content differs from the committed DVC version.")
    return actual


def validate_dataset(frame: pd.DataFrame, config: dict) -> None:
    required = feature_names(config) + [config["data"]["target"], "PID"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(frame) != config["data"]["expected_rows"]:
        raise ValueError("Dataset row count differs from the documented source version.")
    if frame["PID"].isna().any() or not frame["PID"].is_unique:
        raise ValueError("Property identifiers must be present and unique.")
    target = pd.to_numeric(frame[config["data"]["target"]], errors="raise")
    if not target.between(*config["data"]["target_range"]).all():
        raise ValueError("Sale prices are missing or outside the configured range.")
    for name, bounds in config["data"]["numeric_ranges"].items():
        values = pd.to_numeric(frame[name], errors="raise").dropna()
        if not np.isfinite(values).all() or not values.between(*bounds).all():
            raise ValueError(f"{name} contains values outside its configured range.")


def load_dataset(config: dict, *, verify_hash: bool = True) -> pd.DataFrame:
    if verify_hash:
        dataset_hash(config)
    frame = pd.read_csv(project_path(config["data"]["raw_path"]), sep="\t", dtype={"PID": str})
    validate_dataset(frame, config)
    return frame


def split_dataset(frame: pd.DataFrame, config: dict) -> DataSplits:
    X = frame.loc[:, feature_names(config)].copy(deep=True)
    # Float columns retain missing values and match the exported MLflow schema.
    numeric = config["data"]["numeric_features"]
    X[numeric] = X[numeric].astype(float)
    y = frame[config["data"]["target"]].astype(float).copy()
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=config["split"]["test_size"], random_state=config["random_seed"]
    )
    train_X, val_X, train_y, val_y = train_test_split(
        train_X,
        train_y,
        test_size=config["split"]["validation_size"] / (1 - config["split"]["test_size"]),
        random_state=config["random_seed"],
    )
    return DataSplits(train_X, val_X, test_X, train_y, val_y, test_y)
