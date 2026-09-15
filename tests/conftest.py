"""Shared fixtures use the actual versioned dataset and deterministic splits."""

import pytest

from src.config import load_config
from src.data import load_dataset, split_dataset
from src.train import build_model


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def actual_data(config):
    return load_dataset(config)


@pytest.fixture(scope="session")
def splits(actual_data, config):
    return split_dataset(actual_data, config)


@pytest.fixture(scope="session")
def small_model(splits, config):
    X = splits.X_train.sample(n=1000, random_state=config["random_seed"])
    model = build_model(config)
    model.fit(X, splits.y_train.loc[X.index])
    return model
