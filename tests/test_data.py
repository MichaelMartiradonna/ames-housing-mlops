"""Validate the actual downloaded Ames dataset, not a fabricated fixture."""

import numpy as np

from src.config import feature_names


def test_expected_columns_and_dataset_eligibility(actual_data, config):
    assert set(feature_names(config) + [config["data"]["target"], "PID"]) <= set(actual_data.columns)
    assert len(actual_data) == 2930
    assert actual_data["PID"].is_unique
    assert len(feature_names(config)) == 14
    assert actual_data[feature_names(config)].isna().any().any()


def test_target_is_present_and_within_documented_range(actual_data, config):
    target = actual_data[config["data"]["target"]]
    assert target.notna().all()
    assert target.between(*config["data"]["target_range"]).all()


def test_numeric_features_have_expected_ranges(actual_data, config):
    for column, bounds in config["data"]["numeric_ranges"].items():
        values = actual_data[column].dropna()
        assert np.isfinite(values).all(), column
        assert values.between(*bounds).all(), column
