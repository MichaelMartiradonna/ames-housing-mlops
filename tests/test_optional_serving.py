"""Verify defaults are the actual fitted values and inference uses them consistently."""

import json

import pytest

from src.config import ROOT
from src.interface import REQUIRED_FEATURES, SAMPLE_FEATURES
from src.serving import predict, training_defaults


def test_partial_prediction_equals_explicit_training_defaults(small_model, splits, config):
    metadata = json.loads((ROOT / "configs" / "model_release.json").read_text())
    defaults = training_defaults(small_model)
    training = splits.X_train.sample(n=1000, random_state=config["random_seed"])
    for key in config["data"]["numeric_features"]:
        assert defaults[key] == training[key].median()
    for key in config["data"]["categorical_features"]:
        assert defaults[key] == training[key].mode().iloc[0]
    core = {key: SAMPLE_FEATURES[key] for key in REQUIRED_FEATURES}
    assert predict(small_model, metadata, core) == pytest.approx(
        predict(small_model, metadata, {**defaults, **core}), abs=1e-8
    )
    assert len(core) == 4
    with pytest.raises(ValueError, match="invalid"):
        predict(small_model, metadata, {**core, "Garage Cars": -1})
