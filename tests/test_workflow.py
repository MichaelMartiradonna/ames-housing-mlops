"""Protect the boundary between draft inputs, accepted sources, and saved predictions."""
import json

import numpy as np
import pytest

from src.config import ROOT
from src.interface import Extraction, SAMPLE_FEATURES, review_extraction
from src.workflow import HomeWorkflow


@pytest.fixture
def metadata():
    return json.loads((ROOT / "configs" / "model_release.json").read_text())


def test_late_explanation_cannot_attach_after_input_change(metadata):
    class Model:
        def predict(self, frame):
            return np.array([150000.])
    home = HomeWorkflow(features=SAMPLE_FEATURES.copy())
    result = home.estimate(Model(), metadata, {}, confirmed=True, language_enabled=True)
    home.edit("Garage Cars", 1.)
    assert not home.save_explanation(result["id"], explanation={"summary": "Old result"})
    assert home.result["explanation"] is None
    assert home.result["features"]["Garage Cars"] == 2


def test_unrelated_follow_up_cannot_hide_invalid_manual_input(metadata):
    home = HomeWorkflow(features={**SAMPLE_FEATURES, "Garage Cars": "wrong"})
    review = review_extraction(Extraction(intent="estimate", question="", updates=[
        {"name": "Year Built", "value": 1970, "unit": "none", "evidence": "built in 1970"}
    ]), "It was built in 1970.", {key: value for key, value in SAMPLE_FEATURES.items() if key != "Garage Cars"}, metadata["categories"])
    home.apply(review)
    assert home.features["Garage Cars"] == "wrong"
    assert not home.review(metadata["categories"]).ready
    assert "Garage Cars" in home.review(metadata["categories"]).field_issues


def test_unsupported_extraction_has_visible_general_issue(metadata):
    home = HomeWorkflow()
    review = review_extraction(Extraction(intent="estimate", question="", updates=[
        {"name": "SalePrice", "value": 500000, "unit": "none", "evidence": "500000"}
    ]), "The price is 500000.", {}, metadata["categories"])
    home.apply(review)
    assert home.general_issues
    assert "SalePrice" not in home.features


def test_parse_failure_cannot_unlock_out_of_scope_request(metadata):
    home = HomeWorkflow(features=SAMPLE_FEATURES.copy(), intent="out_of_scope")
    home.fail_parsing("Timeout")
    home.use_manual_form()
    assert home.intent == "out_of_scope"
    assert not home.review(metadata["categories"]).ready
