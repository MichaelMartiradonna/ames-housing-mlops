"""UI state and confirmation checks; the model itself has separate integration tests."""
import json

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.config import ROOT
from src.interface import REQUIRED_FEATURES, SAMPLE_FEATURES, validate_features


@pytest.fixture
def app(monkeypatch):
    from src import app as module
    class PredictableModel:
        def predict(self, frame):
            return np.array([150000.])
    metadata = json.loads((ROOT / "configs" / "model_release.json").read_text())
    monkeypatch.setenv("AMES_LLM_ENABLED", "false")
    monkeypatch.setenv("AMES_HOSTED", "false")
    monkeypatch.setenv("AMES_DEMO_PASSWORD", "")
    monkeypatch.setattr(module, "load_resources", lambda: (PredictableModel(), metadata))
    defaults = json.loads((ROOT / "reports" / "optional_input_evaluation.json").read_text())["training_defaults"]
    monkeypatch.setattr(module, "training_defaults", lambda model: defaults)
    return AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=20).run()


def test_empty_form_does_not_estimate(app):
    assert not app.exception
    app.button(key="estimate").click().run()
    assert app.session_state["result"] is None
    assert any("confirmation" in warning.value for warning in app.warning)


def test_confirmed_form_predicts_and_reset_clears(app):
    for key, value in SAMPLE_FEATURES.items():
        if isinstance(value, str):
            app.selectbox(key="field_" + key).select(value)
        else:
            app.text_input(key="field_" + key).set_value(str(value))
    app.run()
    app.checkbox(key="confirm_details").check().run()
    app.button(key="estimate").click().run()
    assert not app.exception
    assert app.session_state["result"]["price"] == 150000
    next(button for button in app.button if button.label == "Start a new home").click().run()
    assert app.session_state["result"] is None
    assert app.session_state["draft"] == {}
    assert not app.checkbox(key="confirm_details").value


def fill_core(app):
    for key in REQUIRED_FEATURES:
        value = SAMPLE_FEATURES[key]
        if isinstance(value, str):
            app.selectbox(key="field_" + key).select(value)
        else:
            app.text_input(key="field_" + key).set_value(str(value))
    app.run()


def test_partial_form_shows_defaults_and_edits_clear_confirmation(app):
    fill_core(app)
    assert any("10 optional details will use" in info.value for info in app.info)
    app.checkbox(key="confirm_details").check().run()
    app.button(key="estimate").click().run()
    assert not app.exception
    assert len(app.session_state["result"]["features"]) == 4
    assert len(app.session_state["result"]["defaults"]) == 10
    app.text_input(key="field_Garage Cars").set_value("0").run()
    assert app.session_state["result"] is None
    assert not app.checkbox(key="confirm_details").value
    assert any("9 optional details will use" in info.value for info in app.info)
    app.checkbox(key="confirm_details").check().run()
    app.button(key="estimate").click().run()
    assert app.session_state["result"]["features"]["Garage Cars"] == 0
    assert "Garage Cars" not in app.session_state["result"]["defaults"]


def test_invalid_optional_form_value_blocks_estimate(app):
    fill_core(app)
    app.text_input(key="field_Garage Cars").set_value("-1").run()
    app.checkbox(key="confirm_details").check().run()
    app.button(key="estimate").click().run()
    assert app.session_state["result"] is None
    assert any("Garage capacity" in warning.value for warning in app.warning)


def test_missing_core_still_blocks_after_confirmation(app):
    app.checkbox(key="confirm_details").check().run()
    app.button(key="estimate").click().run()
    assert app.session_state["result"] is None
    assert any("Still needed" in warning.value for warning in app.warning)


def test_rejected_optional_extraction_needs_deliberate_correction(app, monkeypatch):
    from src import app as module
    class BadGarageExtraction:
        def __init__(self, settings):
            pass

        def parse(self, message, current, categories):
            core = {key: SAMPLE_FEATURES[key] for key in REQUIRED_FEATURES}
            return validate_features({**core, "Garage Cars": -1}, categories)

    monkeypatch.setenv("AMES_LLM_ENABLED", "true")
    monkeypatch.setattr(module, "local_status", lambda *args: True)
    monkeypatch.setattr(module, "LocalLLM", BadGarageExtraction)
    app.run()
    app.text_area(key="description").set_value("A home with a -1 car garage.")
    app.button(key="FormSubmitter:description_form-Read home details").click().run()
    monkeypatch.setenv("AMES_LLM_ENABLED", "false")
    app.checkbox(key="confirm_details").check().run()
    app.button(key="estimate").click().run()
    assert app.session_state["result"] is None
    assert app.session_state["review_issues"]
    app.text_input(key="field_Garage Cars").set_value("0").run()
    assert app.session_state["review_issues"] == []
    assert not app.checkbox(key="confirm_details").value
    app.checkbox(key="confirm_details").check().run()
    app.button(key="estimate").click().run()
    assert not app.exception
    assert app.session_state["result"]["features"]["Garage Cars"] == 0
