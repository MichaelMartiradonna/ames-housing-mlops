"""UI state and confirmation checks; the model itself has separate integration tests."""
import json

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.config import ROOT
from src.interface import SAMPLE_FEATURES


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
    return AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=20).run()


def test_empty_form_does_not_estimate(app):
    assert not app.exception
    app.button(key="FormSubmitter:home_form-Confirm & estimate").click().run()
    assert app.session_state["result"] is None
    assert any("confirmation" in warning.value for warning in app.warning)


def test_confirmed_form_predicts_and_reset_clears(app):
    for key, value in SAMPLE_FEATURES.items():
        if isinstance(value, str):
            app.selectbox(key="field_" + key).select(value)
        else:
            app.text_input(key="field_" + key).set_value(str(value))
    app.checkbox(key="confirm_details").check()
    app.button(key="FormSubmitter:home_form-Confirm & estimate").click().run()
    assert not app.exception
    assert app.session_state["result"]["price"] == 150000
    next(button for button in app.button if button.label == "Start a new home").click().run()
    assert app.session_state["result"] is None
    assert app.session_state["draft"] == {}
    assert not app.checkbox(key="confirm_details").value
