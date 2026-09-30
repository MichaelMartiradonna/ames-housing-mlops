"""UI workflow contracts; real language behavior is evaluated separately."""
import json

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.config import ROOT
from src.interface import Extraction, REQUIRED_FEATURES, SAMPLE_FEATURES, review_extraction
from src.llm import Explanation, LLMError


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


def fill(app, features):
    for key, value in features.items():
        if isinstance(value, str):
            app.selectbox(key="field_" + key).select(value)
        else:
            app.text_input(key="field_" + key).set_value(str(value))
    return app.run()


def fill_core(app):
    return fill(app, {key: SAMPLE_FEATURES[key] for key in REQUIRED_FEATURES})


def estimate(app):
    app.checkbox(key="confirm_details").check().run()
    return app.button(key="estimate").click().run()


def language(app, monkeypatch, client):
    from src import app as module
    monkeypatch.setenv("AMES_LLM_ENABLED", "true")
    monkeypatch.setattr(module, "local_status", lambda *args: True)
    monkeypatch.setattr(module, "LocalLLM", client)
    return app.run()


def send(app, message):
    app.text_area(key="description").set_value(message)
    return app.button(key="FormSubmitter:description_form-Read my description").click().run()


def test_empty_form_explains_missing_fields_and_disables_estimate(app):
    assert not app.exception
    assert app.button(key="estimate").disabled
    assert app.checkbox(key="confirm_details").disabled
    assert app.session_state["home"].result is None
    assert any("Still needed" in item.value for item in app.caption)


def test_confirmed_form_predicts_and_reset_clears(app):
    fill(app, SAMPLE_FEATURES)
    assert app.button(key="estimate").disabled
    estimate(app)
    assert not app.exception
    assert app.session_state["home"].result["price"] == 150000
    next(b for b in app.button if b.label == "Start a new home").click().run()
    assert app.session_state["home"].result is None
    assert app.session_state["home"].features == {}
    assert not app.checkbox(key="confirm_details").value


def test_partial_form_shows_defaults_and_edits_mark_previous_estimate(app):
    fill_core(app)
    estimate(app)
    original = app.session_state["home"].result
    assert len(original["defaults"]) == 10
    app.text_input(key="field_Garage Cars").set_value("0").run()
    home = app.session_state["home"]
    assert home.stale
    assert home.result == original
    assert not app.checkbox(key="confirm_details").value
    assert any("Previous estimate" in item.value for item in app.warning)
    estimate(app)
    home = app.session_state["home"]
    assert not home.stale
    assert home.result["features"]["Garage Cars"] == 0
    assert "Garage Cars" not in home.result["defaults"]
    assert home.result["sources"]["Garage Cars"]["source"] == "Edited by you"
    app.text_input(key="field_Garage Cars").set_value("").run()
    estimate(app)
    assert "Garage Cars" in app.session_state["home"].result["defaults"]


def test_invalid_optional_form_value_blocks_estimate_with_inline_error(app):
    fill_core(app)
    app.text_input(key="field_Garage Cars").set_value("-1").run()
    assert app.button(key="estimate").disabled
    assert app.session_state["home"].result is None
    assert any("Garage capacity" in error.value for error in app.error)


def test_nonfinite_edit_is_an_inline_error_not_an_app_crash(app):
    fill_core(app)
    estimate(app)
    app.text_input(key="field_Gr Liv Area").set_value("nan").run()
    assert not app.exception
    assert app.button(key="estimate").disabled
    assert any("finite number" in error.value for error in app.error)
    assert app.session_state["home"].stale


def test_missing_core_cannot_be_bypassed_by_confirmation(app):
    app.checkbox(key="confirm_details").check().run()
    assert app.button(key="estimate").disabled
    assert app.session_state["home"].result is None


def test_rejected_extracted_value_requires_deliberate_correction(app, monkeypatch):
    class BadGarage:
        def __init__(self, settings):
            pass
        def parse(self, message, current, categories, progress=None):
            return review_extraction(Extraction(intent="estimate", question="", updates=[
                {"name": "Garage Cars", "value": -1, "unit": "none", "evidence": "-1 car garage"}
            ]), message, current, categories)
    fill_core(app)
    language(app, monkeypatch, BadGarage)
    send(app, "A -1 car garage.")
    assert not app.exception
    assert app.button(key="estimate").disabled
    # An unrelated manual edit cannot erase the rejected value's issue.
    app.text_input(key="field_Year Built").set_value("1961").run()
    assert app.button(key="estimate").disabled
    app.text_input(key="field_Garage Cars").set_value("0").run()
    assert not app.checkbox(key="confirm_details").disabled
    assert not app.session_state["home"].field_issues


def test_follow_up_has_evidence_preserves_other_fields_and_requires_review(app, monkeypatch):
    class Correction:
        def __init__(self, settings):
            pass
        def parse(self, message, current, categories, progress=None):
            return review_extraction(Extraction(intent="estimate", question="", updates=[
                {"name": "Garage Cars", "value": 1, "unit": "none", "evidence": "one-car garage"}
            ]), message, current, categories)
    fill(app, SAMPLE_FEATURES)
    estimate(app)
    language(app, monkeypatch, Correction)
    send(app, "Actually, a one-car garage.")
    home = app.session_state["home"]
    assert home.features == {**SAMPLE_FEATURES, "Garage Cars": 1}
    assert home.sources["Garage Cars"]["quote"] == "one-car garage"
    assert home.changes["Garage Cars"] == {"before": 2, "after": 1}
    assert home.result["features"]["Garage Cars"] == 2
    assert home.stale and not app.checkbox(key="confirm_details").value
    assert any("one-car garage" in item.value for item in app.caption)


def test_chicago_request_hides_old_result_and_cannot_be_bypassed_by_edit(app, monkeypatch):
    class Chicago:
        def __init__(self, settings):
            pass
        def parse(self, message, current, categories, progress=None):
            return review_extraction(Extraction(intent="out_of_scope", question="Historical Ames only.", updates=[]),
                                     message, current, categories)
    fill(app, SAMPLE_FEATURES)
    estimate(app)
    language(app, monkeypatch, Chicago)
    send(app, "What is my Chicago condo worth today?")
    assert app.session_state["home"].result is None
    assert app.button(key="estimate").disabled
    app.text_input(key="field_Garage Cars").set_value("1").run()
    assert app.button(key="estimate").disabled
    assert not any(b.key == "manual" for b in app.button)


def test_unresolved_language_question_requires_reply_or_explicit_manual_review(app, monkeypatch):
    class Ambiguous:
        def __init__(self, settings):
            pass
        def parse(self, message, current, categories, progress=None):
            return review_extraction(Extraction(intent="clarify", question="Which living area is correct?", updates=[]),
                                     message, current, categories)
    fill_core(app)
    language(app, monkeypatch, Ambiguous)
    send(app, "The living area might be 1500 or 1700 sq ft.")
    assert app.button(key="estimate").disabled
    app.text_input(key="field_Gr Liv Area").set_value("1700").run()
    assert app.button(key="estimate").disabled
    app.button(key="manual").click().run()
    assert not app.checkbox(key="confirm_details").disabled
    assert not app.checkbox(key="confirm_details").value


def test_parse_timeout_keeps_description_fields_and_prior_result(app, monkeypatch):
    class Timeout:
        def __init__(self, settings):
            pass
        def parse(self, *args, **kwargs):
            raise LLMError("The local model timed out.")
    fill_core(app)
    estimate(app)
    language(app, monkeypatch, Timeout)
    send(app, "Add a two-car garage.")
    assert app.text_area(key="description").value == "Add a two-car garage."
    assert app.session_state["home"].features["Year Built"] == 1960
    assert app.session_state["home"].stale
    assert app.session_state["home"].result["price"] == 150000
    assert app.button(key="estimate").disabled


def test_empty_follow_up_does_not_invalidate_a_reviewed_estimate(app, monkeypatch):
    class NoRequest:
        def __init__(self, settings):
            raise AssertionError("Empty text should not call the language model")
    fill_core(app)
    estimate(app)
    language(app, monkeypatch, NoRequest)
    send(app, "  ")
    assert not app.exception
    assert not app.session_state["home"].stale
    assert app.checkbox(key="confirm_details").value


def test_explanation_failure_retains_price_and_retry_does_not_predict_again(app, monkeypatch):
    from src import app as module
    calls = []
    class ExplanationRetry:
        def __init__(self, settings):
            pass
        def explain(self, price, metadata, defaulted, features):
            calls.append(price)
            # The actual price snapshot exists before the language request starts.
            assert module.st.session_state.home.result["price"] == price
            if len(calls) == 1:
                raise LLMError("The local model timed out.")
            return Explanation(estimate_usd=150000,
                               summary="The historical estimate for this Ames home is $150,000.",
                               limitation="Historical training data, not a current appraisal.")
    fill_core(app)
    language(app, monkeypatch, ExplanationRetry)
    estimate(app)
    home = app.session_state["home"]
    result_id = home.result["id"]
    assert home.result["price"] == 150000 and home.result["error"]
    assert any(m.value == "$150,000" for m in app.metric)
    app.button(key="retry_explanation").click().run()
    home = app.session_state["home"]
    assert not app.exception
    assert home.result["id"] == result_id
    assert home.result["explanation_status"] == "complete"
    assert home.result["error"] is None
    assert calls == [150000, 150000]
