"""Hosted demo contracts: AMES_MANUAL_ONLY=1 shows the form and never touches the language model."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.config import ROOT
from src.interface import OPTIONAL_FEATURES, REQUIRED_FEATURES, SAMPLE_FEATURES

HOSTED_DIR = ROOT / "hosted"
HOSTED_ENTRYPOINT = HOSTED_DIR / "streamlit_app.py"


def forbidden(*args, **kwargs):
    raise AssertionError("The language-model path must never be touched in manual-only mode.")


class ForbiddenClient:
    def __init__(self, *args, **kwargs):
        forbidden()


class ForbiddenSettings:
    from_env = staticmethod(forbidden)


@pytest.fixture
def hosted_app(monkeypatch):
    from src import app as module
    class PredictableModel:
        def predict(self, frame):
            return np.array([150000.])
    metadata = json.loads((ROOT / "configs" / "model_release.json").read_text())
    defaults = json.loads((ROOT / "reports" / "optional_input_evaluation.json").read_text())["training_defaults"]
    monkeypatch.setenv("AMES_MANUAL_ONLY", "1")
    # Local-only settings must be ignored: nothing may switch the language model on.
    monkeypatch.setenv("AMES_LLM_ENABLED", "true")
    monkeypatch.setenv("AMES_HOSTED", "false")
    monkeypatch.setenv("AMES_DEMO_PASSWORD", "")
    # Any touch of the Ollama client, its settings, or its HTTP transport fails the test.
    monkeypatch.setattr(module, "LocalLLM", ForbiddenClient)
    monkeypatch.setattr(module, "Settings", ForbiddenSettings)
    monkeypatch.setattr(module, "local_status", forbidden)
    for name in ("get", "post", "stream", "Client"):
        monkeypatch.setattr(module.httpx, name, forbidden)
    monkeypatch.setattr(module, "load_resources", lambda: (PredictableModel(), metadata))
    monkeypatch.setattr(module, "training_defaults", lambda model: defaults)
    return AppTest.from_file(str(HOSTED_ENTRYPOINT), default_timeout=20).run()


def test_manual_only_mode_renders_form_without_ollama(hosted_app):
    from src.app import DEMO_VIDEO_URL, MANUAL_ONLY_BANNER, README_URL
    app = hosted_app
    assert not app.exception
    banner = [item.value for item in app.info if MANUAL_ONLY_BANNER in item.value]
    assert len(banner) == 1
    assert README_URL in banner[0] and DEMO_VIDEO_URL in banner[0]
    # The manual form is shown directly: four required facts and ten optional fields.
    for key in REQUIRED_FEATURES + OPTIONAL_FEATURES:
        if isinstance(SAMPLE_FEATURES[key], str):
            assert app.selectbox(key="field_" + key).value is None
        else:
            assert not app.text_input(key="field_" + key).value
    assert any("Training default if unknown" in item.value for item in app.caption)
    assert any("Still needed" in item.value for item in app.caption)
    # No description step, no parse button, no language-model status.
    assert not any(item.key == "description" for item in app.text_area)
    assert not any(item.label == "Read my description" for item in app.button)
    assert not any(item.key == "check_connection" for item in app.button)
    assert app.button(key="estimate").disabled
    assert app.checkbox(key="confirm_details").disabled


def test_manual_only_mode_estimates_without_explanation(hosted_app):
    app = hosted_app
    for key in REQUIRED_FEATURES:
        value = SAMPLE_FEATURES[key]
        if isinstance(value, str):
            app.selectbox(key="field_" + key).select(value)
        else:
            app.text_input(key="field_" + key).set_value(str(value))
    app.run()
    app.checkbox(key="confirm_details").check().run()
    app.button(key="estimate").click().run()
    assert not app.exception
    home = app.session_state["home"]
    assert home.result["price"] == 150000
    assert home.result["explanation_status"] == "manual"
    assert home.result["explanation"] is None and home.result["error"] is None
    assert len(home.result["defaults"]) == 10
    assert any(m.value == "$150,000" for m in app.metric)
    assert any("manual mode" in item.value for item in app.caption)
    assert not any(item.key == "retry_explanation" for item in app.button)


def test_hosted_mode_import_boundary():
    """A fresh interpreter importing the app in hosted mode loads no language-model or MLOps modules."""
    program = (
        "import os, sys, json\n"
        "os.environ['AMES_MANUAL_ONLY'] = '1'\n"
        "sys.path.insert(0, os.getcwd())\n"
        "import src.app\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    env = {key: value for key, value in os.environ.items() if key != "AMES_LLM_ENABLED"}
    completed = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=120, check=True)
    loaded = set(json.loads(completed.stdout.strip().splitlines()[-1]))
    forbidden_modules = {"ollama", "mlflow", "dvc", "evidently", "dotenv",
                         "src.llm", "src.train", "src.tracking", "src.export_model"}
    assert not {name for name in loaded if name.split(".")[0] in forbidden_modules or name in forbidden_modules}
    # hosted/requirements.txt sits beside the entrypoint, where Community Cloud looks first.
    pins = [line.strip().lower() for line in (HOSTED_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]
    pinned = {line.split("==")[0] for line in pins}
    assert all("==" in line for line in pins)
    assert "scikit-learn==1.7.2" in pins
    assert pinned == {"streamlit", "scikit-learn", "skops", "numpy", "pandas", "httpx", "pydantic", "pyyaml"}
    assert {"streamlit", "sklearn", "skops", "numpy", "pandas", "httpx", "pydantic", "yaml"} <= loaded
