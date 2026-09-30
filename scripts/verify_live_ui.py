"""Replay real-model UI extraction, a controlled explanation outage, and retry.

Uses Streamlit's AppTest renderer, not a browser. Only the first explanation's
model name is intentionally invalid; extraction, price and retry are real.
"""
import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

from src import app as app_module
from src.config import ROOT
from src.evaluate_workflow import FRESH_FEATURES, FRESH_FULL
from src.llm import LocalLLM


def main():
    calls = []

    class ControlledOutage(LocalLLM):
        def explain(self, price, metadata, defaulted=None, features=None):
            calls.append(price)
            if len(calls) == 1:
                # Real HTTP 404 from the same local server; no canned response.
                return LocalLLM(replace(self.settings, model="ames-workflow-missing-model:latest")).explain(
                    price, metadata, defaulted, features)
            return super().explain(price, metadata, defaulted, features)

    with patch.dict(os.environ, {"AMES_LLM_ENABLED": "true", "AMES_HOSTED": "false", "AMES_DEMO_PASSWORD": ""}), \
            patch.object(app_module, "LocalLLM", ControlledOutage):
        app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=240).run()
        app.text_area(key="description").set_value(FRESH_FULL)
        app.button(key="FormSubmitter:description_form-Read my description").click().run()
        assert not app.exception
        assert app.session_state["home"].features == FRESH_FEATURES
        print("Actual local parsing in the UI: PASS", flush=True)
        app.checkbox(key="confirm_details").check().run()
        app.button(key="estimate").click().run()
        result = app.session_state["home"].result
        result_id, price = result["id"], result["price"]
        assert price == 219776.415
        assert result["explanation_status"] == "failed"
        assert any(metric.value == "$219,776" for metric in app.metric)
        failure = result["error"]
        print("Real Ollama explanation failure leaves the price visible: PASS", flush=True)
        app.button(key="retry_explanation").click().run()
        result = app.session_state["home"].result
        assert not app.exception
        assert result["id"] == result_id and result["price"] == price
        assert result["features"] == FRESH_FEATURES
        assert result["explanation_status"] == "complete" and result["error"] is None
        assert calls == [price, price]
        print("Actual local explanation retry preserves the prediction: PASS", flush=True)
        report = {
            "verified_at": datetime.now(timezone.utc).isoformat(), "passed": True,
            "method": "Streamlit AppTest with real local Ollama parsing, released housing model inference and real Ollama explanation retry. A test-only subclass changes the first explanation request to a nonexistent model, causing an actual HTTP 404. No simulated feature values or price responses.",
            "prediction": price, "display_prediction": "$219,776", "features": result["features"],
            "failure_message": failure, "retry_explanation": result["explanation"],
            "same_prediction_id_after_retry": result["id"] == result_id,
            "checks": {"real_parsing": True, "real_prediction": True, "price_visible_after_failure": True,
                       "retry_succeeded": True, "no_new_prediction_on_retry": True, "no_ui_exceptions": True},
            "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                              for path in ["src/app.py", "src/interface.py", "src/llm.py", "src/workflow.py", "scripts/verify_live_ui.py"]},
        }
        (ROOT / "reports" / "ui_recovery_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
