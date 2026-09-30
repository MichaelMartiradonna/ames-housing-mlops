"""Exercise review/correction/recovery with real local inference, outside offline CI."""

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import time

import httpx

from src.config import ROOT
from src.evaluate_interface import cases as regression_cases
from src.interface import REQUIRED_FEATURES, validate_features
from src.llm import LLMError, LocalLLM
from src.serving import load_serving_model, training_defaults
from src.workflow import HomeWorkflow


FRESH_FEATURES = {
    "Neighborhood": "CollgCr", "Year Built": 1998., "Gr Liv Area": 1850., "Overall Qual": 7.,
    "Garage Cars": 2., "Total Bsmt SF": 900., "Full Bath": 2., "Bedroom AbvGr": 3.,
    "Lot Area": 10500., "Lot Frontage": 80., "House Style": "2Story", "Bldg Type": "1Fam",
    "MS Zoning": "RL", "Central Air": "Y",
}
FRESH_QUICK = (
    "For a historical Ames estimate, the home is in College Creek. It was built in 1998, "
    "has 1,850 sq ft of above-ground living area, and an overall material and finish quality of 7 out of 10."
)
FRESH_FULL = FRESH_QUICK + (
    " It is a two-story single-family detached house with low-density residential zoning and central air. "
    "It has a 2-car garage, 900 sq ft of basement area, 2 above-ground full bathrooms, "
    "3 above-ground bedrooms, a 10,500 sq ft lot, and 80 ft of street frontage."
)


def cases():
    return [*regression_cases(),
        {"id": "fresh_quick", "message": FRESH_QUICK, "ready": True,
         "expected": {key: FRESH_FEATURES[key] for key in REQUIRED_FEATURES}},
        {"id": "fresh_full", "message": FRESH_FULL, "ready": True, "expected": FRESH_FEATURES},
        {"id": "fresh_correction", "message": "Correction: the garage fits one car, not two cars.",
         "current": FRESH_FEATURES, "ready": True, "expected": {**FRESH_FEATURES, "Garage Cars": 1.}},
        {"id": "fresh_ambiguous", "message": "I may have misstated the above-ground living area: it is either 1,850 or 2,050 sq ft. I am not sure which.",
         "current": FRESH_FEATURES, "ready": False},
        {"id": "fresh_zero", "message": "The garage capacity is 0 cars, and the basement area is 0 sq ft.",
         "current": FRESH_FEATURES, "ready": True,
         "expected": {**FRESH_FEATURES, "Garage Cars": 0., "Total Bsmt SF": 0.}},
        {"id": "fresh_unknown", "message": "Correction: the garage capacity is unknown and the basement area is unknown. Clear both values.",
         "current": FRESH_FEATURES, "ready": True,
         "expected": {key: value for key, value in FRESH_FEATURES.items() if key not in {"Garage Cars", "Total Bsmt SF"}}},
        {"id": "fresh_chicago", "message": "Different property now: what would my two-bedroom Chicago condo sell for this year?",
         "current": FRESH_FEATURES, "intent": "out_of_scope", "ready": False},
        {"id": "fresh_missing_quality", "message": "For a historical Ames estimate: a lovely home in Brookside, built in 1940, with 1,220 sq ft of above-ground living area.",
         "ready": False, "expected": {"Neighborhood": "BrkSide", "Year Built": 1940., "Gr Liv Area": 1220.},
         "expected_absent": ["Overall Qual"]},
        {"id": "explanation_failure_recovery", "message": FRESH_QUICK, "ready": True,
         "expected": {key: FRESH_FEATURES[key] for key in REQUIRED_FEATURES}, "fault": True},
    ]


def same_features(actual, expected):
    return set(actual) == set(expected) and all(
        abs(actual[key] - value) < .01 if isinstance(value, (int, float)) and isinstance(actual[key], (int, float))
        else actual[key] == value for key, value in expected.items()
    )


def evaluate(case, client, model, metadata, defaults):
    home = HomeWorkflow(features=deepcopy(case.get("current", {})))
    if validate_features(home.features, metadata["categories"]).ready:
        home.estimate(model, metadata, defaults, confirmed=True, language_enabled=False)
    previous = deepcopy(home.result)
    review = client.parse(case["message"], deepcopy(home.features), metadata["categories"])
    parse_usage = deepcopy(client.last_usage)
    home.apply(review)
    actual_review = home.review(metadata["categories"])
    checks = {"ready": actual_review.ready == case["ready"]}
    if "expected" in case:
        checks["features"] = same_features(home.features, case["expected"])
    if "intent" in case:
        checks["intent"] = home.intent == case["intent"]
    checks["invalid_values_not_used"] = all(key not in home.features for key in case.get("expected_absent", []))
    checks["new_values_have_source"] = all(key in home.sources and home.sources[key].get("quote")
                                           for key in review.touched if key in review.features)
    if previous:
        checks["old_estimate_handled"] = home.result is None if home.intent == "out_of_scope" else home.stale
    record = {"features": home.features, "sources": home.sources, "changes": deepcopy(home.changes),
              "missing": actual_review.missing, "defaulted": actual_review.defaulted,
              "intent": home.intent, "question": home.question, "issues": actual_review.issues,
              "parse_usage": parse_usage}
    if actual_review.ready:
        result = home.estimate(model, metadata, defaults, confirmed=True, language_enabled=True)
        result_id, price = result["id"], result["price"]
        checks["price_saved_before_explanation"] = result["explanation"] is None and result["explanation_status"] == "pending"
        if case.get("fault"):
            failed_client = LocalLLM(replace(client.settings, model="ames-workflow-missing-model:latest"))
            try:
                failed_client.explain(price, metadata, list(result["defaults"]), result["features"])
                checks["controlled_failure"] = False
            except LLMError as exc:
                home.save_explanation(result_id, error=str(exc))
                checks["controlled_failure"] = home.result["price"] == price and home.result["explanation_status"] == "failed"
                record["fault_injection"] = {"method": "Real local Ollama request naming a nonexistent model (HTTP 404); no mocked price or extraction.", "error": str(exc)}
        explanation = client.explain(price, metadata, list(result["defaults"]), result["features"])
        checks["saved_explanation"] = home.save_explanation(result_id, explanation=explanation.model_dump())
        checks["price_unchanged"] = home.result["price"] == price and explanation.estimate_usd == round(price)
        checks["retry_keeps_same_prediction"] = home.result["id"] == result_id
        record.update(prediction=price, defaults=result["defaults"], explanation=explanation.model_dump(),
                      explanation_usage=deepcopy(client.last_usage))
    else:
        try:
            home.estimate(model, metadata, defaults, confirmed=True, language_enabled=True)
            checks["blocked_even_if_confirmed"] = False
        except ValueError:
            checks["blocked_even_if_confirmed"] = True
    record.update(checks=checks, passed=all(checks.values()))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case")
    parser.add_argument("--output", default="reports/workflow_evaluation.json")
    args = parser.parse_args()
    selected = [case for case in cases() if not args.case or case["id"] == args.case]
    if not selected:
        parser.error("Unknown case ID.")
    model, metadata = load_serving_model()
    defaults, client = training_defaults(model), LocalLLM()
    with httpx.Client(timeout=10, trust_env=False) as http:
        tags = http.get(client.settings.base_url + "/api/tags").json()["models"]
        runtime = http.get(client.settings.base_url + "/api/version").json()
    model_info = next(item for item in tags if item["name"] == client.settings.model)
    source_files = ["src/app.py", "src/interface.py", "src/llm.py", "src/workflow.py", "src/evaluate_workflow.py"]
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "model": client.settings.model,
              "ollama_version": runtime["version"], "llm_digest": model_info["digest"],
              "housing_model_sha256": metadata["sha256"], "housing_model_run": metadata["run_id"],
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_files},
              "note": "Actual local language model and unchanged released housing pipeline, through the same review state used by the UI. Includes previous development cases and newly added descriptions. This is a development regression check, not an independent accuracy benchmark. Browser and deterministic UI tests are recorded separately. The explanation outage is deliberately induced and retried against real Ollama.",
              "cases": []}
    for case in selected:
        start = time.monotonic()
        try:
            record = evaluate(case, client, model, metadata, defaults)
        except (LLMError, ValueError) as exc:
            record = {"passed": False, "error": str(exc)}
        record.update(id=case["id"], message=case["message"], seconds=round(time.monotonic() - start, 2))
        report["cases"].append(record)
        report.update(passed=sum(item["passed"] for item in report["cases"]), total=len(report["cases"]),
                      updated_at=datetime.now(timezone.utc).isoformat())
        (ROOT / args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{case['id']}: {'PASS' if record['passed'] else 'FAIL'} ({record['seconds']}s)", flush=True)
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
