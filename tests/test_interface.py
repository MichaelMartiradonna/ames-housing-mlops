"""Deterministic boundary tests; real language-model accuracy is evaluated separately."""
import copy
import json

import httpx
import pytest

from src.config import ROOT
from src.interface import Extraction, SAMPLE_FEATURES, review_extraction, validate_features
from src.llm import LLMError, LocalLLM, Settings


@pytest.fixture
def categories():
    return json.loads((ROOT / "configs" / "model_release.json").read_text())["categories"]


def extraction(updates, intent="estimate", question=""):
    return Extraction(intent=intent, updates=updates, question=question)


def test_partial_extraction_merges_without_changing_input(categories):
    old = {"Neighborhood": "NAmes"}
    result = review_extraction(extraction([
        {"name": "Gr Liv Area", "value": 1500, "unit": "sq_ft", "evidence": "1,500 sq ft"},
        {"name": "Year Built", "value": 1960, "unit": "none", "evidence": "built in 1960"},
    ]), "It has 1,500 sq ft above ground and was built in 1960.", old, categories)
    assert result.features == {"Neighborhood": "NAmes", "Gr Liv Area": 1500., "Year Built": 1960.}
    assert not result.ready
    assert "Garage Cars" in result.missing
    assert old == {"Neighborhood": "NAmes"}


def test_units_convert_in_code(categories):
    result = review_extraction(extraction([
        {"name": "Lot Area", "value": .25, "unit": "acres", "evidence": "0.25 acres"},
        {"name": "Lot Frontage", "value": 20, "unit": "m", "evidence": "20 m"},
    ]), "Lot 0.25 acres, frontage 20 m.", {}, categories)
    assert result.features["Lot Area"] == 10890
    assert result.features["Lot Frontage"] == pytest.approx(65.6167979)


def test_unstated_unit_is_not_inferred_even_when_model_supplies_one(categories):
    result = review_extraction(extraction([
        {"name": "Gr Liv Area", "value": 1500, "unit": "sq_ft", "evidence": "living area is 1500"},
    ]), "The living area is 1500.", SAMPLE_FEATURES, categories)
    assert "Gr Liv Area" not in result.features
    assert not result.ready


def test_thousands_separator_variations_preserve_quote_grounding(categories):
    result = review_extraction(extraction([
        {"name": "Lot Area", "value": 9000, "unit": "sq_ft", "evidence": "9000 sq ft lot"},
    ]), "A 9,000 sq ft lot", {}, categories)
    assert result.features["Lot Area"] == 9000
    assert not result.issues


@pytest.mark.parametrize("value", [-1, 1e8, float("nan"), float("inf"), True, "1500"])
def test_invalid_values_block_inference(categories, value):
    features = {**SAMPLE_FEATURES, "Gr Liv Area": value}
    review = validate_features(features, categories)
    assert not review.ready
    assert review.issues


def test_unknown_category_and_fractional_counts_rejected(categories):
    review = validate_features({**SAMPLE_FEATURES, "Neighborhood": "Chicago", "Full Bath": 1.5}, categories)
    assert len(review.issues) == 2
    assert not review.ready


def test_unsupported_or_unquoted_fields_not_trusted(categories):
    review = review_extraction(extraction([
        {"name": "SalePrice", "value": 100000, "unit": "none", "evidence": "price 100000"},
        {"name": "Year Built", "value": 2000, "unit": "none", "evidence": "built 2000"},
    ]), "price 100000", SAMPLE_FEATURES, categories)
    assert len(review.issues) == 2
    assert "Year Built" not in review.features
    assert not review.ready


def test_ambiguity_and_out_of_scope_block_even_complete_previous_draft(categories):
    for intent in ("clarify", "out_of_scope"):
        result = review_extraction(extraction([], intent, "Please clarify."), "What is it worth today?", SAMPLE_FEATURES, categories)
        assert not result.ready


def test_duplicate_update_blocks_stale_value(categories):
    update = {"name": "Garage Cars", "value": 2, "unit": "none", "evidence": "2 cars"}
    result = review_extraction(extraction([update, update]), "2 cars", SAMPLE_FEATURES, categories)
    assert not result.ready
    assert "Garage Cars" not in result.features


def test_valid_features_do_not_mutate_original(categories):
    before = copy.deepcopy(SAMPLE_FEATURES)
    assert validate_features(SAMPLE_FEATURES, categories).ready
    assert SAMPLE_FEATURES == before


def response_transport(content, status=200):
    return httpx.MockTransport(lambda request: httpx.Response(status, json={
        "done": True, "done_reason": "stop", "model": "test-local",
        "message": {"content": json.dumps(content)}, "eval_count": 12,
    }))


def test_client_parsing_contract_and_prompt(categories):
    def handler(request):
        body = json.loads(request.content)
        assert body["stream"] is False and body["think"] is False
        assert body["format"]["additionalProperties"] is False
        assert "latest_message" in body["messages"][1]["content"]
        return httpx.Response(200, json={"done": True, "message": {"content": json.dumps({
            "intent": "estimate", "updates": [{"name": "Garage Cars", "value": 3, "unit": "none", "evidence": "3-car garage"}], "question": ""
        })}})
    client = LocalLLM(Settings(), transport=httpx.MockTransport(handler))
    result = client.parse("Actually a 3-car garage.", SAMPLE_FEATURES, categories)
    assert result.ready and result.features["Garage Cars"] == 3


def test_provider_errors_are_safe_and_no_automatic_retry(categories):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(500, text="private internal details")
    client = LocalLLM(Settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="could not process") as error:
        client.parse("An Ames home", {}, categories)
    assert "private" not in str(error.value)
    assert len(calls) == 1


def test_malformed_response_and_long_input_rejected(categories):
    client = LocalLLM(Settings(), transport=response_transport({"updates": "bad"}))
    with pytest.raises(LLMError, match="validated"):
        client.parse("An Ames home", {}, categories)
    with pytest.raises(LLMError, match="4,000"):
        client.parse("a" * 4001, {}, categories)


def test_explanation_must_preserve_actual_estimate():
    client = LocalLLM(Settings(), transport=response_transport({
        "estimate_usd": 999999, "summary": "The historical estimate is $999,999.",
        "limitation": "This educational model is not a current appraisal.",
    }))
    with pytest.raises(LLMError, match="misstated"):
        client.explain(150000, {"test_metrics": {"mae": 17000}, "test_rows": 586})


def test_remote_and_cloud_backends_rejected():
    with pytest.raises(LLMError, match="local"):
        LocalLLM(Settings(base_url="https://example.com"))
    with pytest.raises(LLMError, match="cloud"):
        LocalLLM(Settings(model="qwen3.5:cloud"))


def test_second_pass_cannot_overwrite_already_valid_fields(categories):
    calls = []
    def handler(request):
        calls.append(request)
        updates = ([{"name": "Year Built", "value": 1960, "unit": "none", "evidence": "built in 1960"}]
                   if len(calls) == 1 else [{"name": "Year Built", "value": 2000, "unit": "none", "evidence": "built in 1960"}])
        return httpx.Response(200, json={"done": True, "message": {"content": json.dumps({
            "intent": "estimate", "updates": updates, "question": ""
        })}})
    client = LocalLLM(Settings(), transport=httpx.MockTransport(handler))
    result = client.parse("It was built in 1960.", {}, categories)
    assert len(calls) == 2
    assert result.features["Year Built"] == 1960
    assert not result.ready


def test_disabled_language_model_makes_no_request(categories):
    calls = []
    client = LocalLLM(Settings(enabled=False), transport=httpx.MockTransport(lambda req: calls.append(req)))
    with pytest.raises(LLMError, match="switched off"):
        client.parse("An Ames home", {}, categories)
    assert calls == []
