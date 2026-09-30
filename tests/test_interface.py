"""Deterministic boundary tests; real language-model accuracy is evaluated separately."""
import copy
import json

import httpx
import pytest

from src.config import ROOT
from src.interface import (Extraction, OPTIONAL_FEATURES, REQUIRED_FEATURES, SAMPLE_FEATURES,
                           review_extraction, validate_features)
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
    assert result.missing == ["Overall Qual"]
    assert "Garage Cars" in result.defaulted
    assert old == {"Neighborhood": "NAmes"}


def test_units_convert_in_code(categories):
    result = review_extraction(extraction([
        {"name": "Lot Area", "value": .25, "unit": "acres", "evidence": "0.25 acres"},
        {"name": "Lot Frontage", "value": 20, "unit": "m", "evidence": "20 m"},
    ]), "Lot 0.25 acres, frontage 20 m.", {}, categories)
    assert result.features["Lot Area"] == 10890
    assert result.features["Lot Frontage"] == pytest.approx(65.6167979)


def test_exact_category_label_is_normalized_without_fuzzy_guessing(categories):
    review = review_extraction(extraction([
        {"name": "Neighborhood", "value": "North Ames", "unit": "none", "evidence": "North Ames"},
        {"name": "House Style", "value": "unusual house", "unit": "none", "evidence": "unusual house"},
    ]), "An unusual house in North Ames.", SAMPLE_FEATURES, categories)
    assert review.features["Neighborhood"] == "NAmes"
    assert "House Style" not in review.features
    assert review.issues and not review.ready


def test_unstated_unit_is_not_inferred_even_when_model_supplies_one(categories):
    result = review_extraction(extraction([
        {"name": "Gr Liv Area", "value": 1500, "unit": "sq_ft", "evidence": "living area is 1500"},
    ]), "The living area is 1500.", SAMPLE_FEATURES, categories)
    assert "Gr Liv Area" not in result.features
    assert not result.ready


def test_extracted_number_must_match_quoted_input(categories):
    result = review_extraction(extraction([
        {"name": "Garage Cars", "value": 2, "unit": "none", "evidence": "3-car garage"},
    ]), "Correction: a 3-car garage.", SAMPLE_FEATURES, categories)
    assert "Garage Cars" not in result.features
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
    unanswered = review_extraction(extraction([], "estimate", "Which units did you mean?"), "1500", SAMPLE_FEATURES, categories)
    assert not unanswered.ready


def test_duplicate_update_blocks_stale_value(categories):
    update = {"name": "Garage Cars", "value": 2, "unit": "none", "evidence": "2 cars"}
    result = review_extraction(extraction([update, update]), "2 cars", SAMPLE_FEATURES, categories)
    assert not result.ready
    assert "Garage Cars" not in result.features


def test_valid_features_do_not_mutate_original(categories):
    before = copy.deepcopy(SAMPLE_FEATURES)
    assert validate_features(SAMPLE_FEATURES, categories).ready
    assert SAMPLE_FEATURES == before


def test_core_facts_allow_defaults_without_inventing_features(categories):
    core = {key: SAMPLE_FEATURES[key] for key in REQUIRED_FEATURES}
    review = validate_features(core, categories)
    assert review.ready
    assert review.features == core
    assert review.defaulted == list(OPTIONAL_FEATURES)


@pytest.mark.parametrize("missing", REQUIRED_FEATURES)
def test_each_core_fact_is_still_required(categories, missing):
    review = validate_features({key: value for key, value in SAMPLE_FEATURES.items() if key != missing}, categories)
    assert not review.ready
    assert review.missing == [missing]


@pytest.mark.parametrize("key,value", [("Garage Cars", -1), ("Full Bath", 1.5),
                                        ("Lot Area", float("nan")), ("Central Air", "unknown")])
def test_invalid_optional_fact_is_not_silently_defaulted(categories, key, value):
    review = validate_features({**SAMPLE_FEATURES, key: value}, categories)
    assert not review.ready
    assert review.issues
    assert key not in review.defaulted


def test_zero_is_known_and_none_is_unknown(categories):
    core = {key: SAMPLE_FEATURES[key] for key in REQUIRED_FEATURES}
    review = validate_features({**core, "Garage Cars": 0, "Total Bsmt SF": None}, categories)
    assert review.ready
    assert review.features["Garage Cars"] == 0
    assert "Garage Cars" not in review.defaulted
    assert "Total Bsmt SF" in review.defaulted


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
        if body["format"].get("title") == "Scope":
            return httpx.Response(200, json={"done": True, "message": {"content": '{"kind":"housing"}'}})
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
        if json.loads(request.content)["format"].get("title") == "Scope":
            return httpx.Response(200, json={"done": True, "message": {"content": '{"kind":"housing"}'}})
        calls.append(request)
        updates = ([{"name": "Year Built", "value": 1960, "unit": "none", "evidence": "built in 1960"}]
                   if len(calls) == 1 else [{"name": "Year Built", "value": 2000, "unit": "none", "evidence": "built in 1960"}])
        return httpx.Response(200, json={"done": True, "message": {"content": json.dumps({
            "intent": "estimate", "updates": updates, "question": ""
        })}})
    client = LocalLLM(Settings(), transport=httpx.MockTransport(handler))
    result = client.parse("It was built in 1960 and has a 2-car garage.", {}, categories)
    assert len(calls) == 2
    assert result.features["Year Built"] == 1960
    assert not result.ready


def test_disabled_language_model_makes_no_request(categories):
    calls = []
    client = LocalLLM(Settings(enabled=False), transport=httpx.MockTransport(lambda req: calls.append(req)))
    with pytest.raises(LLMError, match="switched off"):
        client.parse("An Ames home", {}, categories)
    assert calls == []


def test_scope_gate_blocks_stale_complete_home_before_extraction(categories):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"done": True, "message": {"content": '{"kind":"out_of_scope"}'}})
    client = LocalLLM(Settings(), transport=httpx.MockTransport(handler))
    result = client.parse("What is my Chicago condo worth today?", SAMPLE_FEATURES, categories)
    assert result.intent == "out_of_scope"
    assert not result.ready
    assert "2006" in result.question
    assert len(calls) == 1


def test_explicit_unknown_clears_a_known_optional_value(categories):
    review = review_extraction(extraction([
        {"name": "Garage Cars", "value": None, "unit": "none", "evidence": "garage capacity is unknown"}
    ]), "Actually the garage capacity is unknown.", SAMPLE_FEATURES, categories)
    assert review.ready
    assert "Garage Cars" not in review.features
    assert "Garage Cars" in review.defaulted
    assert review.touched == ["Garage Cars"]
    assert review.evidence == {}


def test_clearing_without_explicit_unknown_is_rejected(categories):
    review = review_extraction(extraction([
        {"name": "Garage Cars", "value": None, "unit": "none", "evidence": "a two-car garage"}
    ]), "It has a two-car garage.", SAMPLE_FEATURES, categories)
    assert not review.ready
    assert "Garage Cars" in review.field_issues


def test_evidence_records_stated_unit_and_validated_conversion(categories):
    review = review_extraction(extraction([
        {"name": "Lot Area", "value": .25, "unit": "acres", "evidence": "0.25 acres"}
    ]), "The lot is 0.25 acres.", SAMPLE_FEATURES, categories)
    assert review.features["Lot Area"] == 10890
    assert review.evidence["Lot Area"] == {"quote": "0.25 acres", "stated_value": .25, "unit": "acres"}


def test_partial_explanation_receives_supplied_facts_without_full_input_mae():
    def handler(request):
        facts = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert facts["home_details"]["Ames neighborhood"] == "North Ames"
        assert "test_mae_usd" not in facts
        return httpx.Response(200, json={"done": True, "message": {"content": json.dumps({
            "estimate_usd": 150000, "summary": "The historical estimate for this North Ames home is $150,000.",
            "limitation": "Unknown optional details use training defaults."
        })}})
    client = LocalLLM(Settings(), transport=httpx.MockTransport(handler))
    result = client.explain(150000, {}, ["Garage Cars"], {"Neighborhood": "NAmes"})
    assert result.estimate_usd == 150000


def test_neighborhood_uses_exact_quoted_name_not_a_conflicting_model_code(categories):
    review = review_extraction(extraction([
        {"name": "Neighborhood", "value": "NWAmes", "unit": "none", "evidence": "North Ames"}
    ]), "A home in North Ames.", SAMPLE_FEATURES, categories)
    assert review.features["Neighborhood"] == "NAmes"
    assert review.ready


def test_two_quoted_neighborhoods_require_clarification(categories):
    review = review_extraction(extraction([
        {"name": "Neighborhood", "value": "NAmes", "unit": "none", "evidence": "North Ames or Northwest Ames"}
    ]), "North Ames or Northwest Ames", SAMPLE_FEATURES, categories)
    assert "Neighborhood" not in review.features
    assert not review.ready


def test_generic_home_description_cannot_establish_style_or_building_type(categories):
    review = review_extraction(extraction([
        {"name": "House Style", "value": "1Story", "unit": "none", "evidence": "a lovely home"},
        {"name": "Bldg Type", "value": "1Fam", "unit": "none", "evidence": "a lovely home"},
    ]), "A lovely home", {}, categories)
    assert review.features == {}
    assert set(review.field_issues) == {"House Style", "Bldg Type"}


def test_negative_central_air_cannot_be_labeled_yes(categories):
    review = review_extraction(extraction([
        {"name": "Central Air", "value": "Y", "unit": "none", "evidence": "no central air"}
    ]), "There is no central air.", SAMPLE_FEATURES, categories)
    assert "Central Air" not in review.features
    assert not review.ready


def test_missing_optional_fields_do_not_trigger_unnecessary_repair(categories):
    from src.interface import QUICK_QUERY
    calls = []
    updates = [
        {"name": "Neighborhood", "value": "NAmes", "unit": "none", "evidence": "North Ames"},
        {"name": "Year Built", "value": 1960, "unit": "none", "evidence": "built in 1960"},
        {"name": "Gr Liv Area", "value": 1500, "unit": "sq_ft", "evidence": "1,500 sq ft"},
        {"name": "Overall Qual", "value": 6, "unit": "none", "evidence": "quality 6 out of 10"},
    ]
    def handler(request):
        calls.append(request)
        body = json.loads(request.content)
        answer = {"kind": "housing"} if body["format"].get("title") == "Scope" else {
            "intent": "estimate", "updates": updates, "question": ""}
        return httpx.Response(200, json={"done": True, "message": {"content": json.dumps(answer)}})
    result = LocalLLM(Settings(), transport=httpx.MockTransport(handler)).parse(QUICK_QUERY, {}, categories)
    assert result.ready and len(result.defaulted) == 10
    assert len(calls) == 2  # Scope + extraction, with no speculative optional-field pass.


def test_no_garage_cannot_establish_central_air_status(categories):
    from src.interface import mentioned_fields
    assert mentioned_fields("There is no garage.", ["Central Air"], categories) == []
    review = review_extraction(extraction([
        {"name": "Central Air", "value": "N", "unit": "none", "evidence": "no garage"}
    ]), "There is no garage.", SAMPLE_FEATURES, categories)
    assert "Central Air" not in review.features
    assert not review.ready


def test_second_pass_clarification_blocks_even_without_any_updates(categories):
    extraction_calls = []
    def handler(request):
        body = json.loads(request.content)
        if body["format"].get("title") == "Scope":
            answer = {"kind": "housing"}
        else:
            extraction_calls.append(request)
            answer = {"intent": "estimate", "updates": [], "question": ""} if len(extraction_calls) == 1 else {
                "intent": "clarify", "updates": [], "question": "How many cars does the garage fit?"}
        return httpx.Response(200, json={"done": True, "message": {"content": json.dumps(answer)}})
    current = {key: value for key, value in SAMPLE_FEATURES.items() if key != "Garage Cars"}
    review = LocalLLM(Settings(), transport=httpx.MockTransport(handler)).parse(
        "The garage fits either two or three cars; I am unsure.", current, categories)
    assert not review.ready
    assert review.intent == "clarify" and review.question
