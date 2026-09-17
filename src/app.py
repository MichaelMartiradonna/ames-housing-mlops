"""Ames home-price lab: describe, review, estimate, and understand."""

import hmac
import os

import httpx
import streamlit as st

from src.config import load_config
from src.interface import (CATEGORY_LABELS, LABELS, OPTIONAL_FEATURES, QUICK_QUERY,
                           REQUIRED_FEATURES, SAMPLE_QUERY, validate_features)
from src.llm import LLMError, LocalLLM, Settings
from src.serving import load_serving_model, predict, training_defaults


@st.cache_resource
def load_resources():
    return load_serving_model()


@st.cache_data(ttl=20)
def local_status(base_url: str, model: str) -> bool:
    try:
        response = httpx.get(base_url + "/api/tags", timeout=2, trust_env=False)
        response.raise_for_status()
        return model in {item["name"] for item in response.json()["models"]}
    except (httpx.HTTPError, ValueError, KeyError):
        return False


def set_fields(features):
    st.session_state.confirm_details = False
    for key in LABELS:
        value = features.get(key)
        st.session_state["field_" + key] = (str(value).removesuffix(".0") if isinstance(value, (float, int)) else value)


def invalidate_confirmation(changed_field):
    st.session_state.confirm_details = False
    st.session_state.result = None
    # A rejected extracted optional value must be deliberately corrected in the
    # form or clarified in a follow-up, not silently replaced by a default.
    st.session_state.review_issues = [issue for issue in st.session_state.review_issues
                                    if changed_field not in issue and LABELS[changed_field] not in issue]


def read_fields(raw):
    features = dict(raw)
    for key in load_config()["data"]["numeric_features"]:
        value = raw.get(key)
        try:
            features[key] = float(value.replace(",", "")) if value and value.strip() else None
        except ValueError:
            features[key] = value
    return features


def display_value(value):
    return f"{value:,.0f}" if isinstance(value, (int, float)) and value == int(value) else CATEGORY_LABELS.get(value, str(value))


def show_defaults(defaults):
    for key, value in defaults.items():
        st.write(f"{LABELS[key]}: **{display_value(value)}**")


def main():
    st.set_page_config(page_title="Ames · Home-price lab", page_icon="🏡", layout="wide")
    settings = Settings.from_env()
    hosted = os.getenv("AMES_HOSTED", "false").lower() == "true"
    password = os.getenv("AMES_DEMO_PASSWORD", "")
    if hosted and settings.enabled:
        st.error("This hosted demo must run with language features disabled. Use the local app for the full workflow.")
        st.stop()
    if password:
        entered = st.sidebar.text_input("Reviewer access", type="password")
        if not hmac.compare_digest(entered, password):
            st.info("Enter the reviewer password to open this demo.")
            st.stop()
    try:
        model, metadata = load_resources()
        defaults = training_defaults(model)
    except (OSError, ValueError, httpx.HTTPError):
        st.error("The housing model is unavailable. Follow the model setup instructions in the README, then restart the app.")
        st.stop()
    for key, value in {"draft": {}, "history": [], "result": None, "review_issues": [], "intent": "estimate", "description": ""}.items():
        st.session_state.setdefault(key, value)
    if "field_Lot Area" not in st.session_state:
        set_fields(st.session_state.draft)

    st.caption("AMES / HOME-PRICE LAB")
    st.title("A home description. A model-backed estimate.")
    st.write("Describe an Ames home, review the details, and explore a predicted historical sale price.")
    st.caption("Educational model · Ames, Iowa sales from 2006–2010 · Not a current appraisal")
    a, b, c = st.columns(3)
    a.metric("Historical sales", "2,930")
    b.metric("Test error · full inputs", f"${metadata['test_metrics']['mae']:,.0f}")
    c.metric("Test homes", f"{metadata['test_rows']:,}")
    estimate_tab, method_tab = st.tabs(["Explore a home", "Model & limitations"])
    with estimate_tab:
        left, right = st.columns([1.05, 1], gap="large")
        with left:
            st.subheader("1. Describe the home")
            if settings.enabled:
                ready = local_status(settings.base_url, settings.model)
                st.caption(f"Local language model: {settings.model} · {'Ready' if ready else 'Waiting for Ollama'}")
            else:
                st.info("Manual demo: the housing model works here. Run the app locally for natural-language parsing and explanations.")
            quick, example, reset = st.columns(3)
            if quick.button("Quick example", width="stretch"):
                st.session_state.description = QUICK_QUERY
            if example.button("Full example", width="stretch"):
                st.session_state.description = SAMPLE_QUERY
            if reset.button("Start a new home", width="stretch"):
                st.session_state.draft = {}
                st.session_state.history = []
                st.session_state.result = None
                st.session_state.review_issues = []
                st.session_state.intent = "estimate"
                st.session_state.description = ""
                set_fields({})
            with st.form("description_form", clear_on_submit=False):
                message = st.text_area("Home description or follow-up", key="description", height=190,
                    placeholder="Start with an Ames neighborhood, above-ground living area, year built, and a material/finish quality rating (1–10)…", max_chars=4000)
                parse_clicked = st.form_submit_button("Read home details", type="primary", disabled=not settings.enabled)
            if parse_clicked:
                st.session_state.result = None
                st.session_state.confirm_details = False
                try:
                    with st.spinner("Reading your description locally…"):
                        current = read_fields({key: st.session_state.get("field_" + key) for key in LABELS})
                        current = validate_features(current, metadata["categories"]).features
                        review = LocalLLM(settings).parse(message, current, metadata["categories"])
                    st.session_state.draft = review.features
                    st.session_state.review_issues = review.issues
                    st.session_state.intent = review.intent
                    set_fields(review.features)
                    if review.intent == "out_of_scope":
                        answer = review.question or "I can help estimate historical Ames home sale prices. Please describe an Ames home."
                    elif review.issues:
                        answer = "Please correct the flagged details in the form or clarify them in a follow-up before estimating."
                    elif review.question:
                        answer = review.question
                    elif review.missing:
                        answer = "Please add: " + ", ".join(LABELS[key] for key in review.missing) + ". You can reply here or fill in the form."
                    else:
                        answer = "The required details are ready to review. "
                        if review.defaulted:
                            answer += f"{len(review.defaulted)} optional details are unknown; review their training-based defaults or add what you know. "
                        answer += "Check the form, then choose Confirm & estimate."
                    st.session_state.history.extend([("user", message), ("assistant", answer)])
                    st.session_state.history = st.session_state.history[-8:]
                except LLMError as exc:
                    st.session_state.intent = "clarify"
                    st.error(str(exc))
            for role, text in st.session_state.history[-4:]:
                with st.chat_message(role):
                    st.write(text)
            for issue in st.session_state.review_issues:
                st.warning(issue)
            if st.session_state.intent != "estimate":
                st.info("Clarify the description before estimating, or review the form and accept any listed defaults manually.")
        with right:
            st.subheader("2. Review the details")
            st.caption("Four details are required. Add anything else you know to refine the estimate. Areas use square feet.")
            config = load_config()
            raw = {}
            with st.container(border=True):
                def render_fields(keys, required):
                    cols = st.columns(2)
                    for index, key in enumerate(keys):
                        with cols[index % 2]:
                            if key in config["data"]["numeric_features"]:
                                help_text = None
                                if key == "Overall Qual":
                                    help_text = "A whole-number rating of overall materials and finish, from 1 (lowest) to 10 (highest). This is not a condition rating. Use a known rating; words such as 'nice' cannot establish it."
                                elif key in {"Garage Cars", "Total Bsmt SF"}:
                                    help_text = "Enter 0 if there is no garage/basement. Leave blank only if unknown."
                                raw[key] = st.text_input(LABELS[key], key="field_" + key,
                                    placeholder="Required" if required else "Optional · unknown",
                                    help=help_text, on_change=invalidate_confirmation, args=(key,))
                            else:
                                raw[key] = st.selectbox(LABELS[key], metadata["categories"][key], index=None,
                                    format_func=lambda item: CATEGORY_LABELS.get(item, item), key="field_" + key,
                                    placeholder="Choose…" if required else "Optional · unknown",
                                    on_change=invalidate_confirmation, args=(key,))
                render_fields(REQUIRED_FEATURES, True)
                with st.expander("Add optional details (10)"):
                    st.caption("Garage capacity and basement area were especially helpful in our validation comparison. Enter 0 for none; blank means unknown.")
                    render_fields(OPTIONAL_FEATURES, False)
                review = validate_features(read_fields(raw), metadata["categories"])
                used_defaults = {key: defaults[key] for key in review.defaulted}
                if used_defaults:
                    st.info(f"{len(used_defaults)} optional details will use training-based defaults. These are typical training values, not known facts about this home.")
                    with st.expander("Review the defaults"):
                        show_defaults(used_defaults)
                    st.caption("Estimates with missing details may be less accurate. The full-input test error above does not measure their accuracy.")
                else:
                    st.caption("All optional details supplied. No optional defaults will be used.")
                confirmed = st.checkbox("I reviewed the details and any listed defaults for a historical Ames estimate.", key="confirm_details")
                submitted = st.button("Confirm & estimate", key="estimate", type="primary", width="stretch")
            if submitted:
                st.session_state.result = None
                if not confirmed:
                    st.warning("Please check the confirmation box after reviewing the details.")
                elif st.session_state.review_issues:
                    st.warning("Correct the flagged extracted details in the form or clarify them in a follow-up before estimating.")
                elif not review.ready:
                    for issue in review.issues:
                        st.warning(issue)
                    if review.missing:
                        st.warning("Still needed: " + ", ".join(LABELS[key] for key in review.missing))
                else:
                    st.session_state.draft = review.features
                    price = predict(model, metadata, review.features)
                    result = {"price": price, "explanation": None, "error": None,
                              "features": review.features, "defaults": used_defaults}
                    if settings.enabled:
                        try:
                            with st.spinner("Explaining the model’s result locally…"):
                                result["explanation"] = LocalLLM(settings).explain(price, metadata, review.defaulted).model_dump()
                        except LLMError as exc:
                            result["error"] = str(exc)
                    st.session_state.result = result
            if st.session_state.result:
                result = st.session_state.result
                with st.container(border=True):
                    st.subheader("3. Your historical estimate")
                    st.metric("Predicted sale price", f"${result['price']:,.0f}")
                    if result["defaults"]:
                        st.info(f"Estimate uses {len(result['defaults'])} training-based defaults for unknown details. Additional facts can refine it.")
                    if result["explanation"]:
                        st.markdown(result["explanation"]["summary"].replace("$", r"\$"))
                        # Render the statistical caveat from trusted metadata. Small
                        # models sometimes confuse prediction values with prediction errors.
                        if not result["defaults"]:
                            st.caption(f"Individual prediction errors may exceed the model’s average full-input test error of \\${metadata['test_metrics']['mae']:,.0f}.")
                    elif result["error"]:
                        st.warning(result["error"])
                        st.caption("The price above comes from the trained housing model; a language explanation is unavailable.")
                    else:
                        st.caption("Manual-mode result from the trained Random Forest. No language model was used.")
                    if result["defaults"]:
                        st.caption("The reported full-input test error does not apply to this partial-input estimate. Missing details can increase error; no per-home confidence interval is available.")
                    st.caption("Based on 2006–2010 Ames sales. Average model error is not a confidence interval for this home.")
                    with st.expander("Details used for this estimate"):
                        st.write("**Details you supplied**")
                        for key, value in result["features"].items():
                            st.write(f"{LABELS[key]}: {display_value(value)}")
                        if result["defaults"]:
                            st.write("**Training defaults for unknown details**")
                            show_defaults(result["defaults"])
    with method_tab:
        st.subheader("Two models, two jobs")
        st.write("The local language model extracts facts and explains the result. A trained Random Forest calculates the price. You review every input before prediction.")
        st.write("The 2,930 sales are split into 1,758 training, 586 validation and 586 test rows. Five configurations are ranked by validation MAE before their test results are reported.")
        st.write("Four facts are required: living area, neighborhood, year built and material/finish quality. Unknown optional values use the saved pipeline's training-set medians or modes, shown before confirmation. Zero means none; it is never treated as unknown. Numeric fields are standardized and categories are one-hot encoded inside the saved pipeline.")
        st.write("The published test metrics use all available dataset inputs. A separate validation comparison checks the effect of masking optional details; it is a development comparison, not a new test benchmark. More details do not guarantee a better prediction for every home.")
        st.write(f"Selected configuration: {metadata['configuration']}. Test RMSE: ${metadata['test_metrics']['rmse']:,.0f}. Test R²: {metadata['test_metrics']['r2']:.3f}.")
        st.warning("This is an educational historical model, not a current valuation or an appraisal. Errors can be much larger for individual homes, unusual properties, or details outside the training data. Language extraction can also make mistakes; always review the form.")
        st.caption(f"Model release: {metadata['version']} · MLflow run: {metadata['run_id']}")
        st.link_button("View project and reproducible results", "https://github.com/MichaelMartiradonna/ames-housing-mlops")


if __name__ == "__main__":
    main()
