"""Ames home-price lab: describe, review, estimate, and understand."""

import hmac
import os

import httpx
import streamlit as st

from src.config import load_config
from src.interface import CATEGORY_LABELS, LABELS, SAMPLE_QUERY, validate_features
from src.llm import LLMError, LocalLLM, Settings
from src.serving import load_serving_model, predict


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
    b.metric("Average test error", f"${metadata['test_metrics']['mae']:,.0f}")
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
            example, reset = st.columns(2)
            if example.button("Use example prompt", width="stretch"):
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
                    placeholder="For a historical Ames estimate: describe the location, size, year, rooms, quality and lot…", max_chars=4000)
                parse_clicked = st.form_submit_button("Read home details", type="primary", disabled=not settings.enabled)
            if parse_clicked:
                st.session_state.result = None
                try:
                    with st.spinner("Reading your description locally…"):
                        review = LocalLLM(settings).parse(message, st.session_state.draft, metadata["categories"])
                    st.session_state.draft = review.features
                    st.session_state.review_issues = review.issues
                    st.session_state.intent = review.intent
                    set_fields(review.features)
                    if review.intent == "out_of_scope":
                        answer = review.question or "I can help estimate historical Ames home sale prices. Please describe an Ames home."
                    elif review.question:
                        answer = review.question
                    elif review.missing:
                        answer = "Please add: " + ", ".join(LABELS[key] for key in review.missing) + ". You can reply here or fill in the form."
                    else:
                        answer = "The details are ready to review. Check the form, then choose Confirm & estimate."
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
                st.info("Clarify the description before estimating, or confirm all details manually below.")
        with right:
            st.subheader("2. Review the details")
            st.caption("All 14 fields are required. No missing detail is silently guessed. Areas use square feet.")
            config = load_config()
            raw = {}
            with st.form("home_form"):
                cols = st.columns(2)
                for index, key in enumerate(config["data"]["numeric_features"]):
                    with cols[index % 2]:
                        raw[key] = st.text_input(LABELS[key], key="field_" + key, placeholder="Required")
                for index, key in enumerate(config["data"]["categorical_features"]):
                    with cols[index % 2]:
                        raw[key] = st.selectbox(LABELS[key], metadata["categories"][key], index=None,
                            format_func=lambda item: CATEGORY_LABELS.get(item, item), key="field_" + key,
                            placeholder="Choose…")
                confirmed = st.checkbox("I checked these details for a historical Ames estimate.", key="confirm_details")
                submitted = st.form_submit_button("Confirm & estimate", type="primary", width="stretch")
            if submitted:
                st.session_state.result = None
                features = dict(raw)
                for key in config["data"]["numeric_features"]:
                    value = raw[key]
                    try:
                        features[key] = float(value.replace(",", "")) if value and value.strip() else None
                    except ValueError:
                        features[key] = value
                review = validate_features(features, metadata["categories"])
                if not confirmed:
                    st.warning("Please check the confirmation box after reviewing the details.")
                elif not review.ready:
                    for issue in review.issues:
                        st.warning(issue)
                    if review.missing:
                        st.warning("Still needed: " + ", ".join(LABELS[key] for key in review.missing))
                else:
                    st.session_state.draft = review.features
                    price = predict(model, metadata, review.features)
                    result = {"price": price, "explanation": None, "error": None, "features": review.features}
                    if settings.enabled:
                        try:
                            with st.spinner("Explaining the model’s result locally…"):
                                result["explanation"] = LocalLLM(settings).explain(price, metadata).model_dump()
                        except LLMError as exc:
                            result["error"] = str(exc)
                    st.session_state.result = result
            if st.session_state.result:
                result = st.session_state.result
                with st.container(border=True):
                    st.subheader("3. Your historical estimate")
                    st.metric("Predicted sale price", f"${result['price']:,.0f}")
                    if result["explanation"]:
                        st.markdown(result["explanation"]["summary"].replace("$", r"\$"))
                        st.caption(result["explanation"]["limitation"].replace("$", r"\$"))
                    elif result["error"]:
                        st.warning(result["error"])
                        st.caption("The price above comes from the trained housing model; a language explanation is unavailable.")
                    else:
                        st.caption("Manual-mode result from the trained Random Forest. No language model was used.")
                    st.caption("Based on 2006–2010 Ames sales. The test MAE describes average error, not a confidence interval for this home.")
                    with st.expander("Details used for this estimate"):
                        st.json(result["features"])
    with method_tab:
        st.subheader("Two models, two jobs")
        st.write("The local language model extracts facts and explains the result. A trained Random Forest calculates the price. You review every input before prediction.")
        st.write("The 2,930 sales are split into 1,758 training, 586 validation and 586 test rows. Five configurations are ranked by validation MAE before their test results are reported.")
        st.write("Missing training values are imputed using training-set medians or modes. Numeric fields are standardized and categories are one-hot encoded inside the saved pipeline.")
        st.write(f"Selected configuration: {metadata['configuration']}. Test RMSE: ${metadata['test_metrics']['rmse']:,.0f}. Test R²: {metadata['test_metrics']['r2']:.3f}.")
        st.warning("This is an educational historical model, not a current valuation or an appraisal. Errors can be much larger for individual homes, unusual properties, or details outside the training data. Language extraction can also make mistakes; always review the form.")
        st.caption(f"Model release: {metadata['version']} · MLflow run: {metadata['run_id']}")
        st.link_button("View project and reproducible results", "https://github.com/MichaelMartiradonna/ames-housing-mlops")


if __name__ == "__main__":
    main()
