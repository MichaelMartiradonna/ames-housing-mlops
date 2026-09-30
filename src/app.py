"""Ames home-price lab: describe, review, estimate, and understand."""

import hmac
import math
import os

import httpx
import streamlit as st

from src.config import load_config
from src.interface import (CATEGORY_LABELS, LABELS, OPTIONAL_FEATURES, QUICK_QUERY,
                           REQUIRED_FEATURES, SAMPLE_QUERY, validate_features)
from src.llm import LLMError, LocalLLM, Settings
from src.serving import load_serving_model, training_defaults
from src.workflow import HomeWorkflow


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


def display_value(value):
    if value is None:
        return "Unknown"
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return "Invalid value"
        return f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"
    return CATEGORY_LABELS.get(value, str(value))


def sync_fields():
    for key in LABELS:
        value = st.session_state.home.features.get(key)
        st.session_state["field_" + key] = str(value).removesuffix(".0") if isinstance(value, (int, float)) else value
    st.session_state.confirm_details = False


def reset_home(description=""):
    st.session_state.home = HomeWorkflow()
    st.session_state.history = []
    st.session_state.description = description
    st.session_state.clear_description = False
    sync_fields()


def edit_field(key):
    value = st.session_state["field_" + key]
    if key in load_config()["data"]["numeric_features"]:
        if value is None or not value.strip():
            value = None
        else:
            try:
                value = float(value.replace(",", ""))
            except ValueError:
                pass
    st.session_state.home.edit(key, value)
    st.session_state.confirm_details = False


def use_manual_form():
    st.session_state.home.use_manual_form()
    st.session_state.confirm_details = False


def retry_explanation():
    home = st.session_state.home
    if home.result and not home.stale:
        home.result["explanation_status"] = "pending"
        home.result["error"] = None


def render_style():
    # Static CSS only. User and model text use Streamlit's escaped widgets.
    st.html("""
    <style>
      .stMainBlockContainer {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
      h1 {font-size: clamp(1.9rem, 3vw, 2.7rem) !important; letter-spacing: -.035em;}
      h2, h3 {letter-spacing: -.02em;}
      [data-testid="stCaptionContainer"] p {color: #58635e;}
      [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
      .st-key-result [data-testid="stMetricValue"] {font-size: 2.8rem; color: #276A59;}
      .st-key-steps {border-top: 1px solid #d8ddd5; border-bottom: 1px solid #d8ddd5; padding: .6rem 0; margin: .5rem 0 1rem;}
      @media (max-width: 900px) {
        .st-key-workspace > div > [data-testid="stHorizontalBlock"] {flex-direction: column;}
        .st-key-workspace > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {width: 100% !important; flex: 1 1 100% !important;}
      }
    </style>
    """)


def render_description(home, settings, metadata):
    st.subheader("1. Describe the home")
    st.caption("Start with what you know. You can correct every detail before estimating.")
    quick, full = st.columns(2)
    quick.button("Quick example", on_click=reset_home, args=(QUICK_QUERY,), width="stretch",
                 help="Start a new example with the four required details.")
    full.button("Full example", on_click=reset_home, args=(SAMPLE_QUERY,), width="stretch",
                help="Start a new example with all fourteen details.")
    ready = settings.enabled and local_status(settings.base_url, settings.model)
    if settings.enabled:
        if ready:
            st.caption("Local language model available · First request may take longer while it loads.")
        else:
            st.info("Language model unavailable. Start Ollama and check the connection, or enter the details manually.")
            if st.button("Check connection", key="check_connection"):
                local_status.clear()
                st.rerun()
    else:
        st.info("Manual mode · Enter the details below to use the trained housing model.")
    with st.form("description_form"):
        message = st.text_area(
            "Add a detail or correction" if home.features else "Your home description",
            key="description", height=165, max_chars=4000,
            placeholder=("For example: Actually, it has a one-car garage." if home.features else
                         "An Ames home in North Ames, built in 1960, with 1,500 sq ft of above-ground living area and material/finish quality 6 out of 10."),
        )
        parse_clicked = st.form_submit_button("Read my description",
                                              type="secondary" if home.features and home.review(metadata["categories"]).ready else "primary",
                                              width="stretch", disabled=not ready)
    if parse_clicked and not message.strip():
        st.info("Enter a description or correction first. Your current details are unchanged.")
    if parse_clicked and message.strip():
        st.session_state.confirm_details = False
        try:
            stage = st.empty()
            with st.spinner("Reading locally…", show_time=True):
                current = validate_features(home.features, metadata["categories"]).features
                review = LocalLLM(settings).parse(message, current, metadata["categories"], progress=stage.caption)
            stage.empty()
            home.apply(review)
            sync_fields()
            st.session_state.history.append(message)
            st.session_state.history = st.session_state.history[-8:]
            st.session_state.clear_description = True
            st.rerun()
        except LLMError as exc:
            stage.empty()
            home.fail_parsing(str(exc))
    if home.error:
        st.error(home.error)
        st.caption("Your description and existing details are saved. Retry above or continue with the form.")
    if home.intent == "out_of_scope":
        st.warning("This request is outside the model’s scope. It covers Ames sales from 2006–2010, not current prices or other cities. No estimate was produced for this request.")
        st.caption("Start a new home or load an Ames example to continue.")
    elif home.question:
        st.warning(home.question)
        st.caption("Reply above, or explicitly review and resolve the question in the form.")
    if home.intent != "out_of_scope":
        if home.question or home.error:
            st.button("Resolve with the form", key="manual", on_click=use_manual_form, width="stretch")
        else:
            st.markdown("[Enter details manually](#2-review-the-details)")
    if st.session_state.history:
        with st.expander("Your descriptions and corrections"):
            for index, text in enumerate(st.session_state.history, 1):
                st.text(f"{index}. {text}")


def render_field(key, home, review, metadata, defaults, required):
    config = load_config()
    help_text = None
    if key == "Overall Qual":
        help_text = "Overall materials and finish, not condition. Dataset anchors: 1 very poor, 3 fair, 5 average, 6 above average, 7 good, 8 very good, 10 very excellent. Use a known rating; 'nice' cannot establish it."
    elif key in {"Garage Cars", "Total Bsmt SF"}:
        help_text = "Enter 0 for no garage/basement. Clear this field if unknown. Unknown uses a disclosed training default."
    elif key == "Gr Liv Area":
        help_text = "Living space above ground, excluding the basement. Use square feet in the form; a description can specify other units."
    elif key == "Year Built":
        help_text = "Original construction year, between 1800 and 2010 for this historical model."
    if key in config["data"]["numeric_features"]:
        st.text_input(LABELS[key], key="field_" + key, help=help_text,
                      placeholder="Required" if required else "Unknown — optional",
                      on_change=edit_field, args=(key,))
    else:
        st.selectbox(LABELS[key], metadata["categories"][key], index=None,
                     format_func=display_value, key="field_" + key,
                     placeholder="Choose a neighborhood" if required else "Unknown — optional",
                     on_change=edit_field, args=(key,))
    errors = review.field_issues.get(key, [])
    if errors:
        for issue in dict.fromkeys(errors):
            st.error(issue)
    elif key in home.features:
        source = home.sources.get(key, {"source": "Entered in the form"})
        st.caption(source["source"])
        if source.get("quote"):
            st.caption('“' + source["quote"] + '”')
    elif not required:
        st.caption(f"Training default if unknown: {display_value(defaults[key])}")
    if key in home.changes and home.changes[key]["before"] is not None:
        change = home.changes[key]
        st.caption(f"Changed: {display_value(change['before'])} → {display_value(change['after'])}")


def render_review(home, settings, model, metadata, defaults):
    st.subheader("2. Review the details")
    review = home.review(metadata["categories"])
    known_optional = sum(key in review.features for key in OPTIONAL_FEATURES)
    st.caption(f"{4 - len(review.missing)} of 4 required details supplied · {known_optional} optional details supplied · {len(review.defaulted)} defaults")
    with st.container(border=True):
        cols = st.columns(2)
        for index, key in enumerate(REQUIRED_FEATURES):
            with cols[index % 2]:
                render_field(key, home, review, metadata, defaults, True)
        with st.expander(f"Optional details · {known_optional} of 10 supplied",
                         expanded=any(key in review.field_issues for key in OPTIONAL_FEATURES)):
            st.caption("Add what you know. Garage capacity and basement area were useful additions in our validation comparison. Zero means none; blank means unknown.")
            cols = st.columns(2)
            for index, key in enumerate(OPTIONAL_FEATURES):
                with cols[index % 2]:
                    render_field(key, home, review, metadata, defaults, False)
        if home.changes:
            changed = [f"{LABELS[key]}: {display_value(change['before'])} → {display_value(change['after'])}"
                       for key, change in home.changes.items() if change["before"] is not None or home.result]
            if changed:
                st.info("Changes to review: " + "; ".join(changed) + ". Other details were kept.")
        if review.defaulted:
            st.caption(f"{len(review.defaulted)} optional details will use typical training values, not known facts about this home. Review them below or add the actual details.")
            with st.expander("Review defaults before estimating"):
                for key in review.defaulted:
                    st.write(f"{LABELS[key]}: **{display_value(defaults[key])}**")
                st.caption("Missing details may increase error. The full-input test MAE does not measure this partial-input estimate.")
        else:
            st.caption("No optional defaults needed.")
        for issue in home.general_issues:
            st.warning(issue)
        if review.field_issues:
            st.caption("Needs correction: " + ", ".join(LABELS[key] for key in review.field_issues if key in LABELS) + ".")
        if review.missing:
            st.caption("Still needed: " + ", ".join(LABELS[key] for key in review.missing) + ".")
        blocked = not review.ready or bool(home.error)
        if blocked and not review.missing and not review.issues:
            st.caption("Resolve the description above before confirming these details.")
        confirmed = st.checkbox("I reviewed these details and any training defaults for a historical Ames estimate.",
                                key="confirm_details", disabled=blocked)
        submitted = st.button("Update estimate" if home.result else "Confirm details & estimate",
                              key="estimate", type="primary", width="stretch", disabled=blocked or not confirmed)
        if not blocked and not confirmed:
            st.caption("Check the review box to enable your estimate.")
        if submitted:
            try:
                home.estimate(model, metadata, defaults, confirmed=confirmed, language_enabled=settings.enabled)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))


def render_result(home, settings, metadata):
    result = home.result
    if not result:
        return
    with st.container(border=True, key="result"):
        st.subheader("3. Your historical estimate", anchor="historical-estimate")
        if home.stale:
            st.warning("Previous estimate · Based on your previous details. Review the changes and choose Update estimate.")
        st.metric("Historical sale price · Ames, 2006–2010", f"${result['price']:,.0f}")
        core = result["features"]
        st.caption(f"{display_value(core['Neighborhood'])} · Built {int(core['Year Built'])} · {display_value(core['Gr Liv Area'])} sq ft above ground · Quality {display_value(core['Overall Qual'])}/10")
        if result["defaults"]:
            st.caption(f"Uses {len(result['defaults'])} training defaults. Missing details can increase error; the full-input test MAE does not measure this estimate’s accuracy.")
        else:
            st.caption(f"All 14 details supplied. Full-input test MAE: \\${metadata['test_metrics']['mae']:,.0f} across {metadata['test_rows']} homes. Individual errors can be larger.")
        st.caption("A historical prediction, not today’s market value or an appraisal. No per-home confidence interval is available.")
        with st.expander("See the details behind this estimate"):
            st.caption("Values used after filling missing optional inputs, before scaling and category encoding. They belong to this estimate, even if the form has since changed.")
            rows = [{"Detail": LABELS[key],
                     "Value used": display_value(result["features"].get(key, result["defaults"].get(key))),
                     "Source": "Training default" if key in result["defaults"] else result["sources"].get(key, {}).get("source", "Entered in the form")}
                    for key in REQUIRED_FEATURES + OPTIONAL_FEATURES]
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption("The saved preprocessing + Random Forest pipeline loads once and predicts. No training or refitting happens here.")
        if result["explanation_status"] == "pending" and not home.stale:
            result_id = result["id"]
            with st.spinner("Price ready · Writing the explanation locally…", show_time=True):
                try:
                    explanation = LocalLLM(settings).explain(result["price"], metadata,
                                                           list(result["defaults"]), result["features"])
                    home.save_explanation(result_id, explanation=explanation.model_dump())
                except LLMError as exc:
                    home.save_explanation(result_id, error=str(exc))
        if result["explanation"]:
            st.markdown(result["explanation"]["summary"].replace("$", r"\$"))
        elif result["error"]:
            st.warning("The price is ready, but the language explanation is unavailable. " + result["error"])
            if not home.stale and settings.enabled:
                st.button("Retry explanation", key="retry_explanation", on_click=retry_explanation)
                st.caption("Retries only the explanation. Your price and reviewed inputs stay the same.")
        elif result["explanation_status"] == "manual":
            st.caption("Calculated by the trained housing model in manual mode. No language explanation was requested.")


def render_method(metadata, settings):
    st.subheader("Two models, two jobs")
    st.write("The local language model extracts stated facts and explains the result. You check the form and its sources. A saved Random Forest calculates the price from the confirmed details and any disclosed training defaults.")
    a, b, c = st.columns(3)
    a.metric("Historical sales", "2,930")
    b.metric("Test MAE · full inputs", f"${metadata['test_metrics']['mae']:,.0f}")
    c.metric("Held-out test homes", f"{metadata['test_rows']:,}")
    st.write("The split is 1,758 training, 586 validation and 586 test homes. Five configurations were ranked by validation MAE before test results were reported. The training-median baseline had a test MAE of $63,823; the selected model reduced that average error by about 72%.")
    st.write("Four facts are required: living area, neighborhood, year built and material/finish quality. Missing optional values use the saved training medians or most-common categories. These are assumptions, not facts about the home. Zero means none; it is never treated as unknown.")
    st.write("The full-input test metrics do not measure partial-input estimates. A separate validation comparison found MAE of $26,783 with just the four required facts, versus $16,639 with all available inputs. More details do not guarantee a better prediction for every home.")
    st.write(f"Selected model: {metadata['configuration']}. Test RMSE: ${metadata['test_metrics']['rmse']:,.0f}. Test R²: {metadata['test_metrics']['r2']:.3f}. Average error is not a confidence interval for an individual home.")
    st.warning("Historical Ames sales do not establish current values or values in other cities. Language extraction can make mistakes. Review the details and defaults; this is an educational application, not an appraisal.")
    st.caption(f"Local language model: {settings.model} · Saved housing model: {metadata['version']} · MLflow run: {metadata['run_id']}")
    st.link_button("View project and reproducible results", "https://github.com/MichaelMartiradonna/ames-housing-mlops")
    st.link_button("Ames field definitions and quality ratings", "https://jse.amstat.org/v19n3/decock/DataDocumentation.txt")


def main():
    st.set_page_config(page_title="Ames · Home-price lab", page_icon="🏡", layout="wide")
    settings = Settings.from_env()
    if os.getenv("AMES_HOSTED", "false").lower() == "true" and settings.enabled:
        st.error("This hosted demo must run with language features disabled. Use the local app for the full workflow.")
        st.stop()
    password = os.getenv("AMES_DEMO_PASSWORD", "")
    if password and not hmac.compare_digest(st.sidebar.text_input("Reviewer access", type="password"), password):
        st.info("Enter the reviewer password to open this demo.")
        st.stop()
    try:
        model, metadata = load_resources()
        defaults = training_defaults(model)
    except (OSError, ValueError, httpx.HTTPError):
        st.error("The housing model is unavailable. Follow the model setup instructions in the README, then restart the app.")
        st.stop()
    if "home" not in st.session_state:
        reset_home()
    if st.session_state.get("clear_description"):
        st.session_state.description = ""
        st.session_state.clear_description = False
    home = st.session_state.home
    render_style()
    st.caption("AMES / HOME-PRICE LAB")
    st.title("Describe a home. Explore its historical price.")
    st.write("Plain English in. Reviewed details. A model-backed estimate.")
    st.caption("Ames, Iowa · Sales from 2006–2010 · Not a current appraisal")
    estimate_tab, method_tab = st.tabs(["Explore a home", "Model & limitations"])
    with estimate_tab:
        with st.container(key="steps"):
            step, reset = st.columns([3, 1])
            step.caption("DESCRIBE  →  REVIEW  →  ESTIMATE")
            reset.button("Start a new home", on_click=reset_home, width="stretch")
        with st.container(key="workspace"):
            left, right = st.columns([.9, 1.1], gap="large")
            with left:
                render_description(home, settings, metadata)
            with right:
                render_review(home, settings, model, metadata, defaults)
        render_result(home, settings, metadata)
    with method_tab:
        render_method(metadata, settings)


if __name__ == "__main__":
    main()
