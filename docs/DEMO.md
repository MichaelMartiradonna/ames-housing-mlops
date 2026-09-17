# Live review walkthrough

Allow about five minutes. Start Ollama and the Streamlit app as described in the README. Wait until the app says the local model is ready. The first request after model unloading may take longer.

## Successful prediction

1. Open **Explore a home** and select **Quick example**.
2. Read the description aloud: it supplies neighborhood, above-ground living area, year built and material/finish quality.
3. Choose **Read home details**. Explain that Qwen runs locally and extracts structured input; it does not set the price.
4. Review the four required fields. Expand **Review the defaults** to show exactly which unknown optional details use training medians or modes. Explain that these are not facts about this home. If a required field is missed, show the clarification workflow.
5. Check the review confirmation and choose **Confirm & estimate**.
6. Show the predicted dollar amount and the generated explanation. Open **Details used for this estimate**.
7. Explain that the complete saved Random Forest pipeline generated the number, including training-fitted imputation, scaling and encoding. The published full-input test error is not the error estimate for this partial-input home.
8. Optionally start a new home and choose **Full example** to show the original all-details workflow with no optional defaults.

## Follow-up

Enter: **Actually, it has a 3-car garage.**

Choose **Read home details**. The garage value should change while other known fields remain; its default should disappear. The earlier result is cleared and confirmation resets. Review and estimate again. A change in model output is not proof of the causal value of adding a garage. Entering 0 means no garage, while leaving it blank means unknown.

## Missing information and scope

1. Start a new home. Enter: **For a historical Ames estimate: it was built in 1995 and has 1,800 sq ft of above-ground living area.**
2. Read the details. Show that the app asks for neighborhood and quality and cannot estimate while those required facts are missing. It should not demand the ten optional details.
3. Enter: **Estimate the current value of my Chicago condo.**
4. Show the scope response. Explain that training on Ames sales from 2006–2010 does not support current Chicago valuations.

## Evidence to show

- **Model & limitations** tab: dataset scope, split, selected configuration and metrics.
- MLflow at localhost:5000: five configurations and complete pipeline artifacts.
- `reports/experiment_comparison.csv`: three test metrics for each candidate, selected by validation MAE first.
- `pytest tests/ -v`: deterministic tests.
- `reports/interface_evaluation.json`: actual local-language evaluation, with inputs, outcomes and timings.

A live demonstration during review satisfies the course's demo option. This document is a runbook, not a claim that a recording was made. If recording later, show the actual app and include one successful estimate and at least one incomplete/out-of-scope example.
