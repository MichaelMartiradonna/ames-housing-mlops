# Capstone verification checklist

## Implemented and checked locally

- [x] Ames dataset: 2,930 rows, mixed types, missing values, SalePrice target.
- [x] Train-fitted imputation, categorical encoding, scaling, immutable inputs.
- [x] Five meaningfully different MLflow configurations, YAML settings, full pipeline artifacts.
- [x] Programmatic validation-MAE selection using mlflow.search_runs().
- [x] MAE, RMSE and R² on held-out test data for every frozen candidate.
- [x] Selected pipeline export with all-586-row prediction parity and checksum manifest.
- [x] Streamlit description, follow-up, review and prediction workflow.
- [x] Local Ollama integration with no API key or paid fallback.
- [x] Missing/invalid/ambiguous input handling and explicit confirmation.
- [x] Deterministic preprocessing, model, interface and UI tests.
- [x] Live local-model evaluation script and report; inspect its results, not just unit-test status.
- [x] Pinned dependencies, environment template, data/model exclusions.
- [x] README, architecture, results, limitations and engineering reflection.
- [x] Success and edge-case live-demo runbook.

## Final verification

- [x] Final live evaluation (12/12 development cases), 45 tests and browser walkthrough verified.
- [x] Capstone branch and selected model release published; a fresh download was verified.
- [x] GitHub tests, training and container checks passed: https://github.com/MichaelMartiradonna/ames-housing-mlops/actions/runs/35251447840
- [ ] Author completes guided explanation checkpoints and personal reflection.
- [ ] Author gives or records the course demo and submits the public repository link.

The local Windows Docker engine failed during startup on its internal socket, before any project container was run. Do not interpret an included Dockerfile as proof of a successful build; use the capstone CI result for container verification.

A full hosted language demo is not part of the chosen no-payment local model setup. Earlier MLOps verification remains in reports/original-mlops and is not evidence for the new interface. The separate independent skills assessment is still planned for after submission.
