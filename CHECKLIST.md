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

- [x] Original required-14-field version: 12/12 live development cases, 45 tests and browser walkthrough verified.
- [x] Capstone branch and selected model release published; a fresh download was verified.
- [x] GitHub tests, training and container checks passed: https://github.com/MichaelMartiradonna/ames-housing-mlops/actions/runs/35251447840
- [ ] Author completes guided explanation checkpoints and personal reflection.
- [ ] Author gives or records the course demo and submits the public repository link.

## Optional-input revision

- [x] Validation-only comparison of six input policies on 586 homes, using the unchanged model.
- [x] Four required facts; ten optional fields; exact training-default preview and explicit confirmation.
- [x] Invalid optional values block prediction; zero and unknown remain distinct.
- [x] 61 automated tests and 16/16 live language development cases pass locally.
- [x] Browser walkthrough: four facts extracted, exact defaults reviewed, real estimate and explanation displayed.
- [x] User reviews the local changes before finalization.
- [x] Local Docker engine, current app build, health, actual model predictions, and browser manual workflow verified.
- [x] Publish the reviewed changes in [PR #2](https://github.com/MichaelMartiradonna/ames-housing-mlops/pull/2); [GitHub tests, training and container checks passed](https://github.com/MichaelMartiradonna/ames-housing-mlops/actions/runs/35263928912) for the updated application.

The earlier Windows Docker startup issue is no longer present. Local verification on Docker Desktop 4.91.0 / Engine 29.8.0 passed for the current app in manual mode. See [reports/docker_verification.json](reports/docker_verification.json) for source hashes and results, and [docs/DOCKER_WALKTHROUGH.md](docs/DOCKER_WALKTHROUGH.md) to repeat the steps. The full Ollama Compose stack remains unverified; the native app provides the tested natural-language workflow.

A full hosted language demo is not part of the chosen no-payment local model setup. Earlier MLOps verification remains in reports/original-mlops and is not evidence for the new interface. The separate independent skills assessment is still planned for after submission.
