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
- [x] README reflection covers lessons, challenges, tradeoffs, improvements, and AI assistance.
- [x] Course requirements reviewed against the implementation and evidence; see [docs/SUBMISSION.md](docs/SUBMISSION.md).
- [x] Live-demo walkthrough prepared with natural-language input, actual prediction, explanation, and edge cases.
- [x] Include the completed [screen recording on YouTube (5:45)](https://youtu.be/1mW6WnJn5k4); linked from the README and submission guide.
- [x] Submitted and approved: 91/100 including the Docker bonus (Excellent), per reviewer feedback.

## Optional-input revision

- [x] Validation-only comparison of six input policies on 586 homes, using the unchanged model.
- [x] Four required facts; ten optional fields; exact training-default preview and explicit confirmation.
- [x] Invalid optional values block prediction; zero and unknown remain distinct.
- [x] 61 automated tests and 16/16 live language development cases pass locally.
- [x] Browser walkthrough: four facts extracted, exact defaults reviewed, real estimate and explanation displayed.
- [x] User reviews the local changes before finalization.
- [x] Local Docker engine, current app build, health, actual model predictions, and browser manual workflow verified.
- [x] Publish the reviewed changes in [PR #2](https://github.com/MichaelMartiradonna/ames-housing-mlops/pull/2); [GitHub tests, training and container checks passed](https://github.com/MichaelMartiradonna/ames-housing-mlops/actions/runs/35263928912) for the updated application.

The optional-input course version passed Docker verification. Its evidence is preserved in [reports/development/course-docker-verification.json](reports/development/course-docker-verification.json).

## Portfolio interface revision

- [x] Sources and quoted evidence visible beside extracted fields; readable category names.
- [x] Corrections preserve unrelated inputs, reset confirmation, and mark the old estimate as outdated.
- [x] Price renders before the explanation; an explanation retry preserves the prediction.
- [x] 83 automated tests and 25/25 real local-model development cases passed, including fresh examples and controlled explanation failure/recovery.
- [x] README hiring-manager introduction, worked example/screenshots, five experiment rows, and explicit saved-model inference path.
- [x] Rebuilt Linux image: health, source-file parity, real released-model predictions, and manual review/correction checks passed.
- [x] Docker browser flow with native Ollama: all fourteen details extracted, confirmed estimate of $219,776, and actual language explanation displayed.
- [x] GitHub's container job now runs the same repeatable check and saves its JSON evidence.

See [reports/workflow_verification.json](reports/workflow_verification.json), [reports/docker_verification.json](reports/docker_verification.json), and the [Docker walkthrough](docs/DOCKER_WALKTHROUGH.md). The complete separate Ollama Compose stack remains unverified. Current remote check results are available in [GitHub Actions](https://github.com/MichaelMartiradonna/ames-housing-mlops/actions/workflows/mlops.yml).

A full hosted language demo is not part of the chosen no-payment local model setup. Earlier MLOps verification remains in reports/original-mlops and is not evidence for the new interface. Guided explanation questions and an extra independently written test are optional learning activities, not requirements in the supplied course rubric. They do not block submission. The separate independent skills assessment remains a possible activity after submission; it has not been completed.
