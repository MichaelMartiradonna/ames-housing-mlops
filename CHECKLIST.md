# Completion Checklist

This checklist maps the repository to the required project components. Checked items have implementation or verification evidence; publication checks remain open until the corresponding external result is confirmed.

## Dataset and repository

- [x] Tabular regression dataset with at least 1,000 rows, eight predictors, mixed types, and missing values: [dataset findings](reports/experiment_summary.json).
- [x] Source, target, selected predictors, and limitations documented in the [README](README.md).
- [x] Separate source modules, YAML configuration, tests, workflow, pinned dependencies, and Git exclusions.
- [x] Git and DVC initialized; raw data excluded from Git and [DVC pointer committed](data/raw/AmesHousing.tsv.dvc).
- [x] Portable local remote configured and [release manifest generated](configs/data_release.json).
- [ ] Public GitHub repository and data release attachment available.
- [ ] Clean checkout restores the remote and retrieves the dataset with `dvc pull`.

## Training and MLflow

- [x] Training reads paths, features, splits, model settings, and thresholds from YAML.
- [x] Training-only imputation and encoding; inputs remain unchanged.
- [x] Five completed experiments log model parameters, data hash, three metrics, and complete pipeline artifacts.
- [x] `mlflow.search_runs()` ranks the comparable batch by validation MAE.
- [x] Selected configuration and held-out test metrics recorded; planned thresholds passed.
- [x] Training returns failure when a performance threshold is missed.
- [ ] Final saved-model reload contract and full test suite verified together.
- [ ] Five-run MLflow screenshot included with reproduction instructions.

## pytest

- [x] Six preprocessing tests: numeric missingness, categorical missingness, categorical encoding, unseen categories, input immutability, and invalid inputs.
- [x] Three actual-data tests: columns and eligibility, target range, and numeric ranges.
- [x] Two model tests: prediction type/shape and minimum performance on a held-out set after small-sample training.
- [x] Failure-path tests cover performance gates, checksums, and archive extraction boundaries.
- [x] Monitoring tests cover unchanged data and the expected synthetic drift.
- [ ] Full suite passes with `pytest tests/ -v` after final artifact-contract changes.

## GitHub Actions

- [x] Workflow triggers on pushes to `main` and pull requests targeting `main`.
- [x] Test job installs dependencies, restores data, and runs the full suite.
- [x] Dependent training job installs dependencies, restores data, trains, and enforces thresholds.
- [x] Workflow artifacts preserve test and training evidence.
- [ ] Successful complete pipeline visible in the Actions history.

## Drift monitoring

- [x] Training features used as the reference; production features generated reproducibly.
- [x] Evidently evaluates all 14 predictors.
- [x] Summary identifies drifted features and overall drift share.
- [x] HTML and JSON reports generated for control and shifted scenarios.
- [x] Control exits 0; 35.71% simulated drift exceeds 30% and exits 1.
- [x] [Monitoring analysis](MONITORING.md) answers all three required questions.

## Documentation and delivery

- [x] README describes setup, restoration, training, experiments, tests, CI, and monitoring.
- [x] [Retrospective](PROJECT_SUMMARY.md) explains decisions, findings, lessons, safeguards, limitations, and future improvements.
- [x] Future improvements are separated from implemented capabilities.
- [ ] Grader workflow rehearsed from a clean checkout.
- [ ] Final documentation, evidence, repository state, and public links checked.
- [ ] Public repository URL ready for submission.
