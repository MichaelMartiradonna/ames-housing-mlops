# Completion Checklist

All required project components were verified on September 16, 2026. Supporting evidence is linked below and recorded in [the verification summary](reports/verification.json).

## Dataset and repository

- [x] Tabular regression dataset with at least 1,000 rows, eight predictors, mixed types, and missing values: [dataset findings](reports/experiment_summary.json).
- [x] Source, target, selected predictors, and limitations documented in the [README](README.md).
- [x] Separate source modules, YAML configuration, tests, workflow, pinned dependencies, and Git exclusions.
- [x] Git and DVC initialized; raw data excluded from Git and [DVC pointer committed](data/raw/AmesHousing.tsv.dvc).
- [x] Portable local remote configured and [release manifest generated](configs/data_release.json).
- [x] [Public GitHub repository](https://github.com/MichaelMartiradonna/ames-housing-mlops) and [data release attachment](https://github.com/MichaelMartiradonna/ames-housing-mlops/releases/tag/data-v1) available.
- [x] Clean checkout restores the packaged remote and retrieves the dataset with `dvc pull`.
- [x] Clean checkout downloads the published release attachment automatically without credentials, verifies its checksum, and restores the dataset.

## Training and MLflow

- [x] Training reads paths, features, splits, model settings, and thresholds from YAML.
- [x] Training-only imputation and encoding; inputs remain unchanged.
- [x] Five completed experiments log model parameters, data hash, three metrics, and complete pipeline artifacts.
- [x] `mlflow.search_runs()` ranks the comparable batch by validation MAE.
- [x] Selected configuration and held-out test metrics recorded; planned thresholds passed.
- [x] Training returns failure when a performance threshold is missed.
- [x] Final saved-model reload contract and full test suite verified together.
- [x] [Five-run MLflow screenshot](reports/mlflow_experiments.png) included with reproduction instructions.

## pytest

- [x] Six preprocessing tests: numeric missingness, categorical missingness, categorical encoding, unseen categories, input immutability, and invalid inputs.
- [x] Three actual-data tests: columns and eligibility, target range, and numeric ranges.
- [x] Two model tests: prediction type/shape and minimum performance on a held-out set after small-sample training.
- [x] Failure-path tests cover performance gates, checksums, and archive extraction boundaries.
- [x] Monitoring tests cover unchanged data and the expected synthetic drift.
- [x] Full suite passes with `pytest tests/ -v` after final artifact-contract changes: 18 tests passed.

## GitHub Actions

- [x] Workflow triggers on pushes to `main` and pull requests targeting `main`.
- [x] Test job installs dependencies, restores data, and runs the full suite.
- [x] Dependent training job installs dependencies, restores data, trains, and enforces thresholds.
- [x] Workflow artifacts preserve test and training evidence.
- [x] [Successful complete pipeline](https://github.com/MichaelMartiradonna/ames-housing-mlops/actions/runs/35099450327) visible in the Actions history; both jobs passed and retained their artifacts.

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
- [x] Local grader commands rehearsed from a clean checkout: tests, five experiments, comparison, training, and monitoring.
- [x] External grader checks verified: public release and successful GitHub Actions run.
- [x] Final documentation, evidence, repository state, and public links checked; decision comments explain the consequential implementation choices.
- [x] Public repository URL ready for submission: https://github.com/MichaelMartiradonna/ames-housing-mlops
