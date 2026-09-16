# Project Summary and Engineering Retrospective

## Outcome

This project implements a reproducible local and CI workflow for historical Ames home-price prediction. It connects versioned data, configuration-driven training, five MLflow experiments, automated validation, and an Evidently drift demonstration.

The selected **200-tree random forest** uses 14 property characteristics and was chosen by validation MAE. On the reserved 586-record test set, it achieved:

- **MAE:** $17,772.88
- **RMSE:** $30,214.75
- **R²:** 0.8861

Both planned acceptance gates passed: MAE no higher than $35,000 and R² no lower than 0.75. A training-median baseline had a test MAE of $63,822.74; the selected model reduced that error by approximately **72.2%**. Prices and errors are expressed in the historical dataset's nominal dollars.

Evidence is recorded in [the experiment summary](reports/experiment_summary.json), [the ranked runs](reports/experiment_comparison.csv), [the completion checklist](CHECKLIST.md), and [GitHub Actions](https://github.com/MichaelMartiradonna/ames-housing-mlops/actions/workflows/mlops.yml).

### Final verification

A fresh clone of the public repository downloaded the release attachment without credentials, verified its checksums, restored the dataset with DVC, and passed all 18 tests on Windows. Both monitoring scenarios returned the expected exit codes. The [GitHub Actions run](https://github.com/MichaelMartiradonna/ames-housing-mlops/actions/runs/35099450327) independently installed dependencies and restored data on separate Ubuntu runners; testing and gated training both passed. Its saved test results and model metrics matched the documented outcomes.

The final requirements review found no outstanding required components. Documentation links, data and model exclusions, configuration use, and decision comments were checked. The implementation remains within the agreed scope; the recommendations below are potential extensions.

## Decisions and rationale

### Select a dataset with a clear target and modest preparation

Ames provides 2,930 labeled observations, categorical and numeric inputs, and genuine missing values. This meets the dataset requirements without developing a separate collection system. Local tax records, Indiana sales data, and AoE4 match archives offered interesting extensions but would have added acquisition or target-definition work. The scope remained focused on MLOps.

### Use a compact feature set and one model family

Fourteen property characteristics provide an interpretable interface and keep training inexpensive. A random forest handles nonlinear relationships without requiring a more elaborate modeling framework. The five configurations vary capacity and regularization while retaining the same dataset, split, and preprocessing.

The unrestricted 200-tree model had the lowest validation MAE, **$16,634.13**. The unrestricted 100-tree model was close at **$16,703.16**—a difference of about $69. The selection follows the predefined metric; this small difference is not presented as proof of a statistically significant improvement.

### Separate fitting, selection, and final evaluation

The deterministic split uses 1,758 training, 586 validation, and 586 test records. Imputation and encoding are learned only from training data. Validation MAE selects the configuration, and only that winner receives a final test evaluation within each experiment batch.

CI subsequently reuses the fixed test set as a regression gate. This is useful for catching changes in behavior, but repeated use must not become an informal optimization loop against the holdout. A future model-development cycle would need a fresh evaluation strategy.

### Package preprocessing with the model

A saved estimator alone would depend on undocumented external transformations. The MLflow artifact therefore includes validation, imputation, encoding, and the regressor as one pipeline. Numeric inputs use floating-point types so missing values remain representable in the saved schema.

The model uses Skops serialization with an explicit allowlist for the project's validation transformer and NumPy's dtype. A persistence test checks that a reloaded artifact produces matching predictions for missing numeric values and previously unseen categories.

### Keep data and generated artifacts outside source history

Git versions code and small metadata; DVC identifies the source dataset. A checksummed GitHub Release attachment distributes the local DVC remote. This satisfies a local development workflow without adding another service account. The setup step is explicit: restore the remote, then run `dvc pull`.

MLflow stores experiment metadata and models locally. The repository retains compact comparison evidence and viewing instructions. A fresh checkout can recreate the five-run batch rather than depending on absolute paths inside another computer's tracking database.

### Make acceptance decisions executable

Performance gates fail the training command when MAE or R² misses its configured threshold. The training CI job depends on the full test job. Monitoring saves its report and returns a distinct drift-alert exit code when the configured drift share is exceeded. These decisions are automated and observable.

## Findings and lessons

**Missingness must be measured, not assumed.** Lot frontage is missing in 490 records—approximately 16.7% of the dataset. Two other selected numeric features have one missing value each. Median imputation keeps those observations available while avoiding test-data influence on fitted preprocessing.

**An apparently working model can still have an incomplete artifact contract.** In-memory prediction is only part of reproducibility. Explicit type handling, serialization dependencies, and a reload test are needed to preserve the same behavior after saving.

**Data retrieval must be tested from an empty checkout.** Local caches can conceal missing remote objects or discovery problems. Ignore rules must exclude data bytes while allowing DVC to discover its pointers. The release manifest and clean-checkout verification make this dependency visible.

**Comparable experiments need shared context.** Run records include the data hash, split settings, model parameters, source metadata, and a batch identifier. Comparison rejects incomplete batches and selects runs for the current dataset version. This avoids treating unrelated results as a fair comparison.

**Large errors deserve separate attention.** RMSE is materially higher than MAE, indicating that some misses are substantially larger than the typical absolute error. The metric set makes that limitation visible even though the acceptance gates pass.

**Drift is a prompt to investigate.** The shifted simulation flagged exactly five predictors and produced a 35.71% drift share; the unchanged control produced 0%. This verifies the monitoring and alert behavior. It does not establish real-world performance degradation without new outcome labels.

## Implemented reliability and security controls

- Pinned direct dependencies, deterministic splits, configuration-driven settings, and dataset hash verification.
- Separate SHA-256 verification of the release archive and dataset object; archive extraction is constrained to the intended directory.
- Tests for malformed input, performance-gate rejection, altered downloads, unsafe archive paths, and model persistence.
- Read-only GitHub Actions permissions and public data retrieval without CI credentials.
- Exclusion of data, models, local tracking databases, caches, and credential files from source history.
- Explicit serialization trust configuration and a locally bound MLflow interface.

These controls address the demonstrated workflow. They do not constitute a comprehensive production security assessment.

## Limitations

The dataset covers one city and an older period. It does not support claims about current prices, Indiana housing, or national market trends. A random split tests performance within the historical sample; it does not directly measure forecasting performance across time.

The model is deliberately simple, with five bounded experiments rather than an extensive tuning exercise. The drifted data is synthetic. There is no prediction service, live data ingestion, retraining scheduler, or operational incident process. Local MLflow paths are machine-specific, and the release-backed DVC remote requires the documented bootstrap step.

## Prioritized future improvements

1. **Strengthen evaluation before expanding use.** Add chronological validation, repeated splits, and residual checks by price range and neighborhood. This would estimate temporal generalization and selection uncertainty before making broader valuation claims.
2. **Improve monitoring before collecting live traffic.** Calibrate thresholds against independent batches, add missingness checks, and evaluate labeled performance. Define who receives alerts and how investigation or retraining is approved.
3. **Harden dependency and release management for ongoing maintenance.** Add a complete transitive dependency lock, vulnerability review, immutable action references, and signed release provenance. These would reduce environment drift and improve supply-chain accountability.
4. **Move to shared infrastructure only when collaboration requires it.** A durable shared DVC remote and hosted tracking service could remove the bootstrap step and centralize experiment history. Access control, backups, and retention would then need explicit ownership.

These improvements are recommendations. The implemented scope remains the complete local and GitHub-based workflow described in the README.
