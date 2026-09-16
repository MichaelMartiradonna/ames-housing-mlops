# Drift Monitoring Analysis

## Evaluation setup

The reference distribution contains the **1,758 training records** and the same 14 predictors used by the model. The monitor compares raw predictor distributions before imputation or encoding, so the feature names remain interpretable.

Two reproducible scenarios exercise the monitor:

- **Control:** an unchanged copy of the reference data. This verifies that identical inputs do not trigger an alert.
- **Drifted:** a copy with four numeric distributions shifted and the neighborhood mix changed. The raw DVC dataset is never modified.

Evidently uses a Kolmogorov–Smirnov test for numeric features and a chi-square test for categorical features. Each feature is flagged when its p-value is below **0.05**. The command returns an alert when **more than 30%** of predictors drift. These settings are explicit in [the configuration](configs/model.yaml).

## 1. Which features showed drift, and why?

The control detected **0 of 14 features**, a **0% drift share**, and returned exit code 0.

The shifted scenario detected **5 of 14 features**, a **35.71% drift share**, and returned exit code 1:

- **Lot Area:** multiplied by 1.8, shifting the simulated population toward larger lots.
- **Lot Frontage:** multiplied by 1.4, shifting frontage measurements upward while preserving missing entries.
- **Gr Liv Area:** multiplied by 1.5, increasing above-ground living area.
- **Total Bsmt SF:** multiplied by 1.5, increasing basement area.
- **Neighborhood:** 1,406 records (approximately 80%) were selected using the fixed random seed and assigned `NAmes`. Because some records already had that label, the final `NAmes` share was **82.42%**, compared with **15.53%** in the reference data.

The remaining nine feature columns were unchanged and did not drift. Tests for the five modified features returned extremely small p-values; `Lot Area` and `Neighborhood` were reported as zero at machine precision. These values indicate strong distribution differences in this simulation. They do not quantify the probability that the model has lost accuracy.

Evidence: [control summary](reports/drift_control.json), [shifted summary](reports/drift_drifted.json), [control HTML](reports/drift_control.html), and [shifted HTML](reports/drift_drifted.html). Download the HTML files and open them in a browser; GitHub's source view does not render the interactive report.

## 2. Would this drift likely affect model performance?

These large shifts create a plausible risk to prediction accuracy. Property size and neighborhood are relevant inputs to a sale-price model. Material shifts can move incoming properties away from the combinations represented during training, changing predictions and potentially increasing error.

However, **feature drift alone does not establish a loss of predictive accuracy**. The simulation does not provide new, trustworthy sale-price labels. It also changes individual distributions without reconstructing all realistic relationships between home characteristics. Therefore, this project does not report a production MAE or claim that the alert proves model degradation.

The final model's held-out test metrics describe the original historical dataset. They are separate from the drift demonstration.

## 3. What action is recommended?

**Investigate first.** Check whether the changed measurements represent a unit conversion or upstream transformation error, or a genuine change in the properties entering the system. Confirm geographic coverage and the neighborhood mapping as well.

If the change is a data-quality issue, correct the source or transformation and rerun monitoring. If it reflects a persistent population change, obtain representative labeled sales and evaluate the existing model on them. Retraining becomes appropriate when that evaluation shows unacceptable error and suitable updated data is available.

For the unchanged control, continue monitoring. In this repository, the intentional drift alert is a successful demonstration of the detection path; it does not initiate retraining.

## Limits and next steps

- The identical control is a functional check, not an estimate of the false-alert rate on independent incoming batches.
- Fourteen feature tests at an unadjusted 0.05 threshold need calibration before operational use. Batch size and normal seasonal variation affect sensitivity.
- Missingness is preserved in the simulation. Dedicated missing-value-rate checks would complement distribution monitoring in an operational workflow.
- A future deployment would need alert ownership, collection windows, labeled performance monitoring, and an approved response process. These are documented extensions, not implemented services.
