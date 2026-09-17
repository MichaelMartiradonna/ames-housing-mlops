# Capstone implementation summary

The Ames Home-Price Lab adds a Streamlit interface and a local Qwen language model to the existing Ames MLOps project. Natural-language descriptions become validated, reviewable features, the selected Random Forest predicts a historical sale price, and Qwen explains that actual result.

The user selected local inference to avoid payment-card and API-credit requirements. The complete application therefore runs locally with Ollama. A hosted manual-only version is possible, but would not independently demonstrate the required LLM workflow.

## Model results

Five configurations were trained with training-fitted preprocessing. Selection used validation MAE before all candidates received held-out test evaluation. The selected 200-tree model produced test MAE $17,764.88, RMSE $30,245.14 and R² 0.8859 on 586 homes. Its exported pipeline reproduced the original artifact's predictions on all 586 test rows.

## Verification

Current evidence is linked from [CHECKLIST.md](CHECKLIST.md), [the experiment comparison](reports/experiment_comparison.csv), and [the live language evaluation](reports/interface_evaluation.json). Unit tests of transport use synthetic responses; live evaluation uses the real downloaded model. These are deliberately distinguished.

Docker packaging includes a non-root app image and optional local Ollama Compose stack. Docker Desktop on the reference Windows machine failed before loading this project because its internal sailor-ingest.sock could not be accessed. Container verification is therefore also performed on GitHub's Ubuntu runner. No Docker factory reset was performed.

## Scope

This is a historical educational model, not a valuation service. The app has no paid model fallback, ad targeting, CRM, trainer logic, current-price feed, or commercial deployment infrastructure.

The original MLOps write-up and evidence are retained in docs/original-mlops-summary.md and reports/original-mlops. Guided explanation checkpoints remain for the project author before submission. The independent employment-focused skills assessment is planned after the capstone, not marked complete here.
