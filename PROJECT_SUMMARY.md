# Capstone implementation summary

The Ames Home-Price Lab adds a Streamlit interface and a local Qwen language model to the existing Ames MLOps project. Natural-language descriptions become validated, reviewable features, the selected Random Forest predicts a historical sale price, and Qwen explains that actual result.

The user selected local inference to avoid payment-card and API-credit requirements. The complete application therefore runs locally with Ollama. A hosted manual-only version is possible, but would not independently demonstrate the required LLM workflow.

## Model results

Five configurations were trained with training-fitted preprocessing. Selection used validation MAE before all candidates received held-out test evaluation. The selected 200-tree model produced test MAE $17,764.88, RMSE $30,245.14 and R² 0.8859 on 586 homes. Its exported pipeline reproduced the original artifact's predictions on all 586 test rows.

## Verification

Current evidence is linked from [CHECKLIST.md](CHECKLIST.md), [the experiment comparison](reports/experiment_comparison.csv), and [the live language evaluation](reports/interface_evaluation.json). Unit tests of transport use synthetic responses; live evaluation uses the real downloaded model. These are deliberately distinguished.

Docker packaging includes a non-root app image and optional local Ollama Compose stack. The earlier Windows startup issue is no longer present. On Docker Desktop 4.91.0 / Engine 29.8.0, the current image built successfully, became healthy on localhost:8502, and produced matching full-input and four-field predictions. The four-field result was also verified through the browser. This local verification used manual mode; the full Ollama Compose stack has not been run end to end. Source hashes and results are in reports/docker_verification.json.

## Scope

The revised form requires four core facts and accepts ten optional details. It previews the exact training-fitted defaults before confirmation and marks estimates that use them. On the same 586 validation homes, masking all ten optional fields raises MAE from $16,639 to $26,783; adding garage capacity and basement area reduces it to $19,861. This is a development comparison, not a new test benchmark. Four-only R² is 0.739, below the original full-input 0.75 gate, so this mode is presented as a rough estimate. The model and its original full-input test evidence are unchanged. See reports/optional_input_evaluation.json.

This is a historical educational model, not a valuation service. The app has no paid model fallback, ad targeting, CRM, trainer logic, current-price feed, or commercial deployment infrastructure.

The original MLOps write-up and evidence are retained in docs/original-mlops-summary.md and reports/original-mlops. Guided explanation checkpoints remain for the project author before submission. The independent employment-focused skills assessment is planned after the capstone, not marked complete here.
## Language-model evaluation note

Qwen3.5 4B initially missed or misquoted details in complete descriptions. Those development results are retained in reports/development/qwen3.5-evaluation.json. The selected Qwen3 4B is evaluated in reports/interface_evaluation.json, including its exact downloaded digest and Ollama version.

The final flow makes a short scope-classification call, extracts structured facts, and permits one bounded pass over missing fields. Python merges updates and rejects a number that does not occur in its quote. Previous numerical values are withheld from the extraction prompt so they cannot be copied into an explicitly corrected field. Any unanswered clarification blocks readiness. These are application controls, not a claim that every possible language mistake has been eliminated.
