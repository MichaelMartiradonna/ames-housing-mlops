# Ames Home-Price Lab

A capstone application that turns an ordinary home description into a **reviewable, model-backed historical price estimate**. A local language model extracts the details; the selected Random Forest predicts the sale price; the language model explains that actual result.

This educational tool explores Ames, Iowa sales from **2006–2010**. It is not a current appraisal or an investment tool.

[![Validation](https://github.com/MichaelMartiradonna/ames-housing-mlops/actions/workflows/mlops.yml/badge.svg)](https://github.com/MichaelMartiradonna/ames-housing-mlops/actions/workflows/mlops.yml)

## Run the app

Use Python 3.12. Create and activate an environment:

Windows:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

macOS / Linux:

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
~~~

Install the smaller inference environment:

~~~bash
python -m pip install -r app/requirements.txt
~~~

Copy `.env.example` to `.env` (`Copy-Item .env.example .env` on Windows or `cp .env.example .env` on macOS/Linux). Install [Ollama](https://ollama.com/download), start it, and download the local model:

~~~bash
ollama pull qwen3.5:4b
~~~

On Windows, an optional project-contained setup is provided:

~~~powershell
python scripts/setup_local_llm.py
.\scripts\start_local_llm.ps1
.\.tools\ollama\ollama.exe pull qwen3.5:4b
~~~

The portable setup verifies the official Ollama 0.34.1 archive before extracting it. Downloads are approximately 1.5 GB for the runtime plus 3.4 GB for the model. It creates Ollama's usual local identity file in the user's `.ollama` directory; model weights stay in this project's ignored `.tools/models`. Do not start two servers on the same port.

Start the interface:

~~~bash
streamlit run app/streamlit_app.py
~~~

Open http://localhost:8501. On first use, the app downloads the **35.5 MB selected housing model** from its release, checks its SHA-256, and loads its complete preprocessing pipeline. Training data and MLflow are not needed to use that exported model.

Local inference needs **no API key, payment card, or paid provider**. The app permits only local Ollama addresses and has no paid fallback. The reference machine has a 6 GB NVIDIA GPU. CPU inference is supported but can be considerably slower; the first request also includes model loading. Settings are documented in `.env.example`; credentials are never committed.

To use the housing model without Ollama, set `AMES_LLM_ENABLED=false`. This is clearly labeled **manual mode** and does not fulfill the natural-language demo by itself.

## Try it

1. Choose **Use example prompt**, then **Read home details**.
2. Review the extracted fields. Supply missing details in a follow-up or in the form.
3. Check the confirmation box and choose **Confirm & estimate**.
4. Read the trained model's price and the local language model's explanation.
5. Try “Actually, it has a 3-car garage.” Review and confirm the changed input.
6. Try “Estimate the current value of my Chicago condo.” The language layer should explain the scope instead of producing a new estimate.

All 14 fields are required: the app never silently assigns an unknown home's quality, neighborhood, size, or amenities. Quality is an explicit 1–10 material/finish rating; “nice” is not a measured rating. Bedrooms and full bathrooms refer to those above ground.

A prepared success-and-edge-case walkthrough is in [docs/DEMO.md](docs/DEMO.md).

## Architecture

~~~mermaid
flowchart LR
    A[Home description] --> B[Local Qwen via Ollama]
    B --> C[Structured extraction and evidence]
    C --> D[Units, ranges, categories, completeness]
    D --> E[User reviews and confirms]
    E --> F[Saved preprocessing + Random Forest]
    F --> G[Actual historical price estimate]
    G --> H[Local language explanation]
    H --> I[Streamlit result and limitations]
~~~

- `src/interface.py`: feature contract, unit conversion, validation, conversation updates.
- `src/llm.py`: local requests, schema output, validation, timeouts, safe errors.
- `src/serving.py`: checksum-verified model retrieval and inference.
- `src/app.py`: Streamlit UI; `app/streamlit_app.py` is the deployment entry point.
- `src/train.py`, `src/run_experiments.py`, `src/compare_experiments.py`: training, tracking and model selection.
- `src/export_model.py`: exports the actual MLflow winner and checks parity on all test rows.
- `src/evaluate_interface.py`: live language evaluation, separate from offline tests.

Code rejects unsupported features, nonfinite values, unknown categories, out-of-range measurements and fractional room counts. Units must occur in the supporting quote; conversions happen in Python. Quotes accept only whitespace and thousands-separator differences. Duplicate updates block prediction and remove stale values. Missing or ambiguous inputs require clarification.

These checks cannot prove semantic correctness of every extraction. The user therefore confirms the full form. Language failures never substitute a fabricated price. Explanations must echo the actual rounded estimate, and the UI always labels it as historical.

## Dataset and preprocessing

The [Ames Housing dataset](https://jse.amstat.org/v19n3/decock/AmesHousing.txt), described by [Dean De Cock (2011)](https://jse.amstat.org/v19n3/decock.pdf), has **2,930 sales, 82 source columns, and the `SalePrice` target**. Ames is an alternative to the course's suggested datasets; its documented target, mixed features and missing data make it suitable for this regression project.

Fourteen predictors are used:

- Numeric: lot area, lot frontage, above-ground living area, quality, year built, above-ground full bathrooms, garage capacity, basement area, and above-ground bedrooms.
- Categorical: neighborhood, house style, building type, zoning, central air.

Identifiers and target values are excluded. Lot frontage has 490 missing values; garage capacity and basement area each have one. The fixed split is **1,758 train / 586 validation / 586 test**. All learned transformations fit exclusively on training data: numeric median imputation and standardization; categorical mode imputation and one-hot encoding. Inputs are copied rather than modified, and unseen categories retain the fitted output shape.

Random Forest does not require standardization. It is included to satisfy and test the course's scaling requirement, not claimed as an accuracy improvement. The fitted transformations and regressor are saved together.

## Reproduce training

Install the complete training/test environment and restore the DVC data:

~~~bash
python -m pip install -r requirements.txt
python -m pip check
python -m src.data_storage bootstrap
python -m dvc pull
python -m src.run_experiments
python -m src.compare_experiments
python -m src.export_model
~~~

Bootstrap restores the published, checksum-verified local DVC remote. A new checkout needs it **before** `dvc pull`; no private storage account is required. Raw data and model artifacts are excluded from Git. Features, split settings, hyperparameters and acceptance gates live in [configs/model.yaml](configs/model.yaml).

Five Random Forest configurations vary tree count, maximum depth and minimum leaf size. Each logs effective parameters, preprocessing settings, data hash, split sizes, source revision, complete model, validation metrics and three test metrics to MLflow.

**Selection uses validation MAE before any test metrics are calculated.** All frozen candidates are then evaluated on the test set to meet the rubric. The winner is not reselected from test results. MAE/RMSE are in dollars; R² is not a confidence percentage. Acceptance requires MAE ≤ $35,000 and R² ≥ 0.75.

View the runs:

~~~bash
python -m mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5000
~~~

Open http://localhost:5000 and select `ames-housing`. The comparison script uses `mlflow.search_runs()`, rejects incomplete batches, and ranks comparable runs for the current data version. The local database and binaries are intentionally excluded from Git; rerunning the five experiments recreates them.

## Results

The validation-selected model uses **200 trees, unrestricted depth, minimum leaf size 1**:

- Validation MAE: **$16,639**
- Test MAE: **$17,765**
- Test RMSE: **$30,245**
- Test R²: **0.886**
- Training-median baseline test MAE: **$63,823**

Test MAE is about 72% lower than that simple baseline. Some other configurations have better test results; retaining the validation-selected winner avoids choosing based on the test set.

All five configurations and their test metrics are in [reports/experiment_comparison.csv](reports/experiment_comparison.csv). [reports/experiment_summary.json](reports/experiment_summary.json) records data/split evidence; [configs/model_release.json](configs/model_release.json) identifies the selected run, checksum and training categories.

[reports/interface_evaluation.json](reports/interface_evaluation.json) records actual local-model evaluation. The small development set covers complete descriptions, unit conversion, follow-ups, missing data, ambiguity, contradictions, invalid inputs and scope. It is **not a general accuracy benchmark**: prompts and validation were improved using these cases.

## Testing and CI

~~~bash
pytest tests/ -v
python -m src.evaluate_interface
~~~

The first command runs deterministic tests for preprocessing, actual model shape/performance, artifact reload, validation, HTTP failures, explanation consistency, and UI confirmation/reset. Transport responses are mocked in unit tests; those tests do not measure language accuracy.

The second requires the real local model and evaluates actual extraction, housing inference and explanations. It is separate because hardware availability and latency should not make ordinary CI unreliable. No paid calls are made.

GitHub Actions runs the tests, gated training, and a container build with health and actual released-model prediction checks. See the [workflow](.github/workflows/mlops.yml) and [verification checklist](CHECKLIST.md) for current status.

## Docker

Build and launch manual mode:

~~~bash
docker build -t ames-capstone .
docker run --rm -p 127.0.0.1:8501:8501 -e AMES_LLM_ENABLED=false ames-capstone
~~~

For the full local-language workflow:

~~~bash
docker compose up --build
~~~

Compose downloads Qwen into a named volume, keeps Ollama on the internal container network, and exposes the app only on localhost. The default uses CPU inference and may be slow; provide adequate RAM and disk space. Native Ollama can use supported GPUs without GPU container configuration. Stop with `docker compose down`; model volumes remain.

The app image runs as a non-root user, uses inference-only dependencies, and excludes secrets, raw data, MLflow artifacts and local LLM weights. It retrieves and verifies the selected housing model at runtime.

## Deployment and limitations

The complete demonstration runs locally. Streamlit Community Cloud can host the **manual housing-model demo** using entry point `app/streamlit_app.py`, Python 3.12, `AMES_HOSTED=true`, and `AMES_LLM_ENABLED=false`. A cloud app cannot reach a reviewer's own localhost. Do not expose an unauthenticated local Ollama server to solve that problem. Full hosted language features require a separately provisioned model service and are outside this no-payment local setup.

Other limits:

- Historical Ames sales do not establish present prices or generalize to other locations.
- A random split measures within-dataset performance, not future-market performance.
- Fourteen features omit condition details, renovations and other factors.
- Average test error is not a per-home uncertainty interval.
- Range checks do not establish that every permitted combination occurred in training.
- Small local models can misread details or scope. Form review remains necessary.
- This educational app is not a hardened multi-user service.

## Engineering reflection

The main integration challenge was the boundary between language interpretation and numerical prediction. Structured output alone did not prevent missing fields or invented units. Explicit checks, quoted evidence and a review step made those failures visible.

A local model trades hosted convenience for no recurring API cost and local processing. Unit tests make application behavior repeatable; live evaluation exposes mistakes that mocked responses cannot reveal. Selecting on validation before reporting every test result preserves a defensible evaluation procedure.

Useful next improvements are a larger independently written language evaluation set, easier feature collection, calibrated prediction intervals and temporal/generalization evaluation. More interface features would not resolve those measurement limitations.

Development used AI coding assistance. The [guided checkpoints](docs/LEARNING_CHECKPOINTS.md) describe what the author should explain and reproduce independently before submission; they do not claim an independent assessment has already occurred.

## Prior work

This extends an earlier Ames MLOps assignment. DVC integrity checks, MLflow, performance gates, CI and Evidently monitoring were retained. New work adds scaling verification, test metrics for all configurations, the local LLM interface, live evaluation, Streamlit tests, serving/export and Docker.

Earlier evidence is preserved under [reports/original-mlops](reports/original-mlops) and the [original write-up](docs/original-mlops-readme.md). Its old counts, metrics, screenshots and workflow URLs describe that version, not verification of the capstone interface.
