# Capstone submission guide

Public repository: https://github.com/MichaelMartiradonna/ames-housing-mlops

Recorded demo: [Ames Home Price Lab Demo — 5:45](https://youtu.be/1mW6WnJn5k4)

The implementation, required documentation, and recorded demonstration are present. The video is hosted as an unlisted YouTube video and linked from the README. Submitting the public repository link through the course platform remains the final delivery step.

## Data and model quality

- **Dataset and target:** 2,930 Ames sales exceed the 500-row minimum. The prediction target is `SalePrice`; the source and rationale for this alternative dataset are documented in [README.md](../README.md#dataset-and-preprocessing).
- **Preprocessing:** [src/preprocessing.py](../src/preprocessing.py) provides training-fitted median/mode imputation, numeric standardization, and one-hot encoding without modifying the input dataframe. Scaling is included for the course requirement, although Random Forest does not need it.
- **Configurations and evaluation:** [configs/model.yaml](../configs/model.yaml) defines five distinct configurations, exceeding the three-configuration minimum. [The comparison](../reports/experiment_comparison.csv) reports MAE, RMSE, and R² on held-out test data for every candidate.
- **Selection and performance:** Validation MAE selects the winner before test evaluation. The selected model's test MAE is $17,764.88, RMSE $30,245.14, and R² 0.8859, meeting the configured gates. See [the experiment summary](../reports/experiment_summary.json).

## Experiment tracking

- [src/train.py](../src/train.py) logs effective model parameters, preprocessing settings, data version, split sizes, validation/test metrics, and the fitted pipeline to MLflow.
- [src/run_experiments.py](../src/run_experiments.py) creates the five distinct runs; [src/compare_experiments.py](../src/compare_experiments.py) uses `mlflow.search_runs()` to rank a complete comparable batch by validation MAE.
- The [README reproduction steps](../README.md#reproduce-training) restore the versioned data, recreate the runs, and launch the tracking UI. Local tracking databases and binary artifacts are excluded from Git.

## Language interface

- [src/llm.py](../src/llm.py) uses local Qwen via Ollama to extract features and explain the actual model prediction. Alternative providers/models are allowed by the supplied instructions; this setup needs no API key or payment card.
- [src/interface.py](../src/interface.py) checks evidence, units, categories, numeric limits, completeness, and follow-up updates. The Streamlit review step requires confirmation before prediction.
- [src/serving.py](../src/serving.py) loads the checksum-verified pipeline identified in [the release manifest](../configs/model_release.json). The price comes from this trained model.
- [The live evaluation report](../reports/interface_evaluation.json) records 16/16 passing development cases using the actual local language model, including incomplete, ambiguous, invalid, and out-of-scope input. These cases informed development and are not an independent language-accuracy benchmark.
- Four required facts and ten disclosed optional defaults support a shorter form. [The separate validation study](../reports/optional_input_evaluation.json) documents its accuracy tradeoff; full-input test metrics are not presented as measured partial-input accuracy.

## Tests and reproducibility

- [tests/test_preprocessing.py](../tests/test_preprocessing.py) has seven tests covering missing values, encoding, scaling, input immutability, and invalid inputs; the requirement is at least four.
- [tests/test_model.py](../tests/test_model.py) has the required two tests for prediction type/shape and minimum performance using the actual versioned data.
- [tests/test_interface.py](../tests/test_interface.py) exceeds the two-interface-test minimum and checks controlled parsing responses and edge cases. Additional application tests cover review and confirmation. Mocked replies test application logic; the separate live evaluation exercises real language extraction and explanation.
- The complete suite contains 61 passing cases. [The verification report](../reports/verification.json) records the implementation checks, and [the submission review](../reports/submission_review.json) records the latest requirements review and local test run.
- [GitHub Actions](../.github/workflows/mlops.yml) validates the data and tests, enforces training performance gates, and builds and smoke-tests the container with an actual released-model prediction.

## Documentation and packaging

- [README.md](../README.md) covers purpose and audience, setup, usage, architecture, results, limitations, and reflection. AI assistance is disclosed.
- [requirements.txt](../requirements.txt) and [app/requirements.txt](../app/requirements.txt) pin dependencies; [.env.example](../.env.example) documents configuration without credentials.
- Training reads YAML configuration. DVC pointer files are tracked; raw data, model binaries, actual environment files, and local artifacts are excluded from Git.
- **Docker bonus:** [Dockerfile](../Dockerfile), [the walkthrough](DOCKER_WALKTHROUGH.md), and [verification evidence](../reports/docker_verification.json) document a successful local build, app startup, and actual prediction. Docker was tested in manual mode; the complete natural-language workflow was tested natively. The full Ollama Compose stack remains unverified.
- **Deployment scope:** The full application runs locally, and CI runs on GitHub. There is no claim of a full cloud-hosted language application. The supplied detailed delivery requirements allow a Streamlit app and a live review demo; the project uses those options.

## Demo and submission

Watch the [recorded demonstration (5:45)](https://youtu.be/1mW6WnJn5k4) for the natural-language request, extracted values and defaults, actual prediction and generated explanation, and an out-of-scope Chicago query. The recording uses the native app with Ollama enabled; manual Docker mode alone does not demonstrate the required language workflow.

The [demo walkthrough](DEMO.md) and [recording script](RECORDING_SCRIPT.md) remain available for a live review or another recording. The walkthrough includes optional examples beyond the recorded demo.

1. Submit the public repository link above through the course platform.
2. If the platform provides a separate demo-link field, also paste https://youtu.be/1mW6WnJn5k4 there. Otherwise the reviewer can open it from the README.

Explanation quizzes, an extra independently written test, and the employment-focused skills assessment are optional learning activities, not additional course deliverables. No completion or independent-mastery claim is made for those activities.
