# Hosted demo: deploy notes (PLAN.md step 3)

The hosted demo is the same app in manual-only mode: `AMES_MANUAL_ONLY=1` is set by
`hosted/streamlit_app.py` before `src.app` is imported. In that mode the app never imports
`src/llm.py` (the Ollama client), never reads `.env`, never calls the language model for
scope classification, extraction or explanation, and shows the manual form directly with a
banner linking to the README and the 5:45 demo video. With the variable unset, the app is
unchanged.

## Why the files live in hosted/

Community Cloud has no "requirements file" setting. Per its documentation it searches the
entrypoint file's directory first, then the repository root, and the only pip file name it
recognizes is `requirements.txt`. So the entrypoint and its lightweight requirements sit
together in `hosted/`:

- `hosted/streamlit_app.py`: sets the environment variable, finds the repository root by
  walking up to the folder that contains `src/app.py`, then runs `src.app.main()`.
- `hosted/requirements.txt`: picked up automatically because it is next to the entrypoint; the
  root `requirements.txt` (MLflow, DVC, Evidently) is never installed for the demo.

`.streamlit/config.toml` stays at the repository root: Community Cloud reads it from there
regardless of where the entrypoint is, so no copy in `hosted/` is needed.

## share.streamlit.io settings

| Setting | Value |
| --- | --- |
| Repository | `MichaelMartiradonna/ames-housing-mlops` |
| Branch | `hosted-demo` |
| Main file path | `hosted/streamlit_app.py` |
| App URL | your choice, for example `ames-home-price-lab` |
| Advanced settings → Python version | `3.12` (the version the pins are tested with: Dockerfile, CI) |
| Requirements file | no setting exists; `hosted/requirements.txt` is detected automatically |
| Advanced settings → Secrets | none |
| Environment variables | none; the entrypoint sets `AMES_MANUAL_ONLY=1` itself |

`.streamlit/config.toml` at the repository root (theme, `gatherUsageStats = false`,
`server.headless = true`) is read by Community Cloud as is.

Dependencies installed: `hosted/requirements.txt` = `app/requirements.txt` minus
`python-dotenv`: numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, skops 0.15.0, PyYAML 6.0.3,
streamlit 1.55.0, httpx 0.28.1, pydantic 2.13.5. scikit-learn stays at 1.7.2 because the
released `model.skops` embeds `_sklearn_version = "1.7.2"` for every fitted estimator (also the
pin in `requirements.txt` and `app/requirements.txt`, and what `src/train.py` logged to MLflow).
Note that PLAN.md lists `joblib` and `requests`; the code uses `skops` for the model file and
`httpx` for the download, so those are the pins.

## How the saved model gets there

`src/serving.py::ensure_model` runs on first use (inside `st.cache_resource`, so once per
server process):

1. Reads `configs/model_release.json` (committed). It holds the release URL, the SHA-256 and
   byte size of `model.skops`, the category lists, the test metrics and the MLflow run id.
2. If `artifacts/serving/model.skops` is missing or its SHA-256 does not match, it streams
   `https://github.com/MichaelMartiradonna/ames-housing-mlops/releases/download/capstone-model-v1/model.skops`
   (6,214,958 bytes) with httpx, following redirects, 120 s timeout, 100 MB cap, into a
   temporary file, verifies the checksum, then atomically renames it into place. The URL must
   start with the project's GitHub releases prefix, otherwise it refuses.
3. `load_serving_model` checks the skops file's untrusted types against the allow-list
   (`numpy.dtype`, `src.preprocessing.FrameValidator`, `sklearn.tree._tree.Tree`) and loads it.
   Training defaults for the ten optional fields are read from the fitted imputers.

Verified in this session from a clean clone with no DVC and no `artifacts/` folder: the
download completed in about 4 s and matched the manifest checksum. `artifacts/` is gitignored,
so each new Community Cloud container downloads the file once at first request; the
README's worked example ($219,776 for the College Creek home) is computed by this file.

## What must be present in the repository for the hosted app to boot

- `src/` (`app.py`, `config.py`, `interface.py`, `preprocessing.py`, `serving.py`, `workflow.py`
  are imported; `llm.py`, `train.py`, `tracking.py`, `export_model.py`, `monitor_drift.py`,
  `data*.py` are not).
- `configs/model.yaml` (feature lists and numeric ranges) and `configs/model_release.json`
  (release URL and checksum).
- `hosted/streamlit_app.py` and `hosted/requirements.txt`, and optionally the root
  `.streamlit/config.toml`.

Not needed: `data/`, `reports/`, `artifacts/`, `docs/`, DVC, MLflow, Ollama, any secret.

## After deploying (PLAN.md step 3, second bullet)

Replace `HOSTED_URL_TBD` in the README's "Try it live" link with the app URL, and check the
review items: cold load under 60 s, and the worked example returns $219,776 when all fourteen
details are entered in the form.

## Tests

- `tests/test_hosted.py::test_manual_only_mode_renders_form_without_ollama`: AppTest run of the
  hosted entrypoint with every language-model touch point patched to raise; asserts the banner,
  the four required and ten optional fields, the default previews, and that no description
  step or parse button is rendered.
- `tests/test_hosted.py::test_manual_only_mode_estimates_without_explanation`: form estimate
  yields `explanation_status == "manual"` with no explanation call.
- `tests/test_hosted.py::test_hosted_mode_import_boundary`: a fresh interpreter importing
  `src.app` with `AMES_MANUAL_ONLY=1` loads none of `ollama`, `mlflow`, `dvc`, `evidently`,
  `dotenv`, `src.llm`, `src.train`, `src.tracking`, `src.export_model`, and
  `hosted/requirements.txt` pins exactly the eight packages manual mode imports.
