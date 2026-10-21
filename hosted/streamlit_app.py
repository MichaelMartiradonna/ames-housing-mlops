"""Hosted demo entry point (Streamlit Community Cloud): manual entry, saved model, no language model.

Sets AMES_MANUAL_ONLY=1 before src.app is imported, so the language-model client is
never imported or called. The local app (app/streamlit_app.py) is unchanged.
Lives in hosted/ next to its own requirements.txt, which Streamlit Community Cloud picks up
first. The repository root is located by walking up to the folder that holds src/app.py.
"""
import os
import sys
from pathlib import Path

os.environ["AMES_MANUAL_ONLY"] = "1"
HERE = Path(__file__).resolve().parent
ROOT = next(path for path in (HERE, *HERE.parents) if (path / "src" / "app.py").is_file())
sys.path.insert(0, str(ROOT))
from src.app import main  # noqa: E402  (the environment must be set before this import)

main()
