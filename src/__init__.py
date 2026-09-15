"""Reproducible training and monitoring for historical Ames home sales."""

import os

# Keep command-line output focused on model development and validation.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
