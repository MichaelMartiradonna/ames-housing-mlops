"""Verify the logged model, including its input contract, after reloading."""

from copy import deepcopy
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np

from src.tracking import configure_tracking
from src.train import train_run


def test_logged_pipeline_preserves_predictions_with_missing_and_unseen_inputs(config, splits, tmp_path):
    isolated = deepcopy(config)
    isolated["tracking"]["database_path"] = str(tmp_path / "mlflow.db")
    isolated["tracking"]["artifact_path"] = str(tmp_path / "models")
    subset = splits.X_train.sample(n=1000, random_state=config["random_seed"])
    smaller = replace(splits, X_train=subset, y_train=splits.y_train.loc[subset.index])
    previous_uri = mlflow.get_tracking_uri()
    try:
        configure_tracking(isolated, "artifact-contract-test")
        run_id, native, _ = train_run(
            isolated, config["model"]["selected_experiment"], smaller, evaluate_test=False
        )
        loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        artifact_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model"))
        # Recreating the saved model's environment must use the tested serializer.
        requirements = (artifact_dir / "requirements.txt").read_text(encoding="utf-8")
        assert f"skops=={version('skops')}" in requirements.splitlines()
        future = splits.X_test.head(3).copy()
        future.loc[future.index[0], "Full Bath"] = np.nan
        future.loc[future.index[0], "Neighborhood"] = "Previously unseen neighborhood"
        np.testing.assert_allclose(loaded.predict(future), native.predict(future))
        run = mlflow.get_run(run_id)
        assert {"validation_mae", "validation_rmse", "validation_r2"} <= set(run.data.metrics)
        assert "data_version" in run.data.params
    finally:
        mlflow.set_tracking_uri(previous_uri)
