"""Local MLflow configuration and consistent run metadata."""

import subprocess

import mlflow

from src.config import ROOT, project_path


def configure_tracking(config: dict, experiment_name: str | None = None) -> str:
    database = project_path(config["tracking"]["database_path"]).resolve()
    artifacts = project_path(config["tracking"]["artifact_path"]).resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + database.as_posix())
    name = experiment_name or config["tracking"]["experiment_name"]
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def source_metadata() -> dict[str, str]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        return {"source_revision": revision, "source_tree_dirty": str(bool(dirty)).lower()}
    except (OSError, subprocess.CalledProcessError):
        return {"source_revision": "uncommitted", "source_tree_dirty": "true"}
