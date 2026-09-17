"""Train a configured pipeline, log its evidence, and enforce acceptance gates."""

import argparse
import json
import sys

import mlflow
import mlflow.sklearn
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.config import DEFAULT_CONFIG, ROOT, load_config
from src.data import DataSplits, dataset_hash, load_dataset, split_dataset
from src.evaluate import PerformanceGateError, enforce_performance, regression_metrics
from src.preprocessing import build_preprocessor
from src.tracking import configure_tracking, source_metadata


def build_model(config: dict, experiment_key: str | None = None) -> Pipeline:
    key = experiment_key or config["model"]["selected_experiment"]
    if key not in config["experiments"]:
        raise ValueError(f"Unknown experiment: {key}")
    preprocessing = build_preprocessor(
        config["data"]["numeric_features"], config["data"]["categorical_features"]
    )
    model = RandomForestRegressor(
        **config["experiments"][key],
        random_state=config["random_seed"],
        n_jobs=config["model"]["n_jobs"],
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def evaluate_test_run(model: Pipeline, splits: DataSplits, config: dict, run_id: str, *, enforce_gate: bool = True) -> dict:
    metrics = regression_metrics(splits.y_test, model.predict(splits.X_test))
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("test_evaluated", "true")
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        try:
            enforce_performance(metrics, config)
        except PerformanceGateError:
            mlflow.set_tag("performance_gate", "failed")
            if enforce_gate:
                raise
            return metrics
        mlflow.set_tag("performance_gate", "passed")
    return metrics


def train_run(
    config: dict,
    experiment_key: str,
    splits: DataSplits,
    *,
    run_kind: str = "training",
    batch_id: str = "standalone",
    evaluate_test: bool = True,
) -> tuple[str, Pipeline, dict]:
    version = dataset_hash(config)
    model = build_model(config, experiment_key)
    tags = {
        "run_kind": run_kind,
        "batch_id": batch_id,
        "data_version": version,
        "configuration": experiment_key,
        "test_evaluated": "false",
        **source_metadata(),
    }
    with mlflow.start_run(run_name=experiment_key, tags=tags) as run:
        mlflow.log_params({"model_type": config["model"]["type"], "data_version": version})
        mlflow.log_params(model.named_steps["model"].get_params())
        mlflow.log_params({
            "random_seed": config["random_seed"],
            "test_size": config["split"]["test_size"],
            "validation_size": config["split"]["validation_size"],
            "numeric_imputation": "median",
            "numeric_scaling": "standard_scaler_train_only",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "one_hot_ignore_unknown",
            "numeric_features": json.dumps(config["data"]["numeric_features"]),
            "categorical_features": json.dumps(config["data"]["categorical_features"]),
            "train_rows": len(splits.X_train),
            "validation_rows": len(splits.X_validation),
            "test_rows": len(splits.X_test),
        })
        mlflow.log_text(yaml.safe_dump(config, sort_keys=False), "config.yaml")
        model.fit(splits.X_train, splits.y_train)
        metrics = regression_metrics(splits.y_validation, model.predict(splits.X_validation))
        mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
        # Float inputs preserve missing-value support in MLflow's saved schema.
        sample = model.named_steps["preprocess"].named_steps["validate"].transform(splits.X_train.head(5))
        requirements = [
            line for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.startswith(("numpy==", "pandas==", "scikit-learn==", "mlflow==", "skops=="))
        ]
        # Trust only the types in this locally fitted pipeline. Skops 0.15 requires
        # explicit trust for Tree; this is not approval to load external model files.
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(sample, model.predict(sample)),
            input_example=sample,
            pip_requirements=requirements,
            code_paths=[str(ROOT / "src")],
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            skops_trusted_types=[
                "numpy.dtype", "src.preprocessing.FrameValidator", "sklearn.tree._tree.Tree"
            ],
        )
        run_id = run.info.run_id
    print(f"{experiment_key}: validation MAE ${metrics['mae']:,.2f}; RMSE ${metrics['rmse']:,.2f}; R² {metrics['r2']:.4f}")
    if evaluate_test:
        test_metrics = evaluate_test_run(model, splits, config, run_id)
        print("Test metrics: " + json.dumps(test_metrics, indent=2))
        output = ROOT / "artifacts" / "training_result.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "run_id": run_id, "configuration": experiment_key,
            "data_version": version, "validation": metrics, "test": test_metrics,
            "performance_gate": "passed",
        }, indent=2) + "\n", encoding="utf-8")
    return run_id, model, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment", help="Configuration key; defaults to the selected configuration.")
    parser.add_argument("--experiment-name", help="Separate MLflow experiment name, for example ames-housing-ci.")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_tracking(config, args.experiment_name)
    splits = split_dataset(load_dataset(config), config)
    try:
        train_run(config, args.experiment or config["model"]["selected_experiment"], splits)
    except PerformanceGateError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
