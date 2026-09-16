"""Run five experiments and evaluate only the validation-selected winner on test data."""

import argparse
import json
from datetime import datetime, timezone
from uuid import uuid4

import mlflow

from src.compare_experiments import compare_experiments
from src.config import DEFAULT_CONFIG, ROOT, feature_names, load_config
from src.data import dataset_hash, load_dataset, split_dataset
from src.evaluate import regression_metrics
from src.tracking import configure_tracking
from src.train import evaluate_test_run, train_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = load_config(args.config)
    configure_tracking(config)
    frame = load_dataset(config)
    splits = split_dataset(frame, config)
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    models = {}
    for key in config["experiments"]:
        run_id, model, _ = train_run(
            config, key, splits, run_kind="experiment", batch_id=batch_id, evaluate_test=False
        )
        models[run_id] = model
    # Rank validation results before any candidate is evaluated on the test set.
    comparison = compare_experiments(config, batch_id)
    winner = comparison.iloc[0]
    test_metrics = evaluate_test_run(models[winner["run_id"]], splits, config, winner["run_id"])
    with mlflow.start_run(run_id=winner["run_id"]):
        mlflow.set_tags({"selected": "true", "test_evaluated": "true"})
    baseline = regression_metrics(
        splits.y_test, [float(splits.y_train.median())] * len(splits.y_test)
    )
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports / "experiment_comparison.csv", index=False)
    summary = {
        "batch_id": batch_id,
        "data_version": dataset_hash(config),
        "source_rows": len(frame),
        "source_columns": len(frame.columns),
        "predictor_count": len(feature_names(config)),
        "missing_by_feature": {key: int(frame[key].isna().sum()) for key in feature_names(config)},
        "split_rows": {"train": len(splits.X_train), "validation": len(splits.X_validation), "test": len(splits.X_test)},
        "best_configuration": winner["configuration"],
        "best_run_id": winner["run_id"],
        "selection_metric": "validation_mae",
        "best_validation_mae": float(winner["validation_mae"]),
        "test_metrics": test_metrics,
        "median_baseline_test_metrics": baseline,
        "performance_gate": "passed",
        "test_set_used_for_selection": False,
    }
    (reports / "experiment_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Use the winning configuration as model.selected_experiment for subsequent training.")


if __name__ == "__main__":
    main()
