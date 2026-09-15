"""Query MLflow and rank one comparable experiment batch by validation MAE."""

import argparse

import mlflow

from src.config import DEFAULT_CONFIG, ROOT, load_config
from src.data import dataset_hash
from src.tracking import configure_tracking


def compare_experiments(config: dict, batch_id: str | None = None):
    experiment_id = configure_tracking(config)
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' AND tags.run_kind = 'experiment' "
            f"AND tags.data_version = '{dataset_hash(config)}'"
        ),
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        raise ValueError("No completed experiment batch found. Run python -m src.run_experiments first.")
    batch_id = batch_id or runs.iloc[0]["tags.batch_id"]
    runs = runs[runs["tags.batch_id"] == batch_id].copy()
    expected = set(config["experiments"])
    if set(runs["tags.configuration"]) != expected or len(runs) != len(expected):
        raise ValueError("The batch must contain exactly one completed run per configured experiment.")
    columns = {
        "run_id": "run_id", "tags.batch_id": "batch_id", "tags.configuration": "configuration",
        "params.n_estimators": "n_estimators", "params.max_depth": "max_depth",
        "params.min_samples_leaf": "min_samples_leaf",
        "metrics.validation_mae": "validation_mae",
        "metrics.validation_rmse": "validation_rmse",
        "metrics.validation_r2": "validation_r2",
    }
    result = runs[list(columns)].rename(columns=columns).sort_values(
        ["validation_mae", "configuration"], kind="stable"
    ).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--batch", help="Defaults to the latest batch for the current DVC dataset.")
    args = parser.parse_args()
    result = compare_experiments(load_config(args.config), args.batch)
    output = ROOT / "reports" / "experiment_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Best configuration: {result.iloc[0]['configuration']}")
    print(f"Saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
