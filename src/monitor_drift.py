"""Compare training features with a controlled simulation using Evidently."""

import argparse
import json
import sys

import numpy as np
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.config import DEFAULT_CONFIG, feature_names, load_config, project_path
from src.data import dataset_hash, load_dataset, split_dataset


def simulate_production(reference, config: dict, scenario: str):
    current = reference.copy(deep=True)
    if scenario == "control":
        return current
    if scenario != "drifted":
        raise ValueError("Scenario must be control or drifted.")
    settings = config["monitoring"]
    for column, multiplier in settings["numeric_multipliers"].items():
        current[column] = current[column].astype(float) * multiplier
    rng = np.random.default_rng(config["random_seed"])
    count = int(len(current) * settings["neighborhood_fraction"])
    indices = rng.choice(current.index.to_numpy(), size=count, replace=False)
    current.loc[indices, "Neighborhood"] = settings["neighborhood_value"]
    return current


def run_monitor(config: dict, scenario: str, *, write_reports: bool = True) -> dict:
    settings = config["monitoring"]
    threshold = settings["drift_share_threshold"]
    if not 0 <= threshold <= 1:
        raise ValueError("Drift-share threshold must lie between 0 and 1.")
    if settings["numeric_method"] != "ks" or settings["categorical_method"] != "chisquare":
        raise ValueError("This monitor summarizes p-values from ks and chisquare methods.")
    reference = split_dataset(load_dataset(config), config).X_train
    current = simulate_production(reference, config, scenario)
    definition = DataDefinition(
        numerical_columns=config["data"]["numeric_features"],
        categorical_columns=config["data"]["categorical_features"],
    )
    # Explicit methods keep scores interpretable as p-values for every batch size.
    report = Report([
        DataDriftPreset(
            columns=feature_names(config),
            drift_share=threshold,
            num_method=settings["numeric_method"],
            cat_method=settings["categorical_method"],
            num_threshold=settings["feature_threshold"],
            cat_threshold=settings["feature_threshold"],
            include_tests=False,
        )
    ], include_tests=False)
    snapshot = report.run(
        current_data=Dataset.from_pandas(current, data_definition=definition),
        reference_data=Dataset.from_pandas(reference, data_definition=definition),
    )
    columns = {}
    for metric in snapshot.dict()["metrics"]:
        if metric["metric_name"].startswith("ValueDrift("):
            column = metric["config"]["column"]
            p_value = float(metric["value"])
            if not np.isfinite(p_value):
                raise ValueError(f"Drift detection produced an invalid p-value for {column}.")
            columns[column] = {
                "method": settings["numeric_method"] if column in config["data"]["numeric_features"] else settings["categorical_method"],
                "p_value": p_value,
                "drifted": p_value < settings["feature_threshold"],
            }
    if set(columns) != set(feature_names(config)):
        raise ValueError("Evidently did not return a drift result for every configured predictor.")
    drifted = [column for column in feature_names(config) if columns[column]["drifted"]]
    share = len(drifted) / len(columns)
    summary = {
        "scenario": scenario,
        "data_version": dataset_hash(config),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "feature_count": len(columns),
        "feature_p_value_threshold": settings["feature_threshold"],
        "drifted_features": drifted,
        "drift_share": share,
        "drift_share_threshold": threshold,
        # The project requires an alert only when the share exceeds the threshold.
        "alert": share > threshold,
        "columns": columns,
    }
    if write_reports:
        directory = project_path(settings["reports_path"])
        directory.mkdir(parents=True, exist_ok=True)
        html_path = directory / f"drift_{scenario}.html"
        snapshot.save_html(str(html_path))
        # Normalize whitespace-only template lines for clean version-control diffs.
        html_lines = html_path.read_text(encoding="utf-8").splitlines()
        html_path.write_text("\n".join(line if line.strip() else "" for line in html_lines) + "\n", encoding="utf-8")
        (directory / f"drift_{scenario}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scenario", choices=["control", "drifted"], default="drifted")
    parser.add_argument("--threshold", type=float, help="Override the configured maximum drift share.")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.threshold is not None:
            config["monitoring"]["drift_share_threshold"] = args.threshold
        result = run_monitor(config, args.scenario)
    except Exception as exc:
        print(f"Monitoring failed: {exc}", file=sys.stderr)
        return 2
    print(f"Scenario: {result['scenario']}; reference/current rows: {result['reference_rows']}/{result['current_rows']}")
    for name in feature_names(config):
        column = result["columns"][name]
        print(f"{name}: {'DRIFT' if column['drifted'] else 'stable'} (p={column['p_value']:.6g}; {column['method']})")
    print(f"Drift share: {result['drift_share']:.2%}; alert threshold: > {result['drift_share_threshold']:.0%}")
    print(f"HTML and JSON saved to {config['monitoring']['reports_path']}/drift_{args.scenario}.*")
    return 1 if result["alert"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
