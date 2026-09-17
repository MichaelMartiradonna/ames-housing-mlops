"""Compare missing-input policies on validation homes without retraining or test tuning."""

import json
from datetime import datetime, timezone

import numpy as np

from src.config import ROOT, load_config
from src.data import dataset_hash, load_dataset, split_dataset
from src.evaluate import regression_metrics
from src.interface import REQUIRED_FEATURES
from src.serving import load_serving_model, training_defaults


def main():
    config = load_config()
    splits = split_dataset(load_dataset(config), config)
    model, metadata = load_serving_model()
    core = list(REQUIRED_FEATURES)
    policies = {
        "all_available": list(splits.X_validation.columns),
        "required_four": core,
        "four_plus_garage": core + ["Garage Cars"],
        "four_plus_basement": core + ["Total Bsmt SF"],
        "four_plus_garage_and_basement": core + ["Garage Cars", "Total Bsmt SF"],
        "three_without_quality": [key for key in core if key != "Overall Qual"],
    }
    results = []
    for name, supplied in policies.items():
        masked = splits.X_validation.copy(deep=True)
        missing = [key for key in masked if key not in supplied]
        for key in missing:
            masked[key] = np.nan
        metrics = regression_metrics(splits.y_validation, model.predict(masked))
        results.append({"policy": name, "supplied_features": supplied,
                        "masked_features": missing, "validation_metrics": metrics})
        print(f"{name}: MAE ${metrics['mae']:,.0f}; RMSE ${metrics['rmse']:,.0f}; R2 {metrics['r2']:.3f}")
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_run_id": metadata["run_id"], "model_sha256": metadata["sha256"],
        "data_version": dataset_hash(config), "split": "validation", "rows": len(splits.X_validation),
        "random_seed": config["random_seed"], "retrained": False, "test_set_evaluated": False,
        "method": "Mask entire optional columns on the same validation homes; use unchanged training-fitted imputers and model.",
        "selected_policy": "required_four",
        "decision": "Four core facts enable a clearly labeled rough estimate. Recommend garage and basement as optional refinements. Preserve the trained model and its original full-input performance gates.",
        "limitations": "Development comparison used to choose the input policy, not an independent test benchmark or per-home uncertainty interval. Simulated missingness and correct core values do not establish real-user accuracy. More supplied fields need not improve every prediction. Four-only validation R2 is below the original full-input 0.75 gate; the partial-input mode is not claimed to pass that gate.",
        "training_defaults": training_defaults(model), "results": results,
    }
    path = ROOT / "reports" / "optional_input_evaluation.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
