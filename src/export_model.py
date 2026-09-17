"""Export the actual MLflow winner, preserving its provenance and prediction parity."""

import json
from datetime import datetime, timezone

import mlflow.sklearn
import numpy as np
import skops.io as sio

from src.config import ROOT, load_config
from src.data import load_dataset, split_dataset
from src.serving import MODEL_DIR, TRUSTED_TYPES, sha256
from src.tracking import configure_tracking


def main():
    config = load_config()
    configure_tracking(config)
    summary = json.loads((ROOT / "reports" / "experiment_summary.json").read_text())
    run_id = summary["best_run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    splits = split_dataset(load_dataset(config), config)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / "model.skops"
    sio.dump(model, path)
    reloaded = sio.load(path, trusted=list(TRUSTED_TYPES))
    np.testing.assert_allclose(model.predict(splits.X_test), reloaded.predict(splits.X_test), rtol=0, atol=1e-8)
    metadata = {
        "version": "capstone-model-v1", "run_id": run_id,
        "configuration": summary["best_configuration"], "data_version": summary["data_version"],
        "sha256": sha256(path), "bytes": path.stat().st_size,
        "url": "https://github.com/MichaelMartiradonna/ames-housing-mlops/releases/download/capstone-model-v1/model.skops",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "categories": {key: sorted(splits.X_train[key].dropna().unique().tolist()) for key in config["data"]["categorical_features"]},
        "test_metrics": summary["test_metrics"], "test_rows": len(splits.X_test),
        "selection_metric": "validation_mae", "test_set_used_for_selection": False,
        "source_years": [2006, 2010], "export_parity_rows": len(splits.X_test),
    }
    (ROOT / "configs" / "model_release.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
