"""Regression metrics and explicit acceptance gates."""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


class PerformanceGateError(ValueError):
    """A trained model does not satisfy the configured acceptance criteria."""


def regression_metrics(actual, predicted) -> dict[str, float]:
    predicted = np.asarray(predicted)
    if predicted.ndim != 1 or len(predicted) != len(actual) or not np.isfinite(predicted).all():
        raise ValueError("Predictions must be finite and have one value per observation.")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def enforce_performance(metrics: dict[str, float], config: dict) -> None:
    if not all(np.isfinite(value) for value in metrics.values()):
        raise PerformanceGateError("Model metrics must be finite.")
    failures = []
    if metrics["mae"] > config["evaluation"]["max_mae"]:
        failures.append(f"MAE ${metrics['mae']:,.2f} exceeds ${config['evaluation']['max_mae']:,.2f}")
    if metrics["r2"] < config["evaluation"]["min_r2"]:
        failures.append(f"R² {metrics['r2']:.4f} is below {config['evaluation']['min_r2']}")
    if failures:
        raise PerformanceGateError("Performance gate failed: " + "; ".join(failures))
