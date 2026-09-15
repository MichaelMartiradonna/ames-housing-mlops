"""Validate a model trained on a fixed 1,000-row training subset."""

import numpy as np

from src.evaluate import enforce_performance, regression_metrics


def test_model_predictions_have_expected_type_and_shape(small_model, splits):
    predictions = small_model.predict(splits.X_test)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(splits.X_test),)
    assert np.issubdtype(predictions.dtype, np.floating)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


def test_small_model_meets_performance_threshold(small_model, splits, config):
    assert set(splits.X_train.index).isdisjoint(splits.X_test.index)
    assert set(splits.X_train.index).isdisjoint(splits.X_validation.index)
    assert set(splits.X_validation.index).isdisjoint(splits.X_test.index)
    metrics = regression_metrics(splits.y_test, small_model.predict(splits.X_test))
    enforce_performance(metrics, config)
