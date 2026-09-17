"""Preprocessing behavior on explicit examples independent of the training file."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.preprocessing import build_preprocessor


def sample():
    return pd.DataFrame({"area": [100.0, 200.0, np.nan], "style": ["ranch", "ranch", "duplex"]})


def test_numeric_missing_values_use_training_median():
    processor = build_preprocessor(["area"], ["style"])
    transformed = processor.fit_transform(sample())
    numeric = processor.named_steps["columns"].named_transformers_["numeric"]
    assert numeric.named_steps["impute"].statistics_[0] == 150.0
    assert transformed[2, 0] == 0.0
    # An inference outlier must not change the training-time imputation value.
    future = pd.DataFrame({"area": [np.nan, 10000.0], "style": ["ranch", "ranch"]})
    assert processor.transform(future)[0, 0] == 0.0


def test_scaling_uses_only_training_statistics():
    frame = pd.DataFrame({"area": [100.0, 200.0, 300.0], "style": ["ranch"] * 3})
    processor = build_preprocessor(["area"], ["style"])
    values = processor.fit_transform(frame)[:, 0]
    assert values.mean() == pytest.approx(0.0)
    assert values.std() == pytest.approx(1.0)
    future = pd.DataFrame({"area": [400.0], "style": ["ranch"]})
    expected = (400.0 - 200.0) / np.std([100.0, 200.0, 300.0])
    assert processor.transform(future)[0, 0] == pytest.approx(expected)


def test_categorical_missing_values_use_training_mode():
    frame = sample()
    frame.loc[2, "style"] = np.nan
    transformed = build_preprocessor(["area"], ["style"]).fit_transform(frame)
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(transformed[2, 1:], transformed[0, 1:])


def test_categories_receive_distinct_one_hot_vectors():
    transformed = build_preprocessor(["area"], ["style"]).fit_transform(sample())
    np.testing.assert_array_equal(transformed[:, 1:].sum(axis=1), [1, 1, 1])
    np.testing.assert_array_equal(transformed[0, 1:], transformed[1, 1:])
    assert not np.array_equal(transformed[0, 1:], transformed[2, 1:])


def test_unseen_category_keeps_feature_shape():
    processor = build_preprocessor(["area"], ["style"])
    trained = processor.fit_transform(sample())
    future = processor.transform(pd.DataFrame({"area": [180.0], "style": ["townhouse"]}))
    assert future.shape == (1, trained.shape[1])
    np.testing.assert_array_equal(future[0, 1:], 0)


def test_preprocessing_does_not_modify_original_dataframe():
    frame = sample()
    original = frame.copy(deep=True)
    processor = build_preprocessor(["area"], ["style"])
    processor.fit_transform(frame)
    processor.transform(frame)
    assert_frame_equal(frame, original)


def test_invalid_input_raises_clear_errors():
    processor = build_preprocessor(["area"], ["style"])
    with pytest.raises(TypeError, match="DataFrame"):
        processor.fit_transform([[100, "ranch"]])
    with pytest.raises(ValueError, match="Missing required feature"):
        processor.fit_transform(pd.DataFrame({"area": [100]}))
    with pytest.raises(ValueError, match="numbers"):
        processor.fit_transform(pd.DataFrame({"area": ["invalid"], "style": ["ranch"]}))
    with pytest.raises(ValueError, match="infinite"):
        processor.fit_transform(pd.DataFrame({"area": [np.inf], "style": ["ranch"]}))
