"""Input validation and train-fitted transformations for tabular features."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


class FrameValidator(TransformerMixin, BaseEstimator):
    """Select named predictors and normalize missing values on a copy."""

    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def _validate(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame with named columns.")
        columns = list(self.numeric_features) + list(self.categorical_features)
        if X.empty or X.columns.duplicated().any():
            raise ValueError("Input must be nonempty and have unique column names.")
        missing = sorted(set(columns) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        result = X.loc[:, columns].copy(deep=True)
        for column in self.numeric_features:
            try:
                result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{column} must contain numbers or missing values.") from exc
            if np.isinf(result[column].to_numpy()).any():
                raise ValueError(f"{column} cannot contain infinite values.")
        for column in self.categorical_features:
            values = result[column].astype(object)
            result[column] = values.where(values.notna(), np.nan)
        return result

    def fit(self, X, y=None):
        result = self._validate(X)
        self.feature_names_in_ = np.asarray(result.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self._validate(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.feature_names_in_.copy()


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Keep learned imputers and encoding inside the model to prevent data leakage."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        # New categories receive zeros instead of changing the fitted feature layout.
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return Pipeline([
        ("validate", FrameValidator(numeric_features, categorical_features)),
        ("columns", ColumnTransformer([
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ], remainder="drop")),
    ])
