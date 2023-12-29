# transformer.py

import numpy as np
import pandas as pd
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin


class EncodeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No training needed, just return self
        return self

    def transform(self, X):
        # This transformer modify the data using factorize
        for column in X.columns[~X.columns.isin(['rate', 'cost_for_two', 'votes'])]:
            X[column] = X[column].factorize()[0]
        return X


class ClipTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        # The clip transformer doesn't require any training, so fit is just a pass-through
        return self

    def transform(self, X):
        # Clip the values in the DataFrame or Series using the upper bounds
        return X.clip(lower=None, upper=self.upper_bounds_, axis=1)


class PositiveTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No training needed, just return self
        return self

    def transform(self, X):
        # Ensure data is strictly positive
        return X + np.abs(X.min()) + 1  # Add 1 to avoid zero values


class OutlierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier

    def fit(self, X, y=None):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_bound = Q1 - self.multiplier * IQR
        self.upper_bound = Q3 + self.multiplier * IQR
        return self

    def transform(self, X):
        # Replace values outside the bounds with the bounds
        X_transformed = np.where(X < self.lower_bound, self.lower_bound, X)
        X_transformed = np.where(X_transformed > self.upper_bound, self.upper_bound, X_transformed)
        return X_transformed
