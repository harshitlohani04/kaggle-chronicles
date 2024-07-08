from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# This code contains the custom transformers that have been used in the main pipeline of pre-processing
class custom_LabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.encoded_cols = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
      # We have to convert the type of the dataset to dataframe because sklearn internally changes the type of the dataset to numpy
        X_new = X.copy()
        X_new = pd.DataFrame(X_new)
        for col in X_new.select_dtypes(include = object).columns:
            X_new[col] = self.label_encoder.fit_transform(X_new[col])
        return X_new