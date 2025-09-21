from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class PerFoldScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        return self.scaler.fit_transform(X_train.values.astype("float32")).astype("float32")

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(X.values.astype("float32")).astype("float32")
