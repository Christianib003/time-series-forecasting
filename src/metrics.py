from __future__ import annotations
import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype("float32")
    y_pred = y_pred.astype("float32")
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))
