# src/features.py
from __future__ import annotations
import pandas as pd

def add_target_lags(df: pd.DataFrame, target: str = "pm2_5", lags: tuple[int, ...] = (1, 24, 168)) -> pd.DataFrame:
    """Add lagged copies of the target (autoregressive features)."""
    out = df.copy()
    for L in lags:
        out[f"{target}_lag{L}"] = out[target].shift(L)
    return out
