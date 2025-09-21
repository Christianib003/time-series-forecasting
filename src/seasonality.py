from __future__ import annotations
import numpy as np
import pandas as pd

def add_time_parts(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out[dt_col].dt.hour.astype("int16")
    out["day_of_week"] = out[dt_col].dt.dayofweek.astype("int16")
    out["month"] = out[dt_col].dt.month.astype("int16")
    out["day_of_year"] = out[dt_col].dt.dayofyear.astype("int16")
    return out

def add_cyclic_encodings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0).astype("float32")
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0).astype("float32")
    out["sin_doy"] = np.sin(2 * np.pi * out["day_of_year"] / 366.0).astype("float32")
    out["cos_doy"] = np.cos(2 * np.pi * out["day_of_year"] / 366.0).astype("float32")
    return out
