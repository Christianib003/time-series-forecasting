# src/policies.py
from __future__ import annotations
import pandas as pd

def count_missing_target(df: pd.DataFrame, target_col: str) -> int:
    return int(df[target_col].isna().sum()) if target_col in df.columns else 0

def dry_run_feature_fill(df: pd.DataFrame) -> dict:
    temp = df.copy()
    before = temp.isna().sum().sum()
    temp = temp.ffill().bfill()
    after = temp.isna().sum().sum()
    return {"total_nan_before": int(before), "total_nan_after_ffill_bfill": int(after)}

def apply_feature_fill_inplace(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    """Fill forward then backward on specified columns without chained assignment."""
    cols = columns if columns is not None else list(df.columns)
    df.loc[:, cols] = df.loc[:, cols].ffill().bfill()
