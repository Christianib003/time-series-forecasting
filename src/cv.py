from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Fold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp

def make_blocked_folds(df_train: pd.DataFrame, dt_col: str, n_folds: int = 3,
                       val_days: int = 30) -> List[Fold]:
    """
    Create 'n_folds' chronological folds near the end of the training range.
    Each fold uses the preceding data as train and the next 'val_days' as val.
    """
    dts = df_train[dt_col]
    t_min, t_max = dts.min(), dts.max()
    # End anchors for each fold (latest first)
    anchors = pd.date_range(end=t_max, periods=n_folds, freq=f"{val_days}D")
    folds: List[Fold] = []
    prev_train_end = None
    for i, anchor in enumerate(reversed(anchors), start=1):
        val_start = anchor - pd.Timedelta(days=val_days-1)
        val_end   = anchor
        train_end = val_start - pd.Timedelta(hours=1)
        train_start = t_min
        folds.append(Fold(
            name=f"fold{i}",
            train_start=train_start, train_end=train_end,
            val_start=val_start,   val_end=val_end
        ))
        prev_train_end = train_end
    return folds

def mask_for_range(df: pd.DataFrame, dt_col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return (df[dt_col] >= start) & (df[dt_col] <= end)
