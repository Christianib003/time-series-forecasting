from __future__ import annotations
import numpy as np
import pandas as pd

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def naive_forecast(series: pd.Series, lag: int = 1) -> pd.Series:
    return series.shift(lag)

def evaluate_baselines(df: pd.DataFrame, dt_col: str, target: str, folds, seasonal_lag: int = 24) -> pd.DataFrame:
    rows = []
    for f in folds:
        tr_mask = (df[dt_col] >= f.train_start) & (df[dt_col] <= f.train_end)
        va_mask = (df[dt_col] >= f.val_start) & (df[dt_col] <= f.val_end)

        y_train = df.loc[tr_mask, target]
        y_val   = df.loc[va_mask, target]

        # Build baselines using the CONTIGUOUS full series up to val
        series_until_val = df.loc[df[dt_col] <= f.val_end, target]

        pred_naive = naive_forecast(series_until_val, lag=1).loc[y_val.index]
        pred_seas  = naive_forecast(series_until_val, lag=seasonal_lag).loc[y_val.index]

        r1 = rmse(y_val, pred_naive)
        r2 = rmse(y_val, pred_seas)

        rows.append({"fold": f.name, "rmse_naive": r1, "rmse_seasonal24": r2,
                     "val_rows": int(y_val.notna().sum())})
    return pd.DataFrame(rows)
