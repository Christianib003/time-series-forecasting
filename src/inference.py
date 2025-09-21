# src/inference.py
from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
import pandas as pd
from .scaling import PerFoldScaler

def _ensure_column_order(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.loc[:, list(cols)]

def final_fit_windows(
    df_train: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    lookback: int,
    max_lag: int,
) -> Tuple[np.ndarray, np.ndarray, PerFoldScaler]:
    """
    Prepare (Xw, yw) and fit a scaler on ALL usable training rows (after lag trimming).
    Returns windows and the fitted scaler for reuse in test rollout.
    """
    # Trim early rows that don't have lag features populated
    df_tr = df_train.iloc[max_lag:].reset_index(drop=True)
    y_tr  = np.log1p(df_tr[target_col].values.astype("float32"))

    scaler = PerFoldScaler()
    X_tr = scaler.fit_transform(_ensure_column_order(df_tr, feature_cols))

    # Build contiguous lookback windows
    T, F = X_tr.shape
    N = T - lookback
    if N <= 0:
        raise ValueError(f"Not enough training rows ({T}) for lookback={lookback}")
    Xw = np.empty((N, lookback, F), dtype=np.float32)
    for i in range(N):
        Xw[i] = X_tr[i : i + lookback]
    yw = y_tr[lookback:].astype(np.float32)
    return Xw, yw, scaler

def rollout_forecast_test(
    model,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    lags: Sequence[int],
    lookback: int,
    scaler: PerFoldScaler,
) -> np.ndarray:
    """
    Iterative forecast over the test horizon (one-step ahead), using:
    - exogenous features from df_test
    - target lag features built from a rolling buffer seeded with last train targets
    - LSTM lookback window on SCALED features (sliding through train→test boundary)
    Returns predictions in ORIGINAL scale (after expm1).
    """
    max_lag = max(lags)

    # Seed lag buffer with the last max_lag true targets from train
    lag_buffer = pd.Series(df_train[target_col].tail(max_lag).values.astype("float32"))

    # Build a continuous frame of (last lookback rows from train) + full test exogenous
    # We'll regenerate the lagged target columns on the fly for each test step.
    # Start window: last `lookback` rows of TRAIN feature matrix
    train_tail = _ensure_column_order(df_train, feature_cols).tail(lookback).copy()
    Xwin = scaler.transform(train_tail).astype("float32")  # (lookback, F)

    preds = []
    # Iterate over test rows chronologically
    for idx in range(len(df_test)):
        # Build the raw feature row for this test timestamp
        row_exog = df_test.iloc[[idx]].copy()  # DataFrame shape (1, ...)
        # Populate lag features from current lag_buffer
        for L in lags:
            row_exog[f"{target_col}_lag{L}"] = lag_buffer.iloc[-L]

        # Order columns & scale
        row_feat = _ensure_column_order(row_exog, feature_cols)
        x_scaled = scaler.transform(row_feat).astype("float32")  # (1, F)

        # Slide window: append x_scaled, drop the oldest row
        Xwin = np.vstack([Xwin[1:], x_scaled])  # (lookback, F)
        x_input = Xwin[None, :, :]              # (1, lookback, F)

        # Predict and inverse transform
        y_log = model.predict(x_input, verbose=0).reshape(-1)[0]
        y_hat = float(np.expm1(y_log))
        # Non-negativity guard
        if y_hat < 0:
            y_hat = 0.0

        preds.append(y_hat)

        # Update lag buffer with the predicted value
        lag_buffer = pd.concat([lag_buffer, pd.Series([y_hat])])
        # Keep only the last max_lag values in buffer
        lag_buffer = lag_buffer.iloc[-max_lag:]

    return np.array(preds, dtype="float32")
