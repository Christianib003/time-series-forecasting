from __future__ import annotations
import numpy as np

def make_windows(X: np.ndarray, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """
    X: (T, F) features sorted by time
    y: (T,) target aligned with X rows
    lookback: L time steps for each sample window (predict y at t using t-L..t-1)
    Returns: Xw (N, L, F), yw (N,)
    """
    assert X.shape[0] == y.shape[0], "X and y length mismatch"
    T, F = X.shape
    N = T - lookback
    if N <= 0:
        raise ValueError(f"Not enough rows ({T}) for lookback={lookback}")
    Xw = np.empty((N, lookback, F), dtype=np.float32)
    for i in range(N):
        Xw[i] = X[i:i+lookback]
    yw = y[lookback:].astype(np.float32)
    return Xw, yw
