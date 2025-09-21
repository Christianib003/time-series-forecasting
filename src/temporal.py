from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import FIGURES_DIR
from .utils import get_logger

log = get_logger("temporal")

def autocorr_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Robust autocorrelation (biased) for lags 0..max_lag, handling NaNs, short series,
    and near-zero variance.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n <= 1:
        return np.zeros(1, dtype=float)  # only lag 0

    # center
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 1e-12:  # all nearly equal
        return np.zeros(1, dtype=float)

    # don't request lags longer than n-1
    max_k = min(max_lag, n - 1)
    ac = np.empty(max_k + 1, dtype=float)
    for k in range(max_k + 1):
        ac[k] = float(np.dot(x[: n - k], x[k:]) / denom)
    return ac


def plot_autocorr(series: pd.Series, max_lag: int, fname: str, title: str | None = None) -> Path:
    ac = autocorr_1d(series.astype(float).values, max_lag=max_lag)
    figpath = FIGURES_DIR / fname
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    lags = np.arange(max_lag + 1)

    # Matplotlib ≥3.6: do NOT pass use_line_collection
    markerline, stemlines, baseline = ax.stem(lags, ac, basefmt=" ")
    ax.set_xlabel("lag (hours)")
    ax.set_ylabel("autocorr")
    ax.set_title(title or f"Autocorrelation up to {max_lag}h")
    fig.tight_layout()
    fig.savefig(figpath, dpi=120)
    plt.close(fig)
    log.info("Saved %s", figpath)
    return figpath



def seasonal_lag_strength(series: pd.Series, periods: list[int]) -> dict[int, float]:
    """Return autocorr at specific lags that fit within the computed array."""
    x = pd.to_numeric(series, errors="coerce").to_numpy()
    ac = autocorr_1d(x, max_lag=max(periods))
    strengths = {}
    for p in periods:
        if p <= len(ac) - 1:
            strengths[p] = float(ac[p])
    return strengths

# Optional PACF via statsmodels if available
def plot_pacf_optional(series: pd.Series, max_lag: int, fname: str) -> Path | None:
    try:
        from statsmodels.graphics.tsaplots import plot_pacf  # type: ignore
    except Exception:
        log.info("statsmodels not available; skipping PACF.")
        return None
    figpath = FIGURES_DIR / fname
    plt.figure()
    plot_pacf(series.dropna().astype(float), lags=max_lag, method="ywm")
    plt.tight_layout()
    plt.savefig(figpath, dpi=120)
    plt.close()
    log.info("Saved %s", figpath)
    return figpath
