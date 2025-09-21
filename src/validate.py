from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import FIGURES_DIR
from .utils import get_logger

log = get_logger("validate")

def parse_sort_assert(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """Parse datetime col, sort ascending, assert uniqueness & monotonicity."""
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col])
    out = out.sort_values(dt_col).reset_index(drop=True)
    assert out[dt_col].is_monotonic_increasing, f"{dt_col}: not increasing"
    assert out[dt_col].is_unique, f"{dt_col}: duplicates present"
    return out

def boundary_check(train_dt_end: pd.Timestamp, test_dt_start: pd.Timestamp) -> bool:
    ok = (train_dt_end + pd.Timedelta(hours=1) == test_dt_start)
    msg = "PASS ✅" if ok else "FAIL ❌"
    log.info("Boundary check (last train + 1h == first test): %s", msg)
    return ok

def save_continuity_hist(df: pd.DataFrame, dt_col: str, tag: str) -> Path:
    """Save histogram of Δtime in hours to figures directory."""
    deltas = df[dt_col].diff().dropna().dt.total_seconds().div(3600)
    figpath = FIGURES_DIR / f"eda_continuity_{tag}.png"
    plt.figure()
    plt.hist(deltas, bins=np.arange(deltas.min(), deltas.max() + 1.5) - 0.5)
    plt.title(f"Time-step continuity — {tag}")
    plt.xlabel("Δ time (hours)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figpath, dpi=120)
    plt.close()
    log.info("Saved %s", figpath)
    return figpath

def snapshot(df: pd.DataFrame, dt_col: str, y_col: str | None = None) -> dict:
    """Quick snapshot of ranges and columns (for printouts)."""
    d = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "first_ts": df[dt_col].iloc[0].isoformat(),
        "last_ts": df[dt_col].iloc[-1].isoformat(),
        "columns": list(df.columns),
    }
    if y_col and y_col in df.columns:
        d["target_non_null"] = int(df[y_col].notna().sum())
        d["target_null"] = int(df[y_col].isna().sum())
    return d
