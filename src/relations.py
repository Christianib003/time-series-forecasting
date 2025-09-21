from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import FIGURES_DIR
from .utils import get_logger

log = get_logger("relations")

# ------------------------------
# Column selection helpers
# ------------------------------
def numeric_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> List[str]:
    ex = set(exclude)
    return [c for c in df.columns if c not in ex and pd.api.types.is_numeric_dtype(df[c])]

def candidate_exogenous(df: pd.DataFrame) -> List[str]:
    """Heuristic: common weather/exogenous columns seen in this dataset."""
    candidates = ["TEMP", "DEWP", "PRES", "Iws", "Is", "Ir"]
    return [c for c in candidates if c in df.columns]

# ------------------------------
# Correlations
# ------------------------------
def corr_with_target(
    df: pd.DataFrame, target: str, features: Iterable[str], method: str = "pearson"
) -> pd.DataFrame:
    """
    Returns a DataFrame with feature, corr, and |corr| sorted descending.
    Drops rows with NaN in feature or target before computing corr.
    """
    rows = []
    for f in features:
        sub = df[[f, target]].dropna()
        if len(sub) == 0:
            corr = np.nan
        else:
            corr = sub[f].corr(sub[target], method=method)
        rows.append({"feature": f, "corr": corr, "abs_corr": abs(corr) if pd.notna(corr) else np.nan})
    out = pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return out

def corr_matrix(df: pd.DataFrame, cols: Iterable[str], method: str = "pearson") -> pd.DataFrame:
    return df[list(cols)].corr(method=method)

# ------------------------------
# Plots
# ------------------------------
def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def scatter_xy(
    df: pd.DataFrame,
    x: str,
    y: str,
    fname: str,
    title: Optional[str] = None,
    sample: Optional[int] = 10000,
    alpha: float = 0.3,
    s: float = 6.0,
) -> Path:
    """Matplotlib scatter with optional row sampling (to keep plots light)."""
    figpath = FIGURES_DIR / fname
    _ensure_dir(figpath)

    data = df[[x, y]].dropna()
    if sample and len(data) > sample:
        data = data.sample(sample, random_state=42)

    plt.figure()
    plt.scatter(data[x].values, data[y].values, s=s, alpha=alpha)
    plt.xlabel(x)
    plt.ylabel(y)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(figpath, dpi=120)
    plt.close()
    log.info("Saved %s", figpath)
    return figpath

def hexbin_xy(
    df: pd.DataFrame,
    x: str,
    y: str,
    fname: str,
    title: Optional[str] = None,
    gridsize: int = 40,
) -> Path:
    """Hexbin for dense clouds (no seaborn)."""
    figpath = FIGURES_DIR / fname
    _ensure_dir(figpath)
    data = df[[x, y]].dropna()

    plt.figure()
    plt.hexbin(data[x].values, data[y].values, gridsize=gridsize)
    plt.xlabel(x)
    plt.ylabel(y)
    if title:
        plt.title(title)
    cb = plt.colorbar()
    cb.set_label("count")
    plt.tight_layout()
    plt.savefig(figpath, dpi=120)
    plt.close()
    log.info("Saved %s", figpath)
    return figpath
