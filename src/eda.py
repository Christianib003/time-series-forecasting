from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import FIGURES_DIR
from .utils import get_logger

log = get_logger("eda")

@dataclass(frozen=True)
class MissingnessResult:
    table: pd.DataFrame
    figure_path: Path

def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return counts and percentages of NaNs per column."""
    na_counts = df.isna().sum()
    na_pct = (na_counts / len(df) * 100.0)
    out = pd.DataFrame({"na_count": na_counts, "na_pct": na_pct})
    return out.sort_values("na_count", ascending=False)

def plot_missingness_bar(miss_tab: pd.DataFrame, tag: str, top_n: int = 20) -> Path:
    """Save a simple bar chart of top-N missing columns."""
    top = miss_tab.head(top_n)
    figpath = FIGURES_DIR / f"eda_missing_{tag}.png"
    plt.figure()
    x = np.arange(len(top))
    plt.bar(x, top["na_count"].values)
    plt.xticks(x, top.index.tolist(), rotation=45, ha="right")
    plt.ylabel("Missing count")
    plt.title(f"Missingness — {tag}")
    plt.tight_layout()
    plt.savefig(figpath, dpi=120)
    plt.close()
    log.info("Saved %s", figpath)
    return figpath

def profile_missingness(df: pd.DataFrame, tag: str, top_n: int = 20) -> MissingnessResult:
    tab = missingness_summary(df)
    fig = plot_missingness_bar(tab, tag=tag, top_n=top_n)
    return MissingnessResult(table=tab, figure_path=fig)
