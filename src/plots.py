from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import FIGURES_DIR
from .utils import get_logger

log = get_logger("plots")

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def hist_target(series: pd.Series, title: str, fname: str, bins: int = 60) -> Path:
    figpath = FIGURES_DIR / fname
    _ensure_dir(figpath)
    plt.figure()
    series.dropna().astype(float).hist(bins=bins)
    plt.xlabel("pm2_5 (µg/m³)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figpath, dpi=120)
    plt.close()
    log.info("Saved %s", figpath)
    return figpath

def box_target(series: pd.Series, title: str, fname: str) -> Path:
    figpath = FIGURES_DIR / fname
    _ensure_dir(figpath)
    plt.figure()
    plt.boxplot([series.dropna().astype(float).values], labels=["pm2_5"])
    plt.ylabel("µg/m³")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figpath, dpi=120)
    plt.close()
    log.info("Saved %s", figpath)
    return figpath

def seasonal_mean(df: pd.DataFrame, by: str, target: str, title: str, fname: str) -> Path:
    figpath = FIGURES_DIR / fname
    _ensure_dir(figpath)
    g = df.groupby(by)[target].mean()
    plt.figure()
    plt.plot(g.index.values, g.values)
    plt.xlabel(by)
    plt.ylabel(f"Mean {target}")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figpath, dpi=120)
    plt.close()
    log.info("Saved %s", figpath)
    return figpath
