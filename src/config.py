from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

# Project roots
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES_DIR = OUTPUTS / "figures"
MODELS_DIR = OUTPUTS / "models"
SUBMISSIONS_DIR = OUTPUTS / "submissions"

# Ensure output dirs exist (safe no-op if they do)
for p in (OUTPUTS, FIGURES_DIR, MODELS_DIR, SUBMISSIONS_DIR, DATA_INTERIM, DATA_PROCESSED):
    p.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = 42

@dataclass(frozen=True)
class ColumnNames:
    datetime: str = "datetime"  # our canonical name (used later)
    target: str = "pm2_5"       # our canonical name (used later)
