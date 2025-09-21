from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .config import DATA_INTERIM
from .policies import apply_feature_fill_inplace
from .seasonality import add_time_parts, add_cyclic_encodings

@dataclass
class CleanMeta:
    n_train_rows: int
    n_test_rows: int
    dropped_train_missing_target: int
    train_time_min: str
    train_time_max: str
    test_time_min: str
    test_time_max: str
    columns: list[str]

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Canonical target name
    if "pm2_5" not in out.columns and "pm2.5" in out.columns:
        out["pm2_5"] = out["pm2.5"]
    # Canonical datetime name
    if "datetime" not in out.columns:
        for c in ["Date Time", "date_time", "timestamp", "date", "time"]:
            if c in out.columns:
                out = out.rename(columns={c: "datetime"})
                break
    return out

def _ffill_bfill_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    feature_cols = [c for c in out.columns if c != target_col]
    from .policies import apply_feature_fill_inplace
    apply_feature_fill_inplace(out, columns=feature_cols)
    return out


def build_interim(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    target_raw: str = "pm2.5",
    target_clean: str = "pm2_5",
    dt_col_name: str = "datetime",
    save_csv: bool = True,
    csv_path: Path | None = None,
) -> tuple[pd.DataFrame, CleanMeta]:
    """Create clean, feature-augmented dataset with a 'split' column and save."""
    tr = _standardize_columns(train_raw)
    te = _standardize_columns(test_raw)

    # Parse time & sort
    tr[dt_col_name] = pd.to_datetime(tr[dt_col_name])
    te[dt_col_name] = pd.to_datetime(te[dt_col_name])
    tr = tr.sort_values(dt_col_name).reset_index(drop=True)
    te = te.sort_values(dt_col_name).reset_index(drop=True)

    # Ensure clean target column exists in both (test will be NaN)
    tr[target_clean] = tr[target_clean] if target_clean in tr.columns else tr[target_raw]
    if target_clean not in te.columns:
        te[target_clean] = np.nan

    # Drop rows with missing target in train
    n_before = len(tr)
    tr = tr[~tr[target_clean].isna()].reset_index(drop=True)
    dropped = n_before - len(tr)

    # Fill features separately in train/test
    tr = _ffill_bfill_features(tr, target_col=target_clean)
    te = _ffill_bfill_features(te, target_col=target_clean)

    # Add calendar parts + cyclic encodings
    tr = add_time_parts(tr, dt_col=dt_col_name)
    te = add_time_parts(te, dt_col=dt_col_name)
    tr = add_cyclic_encodings(tr)
    te = add_cyclic_encodings(te)

    # Split flag + concat
    tr["split"] = "train"
    te["split"] = "test"
    interim = pd.concat([tr, te], ignore_index=True, sort=False)

    meta = CleanMeta(
        n_train_rows=int(len(tr)),
        n_test_rows=int(len(te)),
        dropped_train_missing_target=int(dropped),
        train_time_min=tr[dt_col_name].min().isoformat(),
        train_time_max=tr[dt_col_name].max().isoformat(),
        test_time_min=te[dt_col_name].min().isoformat(),
        test_time_max=te[dt_col_name].max().isoformat(),
        columns=list(interim.columns),
    )

    if save_csv:
        DATA_INTERIM.mkdir(parents=True, exist_ok=True)
        out_csv = csv_path if csv_path is not None else DATA_INTERIM / "clean.csv"
        interim.to_csv(out_csv, index=False)
        with open(DATA_INTERIM / "clean_meta.json", "w") as f:
            json.dump(asdict(meta), f, indent=2)

    return interim, meta
