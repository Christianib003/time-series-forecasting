from __future__ import annotations
import pandas as pd

_DT_CANDS = ["datetime", "Date Time", "date_time", "timestamp", "date", "time"]
_Y_CANDS  = ["pm2_5", "pm2.5", "PM2.5", "PM2_5", "pm25", "PM25", "PM_2_5"]

def detect_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Best-effort detection of datetime and target columns from a DataFrame."""
    dt_col = next((c for c in df.columns if c in _DT_CANDS), None)
    if dt_col is None:
        dt_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
    y_col = next((c for c in df.columns if c in _Y_CANDS), None)
    if y_col is None:
        y_col = next((c for c in df.columns if "pm2" in c.lower()), None)
    return dt_col, y_col
