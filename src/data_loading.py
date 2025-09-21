from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import DATA_RAW

def load_raw(train_name: str = "train.csv", test_name: str = "test.csv",
             sample_name: str = "sample_submission.csv") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = DATA_RAW / train_name
    test_path = DATA_RAW / test_name
    sub_path = DATA_RAW / sample_name
    assert train_path.exists(), f"Missing: {train_path}"
    assert test_path.exists(), f"Missing: {test_path}"
    assert sub_path.exists(), f"Missing: {sub_path}"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sub_path)
    return train, test, sample_sub
