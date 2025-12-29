"""
CSV to Qlib binary converter.

Reads CSV files from /app/data/csv_source/ and converts them to Qlib binary
format under /app/data/qlib_bin/, generating the required calendars/ and
features/ directories.
"""

import os
import struct
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Default paths
DEFAULT_CSV_DIR = "/app/data/csv_source"
DEFAULT_BIN_DIR = "/app/data/qlib_bin"

# Columns expected in source CSV (output from downloader.py)
REQUIRED_COLUMNS = ["date", "open", "close", "high", "low", "volume"]


def _ensure_dirs(bin_dir: str) -> None:
    """Create calendars/ and features/ directories under bin_dir."""
    Path(bin_dir, "calendars").mkdir(parents=True, exist_ok=True)
    Path(bin_dir, "features").mkdir(parents=True, exist_ok=True)


def _read_csv(csv_path: str) -> pd.DataFrame:
    """Read a CSV and validate required columns."""
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _dump_bin(arr: np.ndarray, out_path: str) -> None:
    """
    Write a 1-D float32 numpy array to Qlib binary format.

    Qlib binary format: little-endian float32 sequence.
    """
    arr = arr.astype(np.float32)
    with open(out_path, "wb") as f:
        f.write(arr.tobytes())


def _convert_symbol(df: pd.DataFrame, symbol: str, features_dir: str) -> List[str]:
    """
    Convert a single symbol DataFrame to Qlib binary features.

    Returns the list of dates (as strings) for calendar aggregation.
    """
    symbol_dir = Path(features_dir, symbol)
    symbol_dir.mkdir(parents=True, exist_ok=True)

    # Feature columns to dump (exclude date)
    feature_cols = ["open", "close", "high", "low", "volume"]
    for col in feature_cols:
        arr = df[col].values
        out_path = symbol_dir / f"{col}.day.bin"
        _dump_bin(arr, str(out_path))

    # Return date strings for calendar
    return df["date"].dt.strftime("%Y-%m-%d").tolist()


def convert_csv_to_qlib(
    csv_dir: str = DEFAULT_CSV_DIR,
    bin_dir: str = DEFAULT_BIN_DIR,
    calendar_name: str = "day.txt",
) -> None:
    """
    Convert all CSV files in csv_dir to Qlib binary format in bin_dir.

    Creates:
      - bin_dir/calendars/{calendar_name}: sorted unique trading dates
      - bin_dir/features/{symbol}/{feature}.day.bin: binary feature files
    """
    _ensure_dirs(bin_dir)

    csv_dir_path = Path(csv_dir)
    features_dir = Path(bin_dir, "features")
    calendars_dir = Path(bin_dir, "calendars")

    all_dates: set = set()

    csv_files = list(csv_dir_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    for csv_file in csv_files:
        symbol = csv_file.stem  # e.g., 600519
        df = _read_csv(str(csv_file))
        dates = _convert_symbol(df, symbol, str(features_dir))
        all_dates.update(dates)

    # Write calendar file
    sorted_dates = sorted(all_dates)
    calendar_path = calendars_dir / calendar_name
    with open(calendar_path, "w") as f:
        f.write("\n".join(sorted_dates))

    print(f"Converted {len(csv_files)} symbols to {bin_dir}")
    print(f"Calendar ({len(sorted_dates)} days) written to {calendar_path}")


if __name__ == "__main__":
    convert_csv_to_qlib()

