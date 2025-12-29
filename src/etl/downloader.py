import os
import time
from typing import Iterable, List

import pandas as pd
from akshare import stock_zh_a_hist

COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
}

REQUIRED_COLUMNS: List[str] = ["date", "open", "close", "high", "low", "volume"]


def _fetch_with_retry(symbol: str, max_retries: int = 3, retry_delay: float = 1.0) -> pd.DataFrame:
    """Fetch a symbol with retries for transient network errors."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_retries:
                raise
            time.sleep(retry_delay)
    # Should not reach here
    raise last_exc  # type: ignore[misc]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns=COLUMN_MAP)
    missing = [col for col in REQUIRED_COLUMNS if col not in renamed.columns]
    if missing:
        raise ValueError(f"Missing required columns after rename: {missing}")
    return renamed[REQUIRED_COLUMNS]


def download_stock_history(
    symbols: Iterable[str],
    out_dir: str = "/app/data/csv_source",
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> None:
    """Download historical daily data for given symbols and save as CSV."""
    os.makedirs(out_dir, exist_ok=True)
    for symbol in symbols:
        df = _fetch_with_retry(symbol, max_retries=max_retries, retry_delay=retry_delay)
        normalized = _normalize_columns(df)
        out_path = os.path.join(out_dir, f"{symbol}.csv")
        normalized.to_csv(out_path, index=False)


if __name__ == "__main__":
    try:
        download_stock_history(["600519"])
        print("Download completed for 600519")
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {exc}")

