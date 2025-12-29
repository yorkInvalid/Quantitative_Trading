import os

import pandas as pd
import pytest

from src.etl import downloader


def test_download_creates_csv(tmp_path, monkeypatch):
    calls = []

    def fake_stock_zh_a_hist(symbol: str, period: str = "daily", adjust: str = "qfq"):
        calls.append((symbol, period, adjust))
        return pd.DataFrame(
            {
                "日期": ["2024-01-02"],
                "开盘": [100.0],
                "收盘": [110.0],
                "最高": [115.0],
                "最低": [95.0],
                "成交量": [123456],
            }
        )

    monkeypatch.setattr(downloader, "stock_zh_a_hist", fake_stock_zh_a_hist)

    out_dir = tmp_path / "csv_source"
    downloader.download_stock_history(["600519"], out_dir=str(out_dir), retry_delay=0)

    target = out_dir / "600519.csv"
    assert target.exists()

    df = pd.read_csv(target)
    assert set(["date", "open", "close", "high", "low", "volume"]).issubset(df.columns)
    assert not df.empty
    assert calls == [("600519", "daily", "qfq")]

