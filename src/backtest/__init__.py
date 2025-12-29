"""
Backtest module for quantitative trading.

Contains backtesting framework and analysis tools.
"""

from src.backtest.run_backtest import (
    run_backtest,
    analyze_backtest_result,
    BacktestConfig,
)

__all__ = [
    "run_backtest",
    "analyze_backtest_result",
    "BacktestConfig",
]

