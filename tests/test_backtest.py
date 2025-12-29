"""
Tests for backtest module.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.backtest.run_backtest import BacktestConfig

        config = BacktestConfig()

        assert config.topk == 50
        assert config.n_drop == 100
        assert config.init_cash == 1_000_000.0
        assert config.risk_degree == 0.95
        assert config.open_cost == 0.0002
        assert config.close_cost == 0.0012

    def test_custom_config(self):
        """Test custom configuration."""
        from src.backtest.run_backtest import BacktestConfig

        config = BacktestConfig(
            start_time="2022-01-01",
            end_time="2022-12-31",
            topk=30,
            n_drop=60,
            init_cash=500_000.0,
        )

        assert config.start_time == "2022-01-01"
        assert config.topk == 30
        assert config.init_cash == 500_000.0


class TestPredictionsLoading:
    """Tests for predictions loading functions."""

    def test_load_predictions(self, tmp_path):
        """Test loading predictions from CSV."""
        from src.backtest.run_backtest import load_predictions

        # Create mock predictions
        pred_df = pd.DataFrame({
            "datetime": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
            "instrument": ["A", "B", "A", "B"],
            "score": [0.9, 0.8, 0.7, 0.6],
        })
        pred_path = tmp_path / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        loaded = load_predictions(str(pred_path))

        assert isinstance(loaded.index, pd.MultiIndex)
        assert len(loaded) == 4
        assert "score" in loaded.columns

    def test_load_predictions_with_time_filter(self, tmp_path):
        """Test loading predictions with time filter."""
        from src.backtest.run_backtest import load_predictions

        pred_df = pd.DataFrame({
            "datetime": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "instrument": ["A", "A", "A"],
            "score": [0.9, 0.8, 0.7],
        })
        pred_path = tmp_path / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        loaded = load_predictions(
            str(pred_path),
            start_time="2023-01-02",
            end_time="2023-01-02",
        )

        assert len(loaded) == 1

    def test_predictions_to_signal(self, tmp_path):
        """Test converting predictions to signal format."""
        from src.backtest.run_backtest import load_predictions, predictions_to_signal

        pred_df = pd.DataFrame({
            "datetime": ["2023-01-01", "2023-01-01"],
            "instrument": ["A", "B"],
            "score": [0.9, 0.8],
        })
        pred_path = tmp_path / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        loaded = load_predictions(str(pred_path))
        signal = predictions_to_signal(loaded)

        assert isinstance(signal, pd.Series)
        assert len(signal) == 2


class TestBacktestAnalysis:
    """Tests for backtest analysis functions."""

    def test_analyze_backtest_result(self):
        """Test backtest result analysis."""
        from src.backtest.run_backtest import analyze_backtest_result, BacktestConfig

        # Create mock portfolio metric
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns = np.random.randn(100) * 0.01  # 1% daily volatility
        portfolio_metric = pd.DataFrame({
            "return": returns,
        }, index=dates)

        config = BacktestConfig()
        result = analyze_backtest_result(
            portfolio_metric=portfolio_metric,
            indicator_dict={},
            config=config,
        )

        assert "total_return" in result
        assert "annual_return" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "calmar_ratio" in result
        assert "win_rate" in result
        assert "trading_days" in result

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        from src.backtest.run_backtest import analyze_backtest_result, BacktestConfig

        # Create positive returns
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        returns = np.ones(252) * 0.001  # 0.1% daily return
        portfolio_metric = pd.DataFrame({"return": returns}, index=dates)

        config = BacktestConfig()
        result = analyze_backtest_result(portfolio_metric, {}, config)

        # Positive returns should have positive Sharpe
        assert result["sharpe_ratio"] > 0

    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation."""
        from src.backtest.run_backtest import analyze_backtest_result, BacktestConfig

        # Create returns with a clear drawdown
        returns = [0.1, 0.1, -0.3, -0.2, 0.1]  # Clear drawdown in middle
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        portfolio_metric = pd.DataFrame({"return": returns}, index=dates)

        config = BacktestConfig()
        result = analyze_backtest_result(portfolio_metric, {}, config)

        # Max drawdown should be negative
        assert result["max_drawdown"] < 0


class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_report(self):
        """Test report generation."""
        from src.backtest.run_backtest import generate_report, BacktestConfig

        portfolio_metric = pd.DataFrame({
            "return": [0.01, 0.02, -0.01],
        })

        analysis_result = {
            "total_return": 0.15,
            "annual_return": 0.20,
            "volatility": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.10,
            "calmar_ratio": 2.0,
            "win_rate": 0.55,
            "trading_days": 100,
        }

        config = BacktestConfig()
        report = generate_report(portfolio_metric, analysis_result, config)

        assert "BACKTEST REPORT" in report
        assert "夏普比率" in report
        assert "最大回撤" in report
        assert "1.50" in report  # Sharpe ratio

    def test_save_report_to_file(self, tmp_path):
        """Test saving report to file."""
        from src.backtest.run_backtest import generate_report, BacktestConfig

        portfolio_metric = pd.DataFrame({"return": [0.01]})
        analysis_result = {
            "total_return": 0.1,
            "annual_return": 0.1,
            "volatility": 0.1,
            "sharpe_ratio": 1.0,
            "max_drawdown": -0.05,
            "calmar_ratio": 2.0,
            "win_rate": 0.5,
            "trading_days": 10,
        }

        config = BacktestConfig()
        report_path = tmp_path / "report.txt"
        generate_report(portfolio_metric, analysis_result, config, save_path=str(report_path))

        assert report_path.exists()
        content = report_path.read_text()
        assert "BACKTEST REPORT" in content


class TestExchangeConfig:
    """Tests for exchange configuration."""

    def test_create_exchange_config(self):
        """Test exchange config creation."""
        from src.backtest.run_backtest import create_exchange_config, BacktestConfig

        config = BacktestConfig(
            open_cost=0.0003,
            close_cost=0.0015,
            limit_threshold=0.10,
        )

        exchange_config = create_exchange_config(config)

        assert exchange_config["open_cost"] == 0.0003
        assert exchange_config["close_cost"] == 0.0015
        assert exchange_config["limit_threshold"] == 0.10
        assert exchange_config["deal_price"] == "close"


class TestTopKStrategy:
    """Tests for TopKStrategy."""

    def test_generate_target_stocks(self):
        """Test target stock generation."""
        # This is a simplified test without full Qlib initialization
        pass  # Requires Qlib environment

    def test_strategy_parameters(self):
        """Test strategy parameter initialization."""
        # Import would require Qlib, so we test the concept
        pass  # Requires Qlib environment

