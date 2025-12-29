"""
Tests for paper trading / dry run module.
"""

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# Simple pickleable model for testing
class SimpleModel:
    """Simple mock model that can be pickled."""
    def predict(self, X):
        return [0.1] * len(X)


class TestPosition:
    """Tests for Position dataclass."""

    def test_create_position(self):
        """Test position creation."""
        from src.dry_run import Position

        pos = Position(
            instrument="600519",
            amount=100,
            cost_price=1800.0,
        )

        assert pos.instrument == "600519"
        assert pos.amount == 100
        assert pos.cost_price == 1800.0

    def test_update_price(self):
        """Test price update."""
        from src.dry_run import Position

        pos = Position(
            instrument="600519",
            amount=100,
            cost_price=1800.0,
        )

        pos.update_price(2000.0)

        assert pos.current_price == 2000.0
        assert pos.market_value == 200_000.0


class TestPortfolio:
    """Tests for Portfolio dataclass."""

    def test_create_portfolio(self):
        """Test portfolio creation."""
        from src.dry_run import Portfolio

        portfolio = Portfolio(cash=1_000_000.0)

        assert portfolio.cash == 1_000_000.0
        assert portfolio.positions == {}

    def test_update_total_value(self):
        """Test total value update."""
        from src.dry_run import Portfolio, Position

        portfolio = Portfolio(cash=500_000.0)
        portfolio.positions = {
            "600519": Position("600519", 100, 1800.0, 2000.0, 200_000.0),
            "000001": Position("000001", 1000, 10.0, 12.0, 12_000.0),
        }

        portfolio.update_total_value()

        assert portfolio.total_value == 712_000.0

    def test_to_dict(self):
        """Test serialization to dict."""
        from src.dry_run import Portfolio, Position

        portfolio = Portfolio(cash=1_000_000.0)
        portfolio.positions["600519"] = Position("600519", 100, 1800.0)

        data = portfolio.to_dict()

        assert "cash" in data
        assert "positions" in data
        assert "600519" in data["positions"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        from src.dry_run import Portfolio

        data = {
            "cash": 1_000_000.0,
            "positions": {
                "600519": {
                    "instrument": "600519",
                    "amount": 100,
                    "cost_price": 1800.0,
                    "current_price": 2000.0,
                    "market_value": 200_000.0,
                }
            },
            "total_value": 1_200_000.0,
            "last_update": "2024-12-29",
        }

        portfolio = Portfolio.from_dict(data)

        assert portfolio.cash == 1_000_000.0
        assert "600519" in portfolio.positions
        assert portfolio.positions["600519"].amount == 100


class TestVirtualExchange:
    """Tests for VirtualExchange."""

    def test_execute_buy(self):
        """Test buy order execution."""
        from src.dry_run import VirtualExchange

        exchange = VirtualExchange(
            buy_slippage=0.0002,
            buy_commission=0.0002,
            min_commission=5.0,
        )

        deal_price, deal_amount, cost = exchange.execute_buy(
            "600519", 100, 1800.0
        )

        # 成交价 = 1800 * (1 + 0.0002) = 1800.36
        assert deal_price == pytest.approx(1800.36, abs=0.01)
        # 成交金额 = 100 * 1800.36 = 180036
        assert deal_amount == pytest.approx(180036.0, abs=1.0)
        # 佣金 = max(180036 * 0.0002, 5) = 36.0072
        assert cost >= 5.0

    def test_execute_sell(self):
        """Test sell order execution."""
        from src.dry_run import VirtualExchange

        exchange = VirtualExchange(
            sell_slippage=0.0002,
            sell_commission=0.0012,
            min_commission=5.0,
        )

        deal_price, deal_amount, cost = exchange.execute_sell(
            "600519", 100, 1800.0
        )

        # 成交价 = 1800 * (1 - 0.0002) = 1799.64
        assert deal_price == pytest.approx(1799.64, abs=0.01)
        # 成交金额 = 100 * 1799.64 = 179964
        assert deal_amount == pytest.approx(179964.0, abs=1.0)
        # 佣金 = 179964 * 0.0012 = 215.9568
        assert cost >= 5.0


class TestPaperTrader:
    """Tests for PaperTrader."""

    @pytest.fixture
    def mock_trader(self, tmp_path):
        """Create a mock trader for testing."""
        from src.dry_run import PaperTrader

        # Create simple mock model (pickleable)
        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(SimpleModel(), f)

        portfolio_path = tmp_path / "portfolio.json"
        reports_dir = tmp_path / "reports"

        trader = PaperTrader(
            model_path=str(model_path),
            portfolio_path=str(portfolio_path),
            reports_dir=str(reports_dir),
            topk=50,
            n_drop=100,
            init_cash=1_000_000.0,
        )

        return trader

    def test_create_trader(self, mock_trader):
        """Test trader creation."""
        assert mock_trader.topk == 50
        assert mock_trader.n_drop == 100
        assert mock_trader.init_cash == 1_000_000.0

    def test_load_portfolio_creates_new(self, mock_trader):
        """Test loading portfolio creates new if not exists."""
        portfolio = mock_trader.load_portfolio()

        assert portfolio.cash == 1_000_000.0
        assert portfolio.positions == {}

    def test_save_and_load_portfolio(self, mock_trader):
        """Test saving and loading portfolio."""
        from src.dry_run import Portfolio, Position

        # Create portfolio
        portfolio = Portfolio(cash=500_000.0)
        portfolio.positions["600519"] = Position("600519", 100, 1800.0)

        # Save
        mock_trader.save_portfolio(portfolio)

        # Load
        loaded = mock_trader.load_portfolio()

        assert loaded.cash == 500_000.0
        assert "600519" in loaded.positions

    def test_load_model(self, mock_trader):
        """Test loading model."""
        model = mock_trader.load_model()
        assert model is not None

    def test_normalize_code(self, mock_trader):
        """Test stock code normalization."""
        assert mock_trader._normalize_code("SH600519") == "600519"
        assert mock_trader._normalize_code("SZ000001") == "000001"
        assert mock_trader._normalize_code("600519.SH") == "600519"

    def test_generate_orders_buy_signal(self, mock_trader):
        """Test order generation for buy signals."""
        from src.dry_run import Portfolio

        predictions = pd.DataFrame({
            "instrument": ["600519", "000001", "600000"],
            "score": [0.05, 0.03, 0.02],
        })

        portfolio = Portfolio(cash=1_000_000.0)

        orders = mock_trader.generate_orders(predictions, portfolio)

        # Should generate buy orders for top stocks
        buy_orders = [o for o in orders if o["direction"] == "BUY"]
        assert len(buy_orders) > 0

    def test_generate_orders_sell_signal(self, mock_trader):
        """Test order generation for sell signals."""
        from src.dry_run import Portfolio, Position

        # Create predictions where current holding is out of top-N
        predictions = pd.DataFrame({
            "instrument": [f"60{i:04d}" for i in range(150)],
            "score": [0.1 - i * 0.001 for i in range(150)],
        })

        # Portfolio holds a stock out of top-100
        portfolio = Portfolio(cash=1_000_000.0)
        portfolio.positions["600120"] = Position("600120", 1000, 10.0)

        mock_trader.n_drop = 100
        orders = mock_trader.generate_orders(predictions, portfolio)

        # Should generate sell order for the holding
        sell_orders = [o for o in orders if o["direction"] == "SELL"]
        assert len(sell_orders) > 0

    def test_execute_orders_with_mock_prices(self, mock_trader):
        """Test order execution with mocked prices."""
        from src.dry_run import Portfolio, Position

        portfolio = Portfolio(cash=1_000_000.0)
        portfolio.positions["600519"] = Position("600519", 100, 1800.0)

        orders = [
            {
                "instrument": "600519",
                "direction": "SELL",
                "amount": 100,
                "reason": "Test",
            }
        ]

        # Mock get_latest_prices
        with patch.object(
            mock_trader, "get_latest_prices", return_value={"600519": 2000.0}
        ):
            trades = mock_trader.execute_orders(orders, portfolio)

        assert len(trades) >= 1
        assert "600519" not in portfolio.positions

    def test_apply_risk_rules(self, mock_trader):
        """Test risk rules application."""
        orders = [
            {
                "instrument": "600519",
                "direction": "BUY",
                "amount": 100,
            }
        ]

        filtered = mock_trader.apply_risk_rules(orders)

        # Currently just returns orders as-is
        assert len(filtered) == len(orders)


class TestDailyReport:
    """Tests for DailyReport."""

    def test_create_report(self):
        """Test report creation."""
        from src.dry_run import DailyReport

        report = DailyReport(
            date="2024-12-29",
            portfolio_value=1_200_000.0,
            cash=500_000.0,
            positions_value=700_000.0,
            positions_count=5,
        )

        assert report.date == "2024-12-29"
        assert report.portfolio_value == 1_200_000.0

    def test_report_to_dict(self):
        """Test report serialization."""
        from src.dry_run import DailyReport

        report = DailyReport(
            date="2024-12-29",
            portfolio_value=1_200_000.0,
            cash=500_000.0,
            positions_value=700_000.0,
            positions_count=5,
        )

        data = report.to_dict()

        assert "date" in data
        assert "portfolio_value" in data
        assert data["date"] == "2024-12-29"


class TestEndToEnd:
    """End-to-end tests."""

    def test_run_daily_cycle_mock(self, tmp_path):
        """Test running daily cycle with mocks."""
        from src.dry_run import PaperTrader

        # Setup paths
        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(SimpleModel(), f)

        trader = PaperTrader(
            model_path=str(model_path),
            portfolio_path=str(tmp_path / "portfolio.json"),
            reports_dir=str(tmp_path / "reports"),
            topk=5,
            n_drop=10,
            init_cash=100_000.0,
        )

        # Mock methods
        with patch.object(trader, "predict") as mock_predict, \
             patch.object(trader, "get_latest_prices") as mock_prices:
            
            mock_predict.return_value = pd.DataFrame({
                "instrument": ["600519", "000001"],
                "score": [0.05, 0.03],
            })
            
            mock_prices.return_value = {
                "600519": 1800.0,
                "000001": 10.0,
            }
            
            report = trader.run_daily_cycle("2024-12-29")

        assert report.date == "2024-12-29"
        assert report.portfolio_value > 0
        assert (tmp_path / "portfolio.json").exists()
        assert (tmp_path / "reports" / "report_2024-12-29.json").exists()

