"""
Tests for risk management module.
"""

import pandas as pd
import pytest


class TestOrder:
    """Tests for Order dataclass."""

    def test_create_order(self):
        """Test order creation."""
        from src.risk.rules import Order

        order = Order(
            instrument="600519",
            direction="BUY",
            amount=100,
            price=1800.0,
            reason="Test order",
        )

        assert order.instrument == "600519"
        assert order.direction == "BUY"
        assert order.amount == 100
        assert order.price == 1800.0

    def test_create_order_helper(self):
        """Test create_order helper function."""
        from src.risk.rules import create_order

        order = create_order("600519", "buy", 100, 1800.0)

        assert order.instrument == "600519"
        assert order.direction == "BUY"  # Should be uppercase


class TestRiskCheckResult:
    """Tests for RiskCheckResult dataclass."""

    def test_result_creation(self):
        """Test result creation."""
        from src.risk.rules import Order, RiskCheckResult

        order = Order("600519", "BUY", 100)
        result = RiskCheckResult(
            passed=True,
            order=order,
            rule_name="TestRule",
            message="Test passed",
        )

        assert result.passed is True
        assert result.rule_name == "TestRule"


class TestStopSignRule:
    """Tests for StopSignRule."""

    def test_normalize_code(self):
        """Test stock code normalization."""
        from src.risk.rules import StopSignRule

        rule = StopSignRule()

        assert rule._normalize_code("SH600519") == "600519"
        assert rule._normalize_code("SZ000001") == "000001"
        assert rule._normalize_code("600519.SH") == "600519"
        assert rule._normalize_code("000001.SZ") == "000001"
        assert rule._normalize_code("600519") == "600519"

    def test_rule_name(self):
        """Test rule name property."""
        from src.risk.rules import StopSignRule

        rule = StopSignRule()
        assert rule.name == "StopSignRule"

    def test_empty_orders(self):
        """Test with empty order list."""
        from src.risk.rules import StopSignRule

        rule = StopSignRule()
        passed, results = rule.check([])

        assert passed == []
        assert results == []


class TestPositionLimitRule:
    """Tests for PositionLimitRule."""

    def test_rule_name(self):
        """Test rule name property."""
        from src.risk.rules import PositionLimitRule

        rule = PositionLimitRule()
        assert rule.name == "PositionLimitRule"

    def test_buy_within_limit(self):
        """Test buy order within position limit."""
        from src.risk.rules import PositionLimitRule, Order

        rule = PositionLimitRule(
            max_position_ratio=0.10,
            total_value=1_000_000.0,
            current_positions={},
        )

        order = Order("600519", "BUY", 50, 1800.0)  # 90,000 < 100,000 limit
        passed, results = rule.check([order])

        assert len(passed) == 1
        assert results[0].passed is True

    def test_buy_exceeds_limit(self):
        """Test buy order exceeding position limit."""
        from src.risk.rules import PositionLimitRule, Order

        rule = PositionLimitRule(
            max_position_ratio=0.10,
            total_value=1_000_000.0,
            current_positions={"600519": 80_000},  # Already 80k
        )

        order = Order("600519", "BUY", 50, 1800.0)  # Would add 90k -> 170k > 100k limit
        passed, results = rule.check([order])

        assert len(passed) == 0
        assert results[0].passed is False
        assert "超过限制" in results[0].message

    def test_sell_order_bypasses_limit(self):
        """Test that sell orders bypass position limit check."""
        from src.risk.rules import PositionLimitRule, Order

        rule = PositionLimitRule(
            max_position_ratio=0.10,
            total_value=1_000_000.0,
        )

        order = Order("600519", "SELL", 100, 1800.0)
        passed, results = rule.check([order])

        assert len(passed) == 1
        assert results[0].passed is True


class TestPriceLimitRule:
    """Tests for PriceLimitRule."""

    def test_rule_name(self):
        """Test rule name property."""
        from src.risk.rules import PriceLimitRule

        rule = PriceLimitRule()
        assert rule.name == "PriceLimitRule"

    def test_empty_orders(self):
        """Test with empty order list."""
        from src.risk.rules import PriceLimitRule

        rule = PriceLimitRule()
        passed, results = rule.check([])

        assert passed == []
        assert results == []


class TestRiskManager:
    """Tests for RiskManager."""

    def test_add_rule(self):
        """Test adding rules to manager."""
        from src.risk.rules import RiskManager, PositionLimitRule

        manager = RiskManager()
        rule = PositionLimitRule()

        manager.add_rule(rule)

        assert len(manager.rules) == 1

    def test_check_orders_with_no_rules(self):
        """Test checking orders with no rules."""
        from src.risk.rules import RiskManager, Order

        manager = RiskManager()
        orders = [Order("600519", "BUY", 100)]

        passed, results = manager.check_orders(orders)

        assert len(passed) == 1
        assert results == {}

    def test_check_orders_with_rules(self):
        """Test checking orders with rules."""
        from src.risk.rules import RiskManager, PositionLimitRule, Order

        manager = RiskManager()
        manager.add_rule(PositionLimitRule(
            max_position_ratio=0.10,
            total_value=1_000_000.0,
        ))

        orders = [Order("600519", "BUY", 50, 1800.0)]
        passed, results = manager.check_orders(orders)

        assert len(passed) == 1
        assert "PositionLimitRule" in results

    def test_get_summary(self):
        """Test summary generation."""
        from src.risk.rules import RiskManager, PositionLimitRule, Order

        manager = RiskManager()
        manager.add_rule(PositionLimitRule())

        orders = [Order("600519", "BUY", 50, 1800.0)]
        passed, results = manager.check_orders(orders)
        summary = manager.get_summary(results)

        assert "total_rules" in summary
        assert "by_rule" in summary
        assert summary["total_rules"] == 1


class TestMarketDataFetcher:
    """Tests for MarketDataFetcher."""

    def test_cache_validity(self):
        """Test cache validity check."""
        from src.risk.rules import MarketDataFetcher
        from datetime import datetime

        fetcher = MarketDataFetcher(cache_ttl_seconds=60)

        # No cache time should be invalid
        assert fetcher._is_cache_valid(None) is False

        # Recent cache should be valid
        assert fetcher._is_cache_valid(datetime.now()) is True

    def test_normalize_code_in_stop_sign_rule(self):
        """Test code normalization in StopSignRule."""
        from src.risk.rules import StopSignRule

        rule = StopSignRule()

        # Test various formats
        test_cases = [
            ("600519", "600519"),
            ("SH600519", "600519"),
            ("SZ000001", "000001"),
            ("600519.SH", "600519"),
            ("000001.SZ", "000001"),
        ]

        for input_code, expected in test_cases:
            assert rule._normalize_code(input_code) == expected


class TestApplyRiskRules:
    """Tests for apply_risk_rules function."""

    def test_apply_with_position_limit_only(self):
        """Test applying only position limit rule."""
        from src.risk.rules import apply_risk_rules, Order

        orders = [Order("600519", "BUY", 50, 1800.0)]

        passed, summary = apply_risk_rules(
            orders=orders,
            enable_st_filter=False,
            enable_suspend_filter=False,
            enable_position_limit=True,
            enable_price_limit=False,
            max_position_ratio=0.10,
            total_value=1_000_000.0,
        )

        assert len(passed) == 1
        assert "by_rule" in summary

    def test_apply_with_empty_orders(self):
        """Test applying rules to empty order list."""
        from src.risk.rules import apply_risk_rules

        passed, summary = apply_risk_rules(
            orders=[],
            enable_st_filter=True,
            enable_position_limit=True,
        )

        assert passed == []

