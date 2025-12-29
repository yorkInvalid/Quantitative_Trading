"""
Tests for Top-K Dropout strategy module.
"""

import pandas as pd
import pytest

from src.strategy.topk_dropout import (
    TopKDropoutStrategy,
    TradeSignal,
    RebalanceResult,
    load_predictions,
    load_holdings,
    save_holdings,
)


class TestTopKDropoutStrategy:
    """Tests for TopKDropoutStrategy class."""

    @pytest.fixture
    def strategy(self):
        """Create a strategy instance for testing."""
        return TopKDropoutStrategy(
            top_k=3,
            dropout_threshold=5,
            sentiment_blacklist_threshold=-0.5,
        )

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions DataFrame."""
        return pd.DataFrame({
            "instrument": ["A", "B", "C", "D", "E", "F", "G"],
            "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        })

    @pytest.fixture
    def sample_sentiments(self):
        """Create sample sentiments DataFrame."""
        return pd.DataFrame({
            "instrument": ["A", "B", "C", "D", "E", "F", "G"],
            "sentiment_score": [0.5, -0.6, 0.3, 0.1, -0.7, 0.2, 0.0],
        })

    def test_rank_by_score(self, strategy, sample_predictions):
        """Test that ranking is correct (descending by score)."""
        ranked = strategy.rank_by_score(sample_predictions)

        assert "rank" in ranked.columns
        assert ranked.iloc[0]["rank"] == 1
        assert ranked.iloc[0]["instrument"] == "A"
        assert ranked.iloc[0]["score"] == 0.9
        assert ranked.iloc[-1]["rank"] == 7

    def test_get_sentiment_blacklist(self, strategy, sample_sentiments):
        """Test that blacklist correctly identifies low sentiment stocks."""
        blacklist = strategy.get_sentiment_blacklist(sample_sentiments)

        # B (-0.6) and E (-0.7) should be in blacklist (< -0.5)
        assert "B" in blacklist
        assert "E" in blacklist
        assert "A" not in blacklist
        assert "C" not in blacklist

    def test_empty_sentiment_returns_empty_blacklist(self, strategy):
        """Test that empty sentiments returns empty blacklist."""
        blacklist = strategy.get_sentiment_blacklist(None)
        assert blacklist == set()

        blacklist = strategy.get_sentiment_blacklist(pd.DataFrame())
        assert blacklist == set()

    def test_generate_buy_signals_for_empty_holdings(self, strategy, sample_predictions):
        """Test buy signals when starting with empty portfolio."""
        ranked = strategy.rank_by_score(sample_predictions)
        buy, sell, hold = strategy.generate_signals(
            ranked_predictions=ranked,
            current_holdings=set(),
            blacklist=set(),
        )

        # Should buy top 3: A, B, C
        assert len(buy) == 3
        assert len(sell) == 0
        assert len(hold) == 0

        instruments = {s.instrument for s in buy}
        assert instruments == {"A", "B", "C"}

    def test_generate_sell_signals_for_dropout(self, strategy, sample_predictions):
        """Test sell signals when stocks drop out of top N."""
        ranked = strategy.rank_by_score(sample_predictions)

        # Holdings include G which is rank 7 (outside top 5)
        buy, sell, hold = strategy.generate_signals(
            ranked_predictions=ranked,
            current_holdings={"A", "G"},
            blacklist=set(),
        )

        # G should be sold (rank 7 > dropout_threshold 5)
        assert len(sell) == 1
        assert sell[0].instrument == "G"
        assert "跌出" in sell[0].reason

        # A should be held
        assert len(hold) == 1
        assert hold[0].instrument == "A"

    def test_generate_sell_signals_for_blacklist(self, strategy, sample_predictions):
        """Test sell signals when holdings are in blacklist."""
        ranked = strategy.rank_by_score(sample_predictions)

        # Holdings include B which is in blacklist
        buy, sell, hold = strategy.generate_signals(
            ranked_predictions=ranked,
            current_holdings={"A", "B"},
            blacklist={"B"},
        )

        # B should be sold due to blacklist
        assert len(sell) == 1
        assert sell[0].instrument == "B"
        assert "情感利空" in sell[0].reason

        # A should be held
        assert len(hold) == 1
        assert hold[0].instrument == "A"

    def test_blacklist_excludes_from_buy(self, strategy, sample_predictions):
        """Test that blacklisted stocks are not bought."""
        ranked = strategy.rank_by_score(sample_predictions)

        # B is in top 3 but also in blacklist
        buy, sell, hold = strategy.generate_signals(
            ranked_predictions=ranked,
            current_holdings=set(),
            blacklist={"B"},
        )

        # Should buy A, C (not B), and D fills the slot
        instruments = {s.instrument for s in buy}
        assert "B" not in instruments
        assert "A" in instruments
        assert "C" in instruments

    def test_rebalance_full_flow(self, strategy, sample_predictions, sample_sentiments):
        """Test full rebalance flow."""
        result = strategy.rebalance(
            predictions=sample_predictions,
            sentiments=sample_sentiments,
            current_holdings={"A", "E"},  # E is in blacklist
            date="2024-01-01",
        )

        assert isinstance(result, RebalanceResult)
        assert result.date == "2024-01-01"

        # E should be sold (in blacklist)
        sell_instruments = {s.instrument for s in result.sell_signals}
        assert "E" in sell_instruments

        # A should be held
        hold_instruments = {s.instrument for s in result.hold_signals}
        assert "A" in hold_instruments

        # Blacklist should contain B and E
        assert "B" in result.blacklist
        assert "E" in result.blacklist

    def test_to_dataframe(self, strategy, sample_predictions, sample_sentiments):
        """Test conversion to DataFrame."""
        result = strategy.rebalance(
            predictions=sample_predictions,
            sentiments=sample_sentiments,
            current_holdings=set(),
        )

        df = strategy.to_dataframe(result)

        assert isinstance(df, pd.DataFrame)
        assert "instrument" in df.columns
        assert "action" in df.columns
        assert "score" in df.columns
        assert "rank" in df.columns
        assert "sentiment_score" in df.columns
        assert "reason" in df.columns


class TestStrategyHelperFunctions:
    """Tests for helper functions."""

    def test_load_predictions(self, tmp_path):
        """Test loading predictions from CSV."""
        pred_df = pd.DataFrame({
            "datetime": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "instrument": ["A", "A", "B"],
            "score": [0.5, 0.8, 0.6],
        })
        pred_path = tmp_path / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        loaded = load_predictions(str(pred_path))

        # Should get latest prediction for each instrument
        assert len(loaded) == 2
        assert "instrument" in loaded.columns
        assert "score" in loaded.columns

        # A should have score 0.8 (latest)
        a_score = loaded[loaded["instrument"] == "A"]["score"].values[0]
        assert a_score == 0.8

    def test_load_holdings_file_not_exists(self, tmp_path):
        """Test loading holdings when file doesn't exist."""
        holdings = load_holdings(str(tmp_path / "nonexistent.csv"))
        assert holdings == set()

    def test_save_and_load_holdings(self, tmp_path):
        """Test saving and loading holdings."""
        holdings = {"A", "B", "C"}
        path = str(tmp_path / "holdings.csv")

        save_holdings(holdings, path)
        loaded = load_holdings(path)

        assert loaded == holdings


class TestStrategyEdgeCases:
    """Tests for edge cases."""

    def test_empty_predictions(self):
        """Test with empty predictions."""
        strategy = TopKDropoutStrategy(top_k=3, dropout_threshold=5)

        with pytest.raises(Exception):
            strategy.rank_by_score(pd.DataFrame())

    def test_missing_score_column(self):
        """Test with missing score column."""
        strategy = TopKDropoutStrategy(top_k=3, dropout_threshold=5)

        df = pd.DataFrame({"instrument": ["A", "B"]})

        with pytest.raises(ValueError, match="score"):
            strategy.rank_by_score(df)

    def test_all_stocks_in_blacklist(self):
        """Test when all top stocks are in blacklist."""
        strategy = TopKDropoutStrategy(
            top_k=2,
            dropout_threshold=5,
            sentiment_blacklist_threshold=-0.5,
        )

        predictions = pd.DataFrame({
            "instrument": ["A", "B", "C"],
            "score": [0.9, 0.8, 0.7],
        })

        sentiments = pd.DataFrame({
            "instrument": ["A", "B", "C"],
            "sentiment_score": [-0.6, -0.7, -0.8],  # All below threshold
        })

        result = strategy.rebalance(
            predictions=predictions,
            sentiments=sentiments,
            current_holdings=set(),
        )

        # No buy signals since all are in blacklist
        assert len(result.buy_signals) == 0
        assert len(result.blacklist) == 3

    def test_high_score_bad_sentiment_not_bought(self):
        """Test that high score stock with bad sentiment is not bought."""
        strategy = TopKDropoutStrategy(
            top_k=3,
            dropout_threshold=5,
            sentiment_blacklist_threshold=-0.5,
        )

        predictions = pd.DataFrame({
            "instrument": ["A", "B", "C", "D"],
            "score": [0.99, 0.8, 0.7, 0.6],  # A has highest score
        })

        sentiments = pd.DataFrame({
            "instrument": ["A", "B", "C", "D"],
            "sentiment_score": [-0.9, 0.5, 0.3, 0.2],  # A has worst sentiment
        })

        result = strategy.rebalance(
            predictions=predictions,
            sentiments=sentiments,
            current_holdings=set(),
        )

        # A should not be in buy list despite highest score
        buy_instruments = {s.instrument for s in result.buy_signals}
        assert "A" not in buy_instruments
        assert "A" in result.blacklist

    def test_holding_in_blacklist_forced_sell(self):
        """Test that holding in blacklist is forced to sell."""
        strategy = TopKDropoutStrategy(
            top_k=3,
            dropout_threshold=5,
            sentiment_blacklist_threshold=-0.5,
        )

        predictions = pd.DataFrame({
            "instrument": ["A", "B", "C"],
            "score": [0.9, 0.8, 0.7],
        })

        sentiments = pd.DataFrame({
            "instrument": ["A", "B", "C"],
            "sentiment_score": [-0.6, 0.5, 0.3],  # A is in blacklist
        })

        # Currently holding A (rank 1, but in blacklist)
        result = strategy.rebalance(
            predictions=predictions,
            sentiments=sentiments,
            current_holdings={"A"},
        )

        # A should be sold despite being rank 1
        sell_instruments = {s.instrument for s in result.sell_signals}
        assert "A" in sell_instruments

        # B and C should be bought
        buy_instruments = {s.instrument for s in result.buy_signals}
        assert "B" in buy_instruments
        assert "C" in buy_instruments

