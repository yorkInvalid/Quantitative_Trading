"""
Integration tests for the full quantitative trading pipeline.
"""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestMainPipeline:
    """Integration tests for src/main.py pipeline."""

    @pytest.fixture
    def mock_predictions(self, tmp_path):
        """Create mock predictions CSV."""
        pred_df = pd.DataFrame(
            {
                "datetime": ["2024-01-01"] * 5,
                "instrument": [
                    "SH600519",
                    "SH601318",
                    "SH600036",
                    "SH000858",
                    "SH002415",
                ],
                "score": [0.9, 0.8, 0.7, 0.6, 0.5],
            }
        )
        pred_path = tmp_path / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        return str(pred_path)

    def test_get_top_stocks(self, mock_predictions):
        """Test that get_top_stocks returns correct number of stocks."""
        from src.main import get_top_stocks

        top_stocks = get_top_stocks(mock_predictions, top_n=3)

        assert len(top_stocks) == 3
        assert "score" in top_stocks.columns
        assert "instrument" in top_stocks.columns
        # Should be sorted by score descending
        assert top_stocks.iloc[0]["score"] == 0.9

    def test_apply_strategy_filters_by_sentiment(self):
        """Test that apply_strategy correctly filters by sentiment threshold."""
        from src.main import apply_strategy

        stocks = pd.DataFrame(
            {
                "instrument": ["A", "B", "C", "D"],
                "score": [0.9, 0.8, 0.7, 0.6],
                "sentiment_score": [0.5, -0.1, -0.3, 0.2],
            }
        )

        # Threshold = -0.2, should filter out C (sentiment = -0.3)
        filtered = apply_strategy(stocks, sentiment_threshold=-0.2)

        assert len(filtered) == 3
        assert "C" not in filtered["instrument"].values
        assert "A" in filtered["instrument"].values
        assert "B" in filtered["instrument"].values
        assert "D" in filtered["instrument"].values

    def test_blacklist_logic_high_score_bad_sentiment(self):
        """
        Test blacklist logic: high model score but bad sentiment should be filtered out.
        """
        from src.main import apply_strategy

        # Stock A has highest model score but worst sentiment
        stocks = pd.DataFrame(
            {
                "instrument": ["A", "B", "C"],
                "score": [0.99, 0.5, 0.3],  # A has best model score
                "sentiment_score": [-0.8, 0.5, 0.3],  # A has worst sentiment
            }
        )

        filtered = apply_strategy(stocks, sentiment_threshold=-0.2)

        # A should be filtered out despite high model score
        assert "A" not in filtered["instrument"].values
        assert "B" in filtered["instrument"].values
        assert "C" in filtered["instrument"].values

    def test_save_final_list_creates_csv(self, tmp_path):
        """Test that save_final_list creates a properly formatted CSV."""
        from src.main import save_final_list

        final_stocks = pd.DataFrame(
            {
                "instrument": ["SH600519", "SH601318"],
                "score": [0.9, 0.8],
                "sentiment_score": [0.5, 0.3],
            }
        )

        output_path = save_final_list(final_stocks, output_dir=str(tmp_path))

        # Check file exists
        assert os.path.exists(output_path)

        # Check filename format
        today = datetime.now().strftime("%Y%m%d")
        assert f"final_buy_list_{today}.csv" in output_path

        # Check content
        saved_df = pd.read_csv(output_path)
        assert len(saved_df) == 2
        assert "instrument" in saved_df.columns
        assert "score" in saved_df.columns
        assert "sentiment_score" in saved_df.columns


class TestMockedPipeline:
    """Tests with mocked ETL and NLP operations."""

    @patch("src.main.run_etl")
    @patch("src.main.run_model")
    @patch("src.main.run_nlp_analysis")
    def test_full_pipeline_mocked(
        self,
        mock_nlp,
        mock_model,
        mock_etl,
        tmp_path,
    ):
        """Test full pipeline with mocked heavy operations."""
        from src.main import main, get_top_stocks, apply_strategy, save_final_list

        # Setup mock predictions
        pred_df = pd.DataFrame(
            {
                "datetime": ["2024-01-01"] * 3,
                "instrument": ["SH600519", "SH601318", "SH600036"],
                "score": [0.9, 0.8, 0.7],
            }
        )
        pred_path = tmp_path / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        # Mock NLP to return stocks with sentiment
        def mock_nlp_fn(stocks):
            stocks = stocks.copy()
            stocks["sentiment_score"] = [0.5, -0.5, 0.2]  # 601318 will be filtered
            return stocks

        mock_nlp.side_effect = mock_nlp_fn

        # Run pipeline components manually
        top_stocks = get_top_stocks(str(pred_path), top_n=3)
        stocks_with_sentiment = mock_nlp_fn(top_stocks)
        final_stocks = apply_strategy(stocks_with_sentiment)
        output_path = save_final_list(final_stocks, output_dir=str(tmp_path))

        # Verify
        assert os.path.exists(output_path)
        result_df = pd.read_csv(output_path)

        # 601318 should be filtered (sentiment = -0.5 < -0.2)
        assert "SH601318" not in result_df["instrument"].values
        assert "SH600519" in result_df["instrument"].values
        assert "SH600036" in result_df["instrument"].values

    def test_empty_news_returns_neutral_sentiment(self):
        """Test that stocks with no news get neutral sentiment (0)."""
        from src.main import fetch_stock_news

        # Mock akshare to return empty DataFrame
        with patch("akshare.stock_news_em") as mock_news:
            mock_news.return_value = pd.DataFrame()

            news = fetch_stock_news("600519")
            assert news == []

    def test_nlp_analysis_handles_no_news(self):
        """Test NLP analysis assigns 0 score when no news available."""
        # Mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = []

        stocks = pd.DataFrame(
            {
                "instrument": ["SH600519"],
                "score": [0.9],
            }
        )

        # Mock fetch_stock_news to return empty and SentimentAnalyzer class
        with patch("src.main.fetch_stock_news", return_value=[]):
            with patch("src.nlp.sentiment.SentimentAnalyzer", return_value=mock_analyzer):
                from src.main import run_nlp_analysis

                result = run_nlp_analysis(stocks)

        # Should have sentiment_score = 0 for no news
        assert result.iloc[0]["sentiment_score"] == 0.0


class TestOutputFormat:
    """Tests for output file format validation."""

    def test_final_csv_has_required_columns(self, tmp_path):
        """Test that final CSV has all required columns."""
        from src.main import save_final_list

        final_stocks = pd.DataFrame(
            {
                "instrument": ["SH600519"],
                "score": [0.9],
                "sentiment_score": [0.5],
                "datetime": ["2024-01-01"],
            }
        )

        output_path = save_final_list(final_stocks, output_dir=str(tmp_path))
        saved_df = pd.read_csv(output_path)

        required_columns = {"instrument", "score", "sentiment_score"}
        assert required_columns.issubset(set(saved_df.columns))

    def test_final_csv_sorted_by_score(self, tmp_path):
        """Test that final CSV is sorted by score descending."""
        from src.main import apply_strategy, save_final_list

        stocks = pd.DataFrame(
            {
                "instrument": ["A", "B", "C"],
                "score": [0.5, 0.9, 0.7],
                "sentiment_score": [0.3, 0.3, 0.3],
            }
        )

        filtered = apply_strategy(stocks)
        output_path = save_final_list(filtered, output_dir=str(tmp_path))
        saved_df = pd.read_csv(output_path)

        # Should be sorted by score descending
        assert saved_df.iloc[0]["instrument"] == "B"  # score 0.9
        assert saved_df.iloc[1]["instrument"] == "C"  # score 0.7
        assert saved_df.iloc[2]["instrument"] == "A"  # score 0.5
