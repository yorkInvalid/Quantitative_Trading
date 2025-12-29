"""
Main entry point for the quantitative trading system.

Orchestrates the full pipeline:
1. ETL: Download stock data and convert to Qlib format
2. Model: Train LightGBM and generate predictions
3. NLP: Analyze news sentiment for top stocks
4. Strategy: Filter and generate final buy list
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Default paths
DATA_DIR = "/app/data"
CSV_SOURCE_DIR = f"{DATA_DIR}/csv_source"
QLIB_BIN_DIR = f"{DATA_DIR}/qlib_bin"
PREDICTIONS_PATH = f"{DATA_DIR}/predictions.csv"
CONFIG_PATH = "/app/config/workflow.yaml"

# Strategy parameters
TOP_N_STOCKS = 50
SENTIMENT_THRESHOLD = -0.2  # Filter out stocks with sentiment < this value


def run_etl(symbols: Optional[List[str]] = None) -> None:
    """
    Run ETL pipeline: download data and convert to Qlib format.

    Args:
        symbols: List of stock symbols to download. If None, uses default CSI300.
    """
    from src.etl.downloader import download_stock_history
    from src.etl.converter import convert_csv_to_qlib

    # Default symbols (sample from CSI300)
    if symbols is None:
        symbols = [
            "600519",  # 贵州茅台
            "601318",  # 中国平安
            "600036",  # 招商银行
            "000858",  # 五粮液
            "002415",  # 海康威视
        ]

    print("=" * 60)
    print("Stage 1: ETL - Downloading stock data...")
    print("=" * 60)

    # Download historical data
    download_stock_history(symbols, out_dir=CSV_SOURCE_DIR)

    print("\nStage 1.2: Converting to Qlib format...")
    convert_csv_to_qlib(csv_dir=CSV_SOURCE_DIR, bin_dir=QLIB_BIN_DIR)

    print("ETL completed.\n")


def run_model() -> pd.DataFrame:
    """
    Run model training and prediction.

    Returns:
        DataFrame with predictions (instrument, datetime, score).
    """
    from src.model.trainer import run_workflow

    print("=" * 60)
    print("Stage 2: Model - Training and prediction...")
    print("=" * 60)

    pred_df = run_workflow(config_path=CONFIG_PATH, output_path=PREDICTIONS_PATH)

    print("Model training and prediction completed.\n")
    return pred_df


def get_top_stocks(
    predictions_path: str = PREDICTIONS_PATH, top_n: int = TOP_N_STOCKS
) -> pd.DataFrame:
    """
    Get top N stocks by prediction score.

    Args:
        predictions_path: Path to predictions CSV.
        top_n: Number of top stocks to select.

    Returns:
        DataFrame with top stocks.
    """
    df = pd.read_csv(predictions_path)

    # Get latest prediction for each instrument
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        latest_df = (
            df.sort_values("datetime").groupby("instrument").last().reset_index()
        )
    else:
        latest_df = df

    # Sort by score and get top N
    top_stocks = latest_df.nlargest(top_n, "score")

    return top_stocks


def fetch_stock_news(symbol: str) -> List[str]:
    """
    Fetch recent news for a stock using AkShare.

    Args:
        symbol: Stock symbol (e.g., '600519').

    Returns:
        List of news headlines/content.
    """
    try:
        import akshare as ak

        # stock_news_em returns news for a given stock
        news_df = ak.stock_news_em(symbol=symbol)

        if news_df is None or news_df.empty:
            return []

        # Extract news titles/content
        if "新闻标题" in news_df.columns:
            news_list = news_df["新闻标题"].tolist()
        elif "title" in news_df.columns:
            news_list = news_df["title"].tolist()
        else:
            # Try first text column
            news_list = news_df.iloc[:, 0].tolist()

        # Limit to recent news (e.g., top 10)
        return news_list[:10]

    except Exception as e:
        print(f"Warning: Failed to fetch news for {symbol}: {e}")
        return []


def run_nlp_analysis(top_stocks: pd.DataFrame) -> pd.DataFrame:
    """
    Run NLP sentiment analysis on news for top stocks.

    Args:
        top_stocks: DataFrame with top stocks.

    Returns:
        DataFrame with added sentiment_score column.
    """
    from src.nlp.sentiment import SentimentAnalyzer

    print("=" * 60)
    print("Stage 3: NLP - Analyzing news sentiment...")
    print("=" * 60)

    analyzer = SentimentAnalyzer()

    sentiment_scores = []

    for _, row in top_stocks.iterrows():
        symbol = row["instrument"]

        # Clean symbol (remove exchange prefix if present)
        clean_symbol = symbol.replace("SH", "").replace("SZ", "").replace(".", "")

        # Fetch news
        news_list = fetch_stock_news(clean_symbol)

        if not news_list:
            # No news = neutral sentiment
            avg_score = 0.0
        else:
            # Analyze sentiment for all news
            scores = analyzer.predict(news_list)
            avg_score = sum(scores) / len(scores) if scores else 0.0

        sentiment_scores.append(avg_score)
        print(
            f"  {symbol}: {len(news_list)} news items, avg sentiment = {avg_score:.3f}"
        )

    top_stocks = top_stocks.copy()
    top_stocks["sentiment_score"] = sentiment_scores

    print("NLP analysis completed.\n")
    return top_stocks


def apply_strategy(
    stocks_with_sentiment: pd.DataFrame,
    sentiment_threshold: float = SENTIMENT_THRESHOLD,
) -> pd.DataFrame:
    """
    Apply strategy filter: keep stocks with sentiment >= threshold.

    Args:
        stocks_with_sentiment: DataFrame with sentiment scores.
        sentiment_threshold: Minimum sentiment score to keep.

    Returns:
        Filtered DataFrame.
    """
    print("=" * 60)
    print("Stage 4: Strategy - Filtering by sentiment...")
    print("=" * 60)

    # Filter by sentiment threshold
    filtered = stocks_with_sentiment[
        stocks_with_sentiment["sentiment_score"] >= sentiment_threshold
    ].copy()

    print(f"  Original: {len(stocks_with_sentiment)} stocks")
    print(
        f"  After filter (sentiment >= {sentiment_threshold}): {len(filtered)} stocks"
    )

    # Sort by model score (descending)
    filtered = filtered.sort_values("score", ascending=False)

    print("Strategy filter completed.\n")
    return filtered


def save_final_list(final_stocks: pd.DataFrame, output_dir: str = DATA_DIR) -> str:
    """
    Save final buy list to CSV.

    Args:
        final_stocks: DataFrame with final stock recommendations.
        output_dir: Directory to save the file.

    Returns:
        Path to the saved file.
    """
    today = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join(output_dir, f"final_buy_list_{today}.csv")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    final_stocks.to_csv(output_path, index=False)

    print(f"Final buy list saved to: {output_path}")
    print(f"Total recommendations: {len(final_stocks)}")

    return output_path


def main(
    symbols: Optional[List[str]] = None,
    skip_etl: bool = False,
    skip_model: bool = False,
) -> str:
    """
    Run the full quantitative trading pipeline.

    Args:
        symbols: Stock symbols to process. None for defaults.
        skip_etl: Skip ETL stage (use existing data).
        skip_model: Skip model training (use existing predictions).

    Returns:
        Path to the final buy list CSV.
    """
    print("\n" + "=" * 60)
    print("QUANTITATIVE TRADING SYSTEM - STARTING PIPELINE")
    print("=" * 60 + "\n")

    # Stage 1: ETL
    if not skip_etl:
        run_etl(symbols)
    else:
        print("Skipping ETL stage (using existing data).\n")

    # Stage 2: Model
    if not skip_model:
        run_model()
    else:
        print("Skipping model stage (using existing predictions).\n")

    # Stage 3: Get top stocks
    print("=" * 60)
    print("Selecting top stocks from predictions...")
    print("=" * 60)
    top_stocks = get_top_stocks()
    print(f"Selected top {len(top_stocks)} stocks.\n")

    # Stage 4: NLP analysis
    stocks_with_sentiment = run_nlp_analysis(top_stocks)

    # Stage 5: Apply strategy filter
    final_stocks = apply_strategy(stocks_with_sentiment)

    # Stage 6: Save results
    output_path = save_final_list(final_stocks)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")

    return output_path


if __name__ == "__main__":
    main()
