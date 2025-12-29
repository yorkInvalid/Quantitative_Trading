"""
Top-K Dropout 换仓策略模块。

实现基于模型预测分数的 Top-K 选股策略，结合情感分析进行风险过滤：
1. 读取 LightGBM 模型的预测分数，按降序排列
2. 选取 Top 50 的股票作为买入候选
3. 如果持仓股票跌出 Top 100，则卖出；资金释放后买入新的 Top 50
4. 硬性过滤：情感得分低于 -0.5 的股票强制剔除或卖出
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# =============================================================================
# 策略参数配置
# =============================================================================

@dataclass
class StrategyConfig:
    """Top-K Dropout 策略配置参数"""
    
    # Top-K 选股参数
    top_k: int = 50                      # 买入候选数量
    dropout_threshold: int = 100         # 跌出此排名则卖出
    
    # 情感过滤阈值
    sentiment_blacklist_threshold: float = -0.5  # 强制剔除阈值（利空）
    
    # 文件路径
    predictions_path: str = "/app/data/predictions.csv"
    holdings_path: str = "/app/data/holdings.csv"
    output_dir: str = "/app/data"


@dataclass
class TradeSignal:
    """交易信号"""
    instrument: str
    action: str           # "BUY" | "SELL" | "HOLD"
    score: float          # 模型预测分数
    rank: int             # 当前排名
    sentiment_score: float = 0.0
    reason: str = ""      # 交易原因


@dataclass
class RebalanceResult:
    """换仓结果"""
    date: str
    buy_signals: List[TradeSignal] = field(default_factory=list)
    sell_signals: List[TradeSignal] = field(default_factory=list)
    hold_signals: List[TradeSignal] = field(default_factory=list)
    blacklist: List[str] = field(default_factory=list)  # 情感黑名单


# =============================================================================
# Top-K Dropout 策略核心类
# =============================================================================

class TopKDropoutStrategy:
    """
    Top-K Dropout 换仓策略。
    
    策略逻辑：
    1. 每日读取模型预测分数，按降序排名
    2. 新建仓：买入 Top-K 中未持仓的股票
    3. 换仓卖出：持仓股票跌出 Top-N (dropout_threshold) 则卖出
    4. 情感过滤：sentiment_score < blacklist_threshold 的股票强制剔除
    
    Example:
        >>> strategy = TopKDropoutStrategy(top_k=50, dropout_threshold=100)
        >>> result = strategy.rebalance(predictions_df, sentiments_df, holdings)
        >>> print(f"买入: {len(result.buy_signals)}, 卖出: {len(result.sell_signals)}")
    """
    
    def __init__(
        self,
        top_k: int = 50,
        dropout_threshold: int = 100,
        sentiment_blacklist_threshold: float = -0.5,
    ):
        """
        初始化策略。
        
        Args:
            top_k: 买入候选股票数量 (Top 50)
            dropout_threshold: 卖出阈值，跌出此排名则卖出 (Top 100)
            sentiment_blacklist_threshold: 情感黑名单阈值，低于此值强制剔除
        """
        self.top_k = top_k
        self.dropout_threshold = dropout_threshold
        self.sentiment_blacklist_threshold = sentiment_blacklist_threshold
    
    def rank_by_score(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        按预测分数降序排名。
        
        Args:
            predictions: 包含 instrument, score 列的 DataFrame
            
        Returns:
            添加 rank 列的 DataFrame，rank=1 为最高分
        """
        df = predictions.copy()
        
        # 确保必要列存在
        if "score" not in df.columns:
            raise ValueError("predictions DataFrame must have 'score' column")
        if "instrument" not in df.columns:
            raise ValueError("predictions DataFrame must have 'instrument' column")
        
        # 按分数降序排名
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1  # rank 从 1 开始
        
        return df
    
    def get_sentiment_blacklist(
        self,
        sentiments: pd.DataFrame,
    ) -> Set[str]:
        """
        获取情感黑名单（利空股票）。
        
        Args:
            sentiments: 包含 instrument, sentiment_score 列的 DataFrame
            
        Returns:
            需要强制剔除的股票代码集合
        """
        if sentiments is None or sentiments.empty:
            return set()
        
        if "sentiment_score" not in sentiments.columns:
            return set()
        
        # 筛选情感分数低于阈值的股票
        blacklist = sentiments[
            sentiments["sentiment_score"] < self.sentiment_blacklist_threshold
        ]["instrument"].tolist()
        
        return set(blacklist)
    
    def generate_signals(
        self,
        ranked_predictions: pd.DataFrame,
        current_holdings: Set[str],
        blacklist: Set[str],
    ) -> Tuple[List[TradeSignal], List[TradeSignal], List[TradeSignal]]:
        """
        生成交易信号。
        
        Args:
            ranked_predictions: 已排名的预测 DataFrame (需包含 rank 列)
            current_holdings: 当前持仓股票代码集合
            blacklist: 情感黑名单股票代码集合
            
        Returns:
            (buy_signals, sell_signals, hold_signals) 三元组
        """
        buy_signals = []
        sell_signals = []
        hold_signals = []
        
        # 构建 instrument -> (score, rank) 映射
        score_map: Dict[str, Tuple[float, int]] = {}
        for _, row in ranked_predictions.iterrows():
            score_map[row["instrument"]] = (row["score"], row["rank"])
        
        # 获取 Top-K 候选（排除黑名单）
        top_k_candidates = set(
            ranked_predictions[ranked_predictions["rank"] <= self.top_k]["instrument"]
        ) - blacklist
        
        # 获取 Top-N 范围（用于判断是否 dropout）
        top_n_range = set(
            ranked_predictions[ranked_predictions["rank"] <= self.dropout_threshold]["instrument"]
        )
        
        # =====================================================================
        # 1. 处理当前持仓：判断卖出或持有
        # =====================================================================
        for instrument in current_holdings:
            score, rank = score_map.get(instrument, (0.0, 9999))
            
            # 情况 A: 在黑名单中 -> 强制卖出
            if instrument in blacklist:
                sell_signals.append(TradeSignal(
                    instrument=instrument,
                    action="SELL",
                    score=score,
                    rank=rank,
                    reason=f"情感利空 (sentiment < {self.sentiment_blacklist_threshold})"
                ))
            
            # 情况 B: 跌出 Top-N -> 卖出
            elif instrument not in top_n_range:
                sell_signals.append(TradeSignal(
                    instrument=instrument,
                    action="SELL",
                    score=score,
                    rank=rank,
                    reason=f"跌出 Top-{self.dropout_threshold} (rank={rank})"
                ))
            
            # 情况 C: 仍在 Top-N 范围内 -> 持有
            else:
                hold_signals.append(TradeSignal(
                    instrument=instrument,
                    action="HOLD",
                    score=score,
                    rank=rank,
                    reason=f"维持持仓 (rank={rank})"
                ))
        
        # =====================================================================
        # 2. 处理新买入：Top-K 中未持仓且不在黑名单的股票
        # =====================================================================
        new_candidates = top_k_candidates - current_holdings
        
        for instrument in new_candidates:
            score, rank = score_map.get(instrument, (0.0, 9999))
            buy_signals.append(TradeSignal(
                instrument=instrument,
                action="BUY",
                score=score,
                rank=rank,
                reason=f"进入 Top-{self.top_k} (rank={rank})"
            ))
        
        # 按分数降序排列买入信号
        buy_signals.sort(key=lambda x: x.score, reverse=True)
        
        return buy_signals, sell_signals, hold_signals
    
    def rebalance(
        self,
        predictions: pd.DataFrame,
        sentiments: Optional[pd.DataFrame] = None,
        current_holdings: Optional[Set[str]] = None,
        date: Optional[str] = None,
    ) -> RebalanceResult:
        """
        执行换仓逻辑，生成完整的交易信号。
        
        Args:
            predictions: 模型预测 DataFrame (instrument, score)
            sentiments: 情感分析 DataFrame (instrument, sentiment_score)，可选
            current_holdings: 当前持仓股票代码集合，None 表示空仓
            date: 换仓日期，None 则使用当前日期
            
        Returns:
            RebalanceResult 包含买入/卖出/持有信号及黑名单
        """
        if current_holdings is None:
            current_holdings = set()
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Step 1: 按分数排名
        ranked = self.rank_by_score(predictions)
        
        # Step 2: 获取情感黑名单
        blacklist = self.get_sentiment_blacklist(sentiments) if sentiments is not None else set()
        
        # Step 3: 生成交易信号
        buy_signals, sell_signals, hold_signals = self.generate_signals(
            ranked_predictions=ranked,
            current_holdings=current_holdings,
            blacklist=blacklist,
        )
        
        # Step 4: 为信号补充情感分数
        if sentiments is not None and not sentiments.empty:
            sentiment_map = dict(zip(sentiments["instrument"], sentiments["sentiment_score"]))
            for signal in buy_signals + sell_signals + hold_signals:
                signal.sentiment_score = sentiment_map.get(signal.instrument, 0.0)
        
        return RebalanceResult(
            date=date,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            hold_signals=hold_signals,
            blacklist=list(blacklist),
        )
    
    def to_dataframe(self, result: RebalanceResult) -> pd.DataFrame:
        """
        将换仓结果转换为 DataFrame。
        
        Args:
            result: RebalanceResult 对象
            
        Returns:
            包含所有交易信号的 DataFrame
        """
        all_signals = result.buy_signals + result.sell_signals + result.hold_signals
        
        if not all_signals:
            return pd.DataFrame(columns=[
                "date", "instrument", "action", "score", "rank", 
                "sentiment_score", "reason"
            ])
        
        records = []
        for signal in all_signals:
            records.append({
                "date": result.date,
                "instrument": signal.instrument,
                "action": signal.action,
                "score": signal.score,
                "rank": signal.rank,
                "sentiment_score": signal.sentiment_score,
                "reason": signal.reason,
            })
        
        df = pd.DataFrame(records)
        
        # 按 action 和 score 排序
        action_order = {"BUY": 0, "SELL": 1, "HOLD": 2}
        df["action_order"] = df["action"].map(action_order)
        df = df.sort_values(["action_order", "score"], ascending=[True, False])
        df = df.drop(columns=["action_order"])
        
        return df.reset_index(drop=True)


# =============================================================================
# 便捷函数
# =============================================================================

def load_predictions(path: str) -> pd.DataFrame:
    """
    加载预测结果文件。
    
    Args:
        path: predictions.csv 路径
        
    Returns:
        包含 instrument, score 列的 DataFrame
    """
    df = pd.read_csv(path)
    
    # 如果有 datetime 列，取每只股票的最新预测
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").groupby("instrument").last().reset_index()
    
    return df[["instrument", "score"]]


def load_holdings(path: str) -> Set[str]:
    """
    加载当前持仓。
    
    Args:
        path: holdings.csv 路径
        
    Returns:
        持仓股票代码集合
    """
    if not Path(path).exists():
        return set()
    
    df = pd.read_csv(path)
    
    if "instrument" not in df.columns:
        return set()
    
    return set(df["instrument"].tolist())


def save_holdings(holdings: Set[str], path: str) -> None:
    """
    保存持仓到文件。
    
    Args:
        holdings: 持仓股票代码集合
        path: 输出文件路径
    """
    df = pd.DataFrame({"instrument": list(holdings)})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_rebalance(
    predictions_path: str = "/app/data/predictions.csv",
    sentiments_path: Optional[str] = None,
    holdings_path: str = "/app/data/holdings.csv",
    output_dir: str = "/app/data",
    top_k: int = 50,
    dropout_threshold: int = 100,
    sentiment_blacklist_threshold: float = -0.5,
) -> str:
    """
    执行完整的换仓流程。
    
    Args:
        predictions_path: 预测结果文件路径
        sentiments_path: 情感分析结果文件路径（可选）
        holdings_path: 当前持仓文件路径
        output_dir: 输出目录
        top_k: 买入候选数量
        dropout_threshold: 卖出阈值
        sentiment_blacklist_threshold: 情感黑名单阈值
        
    Returns:
        输出的交易信号文件路径
    """
    print("=" * 60)
    print("Top-K Dropout 换仓策略")
    print("=" * 60)
    print(f"  Top-K: {top_k}")
    print(f"  Dropout Threshold: {dropout_threshold}")
    print(f"  Sentiment Blacklist: < {sentiment_blacklist_threshold}")
    print()
    
    # 加载数据
    predictions = load_predictions(predictions_path)
    print(f"加载预测结果: {len(predictions)} 只股票")
    
    sentiments = None
    if sentiments_path and Path(sentiments_path).exists():
        sentiments = pd.read_csv(sentiments_path)
        print(f"加载情感分析: {len(sentiments)} 只股票")
    
    holdings = load_holdings(holdings_path)
    print(f"当前持仓: {len(holdings)} 只股票")
    print()
    
    # 初始化策略并执行换仓
    strategy = TopKDropoutStrategy(
        top_k=top_k,
        dropout_threshold=dropout_threshold,
        sentiment_blacklist_threshold=sentiment_blacklist_threshold,
    )
    
    result = strategy.rebalance(
        predictions=predictions,
        sentiments=sentiments,
        current_holdings=holdings,
    )
    
    # 输出统计
    print("换仓结果:")
    print(f"  买入信号: {len(result.buy_signals)} 只")
    print(f"  卖出信号: {len(result.sell_signals)} 只")
    print(f"  持有信号: {len(result.hold_signals)} 只")
    print(f"  情感黑名单: {len(result.blacklist)} 只")
    
    if result.blacklist:
        print(f"  黑名单股票: {result.blacklist}")
    print()
    
    # 保存结果
    signals_df = strategy.to_dataframe(result)
    today = datetime.now().strftime("%Y%m%d")
    output_path = Path(output_dir) / f"trade_signals_{today}.csv"
    signals_df.to_csv(output_path, index=False)
    print(f"交易信号已保存: {output_path}")
    
    # 更新持仓
    new_holdings = set()
    for signal in result.hold_signals:
        new_holdings.add(signal.instrument)
    for signal in result.buy_signals:
        new_holdings.add(signal.instrument)
    
    save_holdings(new_holdings, holdings_path)
    print(f"持仓已更新: {len(new_holdings)} 只股票")
    
    return str(output_path)


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    # 示例：使用默认参数运行换仓
    run_rebalance()

