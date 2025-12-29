"""
Backtest Runner Module.

使用 Qlib 回测框架执行策略回测，生成回测报告。

主要功能：
1. 加载模型预测结果
2. 配置回测执行器
3. 运行回测
4. 生成分析报告（夏普比率、最大回撤等）
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy


# =============================================================================
# 默认配置
# =============================================================================

DEFAULT_PREDICTIONS_PATH = "/app/data/predictions.csv"
DEFAULT_REPORT_DIR = "/app/data/backtest_reports"
DEFAULT_PROVIDER_URI = "/app/data/qlib_bin"


# =============================================================================
# 配置数据类
# =============================================================================

@dataclass
class BacktestConfig:
    """回测配置"""
    
    # 时间范围
    start_time: str = "2023-01-01"
    end_time: str = "2023-12-31"
    
    # 策略参数
    topk: int = 50
    n_drop: int = 100
    
    # 交易成本
    open_cost: float = 0.0002      # 买入佣金 0.02%
    close_cost: float = 0.0012     # 卖出佣金+印花税 0.12%
    min_cost: float = 5.0          # 最低佣金 5 元
    
    # 交易限制
    limit_threshold: float = 0.095  # 涨跌停限制 9.5%
    deal_price: str = "close"       # 成交价格 ("close" 或 "open")
    
    # 资金管理
    init_cash: float = 1_000_000.0  # 初始资金 100 万
    risk_degree: float = 0.95       # 仓位比例
    
    # 股票池
    benchmark: str = "SH000300"     # 基准指数 (沪深300)
    
    # 文件路径
    predictions_path: str = DEFAULT_PREDICTIONS_PATH
    report_dir: str = DEFAULT_REPORT_DIR


# =============================================================================
# 预测数据加载
# =============================================================================

def load_predictions(
    predictions_path: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pd.DataFrame:
    """
    加载预测结果文件。
    
    Args:
        predictions_path: 预测文件路径
        start_time: 开始时间（可选）
        end_time: 结束时间（可选）
        
    Returns:
        预测 DataFrame，包含 datetime, instrument, score 列
    """
    df = pd.read_csv(predictions_path)
    
    # 标准化列名
    if "datetime" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "datetime"})
    
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    # 时间过滤
    if start_time:
        df = df[df["datetime"] >= pd.to_datetime(start_time)]
    if end_time:
        df = df[df["datetime"] <= pd.to_datetime(end_time)]
    
    # 设置多级索引
    df = df.set_index(["datetime", "instrument"])
    
    return df


def predictions_to_signal(predictions: pd.DataFrame) -> pd.Series:
    """
    将预测 DataFrame 转换为 Qlib 信号格式。
    
    Args:
        predictions: 预测 DataFrame (index=[datetime, instrument])
        
    Returns:
        预测分数 Series
    """
    if "score" in predictions.columns:
        return predictions["score"]
    else:
        # 假设最后一列是分数
        return predictions.iloc[:, -1]


# =============================================================================
# 回测执行
# =============================================================================

def create_executor_config(config: BacktestConfig) -> Dict[str, Any]:
    """
    创建回测执行器配置。
    
    Args:
        config: 回测配置
        
    Returns:
        执行器配置字典
    """
    return {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
    }


def create_exchange_config(config: BacktestConfig) -> Dict[str, Any]:
    """
    创建交易所配置。
    
    Args:
        config: 回测配置
        
    Returns:
        交易所配置字典
    """
    return {
        "limit_threshold": config.limit_threshold,
        "deal_price": config.deal_price,
        "open_cost": config.open_cost,
        "close_cost": config.close_cost,
        "min_cost": config.min_cost,
    }


def run_backtest(
    config: Optional[BacktestConfig] = None,
    predictions_path: Optional[str] = None,
    provider_uri: str = DEFAULT_PROVIDER_URI,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    运行回测。
    
    Args:
        config: 回测配置，None 则使用默认配置
        predictions_path: 预测文件路径，覆盖 config 中的设置
        provider_uri: Qlib 数据路径
        
    Returns:
        (portfolio_df, analysis_result) 元组
        - portfolio_df: 组合每日收益 DataFrame
        - analysis_result: 分析结果字典
    """
    if config is None:
        config = BacktestConfig()
    
    if predictions_path:
        config.predictions_path = predictions_path
    
    print("\n" + "=" * 60)
    print("BACKTEST - STARTING")
    print("=" * 60)
    print(f"  Period: {config.start_time} ~ {config.end_time}")
    print(f"  Top-K: {config.topk}, N-Drop: {config.n_drop}")
    print(f"  Init Cash: {config.init_cash:,.0f}")
    print(f"  Predictions: {config.predictions_path}")
    
    # 初始化 Qlib
    try:
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        print(f"  Qlib initialized: {provider_uri}")
    except Exception as e:
        print(f"  Qlib already initialized or error: {e}")
    
    # 加载预测
    print("\nLoading predictions...")
    predictions = load_predictions(
        config.predictions_path,
        start_time=config.start_time,
        end_time=config.end_time,
    )
    pred_signal = predictions_to_signal(predictions)
    print(f"  Loaded {len(pred_signal)} predictions")
    
    # 创建策略
    print("\nCreating strategy...")
    strategy = TopkDropoutStrategy(
        signal=pred_signal,
        topk=config.topk,
        n_drop=config.n_drop,
    )
    
    # 创建执行器配置
    executor_config = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
    }
    
    # 创建交易配置
    exchange_config = create_exchange_config(config)
    
    # 运行回测
    print("\nRunning backtest...")
    portfolio_metric, indicator_dict = backtest(
        pred=pred_signal,
        strategy=strategy,
        start_time=config.start_time,
        end_time=config.end_time,
        account=config.init_cash,
        benchmark=config.benchmark,
        exchange_kwargs=exchange_config,
        pos_type="Position",
    )
    
    print("Backtest completed.")
    
    # 分析结果
    print("\nAnalyzing results...")
    analysis_result = analyze_backtest_result(
        portfolio_metric=portfolio_metric,
        indicator_dict=indicator_dict,
        config=config,
    )
    
    return portfolio_metric, analysis_result


# =============================================================================
# 结果分析
# =============================================================================

def analyze_backtest_result(
    portfolio_metric: pd.DataFrame,
    indicator_dict: Dict[str, Any],
    config: BacktestConfig,
) -> Dict[str, Any]:
    """
    分析回测结果。
    
    Args:
        portfolio_metric: 组合指标 DataFrame
        indicator_dict: 指标字典
        config: 回测配置
        
    Returns:
        分析结果字典，包含夏普比率、最大回撤等
    """
    result = {}
    
    # 计算收益率
    if "return" in portfolio_metric.columns:
        returns = portfolio_metric["return"]
    elif "account" in portfolio_metric.columns:
        returns = portfolio_metric["account"].pct_change().dropna()
    else:
        # 尝试从第一列计算
        returns = portfolio_metric.iloc[:, 0].pct_change().dropna()
    
    # 基本统计
    result["total_return"] = float((1 + returns).prod() - 1)
    result["annual_return"] = float(result["total_return"] * 252 / len(returns)) if len(returns) > 0 else 0.0
    result["volatility"] = float(returns.std() * np.sqrt(252))
    
    # 夏普比率 (假设无风险利率为 3%)
    risk_free_rate = 0.03
    excess_return = result["annual_return"] - risk_free_rate
    result["sharpe_ratio"] = float(excess_return / result["volatility"]) if result["volatility"] > 0 else 0.0
    
    # 最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    result["max_drawdown"] = float(drawdown.min())
    
    # Calmar 比率
    result["calmar_ratio"] = float(result["annual_return"] / abs(result["max_drawdown"])) if result["max_drawdown"] != 0 else 0.0
    
    # 胜率
    result["win_rate"] = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0
    
    # 交易统计
    result["trading_days"] = len(returns)
    
    return result


def generate_report(
    portfolio_metric: pd.DataFrame,
    analysis_result: Dict[str, Any],
    config: BacktestConfig,
    save_path: Optional[str] = None,
) -> str:
    """
    生成回测报告。
    
    Args:
        portfolio_metric: 组合指标 DataFrame
        analysis_result: 分析结果字典
        config: 回测配置
        save_path: 保存路径（可选）
        
    Returns:
        报告文本
    """
    report_lines = [
        "=" * 60,
        "BACKTEST REPORT",
        "=" * 60,
        "",
        "【回测配置】",
        f"  回测期间: {config.start_time} ~ {config.end_time}",
        f"  初始资金: {config.init_cash:,.0f} 元",
        f"  策略参数: Top-{config.topk}, N-Drop-{config.n_drop}",
        f"  仓位比例: {config.risk_degree * 100:.0f}%",
        "",
        "【交易成本】",
        f"  买入佣金: {config.open_cost * 100:.2f}%",
        f"  卖出成本: {config.close_cost * 100:.2f}% (含印花税)",
        f"  最低佣金: {config.min_cost:.0f} 元",
        "",
        "【收益指标】",
        f"  总收益率: {analysis_result['total_return'] * 100:.2f}%",
        f"  年化收益: {analysis_result['annual_return'] * 100:.2f}%",
        f"  年化波动: {analysis_result['volatility'] * 100:.2f}%",
        "",
        "【风险指标】",
        f"  夏普比率: {analysis_result['sharpe_ratio']:.2f}",
        f"  最大回撤: {analysis_result['max_drawdown'] * 100:.2f}%",
        f"  Calmar比率: {analysis_result['calmar_ratio']:.2f}",
        "",
        "【交易统计】",
        f"  交易天数: {analysis_result['trading_days']}",
        f"  日胜率: {analysis_result['win_rate'] * 100:.2f}%",
        "",
        "=" * 60,
    ]
    
    report = "\n".join(report_lines)
    
    # 打印报告
    print("\n" + report)
    
    # 保存报告
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to: {save_path}")
    
    return report


def save_backtest_results(
    portfolio_metric: pd.DataFrame,
    analysis_result: Dict[str, Any],
    config: BacktestConfig,
    report_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    保存回测结果到文件。
    
    Args:
        portfolio_metric: 组合指标 DataFrame
        analysis_result: 分析结果字典
        config: 回测配置
        report_dir: 报告目录
        
    Returns:
        保存的文件路径字典
    """
    if report_dir is None:
        report_dir = config.report_dir
    
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {}
    
    # 保存组合指标
    portfolio_path = os.path.join(report_dir, f"portfolio_{timestamp}.csv")
    portfolio_metric.to_csv(portfolio_path)
    paths["portfolio"] = portfolio_path
    
    # 保存分析结果
    analysis_path = os.path.join(report_dir, f"analysis_{timestamp}.csv")
    pd.DataFrame([analysis_result]).to_csv(analysis_path, index=False)
    paths["analysis"] = analysis_path
    
    # 生成并保存报告
    report_path = os.path.join(report_dir, f"report_{timestamp}.txt")
    generate_report(portfolio_metric, analysis_result, config, save_path=report_path)
    paths["report"] = report_path
    
    print(f"\nResults saved to: {report_dir}")
    
    return paths


# =============================================================================
# 便捷函数
# =============================================================================

def quick_backtest(
    predictions_path: str = DEFAULT_PREDICTIONS_PATH,
    start_time: str = "2023-01-01",
    end_time: str = "2023-12-31",
    topk: int = 50,
    n_drop: int = 100,
    save_report: bool = True,
) -> Dict[str, Any]:
    """
    快速回测接口。
    
    Args:
        predictions_path: 预测文件路径
        start_time: 开始时间
        end_time: 结束时间
        topk: Top-K 参数
        n_drop: N-Drop 参数
        save_report: 是否保存报告
        
    Returns:
        分析结果字典
    """
    config = BacktestConfig(
        start_time=start_time,
        end_time=end_time,
        topk=topk,
        n_drop=n_drop,
        predictions_path=predictions_path,
    )
    
    portfolio_metric, analysis_result = run_backtest(config=config)
    
    if save_report:
        save_backtest_results(portfolio_metric, analysis_result, config)
    
    return analysis_result


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Backtest")
    parser.add_argument(
        "--predictions",
        type=str,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path to predictions CSV",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2023-01-01",
        help="Backtest start date",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2023-12-31",
        help="Backtest end date",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Top-K stocks to hold",
    )
    parser.add_argument(
        "--n-drop",
        type=int,
        default=100,
        help="N-Drop threshold",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=DEFAULT_REPORT_DIR,
        help="Directory to save reports",
    )
    
    args = parser.parse_args()
    
    config = BacktestConfig(
        start_time=args.start,
        end_time=args.end,
        topk=args.topk,
        n_drop=args.n_drop,
        predictions_path=args.predictions,
        report_dir=args.report_dir,
    )
    
    portfolio_metric, analysis_result = run_backtest(config=config)
    save_backtest_results(portfolio_metric, analysis_result, config)

