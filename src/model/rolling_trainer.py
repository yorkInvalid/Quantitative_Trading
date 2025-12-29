"""
Rolling (Incremental) Training Module.

实现滚动训练策略：每隔固定交易日重新训练模型，
使用最新的历史数据，保持模型对市场变化的适应性。

滚动训练流程：
1. 初始训练：使用 [start, first_train_end] 的数据训练首个模型
2. 滚动更新：每隔 step 个交易日，向前滑动窗口重新训练
3. 预测输出：每个模型预测其对应的测试窗口

API Reference:
- 使用 Qlib 的 R.start() 和 R.save_objects() 进行实验管理
- 模型按时间戳保存到 /app/data/models/rolling/
"""

import importlib
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

# =============================================================================
# 默认配置
# =============================================================================

DEFAULT_CONFIG_PATH = "/app/config/rolling_workflow.yaml"
DEFAULT_MODEL_DIR = "/app/data/models/rolling"
DEFAULT_PREDICTIONS_DIR = "/app/data/predictions/rolling"


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class RollingWindow:
    """滚动窗口定义"""
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str
    window_id: int
    
    def __repr__(self) -> str:
        return (
            f"Window {self.window_id}: "
            f"Train[{self.train_start}~{self.train_end}] | "
            f"Valid[{self.valid_start}~{self.valid_end}] | "
            f"Test[{self.test_start}~{self.test_end}]"
        )


@dataclass
class RollingResult:
    """单次滚动训练结果"""
    window: RollingWindow
    model_path: str
    predictions_path: str
    metrics: Dict[str, float]


# =============================================================================
# 工具函数
# =============================================================================

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_qlib(config: dict) -> None:
    """初始化 Qlib"""
    qlib_config = config.get("qlib_init", {})
    provider_uri = qlib_config.get("provider_uri", "/app/data/qlib_bin")
    region = qlib_config.get("region", "cn")
    
    qlib.init(provider_uri=provider_uri, region=REG_CN if region == "cn" else region)
    print(f"Qlib initialized with provider_uri: {provider_uri}")


def get_trading_calendar(start_time: str, end_time: str) -> List[str]:
    """
    获取交易日历。
    
    Args:
        start_time: 开始日期
        end_time: 结束日期
        
    Returns:
        交易日列表（字符串格式）
    """
    cal = D.calendar(start_time=start_time, end_time=end_time)
    return [d.strftime("%Y-%m-%d") for d in cal]


def generate_rolling_windows(
    calendar: List[str],
    first_train_end: str,
    train_window: int,
    valid_window: int,
    test_window: int,
    step: int,
) -> List[RollingWindow]:
    """
    生成滚动窗口序列。
    
    Args:
        calendar: 交易日历列表
        first_train_end: 首次训练结束日期
        train_window: 训练窗口大小（交易日数）
        valid_window: 验证窗口大小（交易日数）
        test_window: 测试窗口大小（交易日数）
        step: 滚动步长（交易日数）
        
    Returns:
        RollingWindow 列表
    """
    windows = []
    
    # 找到 first_train_end 在日历中的位置
    try:
        first_end_idx = calendar.index(first_train_end)
    except ValueError:
        # 如果精确日期不在日历中，找最接近的
        for i, d in enumerate(calendar):
            if d >= first_train_end:
                first_end_idx = i
                break
        else:
            first_end_idx = len(calendar) - 1
    
    window_id = 0
    current_test_start_idx = first_end_idx + 1
    
    while current_test_start_idx + test_window <= len(calendar):
        # 计算各窗口边界
        test_end_idx = current_test_start_idx + test_window - 1
        valid_end_idx = current_test_start_idx - 1
        valid_start_idx = valid_end_idx - valid_window + 1
        train_end_idx = valid_start_idx - 1
        train_start_idx = max(0, train_end_idx - train_window + 1)
        
        # 边界检查
        if train_start_idx < 0 or valid_start_idx < 0:
            current_test_start_idx += step
            continue
        
        window = RollingWindow(
            train_start=calendar[train_start_idx],
            train_end=calendar[train_end_idx],
            valid_start=calendar[valid_start_idx],
            valid_end=calendar[valid_end_idx],
            test_start=calendar[current_test_start_idx],
            test_end=calendar[test_end_idx],
            window_id=window_id,
        )
        windows.append(window)
        
        window_id += 1
        current_test_start_idx += step
    
    return windows


# =============================================================================
# 模型构建
# =============================================================================

def build_model(config: dict) -> Any:
    """
    从配置构建模型实例。
    
    Args:
        config: 完整配置字典
        
    Returns:
        模型实例
    """
    task_config = config.get("task", {})
    model_config = task_config.get("model", {})
    
    model_class = model_config.get("class", "LGBModel")
    model_module = model_config.get("module_path", "qlib.contrib.model.gbdt")
    model_kwargs = model_config.get("kwargs", {})
    
    module = importlib.import_module(model_module)
    ModelClass = getattr(module, model_class)
    
    return ModelClass(**model_kwargs)


def build_dataset(config: dict, window: RollingWindow) -> Any:
    """
    从配置和滚动窗口构建数据集。
    
    Args:
        config: 完整配置字典
        window: 滚动窗口定义
        
    Returns:
        数据集实例
    """
    task_config = config.get("task", {})
    dataset_config = task_config.get("dataset", {})
    
    dataset_class = dataset_config.get("class", "DatasetH")
    dataset_module = dataset_config.get("module_path", "qlib.data.dataset")
    
    module = importlib.import_module(dataset_module)
    DatasetClass = getattr(module, dataset_class)
    
    # 构建 handler
    handler_config = dataset_config.get("kwargs", {}).get("handler", {})
    handler_class = handler_config.get("class", "Alpha158")
    handler_module = handler_config.get("module_path", "qlib.contrib.data.handler")
    handler_kwargs = handler_config.get("kwargs", {}).copy()
    
    # 更新 handler 的时间范围
    handler_kwargs["start_time"] = window.train_start
    handler_kwargs["end_time"] = window.test_end
    handler_kwargs["fit_start_time"] = window.train_start
    handler_kwargs["fit_end_time"] = window.train_end
    
    module = importlib.import_module(handler_module)
    HandlerClass = getattr(module, handler_class)
    handler = HandlerClass(**handler_kwargs)
    
    # 构建数据集，使用滚动窗口的时间段
    segments = {
        "train": (window.train_start, window.train_end),
        "valid": (window.valid_start, window.valid_end),
        "test": (window.test_start, window.test_end),
    }
    
    dataset = DatasetClass(handler=handler, segments=segments)
    
    return dataset


# =============================================================================
# 模型保存与加载
# =============================================================================

def save_rolling_model(
    model: Any,
    window: RollingWindow,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> str:
    """
    保存滚动训练的模型。
    
    Args:
        model: 训练好的模型
        window: 对应的滚动窗口
        model_dir: 保存目录
        
    Returns:
        模型文件路径
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # 使用测试窗口开始日期作为模型标识
    model_name = f"rolling_model_{window.test_start.replace('-', '')}_w{window.window_id}.pkl"
    model_path = os.path.join(model_dir, model_name)
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"  Model saved: {model_path}")
    return model_path


def load_rolling_model(model_path: str) -> Any:
    """加载滚动模型"""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def save_predictions(
    predictions: pd.DataFrame,
    window: RollingWindow,
    predictions_dir: str = DEFAULT_PREDICTIONS_DIR,
) -> str:
    """
    保存预测结果。
    
    Args:
        predictions: 预测 DataFrame
        window: 对应的滚动窗口
        predictions_dir: 保存目录
        
    Returns:
        预测文件路径
    """
    Path(predictions_dir).mkdir(parents=True, exist_ok=True)
    
    pred_name = f"pred_{window.test_start.replace('-', '')}_{window.test_end.replace('-', '')}.csv"
    pred_path = os.path.join(predictions_dir, pred_name)
    
    predictions.to_csv(pred_path)
    print(f"  Predictions saved: {pred_path}")
    
    return pred_path


# =============================================================================
# 滚动训练核心
# =============================================================================

def train_single_window(
    config: dict,
    window: RollingWindow,
    model_dir: str,
    predictions_dir: str,
    experiment_name: str,
) -> RollingResult:
    """
    训练单个滚动窗口。
    
    Args:
        config: 配置字典
        window: 滚动窗口
        model_dir: 模型保存目录
        predictions_dir: 预测保存目录
        experiment_name: 实验名称
        
    Returns:
        RollingResult 对象
    """
    print(f"\n{'='*60}")
    print(f"Training {window}")
    print(f"{'='*60}")
    
    # 构建模型和数据集
    model = build_model(config)
    dataset = build_dataset(config, window)
    
    # 使用 Qlib 的实验管理
    with R.start(experiment_name=f"{experiment_name}_w{window.window_id}"):
        # 记录参数
        R.log_params(
            window_id=window.window_id,
            train_start=window.train_start,
            train_end=window.train_end,
            test_start=window.test_start,
            test_end=window.test_end,
        )
        
        # 训练模型
        print("  Training model...")
        model.fit(dataset)
        
        # 保存模型到 Qlib 记录
        R.save_objects(trained_model=model)
        
        # 生成预测
        print("  Generating predictions...")
        pred = model.predict(dataset, segment="test")
        
        # 格式化预测结果
        if isinstance(pred, pd.Series):
            pred_df = pred.reset_index()
            pred_df.columns = ["datetime", "instrument", "score"]
        else:
            pred_df = pred.reset_index()
            if len(pred_df.columns) == 3:
                pred_df.columns = ["datetime", "instrument", "score"]
            else:
                pred_df.columns = list(pred_df.columns[:-1]) + ["score"]
        
        # 记录信号
        recorder = R.get_recorder()
        sr = SignalRecord(model=model, dataset=dataset, recorder=recorder)
        sr.generate()
    
    # 保存模型和预测到文件系统
    model_path = save_rolling_model(model, window, model_dir)
    pred_path = save_predictions(pred_df, window, predictions_dir)
    
    # 计算简单指标
    metrics = {
        "pred_count": len(pred_df),
        "pred_mean": float(pred_df["score"].mean()),
        "pred_std": float(pred_df["score"].std()),
    }
    
    return RollingResult(
        window=window,
        model_path=model_path,
        predictions_path=pred_path,
        metrics=metrics,
    )


def run_rolling_training(
    config_path: str = DEFAULT_CONFIG_PATH,
    max_windows: Optional[int] = None,
) -> List[RollingResult]:
    """
    执行完整的滚动训练流程。
    
    Args:
        config_path: 配置文件路径
        max_windows: 最大训练窗口数（用于测试，None 表示全部）
        
    Returns:
        RollingResult 列表
    """
    print("\n" + "=" * 60)
    print("ROLLING TRAINING - STARTING")
    print("=" * 60)
    
    # 加载配置
    config = load_config(config_path)
    
    # 初始化 Qlib
    init_qlib(config)
    
    # 获取滚动参数
    rolling_config = config.get("rolling", {})
    step = rolling_config.get("step", 20)
    train_window = rolling_config.get("train_window", 480)
    valid_window = rolling_config.get("valid_window", 60)
    test_window = rolling_config.get("test_window", 20)
    start_time = rolling_config.get("start_time", "2015-01-01")
    end_time = rolling_config.get("end_time", "2024-12-31")
    first_train_end = rolling_config.get("first_train_end", "2020-12-31")
    
    # 获取输出目录
    output_config = config.get("output", {})
    model_dir = output_config.get("model_dir", DEFAULT_MODEL_DIR)
    predictions_dir = output_config.get("predictions_dir", DEFAULT_PREDICTIONS_DIR)
    experiment_name = config.get("experiment_name", "rolling_lgb")
    
    print(f"\nRolling Parameters:")
    print(f"  Step: {step} trading days")
    print(f"  Train window: {train_window} trading days")
    print(f"  Valid window: {valid_window} trading days")
    print(f"  Test window: {test_window} trading days")
    print(f"  Data range: {start_time} ~ {end_time}")
    print(f"  First train end: {first_train_end}")
    
    # 获取交易日历
    print("\nFetching trading calendar...")
    calendar = get_trading_calendar(start_time, end_time)
    print(f"  Total trading days: {len(calendar)}")
    
    # 生成滚动窗口
    print("\nGenerating rolling windows...")
    windows = generate_rolling_windows(
        calendar=calendar,
        first_train_end=first_train_end,
        train_window=train_window,
        valid_window=valid_window,
        test_window=test_window,
        step=step,
    )
    
    if max_windows is not None:
        windows = windows[:max_windows]
    
    print(f"  Total windows to train: {len(windows)}")
    
    # 执行滚动训练
    results = []
    for i, window in enumerate(windows):
        print(f"\n[{i+1}/{len(windows)}] Processing window {window.window_id}")
        
        try:
            result = train_single_window(
                config=config,
                window=window,
                model_dir=model_dir,
                predictions_dir=predictions_dir,
                experiment_name=experiment_name,
            )
            results.append(result)
            print(f"  ✓ Window {window.window_id} completed")
            
        except Exception as e:
            print(f"  ✗ Window {window.window_id} failed: {e}")
            continue
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("ROLLING TRAINING - COMPLETED")
    print("=" * 60)
    print(f"  Successful windows: {len(results)}/{len(windows)}")
    print(f"  Models saved to: {model_dir}")
    print(f"  Predictions saved to: {predictions_dir}")
    
    return results


def merge_rolling_predictions(
    predictions_dir: str = DEFAULT_PREDICTIONS_DIR,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    合并所有滚动预测结果。
    
    Args:
        predictions_dir: 预测文件目录
        output_path: 输出文件路径（可选）
        
    Returns:
        合并后的预测 DataFrame
    """
    pred_dir = Path(predictions_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    
    pred_files = sorted(pred_dir.glob("pred_*.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")
    
    print(f"Merging {len(pred_files)} prediction files...")
    
    dfs = []
    for f in pred_files:
        df = pd.read_csv(f)
        dfs.append(df)
    
    merged = pd.concat(dfs, ignore_index=True)
    
    # 去重（保留最新的预测）
    if "datetime" in merged.columns and "instrument" in merged.columns:
        merged = merged.drop_duplicates(
            subset=["datetime", "instrument"],
            keep="last",
        )
    
    merged = merged.sort_values(["datetime", "instrument"]).reset_index(drop=True)
    
    print(f"Merged predictions: {len(merged)} rows")
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
    
    return merged


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rolling Training for Quantitative Trading")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to rolling workflow config",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Maximum number of windows to train (for testing)",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge existing predictions, skip training",
    )
    
    args = parser.parse_args()
    
    if args.merge_only:
        config = load_config(args.config)
        predictions_dir = config.get("output", {}).get("predictions_dir", DEFAULT_PREDICTIONS_DIR)
        merge_rolling_predictions(
            predictions_dir=predictions_dir,
            output_path="/app/data/predictions/rolling_merged.csv",
        )
    else:
        run_rolling_training(
            config_path=args.config,
            max_windows=args.max_windows,
        )

