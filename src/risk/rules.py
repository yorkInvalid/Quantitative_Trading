"""
Risk Management Rules Module.

实现风险控制规则，用于过滤不符合风控要求的交易订单。

主要规则：
1. StopSignRule: 过滤 ST 股票和停牌股票
2. PositionLimitRule: 单只股票持仓比例限制
3. PriceLimitRule: 涨跌停限制

API Reference:
- akshare.stock_zh_a_spot_em(): 获取 A 股实时行情
- akshare.stock_zh_a_st_em(): 获取 ST 股票列表
- akshare.stock_tfp_em(): 获取停复牌信息
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class Order:
    """交易订单"""
    instrument: str          # 股票代码
    direction: str           # "BUY" 或 "SELL"
    amount: int              # 数量（股）
    price: Optional[float] = None  # 价格（可选）
    reason: str = ""         # 订单原因


@dataclass
class RiskCheckResult:
    """风控检查结果"""
    passed: bool             # 是否通过
    order: Order             # 原始订单
    rule_name: str           # 规则名称
    message: str = ""        # 说明信息


# =============================================================================
# 风控规则基类
# =============================================================================

class RiskRule(ABC):
    """
    风控规则基类。
    
    所有风控规则必须继承此类并实现 check 方法。
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """规则名称"""
        pass
    
    @abstractmethod
    def check(self, orders: List[Order]) -> Tuple[List[Order], List[RiskCheckResult]]:
        """
        检查订单列表，返回通过的订单和检查结果。
        
        Args:
            orders: 待检查的订单列表
            
        Returns:
            (passed_orders, check_results) 元组
            - passed_orders: 通过检查的订单列表
            - check_results: 所有订单的检查结果
        """
        pass


# =============================================================================
# 市场数据获取
# =============================================================================

class MarketDataFetcher:
    """
    市场数据获取器。
    
    封装 AkShare API 调用，提供缓存机制。
    """
    
    def __init__(self, cache_ttl_seconds: int = 60):
        """
        初始化数据获取器。
        
        Args:
            cache_ttl_seconds: 缓存有效期（秒）
        """
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._spot_cache: Optional[pd.DataFrame] = None
        self._spot_cache_time: Optional[datetime] = None
        self._st_cache: Optional[Set[str]] = None
        self._st_cache_time: Optional[datetime] = None
        self._suspend_cache: Optional[Set[str]] = None
        self._suspend_cache_time: Optional[datetime] = None
    
    def _is_cache_valid(self, cache_time: Optional[datetime]) -> bool:
        """检查缓存是否有效"""
        if cache_time is None:
            return False
        return datetime.now() - cache_time < self.cache_ttl
    
    def get_realtime_quotes(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        获取 A 股实时行情。
        
        使用 akshare.stock_zh_a_spot_em() 获取实时行情数据。
        
        Returns:
            实时行情 DataFrame，包含列：
            - 代码: 股票代码
            - 名称: 股票名称
            - 最新价: 当前价格
            - 涨跌幅: 涨跌幅百分比
            - 成交量: 成交量
        """
        if not force_refresh and self._is_cache_valid(self._spot_cache_time):
            return self._spot_cache
        
        if ak is None:
            raise ImportError("akshare is not installed")
        
        try:
            df = ak.stock_zh_a_spot_em()
            self._spot_cache = df
            self._spot_cache_time = datetime.now()
            return df
        except Exception as e:
            print(f"Warning: Failed to fetch realtime quotes: {e}")
            if self._spot_cache is not None:
                return self._spot_cache
            return pd.DataFrame()
    
    def get_st_stocks(self, force_refresh: bool = False) -> Set[str]:
        """
        获取 ST 股票代码集合。
        
        优先使用 akshare.stock_zh_a_st_em()，
        如果失败则从实时行情中通过名称筛选。
        
        Returns:
            ST 股票代码集合
        """
        if not force_refresh and self._is_cache_valid(self._st_cache_time):
            return self._st_cache
        
        if ak is None:
            raise ImportError("akshare is not installed")
        
        st_codes = set()
        
        # 方法 1: 尝试使用专用 ST 接口
        try:
            st_df = ak.stock_zh_a_st_em()
            if st_df is not None and not st_df.empty:
                if "代码" in st_df.columns:
                    st_codes = set(st_df["代码"].astype(str).tolist())
                elif "股票代码" in st_df.columns:
                    st_codes = set(st_df["股票代码"].astype(str).tolist())
        except Exception as e:
            print(f"Warning: stock_zh_a_st_em failed: {e}")
        
        # 方法 2: 从实时行情中筛选名称包含 ST 的股票
        if not st_codes:
            try:
                quotes = self.get_realtime_quotes()
                if not quotes.empty and "名称" in quotes.columns and "代码" in quotes.columns:
                    # 筛选名称包含 ST、*ST、S*ST 等的股票
                    st_mask = quotes["名称"].str.contains(
                        r"ST|退市", case=False, na=False, regex=True
                    )
                    st_codes = set(quotes.loc[st_mask, "代码"].astype(str).tolist())
            except Exception as e:
                print(f"Warning: Failed to filter ST from quotes: {e}")
        
        self._st_cache = st_codes
        self._st_cache_time = datetime.now()
        
        return st_codes
    
    def get_suspended_stocks(self, force_refresh: bool = False) -> Set[str]:
        """
        获取停牌股票代码集合。
        
        通过检查成交量为 0 或最新价为空来判断停牌。
        
        Returns:
            停牌股票代码集合
        """
        if not force_refresh and self._is_cache_valid(self._suspend_cache_time):
            return self._suspend_cache
        
        suspended_codes = set()
        
        try:
            quotes = self.get_realtime_quotes(force_refresh=force_refresh)
            
            if quotes.empty:
                return suspended_codes
            
            # 判断停牌条件：
            # 1. 最新价为空 (NaN)
            # 2. 成交量为 0
            if "代码" in quotes.columns:
                code_col = "代码"
            else:
                return suspended_codes
            
            # 条件 1: 最新价为空
            if "最新价" in quotes.columns:
                price_na_mask = quotes["最新价"].isna() | (quotes["最新价"] == 0)
            else:
                price_na_mask = pd.Series([False] * len(quotes))
            
            # 条件 2: 成交量为 0
            if "成交量" in quotes.columns:
                vol_zero_mask = (quotes["成交量"] == 0) | quotes["成交量"].isna()
            else:
                vol_zero_mask = pd.Series([False] * len(quotes))
            
            # 满足任一条件视为停牌
            suspended_mask = price_na_mask | vol_zero_mask
            suspended_codes = set(quotes.loc[suspended_mask, code_col].astype(str).tolist())
            
        except Exception as e:
            print(f"Warning: Failed to get suspended stocks: {e}")
        
        self._suspend_cache = suspended_codes
        self._suspend_cache_time = datetime.now()
        
        return suspended_codes
    
    def get_price_limits(self, force_refresh: bool = False) -> Dict[str, Tuple[float, float]]:
        """
        获取涨跌停价格。
        
        Returns:
            字典 {股票代码: (跌停价, 涨停价)}
        """
        limits = {}
        
        try:
            quotes = self.get_realtime_quotes(force_refresh=force_refresh)
            
            if quotes.empty:
                return limits
            
            code_col = "代码" if "代码" in quotes.columns else None
            if code_col is None:
                return limits
            
            # 尝试获取涨跌停价格列
            up_limit_col = None
            down_limit_col = None
            
            for col in quotes.columns:
                if "涨停" in col:
                    up_limit_col = col
                if "跌停" in col:
                    down_limit_col = col
            
            if up_limit_col and down_limit_col:
                for _, row in quotes.iterrows():
                    code = str(row[code_col])
                    try:
                        up_limit = float(row[up_limit_col])
                        down_limit = float(row[down_limit_col])
                        limits[code] = (down_limit, up_limit)
                    except (ValueError, TypeError):
                        continue
            
        except Exception as e:
            print(f"Warning: Failed to get price limits: {e}")
        
        return limits


# =============================================================================
# 具体风控规则实现
# =============================================================================

class StopSignRule(RiskRule):
    """
    ST 和停牌股票过滤规则。
    
    过滤条件：
    1. 股票名称包含 "ST"（含 *ST、S*ST、退市等）
    2. 成交量为 0（疑似停牌）
    3. 最新价为空或为 0
    """
    
    def __init__(self, market_data: Optional[MarketDataFetcher] = None):
        """
        初始化规则。
        
        Args:
            market_data: 市场数据获取器，None 则创建新实例
        """
        self._market_data = market_data or MarketDataFetcher()
    
    @property
    def name(self) -> str:
        return "StopSignRule"
    
    def check(self, orders: List[Order]) -> Tuple[List[Order], List[RiskCheckResult]]:
        """
        检查订单，过滤 ST 和停牌股票。
        
        Args:
            orders: 待检查的订单列表
            
        Returns:
            (passed_orders, check_results) 元组
        """
        if not orders:
            return [], []
        
        # 获取 ST 和停牌股票列表
        st_stocks = self._get_st_stocks()
        suspended_stocks = self._get_suspended_stocks()
        
        passed_orders = []
        check_results = []
        
        for order in orders:
            # 标准化股票代码（去除前缀）
            code = self._normalize_code(order.instrument)
            
            # 检查是否为 ST 股票
            if code in st_stocks:
                check_results.append(RiskCheckResult(
                    passed=False,
                    order=order,
                    rule_name=self.name,
                    message=f"股票 {order.instrument} 为 ST 股票，禁止交易",
                ))
                continue
            
            # 检查是否停牌
            if code in suspended_stocks:
                check_results.append(RiskCheckResult(
                    passed=False,
                    order=order,
                    rule_name=self.name,
                    message=f"股票 {order.instrument} 疑似停牌（成交量为0），禁止交易",
                ))
                continue
            
            # 通过检查
            passed_orders.append(order)
            check_results.append(RiskCheckResult(
                passed=True,
                order=order,
                rule_name=self.name,
                message="通过 ST/停牌检查",
            ))
        
        return passed_orders, check_results
    
    def _normalize_code(self, instrument: str) -> str:
        """
        标准化股票代码。
        
        移除交易所前缀（如 SH、SZ、.SH、.SZ 等）。
        """
        # 移除常见前缀
        code = instrument.upper()
        code = re.sub(r"^(SH|SZ|\.SH|\.SZ|SSE|SZSE)", "", code)
        code = re.sub(r"(\.SH|\.SZ)$", "", code)
        return code.strip()
    
    def _get_st_stocks(self) -> Set[str]:
        """获取 ST 股票集合"""
        try:
            return self._market_data.get_st_stocks()
        except Exception as e:
            print(f"Warning: Failed to get ST stocks: {e}")
            return set()
    
    def _get_suspended_stocks(self) -> Set[str]:
        """获取停牌股票集合"""
        try:
            return self._market_data.get_suspended_stocks()
        except Exception as e:
            print(f"Warning: Failed to get suspended stocks: {e}")
            return set()


class PositionLimitRule(RiskRule):
    """
    单只股票持仓比例限制规则。
    
    限制单只股票的持仓金额不超过总资产的指定比例。
    """
    
    def __init__(
        self,
        max_position_ratio: float = 0.10,
        total_value: float = 1_000_000.0,
        current_positions: Optional[Dict[str, float]] = None,
    ):
        """
        初始化规则。
        
        Args:
            max_position_ratio: 单只股票最大持仓比例 (0-1)
            total_value: 总资产价值
            current_positions: 当前持仓 {股票代码: 持仓金额}
        """
        self.max_position_ratio = max_position_ratio
        self.total_value = total_value
        self.current_positions = current_positions or {}
    
    @property
    def name(self) -> str:
        return "PositionLimitRule"
    
    def check(self, orders: List[Order]) -> Tuple[List[Order], List[RiskCheckResult]]:
        """
        检查订单，确保不超过持仓限制。
        
        Args:
            orders: 待检查的订单列表
            
        Returns:
            (passed_orders, check_results) 元组
        """
        if not orders:
            return [], []
        
        max_position_value = self.total_value * self.max_position_ratio
        passed_orders = []
        check_results = []
        
        for order in orders:
            # 只检查买入订单
            if order.direction.upper() != "BUY":
                passed_orders.append(order)
                check_results.append(RiskCheckResult(
                    passed=True,
                    order=order,
                    rule_name=self.name,
                    message="卖出订单，跳过持仓限制检查",
                ))
                continue
            
            # 计算订单金额
            order_value = order.amount * (order.price or 0)
            current_value = self.current_positions.get(order.instrument, 0)
            new_value = current_value + order_value
            
            if new_value > max_position_value:
                check_results.append(RiskCheckResult(
                    passed=False,
                    order=order,
                    rule_name=self.name,
                    message=(
                        f"股票 {order.instrument} 持仓将达到 {new_value:,.0f} 元，"
                        f"超过限制 {max_position_value:,.0f} 元 "
                        f"({self.max_position_ratio*100:.0f}%)"
                    ),
                ))
                continue
            
            passed_orders.append(order)
            check_results.append(RiskCheckResult(
                passed=True,
                order=order,
                rule_name=self.name,
                message="通过持仓限制检查",
            ))
        
        return passed_orders, check_results


class PriceLimitRule(RiskRule):
    """
    涨跌停限制规则。
    
    过滤已经涨停或跌停的股票。
    """
    
    def __init__(
        self,
        market_data: Optional[MarketDataFetcher] = None,
        limit_threshold: float = 0.095,
    ):
        """
        初始化规则。
        
        Args:
            market_data: 市场数据获取器
            limit_threshold: 涨跌停阈值 (默认 9.5%)
        """
        self._market_data = market_data or MarketDataFetcher()
        self.limit_threshold = limit_threshold
    
    @property
    def name(self) -> str:
        return "PriceLimitRule"
    
    def check(self, orders: List[Order]) -> Tuple[List[Order], List[RiskCheckResult]]:
        """
        检查订单，过滤涨跌停股票。
        
        Args:
            orders: 待检查的订单列表
            
        Returns:
            (passed_orders, check_results) 元组
        """
        if not orders:
            return [], []
        
        # 获取实时行情
        quotes = self._get_quotes()
        
        passed_orders = []
        check_results = []
        
        for order in orders:
            code = self._normalize_code(order.instrument)
            
            # 查找股票行情
            quote = self._find_quote(quotes, code)
            
            if quote is None:
                # 找不到行情，放行但记录警告
                passed_orders.append(order)
                check_results.append(RiskCheckResult(
                    passed=True,
                    order=order,
                    rule_name=self.name,
                    message=f"未找到股票 {order.instrument} 的行情数据，跳过涨跌停检查",
                ))
                continue
            
            # 获取涨跌幅
            change_pct = quote.get("涨跌幅", 0)
            if pd.isna(change_pct):
                change_pct = 0
            
            # 检查涨停（买入时）
            if order.direction.upper() == "BUY" and change_pct >= self.limit_threshold * 100:
                check_results.append(RiskCheckResult(
                    passed=False,
                    order=order,
                    rule_name=self.name,
                    message=f"股票 {order.instrument} 已涨停 ({change_pct:.2f}%)，禁止买入",
                ))
                continue
            
            # 检查跌停（卖出时）
            if order.direction.upper() == "SELL" and change_pct <= -self.limit_threshold * 100:
                check_results.append(RiskCheckResult(
                    passed=False,
                    order=order,
                    rule_name=self.name,
                    message=f"股票 {order.instrument} 已跌停 ({change_pct:.2f}%)，禁止卖出",
                ))
                continue
            
            passed_orders.append(order)
            check_results.append(RiskCheckResult(
                passed=True,
                order=order,
                rule_name=self.name,
                message="通过涨跌停检查",
            ))
        
        return passed_orders, check_results
    
    def _normalize_code(self, instrument: str) -> str:
        """标准化股票代码"""
        code = instrument.upper()
        code = re.sub(r"^(SH|SZ|\.SH|\.SZ|SSE|SZSE)", "", code)
        code = re.sub(r"(\.SH|\.SZ)$", "", code)
        return code.strip()
    
    def _get_quotes(self) -> pd.DataFrame:
        """获取实时行情"""
        try:
            return self._market_data.get_realtime_quotes()
        except Exception as e:
            print(f"Warning: Failed to get quotes: {e}")
            return pd.DataFrame()
    
    def _find_quote(self, quotes: pd.DataFrame, code: str) -> Optional[Dict]:
        """查找股票行情"""
        if quotes.empty:
            return None
        
        if "代码" not in quotes.columns:
            return None
        
        mask = quotes["代码"].astype(str) == code
        if mask.sum() == 0:
            return None
        
        return quotes.loc[mask].iloc[0].to_dict()


# =============================================================================
# 风控管理器
# =============================================================================

class RiskManager:
    """
    风控管理器。
    
    管理多个风控规则，按顺序执行检查。
    """
    
    def __init__(self, rules: Optional[List[RiskRule]] = None):
        """
        初始化风控管理器。
        
        Args:
            rules: 风控规则列表，按顺序执行
        """
        self.rules = rules or []
    
    def add_rule(self, rule: RiskRule) -> None:
        """添加风控规则"""
        self.rules.append(rule)
    
    def check_orders(
        self, orders: List[Order]
    ) -> Tuple[List[Order], Dict[str, List[RiskCheckResult]]]:
        """
        检查订单列表。
        
        按顺序执行所有规则，返回最终通过的订单。
        
        Args:
            orders: 待检查的订单列表
            
        Returns:
            (passed_orders, all_results) 元组
            - passed_orders: 通过所有规则的订单
            - all_results: 各规则的检查结果 {rule_name: [results]}
        """
        current_orders = orders
        all_results: Dict[str, List[RiskCheckResult]] = {}
        
        for rule in self.rules:
            passed_orders, results = rule.check(current_orders)
            all_results[rule.name] = results
            current_orders = passed_orders
            
            if not current_orders:
                break
        
        return current_orders, all_results
    
    def get_summary(
        self, all_results: Dict[str, List[RiskCheckResult]]
    ) -> Dict[str, Any]:
        """
        生成风控检查摘要。
        
        Args:
            all_results: 各规则的检查结果
            
        Returns:
            摘要字典
        """
        summary = {
            "total_rules": len(self.rules),
            "rules_applied": len(all_results),
            "by_rule": {},
        }
        
        for rule_name, results in all_results.items():
            passed = sum(1 for r in results if r.passed)
            failed = sum(1 for r in results if not r.passed)
            summary["by_rule"][rule_name] = {
                "passed": passed,
                "failed": failed,
                "total": len(results),
            }
        
        return summary


# =============================================================================
# 便捷函数
# =============================================================================

def apply_risk_rules(
    orders: List[Order],
    enable_st_filter: bool = True,
    enable_suspend_filter: bool = True,
    enable_position_limit: bool = True,
    enable_price_limit: bool = True,
    max_position_ratio: float = 0.10,
    total_value: float = 1_000_000.0,
    current_positions: Optional[Dict[str, float]] = None,
) -> Tuple[List[Order], Dict[str, Any]]:
    """
    应用风控规则过滤订单。
    
    Args:
        orders: 待过滤的订单列表
        enable_st_filter: 是否启用 ST 过滤
        enable_suspend_filter: 是否启用停牌过滤
        enable_position_limit: 是否启用持仓限制
        enable_price_limit: 是否启用涨跌停限制
        max_position_ratio: 单只股票最大持仓比例
        total_value: 总资产价值
        current_positions: 当前持仓
        
    Returns:
        (passed_orders, summary) 元组
    """
    market_data = MarketDataFetcher()
    manager = RiskManager()
    
    # 添加规则
    if enable_st_filter or enable_suspend_filter:
        manager.add_rule(StopSignRule(market_data=market_data))
    
    if enable_position_limit:
        manager.add_rule(PositionLimitRule(
            max_position_ratio=max_position_ratio,
            total_value=total_value,
            current_positions=current_positions,
        ))
    
    if enable_price_limit:
        manager.add_rule(PriceLimitRule(market_data=market_data))
    
    # 执行检查
    passed_orders, all_results = manager.check_orders(orders)
    summary = manager.get_summary(all_results)
    
    return passed_orders, summary


def create_order(
    instrument: str,
    direction: str,
    amount: int,
    price: Optional[float] = None,
    reason: str = "",
) -> Order:
    """
    创建订单的便捷函数。
    
    Args:
        instrument: 股票代码
        direction: "BUY" 或 "SELL"
        amount: 数量
        price: 价格（可选）
        reason: 原因
        
    Returns:
        Order 对象
    """
    return Order(
        instrument=instrument,
        direction=direction.upper(),
        amount=amount,
        price=price,
        reason=reason,
    )


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("Risk Management Module - Demo")
    print("=" * 60)
    
    # 创建测试订单
    orders = [
        create_order("600519", "BUY", 100, 1800.0, "贵州茅台"),
        create_order("000001", "BUY", 1000, 10.0, "平安银行"),
        create_order("600000", "SELL", 500, 8.0, "浦发银行"),
    ]
    
    print(f"\n原始订单数: {len(orders)}")
    for order in orders:
        print(f"  - {order.instrument} {order.direction} {order.amount}股")
    
    # 应用风控规则
    passed_orders, summary = apply_risk_rules(
        orders=orders,
        enable_st_filter=True,
        enable_suspend_filter=True,
        enable_position_limit=True,
        max_position_ratio=0.10,
        total_value=1_000_000.0,
    )
    
    print(f"\n通过风控的订单数: {len(passed_orders)}")
    for order in passed_orders:
        print(f"  - {order.instrument} {order.direction} {order.amount}股")
    
    print(f"\n风控摘要: {summary}")

