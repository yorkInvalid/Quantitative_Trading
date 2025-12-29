"""
Paper Trading / Dry Run Module.

实现模拟实盘交易，用于端到端测试和每日自动运行。

核心功能：
1. 加载和保存持仓
2. 获取最新数据并进行模型预测
3. 生成交易信号并应用风控规则
4. 虚拟撮合（模拟订单执行）
5. 更新持仓和资金

注意事项：
- 正确处理 datetime 和 Date 对象转换
- 虚拟撮合需考虑滑点和交易成本
- 保证幂等性（同一天多次运行结果一致）
"""

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from qlib.data import D

try:
    import akshare as ak
except ImportError:
    ak = None


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class Position:
    """持仓信息"""
    instrument: str       # 股票代码
    amount: int          # 持有数量（股）
    cost_price: float    # 成本价
    current_price: float = 0.0  # 当前价格
    market_value: float = 0.0   # 市值
    
    def update_price(self, price: float) -> None:
        """更新当前价格和市值"""
        self.current_price = price
        self.market_value = self.amount * price


@dataclass
class Portfolio:
    """投资组合"""
    cash: float                              # 现金
    positions: Dict[str, Position] = field(default_factory=dict)  # 持仓
    total_value: float = 0.0                 # 总资产
    last_update: str = ""                    # 最后更新时间
    
    def update_total_value(self) -> None:
        """更新总资产"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        self.total_value = self.cash + positions_value
    
    def to_dict(self) -> Dict:
        """转为字典（用于保存 JSON）"""
        return {
            "cash": self.cash,
            "positions": {
                code: asdict(pos) for code, pos in self.positions.items()
            },
            "total_value": self.total_value,
            "last_update": self.last_update,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Portfolio":
        """从字典创建（用于加载 JSON）"""
        positions = {
            code: Position(**pos_data)
            for code, pos_data in data.get("positions", {}).items()
        }
        return cls(
            cash=data.get("cash", 0.0),
            positions=positions,
            total_value=data.get("total_value", 0.0),
            last_update=data.get("last_update", ""),
        )


@dataclass
class Trade:
    """交易记录"""
    timestamp: str       # 时间戳
    instrument: str      # 股票代码
    direction: str       # "BUY" 或 "SELL"
    amount: int          # 数量
    price: float         # 成交价
    cost: float          # 交易成本（佣金+印花税）
    total_value: float   # 交易金额


@dataclass
class DailyReport:
    """每日报告"""
    date: str
    portfolio_value: float
    cash: float
    positions_value: float
    positions_count: int
    trades: List[Trade] = field(default_factory=list)
    predictions: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict:
        """转为字典"""
        return {
            "date": self.date,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "positions_value": self.positions_value,
            "positions_count": self.positions_count,
            "trades": [asdict(t) for t in self.trades],
        }


# =============================================================================
# 虚拟撮合引擎
# =============================================================================

class VirtualExchange:
    """
    虚拟撮合引擎。
    
    模拟订单执行，考虑滑点和交易成本。
    """
    
    def __init__(
        self,
        buy_slippage: float = 0.0002,   # 买入滑点 0.02%
        sell_slippage: float = 0.0002,  # 卖出滑点 0.02%
        buy_commission: float = 0.0002,  # 买入佣金 0.02%
        sell_commission: float = 0.0012, # 卖出成本 0.12% (含印花税)
        min_commission: float = 5.0,     # 最低佣金 5 元
    ):
        """
        初始化虚拟交易所。
        
        Args:
            buy_slippage: 买入滑点比例
            sell_slippage: 卖出滑点比例
            buy_commission: 买入佣金比例
            sell_commission: 卖出成本比例（含印花税）
            min_commission: 最低佣金
        """
        self.buy_slippage = buy_slippage
        self.sell_slippage = sell_slippage
        self.buy_commission = buy_commission
        self.sell_commission = sell_commission
        self.min_commission = min_commission
    
    def execute_buy(
        self, instrument: str, amount: int, reference_price: float
    ) -> Tuple[float, float, float]:
        """
        执行买入订单。
        
        Args:
            instrument: 股票代码
            amount: 数量
            reference_price: 参考价格
            
        Returns:
            (成交价, 成交金额, 交易成本) 元组
        """
        # 买入价 = 参考价 * (1 + 滑点)
        deal_price = reference_price * (1 + self.buy_slippage)
        
        # 成交金额
        deal_amount = amount * deal_price
        
        # 交易成本 = max(成交金额 * 佣金率, 最低佣金)
        commission = max(deal_amount * self.buy_commission, self.min_commission)
        
        return deal_price, deal_amount, commission
    
    def execute_sell(
        self, instrument: str, amount: int, reference_price: float
    ) -> Tuple[float, float, float]:
        """
        执行卖出订单。
        
        Args:
            instrument: 股票代码
            amount: 数量
            reference_price: 参考价格
            
        Returns:
            (成交价, 成交金额, 交易成本) 元组
        """
        # 卖出价 = 参考价 * (1 - 滑点)
        deal_price = reference_price * (1 - self.sell_slippage)
        
        # 成交金额
        deal_amount = amount * deal_price
        
        # 交易成本 = max(成交金额 * (佣金+印花税), 最低佣金)
        commission = max(deal_amount * self.sell_commission, self.min_commission)
        
        return deal_price, deal_amount, commission


# =============================================================================
# 模拟交易器
# =============================================================================

class PaperTrader:
    """
    模拟交易器。
    
    实现每日自动化交易流程：
    1. 加载持仓
    2. 获取最新数据
    3. 模型预测
    4. 策略生成
    5. 风控过滤
    6. 虚拟撮合
    7. 更新持仓
    """
    
    def __init__(
        self,
        model_path: str,
        portfolio_path: str = "/app/data/portfolio.json",
        reports_dir: str = "/app/data/reports",
        qlib_provider_uri: str = "/app/data/qlib_bin",
        topk: int = 50,
        n_drop: int = 100,
        init_cash: float = 1_000_000.0,
    ):
        """
        初始化模拟交易器。
        
        Args:
            model_path: 模型文件路径
            portfolio_path: 持仓文件路径
            reports_dir: 报告保存目录
            qlib_provider_uri: Qlib 数据目录
            topk: Top-K 选股数量
            n_drop: 跌出阈值
            init_cash: 初始资金
        """
        self.model_path = model_path
        self.portfolio_path = Path(portfolio_path)
        self.reports_dir = Path(reports_dir)
        self.qlib_provider_uri = qlib_provider_uri
        self.topk = topk
        self.n_drop = n_drop
        self.init_cash = init_cash
        
        # 创建目录
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.model = None
        self.portfolio: Optional[Portfolio] = None
        self.exchange = VirtualExchange()
    
    def load_portfolio(self) -> Portfolio:
        """
        加载持仓。
        
        如果文件不存在，则创建初始持仓。
        
        Returns:
            Portfolio 对象
        """
        if self.portfolio_path.exists():
            try:
                with open(self.portfolio_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                portfolio = Portfolio.from_dict(data)
                print(f"✓ 加载持仓成功: {self.portfolio_path}")
                print(f"  现金: {portfolio.cash:,.0f} 元")
                print(f"  持仓数: {len(portfolio.positions)}")
                print(f"  总资产: {portfolio.total_value:,.0f} 元")
                return portfolio
            except Exception as e:
                print(f"✗ 加载持仓失败: {e}")
                print(f"  创建新持仓...")
        
        # 创建初始持仓
        portfolio = Portfolio(cash=self.init_cash)
        portfolio.update_total_value()
        portfolio.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"✓ 创建初始持仓: 现金 {self.init_cash:,.0f} 元")
        return portfolio
    
    def save_portfolio(self, portfolio: Portfolio) -> None:
        """
        保存持仓。
        
        Args:
            portfolio: Portfolio 对象
        """
        try:
            portfolio.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.portfolio_path, "w", encoding="utf-8") as f:
                json.dump(portfolio.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"✓ 保存持仓成功: {self.portfolio_path}")
        except Exception as e:
            print(f"✗ 保存持仓失败: {e}")
            raise
    
    def load_model(self) -> Any:
        """
        加载模型。
        
        Returns:
            模型对象
        """
        if self.model is not None:
            return self.model
        
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            print(f"✓ 加载模型成功: {self.model_path}")
            return self.model
        except Exception as e:
            print(f"✗ 加载模型失败: {e}")
            raise
    
    def get_latest_prices(self, instruments: List[str]) -> Dict[str, float]:
        """
        获取最新价格。
        
        使用 AkShare 获取实时行情。
        
        Args:
            instruments: 股票代码列表
            
        Returns:
            {股票代码: 最新价} 字典
        """
        prices = {}
        
        if ak is None:
            print("Warning: akshare not installed, using mock prices")
            # 使用模拟价格
            for code in instruments:
                prices[code] = 10.0
            return prices
        
        try:
            # 获取实时行情
            quotes = ak.stock_zh_a_spot_em()
            
            if quotes.empty:
                print("Warning: Failed to fetch quotes, using last known prices")
                return prices
            
            # 标准化股票代码
            for code in instruments:
                normalized_code = self._normalize_code(code)
                
                # 查找行情
                mask = quotes["代码"].astype(str) == normalized_code
                if mask.sum() > 0:
                    price = quotes.loc[mask, "最新价"].iloc[0]
                    if pd.notna(price) and price > 0:
                        prices[code] = float(price)
                    else:
                        print(f"Warning: Invalid price for {code}")
                else:
                    print(f"Warning: No quote found for {code}")
        
        except Exception as e:
            print(f"Warning: Failed to fetch prices: {e}")
        
        return prices
    
    def update_portfolio_prices(self, portfolio: Portfolio) -> None:
        """
        更新持仓价格。
        
        Args:
            portfolio: Portfolio 对象
        """
        if not portfolio.positions:
            return
        
        instruments = list(portfolio.positions.keys())
        prices = self.get_latest_prices(instruments)
        
        for code, position in portfolio.positions.items():
            if code in prices:
                position.update_price(prices[code])
            else:
                print(f"Warning: No price update for {code}")
        
        portfolio.update_total_value()
    
    def predict(self, date: str) -> pd.DataFrame:
        """
        使用模型进行预测。
        
        Args:
            date: 预测日期 (YYYY-MM-DD)
            
        Returns:
            预测结果 DataFrame (columns: instrument, score)
        """
        # 这里简化实现，实际应该：
        # 1. 从 Qlib 获取最新数据
        # 2. 构造特征
        # 3. 模型预测
        
        # 模拟预测结果
        print(f"  执行模型预测 (日期: {date})...")
        
        # TODO: 实际实现应该调用 Qlib API
        # features_df = D.features(
        #     instruments=['600519', ...],
        #     fields=['$open', '$close', '$volume', ...],
        #     start_time=...,
        #     end_time=date
        # )
        # scores = self.model.predict(dataset)
        
        # 返回模拟数据
        return pd.DataFrame({
            "instrument": ["600519", "000001", "600000"],
            "score": [0.05, 0.03, 0.02],
        })
    
    def generate_orders(
        self, predictions: pd.DataFrame, portfolio: Portfolio
    ) -> List[Dict]:
        """
        根据预测生成订单。
        
        实现 Top-K Dropout 策略。
        
        Args:
            predictions: 预测结果
            portfolio: 当前持仓
            
        Returns:
            订单列表
        """
        orders = []
        
        # 按分数排序
        ranked = predictions.sort_values("score", ascending=False).reset_index(drop=True)
        
        # Top-K 股票
        top_stocks = set(ranked.head(self.topk)["instrument"].tolist())
        
        # 当前持仓
        current_holdings = set(portfolio.positions.keys())
        
        # 卖出信号：跌出 Top-N 的持仓
        if len(ranked) >= self.n_drop:
            top_n_stocks = set(ranked.head(self.n_drop)["instrument"].tolist())
            sell_stocks = current_holdings - top_n_stocks
        else:
            sell_stocks = set()
        
        for stock in sell_stocks:
            position = portfolio.positions[stock]
            orders.append({
                "instrument": stock,
                "direction": "SELL",
                "amount": position.amount,
                "reason": f"跌出 Top-{self.n_drop}",
            })
        
        # 买入信号：Top-K 中不在持仓的股票
        buy_stocks = top_stocks - current_holdings
        
        if buy_stocks:
            # 平均分配可用资金
            available_cash = portfolio.cash
            per_stock_cash = available_cash / len(buy_stocks)
            
            for stock in buy_stocks:
                orders.append({
                    "instrument": stock,
                    "direction": "BUY",
                    "amount": None,  # 金额模式，后续根据价格计算数量
                    "value": per_stock_cash,
                    "reason": f"进入 Top-{self.topk}",
                })
        
        return orders
    
    def apply_risk_rules(self, orders: List[Dict]) -> List[Dict]:
        """
        应用风控规则。
        
        Args:
            orders: 原始订单列表
            
        Returns:
            过滤后的订单列表
        """
        # TODO: 集成 src.risk.rules
        # 目前简化实现，直接返回
        print(f"  应用风控规则: {len(orders)} 个订单")
        return orders
    
    def execute_orders(
        self, orders: List[Dict], portfolio: Portfolio
    ) -> List[Trade]:
        """
        执行订单（虚拟撮合）。
        
        Args:
            orders: 订单列表
            portfolio: 当前持仓
            
        Returns:
            交易记录列表
        """
        trades = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 获取价格
        instruments = [o["instrument"] for o in orders]
        prices = self.get_latest_prices(instruments)
        
        # 执行卖出订单
        for order in orders:
            if order["direction"] == "SELL":
                instrument = order["instrument"]
                
                if instrument not in prices:
                    print(f"  ✗ 跳过 {instrument}: 无法获取价格")
                    continue
                
                reference_price = prices[instrument]
                position = portfolio.positions.get(instrument)
                
                if position is None:
                    print(f"  ✗ 跳过 {instrument}: 无持仓")
                    continue
                
                amount = order["amount"]
                deal_price, deal_amount, cost = self.exchange.execute_sell(
                    instrument, amount, reference_price
                )
                
                # 更新持仓
                del portfolio.positions[instrument]
                portfolio.cash += (deal_amount - cost)
                
                # 记录交易
                trade = Trade(
                    timestamp=timestamp,
                    instrument=instrument,
                    direction="SELL",
                    amount=amount,
                    price=deal_price,
                    cost=cost,
                    total_value=deal_amount,
                )
                trades.append(trade)
                
                print(f"  ✓ 卖出 {instrument}: {amount}股 @ {deal_price:.2f} 元")
        
        # 执行买入订单
        for order in orders:
            if order["direction"] == "BUY":
                instrument = order["instrument"]
                
                if instrument not in prices:
                    print(f"  ✗ 跳过 {instrument}: 无法获取价格")
                    continue
                
                reference_price = prices[instrument]
                
                # 计算可买数量（需向下取整到 100 股的整数倍）
                target_value = order.get("value", 0)
                if target_value <= 0:
                    continue
                
                estimated_shares = int(target_value / (reference_price * 1.0004))  # 考虑滑点和佣金
                amount = (estimated_shares // 100) * 100  # 向下取整到 100 的整数倍
                
                if amount < 100:
                    print(f"  ✗ 跳过 {instrument}: 资金不足 (需要至少 100 股)")
                    continue
                
                deal_price, deal_amount, cost = self.exchange.execute_buy(
                    instrument, amount, reference_price
                )
                
                total_cost = deal_amount + cost
                
                if total_cost > portfolio.cash:
                    print(f"  ✗ 跳过 {instrument}: 资金不足")
                    continue
                
                # 更新持仓
                portfolio.cash -= total_cost
                
                if instrument in portfolio.positions:
                    # 已有持仓，更新成本价
                    pos = portfolio.positions[instrument]
                    total_amount = pos.amount + amount
                    total_cost_value = pos.cost_price * pos.amount + deal_price * amount
                    new_cost_price = total_cost_value / total_amount
                    
                    pos.amount = total_amount
                    pos.cost_price = new_cost_price
                    pos.update_price(reference_price)
                else:
                    # 新建持仓
                    portfolio.positions[instrument] = Position(
                        instrument=instrument,
                        amount=amount,
                        cost_price=deal_price,
                        current_price=reference_price,
                        market_value=amount * reference_price,
                    )
                
                # 记录交易
                trade = Trade(
                    timestamp=timestamp,
                    instrument=instrument,
                    direction="BUY",
                    amount=amount,
                    price=deal_price,
                    cost=cost,
                    total_value=deal_amount,
                )
                trades.append(trade)
                
                print(f"  ✓ 买入 {instrument}: {amount}股 @ {deal_price:.2f} 元")
        
        portfolio.update_total_value()
        
        return trades
    
    def run_daily_cycle(self, date: Optional[str] = None) -> DailyReport:
        """
        运行每日循环。
        
        Args:
            date: 交易日期 (YYYY-MM-DD)，None 则使用今天
            
        Returns:
            每日报告
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        print("=" * 60)
        print(f"Paper Trading - {date}")
        print("=" * 60)
        
        # 1. 加载持仓
        print("\n[1/7] 加载持仓...")
        portfolio = self.load_portfolio()
        self.portfolio = portfolio
        
        # 2. 更新价格
        print("\n[2/7] 更新价格...")
        self.update_portfolio_prices(portfolio)
        
        # 3. 加载模型
        print("\n[3/7] 加载模型...")
        self.load_model()
        
        # 4. 模型预测
        print("\n[4/7] 模型预测...")
        predictions = self.predict(date)
        print(f"  预测数量: {len(predictions)}")
        
        # 5. 生成订单
        print("\n[5/7] 生成订单...")
        orders = self.generate_orders(predictions, portfolio)
        print(f"  订单数量: {len(orders)}")
        
        # 6. 应用风控
        print("\n[6/7] 应用风控...")
        filtered_orders = self.apply_risk_rules(orders)
        print(f"  通过风控: {len(filtered_orders)}")
        
        # 7. 执行订单
        print("\n[7/7] 执行订单...")
        trades = self.execute_orders(filtered_orders, portfolio)
        print(f"  成交数量: {len(trades)}")
        
        # 8. 保存持仓
        print("\n保存持仓...")
        self.save_portfolio(portfolio)
        
        # 9. 生成报告
        positions_value = sum(pos.market_value for pos in portfolio.positions.values())
        report = DailyReport(
            date=date,
            portfolio_value=portfolio.total_value,
            cash=portfolio.cash,
            positions_value=positions_value,
            positions_count=len(portfolio.positions),
            trades=trades,
            predictions=predictions,
        )
        
        # 10. 保存报告
        report_path = self.reports_dir / f"report_{date}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 60)
        print("Daily Report")
        print("=" * 60)
        print(f"日期: {date}")
        print(f"总资产: {portfolio.total_value:,.2f} 元")
        print(f"现金: {portfolio.cash:,.2f} 元")
        print(f"持仓市值: {positions_value:,.2f} 元")
        print(f"持仓数量: {len(portfolio.positions)}")
        print(f"交易笔数: {len(trades)}")
        print("=" * 60)
        
        return report
    
    def _normalize_code(self, code: str) -> str:
        """标准化股票代码"""
        import re
        code = code.upper()
        code = re.sub(r"^(SH|SZ|\.SH|\.SZ|SSE|SZSE)", "", code)
        code = re.sub(r"(\.SH|\.SZ)$", "", code)
        return code.strip()


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper Trading / Dry Run")
    parser.add_argument(
        "--model",
        type=str,
        default="/app/data/models/latest_model.pkl",
        help="模型文件路径",
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        default="/app/data/portfolio.json",
        help="持仓文件路径",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="交易日期 (YYYY-MM-DD)，默认为今天",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Top-K 选股数量",
    )
    parser.add_argument(
        "--n-drop",
        type=int,
        default=100,
        help="跌出阈值",
    )
    parser.add_argument(
        "--init-cash",
        type=float,
        default=1_000_000.0,
        help="初始资金",
    )
    
    args = parser.parse_args()
    
    # 创建交易器
    trader = PaperTrader(
        model_path=args.model,
        portfolio_path=args.portfolio,
        topk=args.topk,
        n_drop=args.n_drop,
        init_cash=args.init_cash,
    )
    
    # 运行每日循环
    trader.run_daily_cycle(date=args.date)


if __name__ == "__main__":
    main()

