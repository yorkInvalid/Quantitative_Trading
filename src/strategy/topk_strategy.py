"""
Top-K Strategy for Qlib Backtest.

实现基于预测分数的 Top-K 选股策略，用于 Qlib 回测框架。
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO


class TopKStrategy(BaseSignalStrategy):
    """
    Top-K 选股策略。
    
    根据模型预测分数选取 Top-K 股票，生成等权重目标持仓。
    支持情感过滤（黑名单机制）。
    
    Args:
        topk: 选取的股票数量
        n_drop: 跌出前 n_drop 名才卖出（dropout 机制）
        risk_degree: 仓位比例 (0-1)
        only_tradable: 是否只交易可交易的股票
    """
    
    def __init__(
        self,
        *,
        topk: int = 50,
        n_drop: Optional[int] = None,
        risk_degree: float = 0.95,
        only_tradable: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.topk = topk
        self.n_drop = n_drop if n_drop is not None else topk * 2
        self.risk_degree = risk_degree
        self.only_tradable = only_tradable
    
    def generate_trade_decision(self, execute_result=None) -> TradeDecisionWO:
        """
        生成交易决策。
        
        根据当前预测分数和持仓状态，生成买卖订单列表。
        
        Args:
            execute_result: 上一步执行结果（可选）
            
        Returns:
            TradeDecisionWO 对象，包含订单列表
        """
        # 获取当前交易步骤
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        
        # 获取当前预测分数
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(
            trade_step, shift=1
        )
        pred_score = self.signal.get_signal(
            start_time=pred_start_time,
            end_time=pred_end_time,
        )
        
        if pred_score is None or pred_score.empty:
            return TradeDecisionWO(order_list=[], strategy=self)
        
        # 获取当前持仓
        current_position = self.trade_position.get_stock_list()
        
        # 生成目标持仓
        target_stocks = self._generate_target_stocks(
            pred_score=pred_score,
            current_position=current_position,
        )
        
        # 生成订单
        order_list = self._generate_orders(
            target_stocks=target_stocks,
            current_position=current_position,
        )
        
        return TradeDecisionWO(order_list=order_list, strategy=self)
    
    def _generate_target_stocks(
        self,
        pred_score: pd.Series,
        current_position: List[str],
    ) -> Dict[str, float]:
        """
        生成目标持仓股票及权重。
        
        使用 Top-K Dropout 逻辑：
        1. 选取预测分数最高的 Top-K 股票
        2. 当前持仓如果跌出 Top-N_DROP，则卖出
        3. 新进入 Top-K 的股票买入
        
        Args:
            pred_score: 预测分数 Series (index=instrument)
            current_position: 当前持仓股票列表
            
        Returns:
            目标持仓字典 {instrument: weight}
        """
        # 按分数降序排列
        if isinstance(pred_score.index, pd.MultiIndex):
            # 处理多级索引 (datetime, instrument)
            pred_score = pred_score.droplevel(0)
        
        sorted_score = pred_score.sort_values(ascending=False)
        
        # 获取 Top-K 和 Top-N_DROP 股票
        topk_stocks = set(sorted_score.head(self.topk).index.tolist())
        top_ndrop_stocks = set(sorted_score.head(self.n_drop).index.tolist())
        
        # 确定目标持仓
        target_stocks = set()
        
        # 保留当前持仓中仍在 Top-N_DROP 的股票
        for stock in current_position:
            if stock in top_ndrop_stocks:
                target_stocks.add(stock)
        
        # 添加新进入 Top-K 的股票（直到达到 topk 数量）
        for stock in sorted_score.head(self.topk).index:
            if len(target_stocks) >= self.topk:
                break
            target_stocks.add(stock)
        
        # 计算等权重
        if len(target_stocks) == 0:
            return {}
        
        weight = self.risk_degree / len(target_stocks)
        
        return {stock: weight for stock in target_stocks}
    
    def _generate_orders(
        self,
        target_stocks: Dict[str, float],
        current_position: List[str],
    ) -> List[Order]:
        """
        生成订单列表。
        
        Args:
            target_stocks: 目标持仓 {instrument: weight}
            current_position: 当前持仓股票列表
            
        Returns:
            订单列表
        """
        order_list = []
        
        # 获取当前总资产
        total_value = self.trade_position.calculate_value()
        
        # 卖出不在目标持仓中的股票
        for stock in current_position:
            if stock not in target_stocks:
                # 获取当前持仓数量
                amount = self.trade_position.get_stock_amount(stock)
                if amount > 0:
                    order = Order(
                        stock_id=stock,
                        amount=amount,
                        direction=OrderDir.SELL,
                    )
                    order_list.append(order)
        
        # 买入目标持仓中的股票
        for stock, weight in target_stocks.items():
            target_value = total_value * weight
            current_amount = self.trade_position.get_stock_amount(stock)
            current_price = self._get_current_price(stock)
            
            if current_price is None or current_price <= 0:
                continue
            
            target_amount = int(target_value / current_price / 100) * 100  # 整手
            diff_amount = target_amount - current_amount
            
            if diff_amount > 0:
                order = Order(
                    stock_id=stock,
                    amount=diff_amount,
                    direction=OrderDir.BUY,
                )
                order_list.append(order)
            elif diff_amount < 0:
                order = Order(
                    stock_id=stock,
                    amount=abs(diff_amount),
                    direction=OrderDir.SELL,
                )
                order_list.append(order)
        
        return order_list
    
    def _get_current_price(self, stock: str) -> Optional[float]:
        """获取股票当前价格"""
        try:
            trade_step = self.trade_calendar.get_trade_step()
            trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
            
            price = self.trade_exchange.get_close(
                stock_id=stock,
                start_time=trade_start_time,
                end_time=trade_end_time,
            )
            return float(price) if price is not None else None
        except Exception:
            return None

