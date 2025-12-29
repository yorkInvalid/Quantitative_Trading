# 🤖 Agent 交接文档 | AGENT HANDOVER

> **文档版本**: v1.5  
> **更新日期**: 2024-12-29  
> **适用对象**: 接手的 AI Agent 或开发者

---

## 1. 项目全景图 (Project Overview)

### 一句话描述

**基于 Docker + Qlib + AkShare + FinBERT 的 A 股量化选股与舆情监控系统**：从数据采集、特征工程、模型预测到新闻情绪分析的端到端量化交易流水线。

### 核心技术栈

| 组件 | 技术选型 | 版本要求 | 用途 |
|------|----------|----------|------|
| 运行环境 | Python | 3.10 | 主语言 |
| 容器化 | Docker + Compose | 3.9+ | 环境隔离与部署 |
| 量化框架 | Qlib (Microsoft) | latest | 因子计算 + 模型训练 |
| 数据源 | AkShare | ≥1.12.0 | A股历史数据 + 新闻 |
| ML 模型 | LightGBM | ≥3.3.0 | 股票收益率预测 |
| NLP 模型 | Transformers + PyTorch | ≥4.30.0 / ≥2.0.0 | 中文金融情感分析 |
| 情感模型 | FinBERT-Chinese | yiyanghkust/finbert-tone-chinese | 新闻舆情打分 |

---

## 2. 当前文件结构 (File Structure)

```
Quantitative_Trading/
├── 📄 Dockerfile                    # ✅ 多阶段构建，Layer Caching 优化
├── 📄 docker-compose.yml            # ✅ 服务编排，挂载 data/src/tests 目录
├── 📄 requirements.txt              # ✅ Python 依赖清单 (含 Qlib/AkShare/Transformers)
├── 📄 AGENT_HANDOVER.md             # ✅ 本文档
│
├── 📁 config/
│   ├── 📄 workflow.yaml             # ✅ Qlib 训练配置 (Alpha158 + LGBModel + CSI300)
│   └── 📄 rolling_workflow.yaml     # ✅ 滚动训练配置 (每 20 交易日重训)
│
├── 📁 src/
│   ├── 📄 main.py                   # ✅ 主入口，串联 ETL→Model→NLP→Strategy 流水线
│   │
│   ├── 📁 etl/
│   │   ├── 📄 downloader.py         # ✅ AkShare 数据下载器 (stock_zh_a_hist)
│   │   └── 📄 converter.py          # ✅ CSV → Qlib 二进制格式转换器
│   │
│   ├── 📁 model/
│   │   ├── 📄 trainer.py            # ✅ Qlib 模型训练 + 预测输出 + 持久化
│   │   └── 📄 rolling_trainer.py    # ✅ 滚动训练模块 (增量更新)
│   │
│   ├── 📁 nlp/
│   │   └── 📄 sentiment.py          # ✅ FinBERT 情感分析器 (Score = P(+) - P(-))
│   │
│   ├── 📁 strategy/
│   │   ├── 📄 __init__.py            # ✅ 模块初始化
│   │   ├── 📄 topk_dropout.py        # ✅ Top-K Dropout 换仓策略
│   │   └── 📄 topk_strategy.py       # ✅ Qlib 回测策略 (BaseStrategy)
│   │
│   ├── 📁 backtest/
│   │   ├── 📄 __init__.py            # ✅ 模块初始化
│   │   └── 📄 run_backtest.py        # ✅ 回测执行器 + 报告生成
│   │
│   ├── 📁 risk/
│   │   ├── 📄 __init__.py            # ✅ 模块初始化
│   │   └── 📄 rules.py               # ✅ 风控规则 (ST/停牌/持仓限制/涨跌停)
│   │
│   └── 📄 dry_run.py                 # ✅ 模拟实盘/端到端测试
│
├── 📁 tests/
│   ├── 📄 test_etl.py               # ✅ ETL 单元测试 (monkeypatch mock)
│   ├── 📄 test_model.py             # ✅ 模型配置/输出格式/持久化测试
│   ├── 📄 test_nlp.py               # ✅ 情感分析测试 (含 mock 版本)
│   ├── 📄 test_strategy.py          # ✅ Top-K Dropout 策略测试
│   └── 📄 test_integration.py       # ✅ 集成测试 (流水线端到端)
│
├── 📁 data/
│   └── 📁 models/                   # FinBERT 模型缓存目录
│       └── models--yiyanghkust--finbert-tone-chinese/  # HuggingFace 缓存
│
├── 📁 logs/                         # (空) 日志输出目录
│
└── 📁 ref_doc/
    ├── 📄 AI 股票预测与新闻监控模型.docx
    └── 📄 AI 股票预测与新闻监控模型.pdf   # 项目需求文档
```

### 文件状态图例

| 标记 | 含义 |
|------|------|
| ✅ | 已完成，代码可运行 |
| ⚠️ | 空目录/待实现 |
| 🔧 | 需要修复或优化 |

---

## 3. Docker 运行指南 (Docker Operations)

### 为什么这样写 Dockerfile？

```dockerfile
# Stage 1: 系统依赖 (几乎不变)
RUN apt-get install build-essential cmake git libgomp1 ...

# Stage 2: Python 依赖 (仅 requirements.txt 变化时重建)
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt  # Qlib 编译耗时，缓存关键！

# Stage 3: 应用代码 (频繁变化)
COPY src/ config/ tests/ /app/
```

**设计原则**: 利用 Docker Layer Caching，将**变化频率低的层放在前面**：
1. 系统依赖变化最少 → 最先安装
2. Python 依赖次之 → 仅 `requirements.txt` 变化时重建
3. 源代码变化最频繁 → 放在最后，避免触发依赖重装

### 标准操作命令

```bash
# 构建镜像 (首次约 10-15 分钟，后续利用缓存 < 1 分钟)
docker compose build

# 后台启动容器
docker compose up -d

# 进入容器交互式 Shell
docker compose exec quant-engine bash

# 在容器内运行测试
pytest tests/ -v

# 在容器内运行主流水线
python -m src.main

# 停止并清理
docker compose down
```

### 数据卷挂载

| 宿主机路径 | 容器路径 | 用途 |
|-----------|---------|------|
| `./data` | `/app/data` | CSV 数据 + Qlib 二进制 + 模型缓存 |
| `./config` | `/app/config` | workflow.yaml 配置 |
| `./src` | `/app/src` | 源代码 (热更新) |
| `./tests` | `/app/tests` | 测试代码 |

---

## 4. 模块化工作流 (Module Workflow)

### 数据流向图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUANTITATIVE TRADING PIPELINE                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Stage 1   │    │   Stage 2   │    │   Stage 3   │    │   Stage 4   │
│     ETL     │───▶│    Model    │───▶│     NLP     │───▶│  Strategy   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  AkShare    │    │   Qlib      │    │  AkShare    │    │  Top-K      │
│  日线数据   │    │  Alpha158   │    │  新闻API    │    │  Dropout    │
│     ↓       │    │     ↓       │    │     ↓       │    │     ↓       │
│  CSV 文件   │    │  LightGBM   │    │  FinBERT    │    │  Sentiment  │
│     ↓       │    │     ↓       │    │     ↓       │    │   Filter    │
│  Qlib .bin  │    │  Score CSV  │    │ Sentiment   │    │     ↓       │
│             │    │             │    │   Score     │    │  Buy List   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 各模块详解

#### Stage 1: ETL (数据抽取-转换-加载)

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `src/etl/downloader.py` | 调用 AkShare `stock_zh_a_hist` 下载日线数据 | 股票代码列表 | `data/csv_source/{symbol}.csv` |
| `src/etl/converter.py` | 将 CSV 转换为 Qlib 二进制格式 | CSV 目录 | `data/qlib_bin/features/` + `calendars/` |

**CSV 标准列**: `date, open, close, high, low, volume`

#### Stage 2: Model (模型训练与预测)

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `config/workflow.yaml` | Qlib 工作流配置 | - | - |
| `src/model/trainer.py` | 初始化 Qlib → 训练 LGBModel → 生成预测 | Qlib 二进制数据 | `data/predictions.csv` + `data/models/trained/*.pkl` |

**workflow.yaml 关键配置**:
- **特征**: Alpha158 (Qlib 内置 158 个量价因子)
- **标签**: 5 日前向收益率 `Ref($close, -5) / $close - 1`
- **模型**: LightGBM (num_leaves=64, n_estimators=500)
- **数据划分**: Train 2020-2022.06 | Valid 2022.07-12 | Test 2023-2024

**模型持久化**:
```python
from src.model.trainer import save_model, load_model, predict_only

# 训练并保存模型
from src.model.trainer import run_workflow
run_workflow(save_model_to_disk=True)  # 自动保存到 /app/data/models/trained/

# 仅预测（使用已保存的模型）
predict_only(model_path="/app/data/models/trained/lgb_model_xxx.pkl")
```

**滚动训练 (Rolling Training)**:
```python
from src.model.rolling_trainer import run_rolling_training, merge_rolling_predictions

# 执行滚动训练（每 20 交易日重训）
results = run_rolling_training(config_path="/app/config/rolling_workflow.yaml")

# 合并所有滚动预测
merged_df = merge_rolling_predictions(
    predictions_dir="/app/data/predictions/rolling",
    output_path="/app/data/predictions/rolling_merged.csv"
)
```

滚动训练参数（在 `rolling_workflow.yaml` 中配置）：
- `step`: 20 交易日（约 1 个月）
- `train_window`: 480 交易日（约 2 年）
- `valid_window`: 60 交易日（约 3 个月）
- `test_window`: 20 交易日（等于 step）

**回测框架**:
```python
from src.backtest.run_backtest import run_backtest, BacktestConfig

# 配置回测参数
config = BacktestConfig(
    start_time="2023-01-01",
    end_time="2023-12-31",
    topk=50,
    n_drop=100,
    init_cash=1_000_000,
    predictions_path="/app/data/predictions.csv",
)

# 运行回测
portfolio, analysis = run_backtest(config=config)

# 查看结果
print(f"夏普比率: {analysis['sharpe_ratio']:.2f}")
print(f"最大回撤: {analysis['max_drawdown']*100:.2f}%")
print(f"年化收益: {analysis['annual_return']*100:.2f}%")
```

或使用命令行：
```bash
python -m src.backtest.run_backtest \
    --predictions /app/data/predictions.csv \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --topk 50
```

**风控模块**:
```python
from src.risk.rules import (
    Order, RiskManager, StopSignRule, 
    PositionLimitRule, PriceLimitRule, apply_risk_rules
)

# 创建订单
orders = [
    Order("600519", "BUY", 100, 1800.0),
    Order("000001", "BUY", 1000, 10.0),
]

# 方式1: 使用便捷函数
passed_orders, summary = apply_risk_rules(
    orders=orders,
    enable_st_filter=True,      # 过滤 ST 股票
    enable_suspend_filter=True,  # 过滤停牌股票
    enable_position_limit=True,  # 持仓限制
    enable_price_limit=True,     # 涨跌停限制
    max_position_ratio=0.10,     # 单只最大 10%
    total_value=1_000_000.0,     # 总资产
)

# 方式2: 使用 RiskManager
manager = RiskManager()
manager.add_rule(StopSignRule())
manager.add_rule(PositionLimitRule(max_position_ratio=0.10))
manager.add_rule(PriceLimitRule())

passed, results = manager.check_orders(orders)
```

**模拟实盘**:
```python
from src.dry_run import PaperTrader

# 创建交易器
trader = PaperTrader(
    model_path="/app/data/models/latest_model.pkl",
    portfolio_path="/app/data/portfolio.json",
    reports_dir="/app/data/reports",
    topk=50,
    n_drop=100,
    init_cash=1_000_000.0,
)

# 运行每日循环
report = trader.run_daily_cycle(date="2024-12-29")

# 查看报告
print(f"总资产: {report.portfolio_value:,.0f} 元")
print(f"交易笔数: {len(report.trades)}")
```

或使用命令行：
```bash
python -m src.dry_run \
    --model /app/data/models/latest_model.pkl \
    --date 2024-12-29 \
    --topk 50 \
    --init-cash 1000000
```

**虚拟撮合规则**：
- 买入价 = 参考价 × (1 + 0.0002)  # 滑点 0.02%
- 卖出价 = 参考价 × (1 - 0.0002)  # 滑点 0.02%
- 买入成本 = max(成交金额 × 0.0002, 5元)  # 佣金
- 卖出成本 = max(成交金额 × 0.0012, 5元)  # 佣金+印花税

#### Stage 3: NLP (新闻情感分析)

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `src/nlp/sentiment.py` | 使用 FinBERT 分析新闻情感 | 新闻文本列表 | 情感分数 [-1, 1] |

**情感分数计算**: `Score = P(Positive) - P(Negative)`
- Score > 0.5: 强正面
- Score < -0.5: 强负面
- Score ≈ 0: 中性

#### Stage 4: Strategy (策略过滤)

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `src/strategy/topk_dropout.py` | Top-K Dropout 换仓策略 | 预测分数 + 情感分数 + 持仓 | `data/trade_signals_{date}.csv` |
| `src/main.py` | 简化版策略 (内置) | 预测分数 + 情感分数 | `data/final_buy_list_{date}.csv` |

**Top-K Dropout 策略逻辑**:
1. 读取 LightGBM 模型预测分数，按降序排名
2. 选取 Top 50 股票作为买入候选
3. 持仓股票跌出 Top 100 则卖出，资金释放后买入新的 Top 50
4. 硬性过滤：情感分数 < -0.5 的股票强制剔除或卖出（利空黑名单）

---

## 5. 当前开发进度 (Current Status)

### 基础设施

- [x] Docker 环境配置 (`Dockerfile` + `docker-compose.yml`)
- [x] Python 依赖管理 (`requirements.txt`)
- [x] Qlib 工作流配置 (`config/workflow.yaml`)

### ETL 模块

- [x] AkShare 数据下载器 (`src/etl/downloader.py`)
  - [x] 日线数据下载 (前复权)
  - [x] 重试机制 (max_retries=3)
  - [x] 列名标准化
- [x] Qlib 格式转换器 (`src/etl/converter.py`)
  - [x] CSV → .bin 二进制转换
  - [x] 交易日历生成 (`calendars/day.txt`)

### Model 模块

- [x] Qlib 初始化 (`src/model/trainer.py`)
- [x] LightGBM 模型训练
- [x] 预测结果输出 (CSV 格式)
- [x] 模型持久化 (保存/加载训练好的模型)
  - [x] `save_model()` - 保存模型到 pickle 文件
  - [x] `load_model()` - 从文件加载模型
  - [x] `get_latest_model()` - 获取最新保存的模型
  - [x] `predict_only()` - 仅预测模式（不训练）
- [x] 增量滚动训练 (`src/model/rolling_trainer.py`)
  - [x] 滚动窗口生成 (每 20 交易日)
  - [x] 自动时间窗口划分 (Train/Valid/Test)
  - [x] 模型按时间戳保存到 `/app/data/models/rolling/`
  - [x] 预测结果合并功能

### NLP 模块

- [x] FinBERT 情感分析器 (`src/nlp/sentiment.py`)
  - [x] 批量推理支持
  - [x] GPU/CPU 自动检测
  - [x] 模型缓存机制
- [x] 新闻抓取 (通过 AkShare `stock_news_em`)

### Strategy 模块

- [x] Top-K 选股 (内置于 `main.py`)
- [x] 情感阈值过滤 (内置于 `main.py`)
- [x] 独立策略模块 (`src/strategy/topk_dropout.py`)
  - [x] Top-K Dropout 换仓逻辑
  - [x] 持仓跟踪与更新
  - [x] 情感黑名单过滤 (sentiment < -0.5)
  - [x] 交易信号生成 (BUY/SELL/HOLD)
- [x] Qlib 回测策略 (`src/strategy/topk_strategy.py`)
  - [x] 继承 BaseStrategy
  - [x] 实现 generate_trade_decision()
- [x] 回测框架 (`src/backtest/run_backtest.py`)
  - [x] 加载预测结果
  - [x] 配置交易成本 (佣金/印花税/涨跌停)
  - [x] 运行回测
  - [x] 生成报告 (夏普比率/最大回撤/Calmar比率)
- [x] 风险控制模块 (`src/risk/rules.py`)
  - [x] StopSignRule: ST 股票和停牌股票过滤
  - [x] PositionLimitRule: 单只股票持仓比例限制
  - [x] PriceLimitRule: 涨跌停限制
  - [x] RiskManager: 多规则管理器
- [x] 模拟实盘 (`src/dry_run.py`)
  - [x] PaperTrader: 模拟交易器
  - [x] VirtualExchange: 虚拟撮合引擎
  - [x] 持仓管理 (加载/保存 JSON)
  - [x] 每日循环 (数据->预测->策略->风控->撮合)
  - [x] 交易成本和滑点模拟

### 测试覆盖

- [x] ETL 单元测试 (`test_etl.py`)
- [x] 模型配置测试 (`test_model.py`)
- [x] 模型持久化测试 (`test_model.py::TestModelPersistence`)
- [x] NLP 情感测试 (`test_nlp.py`)
- [x] 策略模块测试 (`test_strategy.py`)
- [x] 滚动训练测试 (`test_model.py::TestRollingTrainer`)
- [x] 回测模块测试 (`test_backtest.py`)
- [x] 风控模块测试 (`test_risk.py`)
- [x] 模拟实盘测试 (`test_dry_run.py`)
- [x] 集成测试 (`test_integration.py`)
- [x] 端到端测试 (Dry Run with mocks)

**测试统计**: 106 个测试用例全部通过 ✅

### 文档

- [x] 项目交接文档 (`AGENT_HANDOVER.md`)
- [ ] API 文档
- [ ] 用户使用手册

---

## 6. 下一步行动计划 (Next Actions)

### 🎯 接手后立即执行的 3 件事

#### 1️⃣ 验证 Docker 环境可用性

```bash
# 构建并启动容器
docker compose build
docker compose up -d

# 进入容器
docker compose exec quant-engine bash

# 运行测试验证环境
pytest tests/test_etl.py tests/test_model.py -v
```

**预期结果**: 所有测试通过，证明基础环境正常。

#### 2️⃣ 运行 ETL 流水线获取真实数据

```bash
# 在容器内执行
python -c "
from src.etl.downloader import download_stock_history
from src.etl.converter import convert_csv_to_qlib

# 下载 5 只样本股票
symbols = ['600519', '601318', '600036', '000858', '002415']
download_stock_history(symbols)

# 转换为 Qlib 格式
convert_csv_to_qlib()
"
```

**预期结果**: 
- `data/csv_source/` 下生成 5 个 CSV 文件
- `data/qlib_bin/features/` 下生成对应的二进制文件
- `data/qlib_bin/calendars/day.txt` 生成交易日历

#### 3️⃣ 运行 Top-K Dropout 换仓策略

独立策略模块已实现，可直接运行：

```bash
# 在容器内执行换仓
python -c "
from src.strategy.topk_dropout import run_rebalance

# 运行换仓策略
run_rebalance(
    predictions_path='/app/data/predictions.csv',
    holdings_path='/app/data/holdings.csv',
    top_k=50,
    dropout_threshold=100,
    sentiment_blacklist_threshold=-0.5
)
"
```

**策略使用示例**:

```python
from src.strategy.topk_dropout import TopKDropoutStrategy

# 初始化策略
strategy = TopKDropoutStrategy(
    top_k=50,                          # 买入 Top 50
    dropout_threshold=100,             # 跌出 Top 100 卖出
    sentiment_blacklist_threshold=-0.5 # 情感 < -0.5 强制剔除
)

# 执行换仓
result = strategy.rebalance(
    predictions=predictions_df,
    sentiments=sentiments_df,
    current_holdings={"SH600519", "SH601318"}
)

# 查看结果
print(f"买入: {len(result.buy_signals)}")
print(f"卖出: {len(result.sell_signals)}")
print(f"黑名单: {result.blacklist}")
```

---

## 📋 快速参考卡片

### 常用命令

| 操作 | 命令 |
|------|------|
| 构建镜像 | `docker compose build` |
| 启动容器 | `docker compose up -d` |
| 进入容器 | `docker compose exec quant-engine bash` |
| 运行测试 | `pytest tests/ -v` |
| 运行流水线 | `python -m src.main` |
| 查看日志 | `docker compose logs -f` |
| 停止容器 | `docker compose down` |

### 关键路径

| 用途 | 路径 |
|------|------|
| 原始 CSV 数据 | `/app/data/csv_source/` |
| Qlib 二进制数据 | `/app/data/qlib_bin/` |
| 模型预测输出 | `/app/data/predictions.csv` |
| 交易信号输出 | `/app/data/trade_signals_{date}.csv` |
| 持仓记录 | `/app/data/holdings.csv` |
| 训练好的模型 | `/app/data/models/trained/*.pkl` |
| 滚动训练模型 | `/app/data/models/rolling/*.pkl` |
| 滚动预测结果 | `/app/data/predictions/rolling/*.csv` |
| 回测报告 | `/app/data/backtest_reports/` |
| 模拟实盘持仓 | `/app/data/portfolio.json` |
| 每日交易报告 | `/app/data/reports/report_*.json` |
| 最终买入列表 | `/app/data/final_buy_list_{date}.csv` |
| FinBERT 模型缓存 | `/app/data/models/` |
| Qlib 配置 | `/app/config/workflow.yaml` |

### 环境变量

| 变量 | 值 | 用途 |
|------|-----|------|
| `PYTHONPATH` | `/app` | 确保模块导入正常 |
| `PYTHONUNBUFFERED` | `1` | 实时输出日志 |

---

## ⚠️ 已知问题与注意事项

1. **Qlib 首次初始化**: 需要先运行 ETL 生成 `qlib_bin` 目录，否则 `init_qlib()` 会报错
2. **AkShare 限流**: 批量下载时建议添加延时，避免被封 IP
3. **FinBERT 模型下载**: 首次运行需联网下载约 400MB 模型文件
4. **GPU 支持**: 当前 Dockerfile 未配置 CUDA，NLP 推理使用 CPU

---

**祝接手顺利！如有问题，请参考 `ref_doc/` 下的需求文档。** 🚀

