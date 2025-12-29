# 🚀 A股量化选股与舆情监控系统

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![Qlib](https://img.shields.io/badge/Qlib-Microsoft-green.svg)](https://github.com/microsoft/qlib)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于 **Docker + Qlib + AkShare + FinBERT** 的 A 股量化选股与舆情监控系统。从数据采集、特征工程、模型预测到新闻情绪分析的端到端量化交易流水线。

## ✨ 核心功能

- 📊 **数据采集**: 使用 AkShare 自动下载 A 股日线数据
- 🔢 **特征工程**: Qlib Alpha158 量价因子 (158 个技术指标)
- 🤖 **机器学习**: LightGBM 模型预测股票收益率
- 📰 **舆情分析**: FinBERT 中文金融情感分析
- 📈 **选股策略**: Top-K Dropout 换仓策略 + 情感过滤

## 🏗️ 技术架构

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
   AkShare           LightGBM           FinBERT          Top-K Dropout
   日线数据          Alpha158           情感分析          换仓策略
```

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/Quantitative_Trading.git
cd Quantitative_Trading
```

### 2. 构建 Docker 镜像

```bash
docker compose build
```

### 3. 启动容器

```bash
docker compose up -d
```

### 4. 进入容器

```bash
docker compose exec quant-engine bash
```

### 5. 运行测试

```bash
pytest tests/ -v
```

### 6. 运行主流水线

```bash
python -m src.main
```

## 📁 项目结构

```
Quantitative_Trading/
├── Dockerfile                 # Docker 多阶段构建
├── docker-compose.yml         # 服务编排
├── requirements.txt           # Python 依赖
├── config/
│   └── workflow.yaml          # Qlib 训练配置
├── src/
│   ├── main.py                # 主入口
│   ├── etl/
│   │   ├── downloader.py      # AkShare 数据下载
│   │   └── converter.py       # Qlib 格式转换
│   ├── model/
│   │   └── trainer.py         # 模型训练与预测
│   ├── nlp/
│   │   └── sentiment.py       # FinBERT 情感分析
│   └── strategy/
│       └── topk_dropout.py    # Top-K Dropout 策略
├── tests/                     # 测试用例
└── data/                      # 数据目录 (gitignore)
```

## 📊 策略说明

### Top-K Dropout 换仓策略

1. **排名**: 按 LightGBM 预测分数降序排列
2. **买入**: 选取 Top 50 股票作为买入候选
3. **卖出**: 持仓股票跌出 Top 100 则卖出
4. **黑名单**: 情感分数 < -0.5 的股票强制剔除

```python
from src.strategy.topk_dropout import TopKDropoutStrategy

strategy = TopKDropoutStrategy(
    top_k=50,                          # 买入 Top 50
    dropout_threshold=100,             # 跌出 Top 100 卖出
    sentiment_blacklist_threshold=-0.5 # 情感黑名单
)

result = strategy.rebalance(predictions, sentiments, current_holdings)
```

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_strategy.py -v
pytest tests/test_nlp.py -v
```

## 📝 文档

详细的项目交接文档请参考 [AGENT_HANDOVER.md](AGENT_HANDOVER.md)。

## 🛠️ 技术栈

| 组件 | 技术选型 | 版本 |
|------|----------|------|
| 运行环境 | Python | 3.10 |
| 容器化 | Docker + Compose | 3.9+ |
| 量化框架 | Qlib (Microsoft) | latest |
| 数据源 | AkShare | ≥1.12.0 |
| ML 模型 | LightGBM | ≥3.3.0 |
| NLP 模型 | Transformers + PyTorch | ≥4.30.0 |
| 情感模型 | FinBERT-Chinese | yiyanghkust/finbert-tone-chinese |

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

⭐ 如果这个项目对你有帮助，请给一个 Star！

