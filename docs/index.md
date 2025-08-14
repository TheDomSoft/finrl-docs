# FinRL Documentation

Welcome to the comprehensive documentation for **FinRL** - a Deep Reinforcement Learning library for financial markets!

## 🚀 What is FinRL?

FinRL is a powerful Python library that provides implementations of state-of-the-art deep reinforcement learning algorithms for financial applications, including:

- **Stock Trading**: Multi-asset portfolio management
- **Cryptocurrency Trading**: Real-time crypto trading strategies  
- **Portfolio Optimization**: Dynamic portfolio allocation
- **Risk Management**: Advanced risk metrics and controls

## 🏗️ Architecture Overview

FinRL follows a three-layer architecture:

```mermaid
graph TD
    A[Data Layer] --> B[Environment Layer]
    B --> C[Agent Layer]
    C --> D[Trading Results]
```

- **Data Layer**: Data collection, preprocessing, and feature engineering
- **Environment Layer**: Trading environments (Gym-compatible)
- **Agent Layer**: RL algorithms (A2C, PPO, DDPG, SAC, TD3)

## ⚡ Quick Start

```python
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

# 1. Download data
df = YahooDownloader(
    start_date="2020-01-01",
    end_date="2023-01-01", 
    ticker_list=["AAPL", "GOOGL", "MSFT"]
).fetch_data()

# 2. Process features
fe = FeatureEngineer(use_technical_indicator=True)
processed = fe.preprocess_data(df)

# 3. Create environment
env = StockTradingEnv(df=processed, **env_kwargs)

# 4. Train agent
agent = DRLAgent(env=env)
model = agent.get_model("ppo")
trained_model = agent.train_model(model, total_timesteps=50000)
```

## 📚 Documentation Sections

### API Reference
- [Environment Classes](api/environments.md) - Trading environment APIs
- [Agent Classes](api/agents.md) - RL agent implementations

### Data & Features
- [Data Format](data/format-requirements.md) - Data requirements and format specifications

### Advanced Features
- [Ensemble Methods](advanced/ensemble-methods.md) - Multi-model strategies

### Examples & Tutorials
- [Cryptocurrency Trading](examples/crypto-trading.md) - Crypto trading strategies and examples

### Troubleshooting
- [Common Issues](troubleshooting/common-issues.md) - Solutions to frequent problems

## 🎯 Key Features

!!! tip "Multi-Algorithm Support"
    FinRL supports 5 state-of-the-art RL algorithms: A2C, PPO, DDPG, SAC, and TD3

!!! info "Flexible Environments"
    Pre-built environments for stocks, crypto, and portfolio optimization

!!! success "Performance Tracking"
    Comprehensive backtesting with 20+ financial metrics

!!! warning "Risk Management"
    Built-in turbulence detection and risk controls

## 🤝 Contributing

We welcome contributions! Please check our GitHub repository for contribution guidelines.

## 📄 License

This project is licensed under the MIT License.

## 🔗 Links

- [GitHub Repository](https://github.com/AI4Finance-Foundation/FinRL)
- [Paper](https://papers.nips.cc/paper/2020/hash/1577d6b4e6de31bb93e24f14c0a87ee9-Abstract.html)
- [Tutorials](https://github.com/AI4Finance-Foundation/FinRL-Tutorials)