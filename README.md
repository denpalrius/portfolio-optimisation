# Deep Reinforcement Learning (DRL) Portfolio Optimization

A modular implementation of deep reinforcement learning for portfolio optimization using Stable-Baselines3 and FinRL. It supports:

- **MLP, EIIE, EI3** policy architectures
- **A2C, PPO, SAC, DDPG, TD3** algorithms
- **Realistic trading constraints** (transaction costs, slippage, liquidity caps, budget limits)
- **Train / (validate) / test** splits with an automated evaluation callback
- **Backtesting** and **benchmarking** (MVO, equal-weighted, SPY)

## Prerequisites

- Python 3.11
- CUDA-enabled GPU (optional but recommended)

## Installation

```bash
pip install pandas numpy matplotlib \
    stable-baselines3 \
    PyPortfolioOpt \
    pandas_market_calendars quantstats gymnasium \
    git+https://github.com/AI4Finance-Foundation/FinRL.git
```
