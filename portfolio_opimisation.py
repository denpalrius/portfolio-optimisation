# %% [markdown]
# Portfolio Optimization with Enhanced DRL and Risk Controls

This notebook:
1. Downloads & preprocesses Dow 30 data
2. Engineers features & computes covariances
3. Splits into train/test sets with validation
4. Configures Gym environment with tunable parameters
5. Trains A2C, PPO, and SAC with hyperparameter options
6. Backtests strategies and computes risk-adjusted stats
7. Builds a minimum-variance benchmark
8. Plots comparative returns
9. Provides commented options for further tuning and validation

---

# %%
# 0. Install & import dependencies
# ! pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
# ! conda install -n portfolio_opt ipykernel --update-deps --force-reinstall
! pip install pandas_market_calendars quantstats gymnasium -q

# Standard libs
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Stable Baselines3
from stable_baselines3 import A2C, PPO, SAC

# FinRL
from finrl import config, config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, get_daily_return, get_baseline, convert_daily_return_to_pyfolio_ts

# PyPortfolioOpt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models

# Optional performance report
import quantstats as qs

%matplotlib inline

# %% [markdown]
## 1. Download Data

We fetch OHLCV for the Dow 30 using YahooDownloader.  
You can adjust `start_date` or switch to `use_turbulence=True` below.

# %%
# Parameters
ticker_list = config_tickers.DOW_30_TICKER
start_date = '2005-01-01'
end_date = (datetime.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

print(f"Downloading data: {start_date} → {end_date}")
df = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=ticker_list).fetch_data()

# %% [markdown]
## 2. Feature Engineering

Compute technical indicators and optionally add turbulence.  

# %%
print("Preprocessing technical indicators...")
# To include turbulence: use_turbulence=True
fe = FeatureEngineer(use_technical_indicator=True, use_turbulence=False)
df_tech = fe.preprocess_data(df)
print(f"df_tech shape: {df_tech.shape}")

# %% [markdown]
## 3. Covariance & Returns for State

Rolling window of 252 trading days → build `cov_list` and `return_list`.

# %%
print("Computing covariance matrices...")
df_sorted = df_tech.sort_values(['date','tic'], ignore_index=True)
df_sorted.index = df_sorted.date.factorize()[0]

cov_list, return_list = [], []
lookback = 252
unique_dates = df_sorted.date.unique()
for i in range(lookback, len(unique_dates)):
    window = df_sorted.loc[i-lookback:i]
    price_mat = window.pivot_table(index='date', columns='tic', values='close')
    ret_mat = price_mat.pct_change().dropna()
    return_list.append(ret_mat)
    cov_list.append(ret_mat.cov().values)

# Merge back
df_cov = pd.DataFrame({'date': unique_dates[lookback:], 'cov_list': cov_list, 'return_list': return_list})
df_merged = pd.merge(df_tech, df_cov, on='date', how='left')
df_final = df_merged[df_merged['cov_list'].notna()].reset_index(drop=True)
print(f"df_final shape with cov_list: {df_final.shape}")
assert 'cov_list' in df_final.columns, "cov_list missing ― check lookback or merge logic"

# %% [markdown]
## 4. Train/Test Split & Validation

Split by date-range, then ensure neither set is empty.

# %%
train_start, train_end = '2010-01-01', '2024-01-01'
trade_start, trade_end = train_end, end_date
train_data = data_split(df_final, train_start, train_end)
trade_data = data_split(df_final, trade_start, trade_end)

# Validation
def validate_split(df, start, end):
    assert not df.empty, f"DataSplit({start}->{end}) is empty"
    assert df.date.min() >= pd.to_datetime(start), "Train range too early"
    assert df.date.max() <= pd.to_datetime(end), "Train range too late"

validate_split(train_data, train_start, train_end)
validate_split(trade_data, trade_start, trade_end)
print(f"Train: {train_data.shape}, Trade: {trade_data.shape}")

# %% [markdown]
## 5. Environment Setup

Define env parameters.  
Below are commented alternative settings you can toggle.

# %%
stock_dim = len(train_data.tic.unique())
state_space = stock_dim
tech_indicators = config.INDICATORS

env_kwargs = {
    'stock_dim': stock_dim,
    'hmax': 100,                 # max shares per trade
    'initial_amount': 1e6,       # starting capital
    'transaction_cost_pct': 0.001, # slippage cost
    'reward_scaling': 1e-4,      # try 1e-3 or 1e-5
    'state_space': state_space,
    'action_space': stock_dim,
    'tech_indicator_list': tech_indicators,
    # 'turbulence_threshold':...  # to penalize extreme volatility
}
print(f"Env args: {env_kwargs}")

e_train = StockPortfolioEnv(df=train_data, **env_kwargs)
env_train, _ = e_train.get_sb_env()
e_trade = StockPortfolioEnv(df=trade_data, **env_kwargs)

# %% [markdown]
## 6. Train DRL Agents with Hyperparams

We train A2C, PPO, SAC.  Below are sample hyperparameter grids ― uncomment to test.

# %%
# Hyperparameter options:
# A2C_PARAMS = {'learning_rate': 7e-5, 'ent_coef': 0.01, 'n_steps': 5}
# PPO_PARAMS = {'learning_rate': 3e-4, 'ent_coef': 0.02, 'n_steps': 128, 'batch_size':64}
# SAC_PARAMS = {'learning_rate': 3e-4, 'buffer_size':50000, 'learning_starts':1000}

algos = ['a2c', 'ppo', 'sac']
trained_models = {}

for algo in algos:
    print(f"Training {algo.upper()}...")
    agent = DRLAgent(env=env_train)
    if algo == 'a2c':
        # model = agent.get_model('a2c', model_kwargs=A2C_PARAMS)
        model = agent.get_model('a2c')
    elif algo == 'ppo':
        # model = agent.get_model('ppo', model_kwargs=PPO_PARAMS)
        model = agent.get_model('ppo')
    else:
        # model = agent.get_model('sac', model_kwargs=SAC_PARAMS)
        model = agent.get_model('sac')
    trained = agent.train_model(model=model, tb_log_name=algo, total_timesteps=100_000)
    trained_models[algo] = trained

# Save/Load logic
os.makedirs('results/models', exist_ok=True)
for algo, model in trained_models.items():
    model.save(f'results/models/{algo}_model')
print("Models saved.")

# %% [markdown]
## 7. Backtest DRL Strategies

Compute daily returns → reconstruct account value → stats.  
Also generate a QuantStats HTML report (optional).

# %%
results = {}
for algo, model in trained_models.items():
    print(f"Backtesting {algo.upper()}...")
    df_ret, _ = DRLAgent.DRL_prediction(model=model, environment=e_trade)
    df_ret['account_value'] = (df_ret['daily_return'] + 1).cumprod() * env_kwargs['initial_amount']
    stats = backtest_stats(df_ret, value_col_name='account_value')
    results[algo] = {'df': df_ret, 'stats': stats}
    # Optional QuantStats report:
    # qs.reports.html(df_ret['daily_return'], output=f'results/{algo}_quantstats.html')

# %% [markdown]
## 8. Minimum-Variance Benchmark

Construct a rolling min‑variance portfolio for comparison.

# %%
print("Building min‑variance portfolio...")
dates = trade_data.date.unique()
min_vals = [env_kwargs['initial_amount']]
for i in range(len(dates)-1):
    curr = trade_data[trade_data.date==dates[i]]
    nxt  = trade_data[trade_data.date==dates[i+1]]
    cov  = np.array(curr.cov_list.values[0])
    ef   = EfficientFrontier(None, cov, weight_bounds=(0,1))
    ef.min_volatility()
    w    = ef.clean_weights()
    prices = curr.close.values
    nextp  = nxt.close.values
    shares = np.array(list(w.values())) * min_vals[-1] / prices
    min_vals.append(np.dot(shares, nextp))
min_var_df = pd.DataFrame({'date': dates, 'account_value': min_vals})

# %% [markdown]
## 9. DJIA Benchmark

Fetch DJIA and compute daily returns.

# %%
baseline = get_baseline(ticker='^DJI', start=trade_start, end=trade_end)
baseline_ret = get_daily_return(baseline, 'close')

# %% [markdown]
## 10. Plot Cumulative Returns

Visualize DRL vs. min‑var vs. DJIA.  

# %%
plt.figure(figsize=(12,6))
for algo in algos:
    df_ret = results[algo]['df']
    cump   = (df_ret['daily_return'] + 1).cumprod() - 1
    plt.plot(df_ret['date'], cump, label=algo.upper())
# Min-var & DJIA
c_min = (min_var_df['account_value'].pct_change() + 1).cumprod() - 1
plt.plot(min_var_df['date'], c_min, label='MIN_VAR')
c_dji = (baseline_ret + 1).cumprod() - 1
plt.plot(baseline['date'], c_dji, label='DJIA')
plt.legend(); plt.title('Cumulative Return Comparison')
plt.xlabel('Date'); plt.ylabel('Cumulative Return')
plt.savefig('results/cumulative_return.png')
plt.show()

# %% [markdown]
## 11. Performance Summary

Tabulate key metrics (Sharpe, MaxDD, etc.) for each algorithm.

# %%
perf_stats = pd.DataFrame({algo.upper(): results[algo]['stats'] for algo in algos})
perf_stats
