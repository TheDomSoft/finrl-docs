# Environment Classes

## Overview

FinRL provides several pre-built trading environments that are compatible with OpenAI Gym. These environments handle the trading logic, state representation, and reward calculation.

## StockTradingEnv

The primary environment for stock and cryptocurrency trading.

### Class Definition

```python
class StockTradingEnv(gym.Env):
    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration=""
    )
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Price data with OHLCV + indicators |
| `stock_dim` | `int` | Number of assets to trade |
| `hmax` | `int` | Maximum shares per trade |
| `initial_amount` | `int` | Starting cash amount |
| `num_stock_shares` | `list[int]` | Initial holdings per asset |
| `buy_cost_pct` | `list[float]` | Buy transaction costs per asset |
| `sell_cost_pct` | `list[float]` | Sell transaction costs per asset |
| `reward_scaling` | `float` | Reward normalization factor |
| `state_space` | `int` | State vector dimensions |
| `action_space` | `int` | Action vector dimensions |
| `tech_indicator_list` | `list[str]` | Technical indicators to include |
| `turbulence_threshold` | `float` | Risk management threshold (optional) |
| `risk_indicator_col` | `str` | Column name for risk metric (default: "turbulence") |
| `make_plots` | `bool` | Enable automatic plotting (default: False) |
| `print_verbosity` | `int` | Logging frequency (default: 10) |
| `day` | `int` | Starting day index (default: 0) |
| `initial` | `bool` | Initial state flag (default: True) |
| `previous_state` | `list` | Previous state for continuity (default: []) |
| `model_name` | `str` | Model identifier (default: "") |
| `mode` | `str` | Operating mode (default: "") |
| `iteration` | `str` | Iteration identifier (default: "") |

### State Space

The state space consists of:
```
State = [cash, prices[], holdings[], technical_indicators[]]
```

**Dimensions**: `1 + 2*stock_dim + len(tech_indicators)*stock_dim`

**Example**: For 3 stocks with 5 technical indicators:
- Cash: 1 dimension
- Prices: 3 dimensions  
- Holdings: 3 dimensions
- Technical indicators: 15 dimensions (5 indicators × 3 stocks)
- **Total**: 22 dimensions

### Action Space

Actions represent the number of shares to buy/sell for each asset:
- **Positive values**: Buy shares
- **Negative values**: Sell shares
- **Range**: `[-hmax, hmax]` for each asset

### Reward Function

The default reward is the change in portfolio value:

```python
reward = (end_portfolio_value - begin_portfolio_value) * reward_scaling
```

Where portfolio value = `cash + sum(holdings * current_prices)`

### Key Methods

#### reset()
Resets the environment to initial state.

```python
state, info = env.reset()
```

#### step(actions)
Executes actions and returns next state.

```python
state, reward, terminated, truncated, info = env.step(actions)
```

#### render()
Returns the current state for inspection.

```python
current_state = env.render()
```

### Risk Management

#### Turbulence Control
When market turbulence exceeds the threshold, the environment:
1. Liquidates all positions (sells everything)
2. Prevents new purchases
3. Only allows selling

```python
# Enable turbulence control
env = StockTradingEnv(
    df=df,
    turbulence_threshold=140,  # Liquidate when turbulence > 140
    **other_params
)
```

### Example Usage

```python
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# Required parameters (no optional kwargs with invalid parameters)
def create_env(data):
    return StockTradingEnv(
        df=data,
        stock_dim=len(tickers),
        hmax=100,
        initial_amount=1000000,
        num_stock_shares=[0] * len(tickers),  # Start with no holdings
        buy_cost_pct=[0.001] * len(tickers),
        sell_cost_pct=[0.001] * len(tickers),
        reward_scaling=1e-4,
        state_space=state_dimensions,
        action_space=len(tickers),
        tech_indicator_list=indicators,
        turbulence_threshold=140
    )

# Create and wrap environment
vec_env = DummyVecEnv([lambda: create_env(processed_data)])
```

## CryptoEnv

Specialized environment for cryptocurrency trading with advanced features.

### Class Definition

```python
class CryptoEnv:
    def __init__(
        self,
        config,
        lookback=1,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        gamma=0.99
    )
```

### Key Features

- **Lookback Window**: Uses historical data for state representation
- **Action Normalization**: Automatically scales actions based on price levels
- **Multi-Crypto Support**: Handles multiple cryptocurrency pairs
- **Continuous Trading**: Designed for 24/7 crypto markets

### State Representation

```python
state = [
    cash_normalized,
    holdings_normalized, 
    technical_indicators[t-lookback:t]
]
```

### Example Configuration

```python
# Prepare data arrays
price_array = crypto_data.pivot_table(
    index='date', 
    columns='symbol', 
    values='close'
).values

tech_array = get_technical_indicators(crypto_data)

# Environment config
config = {
    "price_array": price_array,
    "tech_array": tech_array
}

# Create crypto environment
crypto_env = CryptoEnv(
    config=config,
    lookbook=24,  # 24-hour lookback
    initial_capital=100000,
    buy_cost_pct=0.001,
    sell_cost_pct=0.001
)
```

## Environment Variants

### StockTradingEnvCashPenalty
Penalizes holding too much cash.

### StockTradingEnvStopLoss
Includes automatic stop-loss functionality.

### PortfolioOptimizationEnv
Optimizes portfolio weights rather than individual trades.

## Best Practices

### Environment Configuration

!!! tip "Reward Scaling"
    Use `reward_scaling=1e-4` for typical stock prices to normalize rewards

!!! warning "Transaction Costs"
    Set realistic transaction costs: 0.1% (0.001) for stocks, 0.25% (0.0025) for crypto

!!! danger "Common Parameter Errors"
    - **TypeError with unexpected keyword arguments**: Ensure you're not passing invalid parameters like `day_trading`, `lookback_window`, etc.
    - **All required parameters must be provided**: `df`, `stock_dim`, `hmax`, `initial_amount`, `num_stock_shares`, `buy_cost_pct`, `sell_cost_pct`, `reward_scaling`, `state_space`, `action_space`, `tech_indicator_list`
    - **List parameters must match stock_dim**: `num_stock_shares`, `buy_cost_pct`, `sell_cost_pct` should have length equal to `stock_dim`

!!! info "State Space Calculation"
    Always verify state space dimensions match your data:
    ```python
    expected_dims = 1 + 2*stock_dim + len(indicators)*stock_dim
    ```

### Data Requirements

Ensure your DataFrame has these required columns:
- `date`: Trading date (YYYY-MM-DD format)
- `tic`: Asset ticker/symbol
- `close`: Closing price
- Technical indicators as specified in `tech_indicator_list`

### Memory Management

For large datasets, consider:
```python
# Use fewer technical indicators
MINIMAL_INDICATORS = ["close_14_sma", "rsi_14"]

# Reduce data frequency
daily_data = resample_to_daily(minute_data)

# Use data chunking for training
def create_env_chunks(df, chunk_size=10000):
    for chunk in df.groupby(df.index // chunk_size):
        yield StockTradingEnv(chunk, **env_kwargs)
```