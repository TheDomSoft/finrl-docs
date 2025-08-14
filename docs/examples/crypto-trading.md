# Cryptocurrency Trading with FinRL

## Overview

This guide demonstrates how to implement cryptocurrency trading strategies using FinRL's specialized crypto environments and deep reinforcement learning algorithms.

## Complete Crypto Trading Example

### 1. Data Preparation

```python
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import ccxt
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.vec_env import DummyVecEnv

# Cryptocurrency symbols to trade
CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "DOT/USDT"]
```

### 2. Data Collection using CCXT

```python
def fetch_crypto_data(symbols, timeframe='1d', limit=1000):
    """Fetch crypto data using ccxt library"""
    
    # Initialize exchange (Binance)
    exchange = ccxt.binance({
        'apiKey': 'your_api_key',        # Optional for public data
        'secret': 'your_secret_key',     # Optional for public data
        'timeout': 30000,
        'enableRateLimit': True,
    })
    
    all_data = []
    
    for symbol in symbols:
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%Y-%m-%d')
            df['tic'] = symbol.replace('/', '-')  # Convert BTC/USDT to BTC-USDT
            
            # Select required columns
            df = df[['date', 'tic', 'open', 'high', 'low', 'close', 'volume']]
            all_data.append(df)
            
            print(f"✅ Fetched {len(df)} records for {symbol}")
            
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df.sort_values(['date', 'tic']).reset_index(drop=True)

# Fetch crypto data
crypto_df = fetch_crypto_data(CRYPTO_SYMBOLS, timeframe='1h', limit=2000)
print(f"Dataset shape: {crypto_df.shape}")
print(crypto_df.head())
```

### 3. Feature Engineering for Crypto

```python
# Define crypto-specific technical indicators
CRYPTO_INDICATORS = [
    "macd", "macdh", "macds",           # MACD family
    "rsi_14", "rsi_30",                 # RSI multiple periods
    "cci_14", "cci_30",                 # CCI multiple periods  
    "dx_14", "dx_30",                   # DX multiple periods
    "boll_ub", "boll_lb",               # Bollinger bands
    "close_7_sma", "close_14_sma",      # Short-term SMA
    "close_30_sma", "close_60_sma",     # Long-term SMA
    "volume_14_sma", "volume_30_sma",   # Volume indicators
    "high_14_sma", "low_14_sma"         # High/Low SMA
]

def add_crypto_features(df):
    """Add crypto-specific features"""
    df = df.copy()
    
    # Price-based features
    df['price_change'] = df.groupby('tic')['close'].pct_change()
    df['volatility'] = df.groupby('tic')['close'].rolling(24).std().values  # 24h volatility
    
    # Volume-based features  
    df['volume_change'] = df.groupby('tic')['volume'].pct_change()
    df['price_volume'] = df['close'] * df['volume']  # Dollar volume
    
    # Time-based features (crypto markets are 24/7)
    df['hour'] = pd.to_datetime(df['date']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    
    return df

# Initialize feature engineer
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=CRYPTO_INDICATORS,
    use_turbulence=True,      # Market stress indicator
    user_defined_feature=False
)

# Process features
processed_crypto = fe.preprocess_data(crypto_df)

# Add custom crypto features
processed_crypto = add_crypto_features(processed_crypto)

print("✅ Feature engineering completed")
print(f"Final dataset shape: {processed_crypto.shape}")
print(f"Available columns: {list(processed_crypto.columns)}")
```

### 4. Data Splitting

```python
from finrl.meta.preprocessor.preprocessors import data_split

# Define time periods
TRAIN_START = "2023-01-01"
TRAIN_END = "2023-09-01"
TEST_START = "2023-09-01"  
TEST_END = "2023-12-31"

# Split data
train_data = data_split(processed_crypto, TRAIN_START, TRAIN_END)
test_data = data_split(processed_crypto, TEST_START, TEST_END)

print(f"Training data: {train_data.shape}")
print(f"Testing data: {test_data.shape}")

# Verify data consistency
print(f"Training period: {train_data['date'].min()} to {train_data['date'].max()}")
print(f"Testing period: {test_data['date'].min()} to {test_data['date'].max()}")
print(f"Assets in training: {sorted(train_data['tic'].unique())}")
print(f"Assets in testing: {sorted(test_data['tic'].unique())}")
```

### 5. Environment Setup

```python
# Calculate environment parameters
crypto_dim = len(processed_crypto['tic'].unique())
state_space = 1 + 2 * crypto_dim + len(CRYPTO_INDICATORS) * crypto_dim

print(f"Number of cryptocurrencies: {crypto_dim}")
print(f"State space dimensions: {state_space}")

# Environment configuration optimized for crypto
env_kwargs = {
    "hmax": 100,                    # Max shares per trade
    "initial_amount": 100000,       # Starting capital (USD)
    "buy_cost_pct": 0.0025,         # 0.25% buy fee (typical for crypto)
    "sell_cost_pct": 0.0025,        # 0.25% sell fee  
    "reward_scaling": 1e-4,         # Reward normalization
    "turbulence_threshold": 99999,  # Disable turbulence for crypto (24/7 markets)
    "print_verbosity": 100,         # Print less frequently
}

# Create training environment
train_env = DummyVecEnv([
    lambda: StockTradingEnv(
        df=train_data,
        stock_dim=crypto_dim,
        state_space=state_space,
        action_space=crypto_dim,
        tech_indicator_list=CRYPTO_INDICATORS,
        **env_kwargs
    )
])

print("✅ Training environment created")
```

### 6. Model Training with Crypto-Optimized Parameters

```python
# SAC is often best for crypto due to sample efficiency and continuous markets
sac_crypto_params = {
    "batch_size": 256,              # Larger batches for stability
    "buffer_size": 200000,          # Large replay buffer
    "learning_rate": 1e-4,          # Conservative learning rate
    "learning_starts": 1000,        # More exploration initially
    "ent_coef": "auto",             # Automatic entropy tuning
    "gamma": 0.99,                  # Standard discount factor
    "tau": 0.01,                    # Faster soft updates for crypto volatility
    "train_freq": (4, "step"),      # Train every 4 steps
    "gradient_steps": 4             # Multiple gradient steps
}

# Initialize agent
agent = DRLAgent(env=train_env)

# Create SAC model
sac_model = agent.get_model(
    "sac",
    model_kwargs=sac_crypto_params,
    tensorboard_log="./crypto_sac_logs/"
)

print("Model parameters:")
for key, value in sac_crypto_params.items():
    print(f"  {key}: {value}")
```

### 7. Training

```python
import time

print("🚀 Starting training...")
start_time = time.time()

# Train the model
trained_sac = DRLAgent.train_model(
    model=sac_model,
    tb_log_name="sac_crypto_trading",
    total_timesteps=200000          # More training steps for crypto volatility
)

training_time = time.time() - start_time
print(f"✅ Training completed in {training_time:.2f} seconds")

# Save the trained model
trained_sac.save("./models/sac_crypto_final")
print("💾 Model saved")
```

### 8. Testing and Backtesting

```python
# Create test environment
test_env = DummyVecEnv([
    lambda: StockTradingEnv(
        df=test_data,
        stock_dim=crypto_dim,
        state_space=state_space,
        action_space=crypto_dim,
        tech_indicator_list=CRYPTO_INDICATORS,
        **env_kwargs
    )
])

# Make predictions
print("📈 Running backtest...")
account_value, actions = DRLAgent.DRL_prediction(
    model=trained_sac,
    environment=test_env.envs[0],
    deterministic=True
)

print("✅ Backtesting completed")
print(f"Account value shape: {account_value.shape}")
print(f"Actions shape: {actions.shape}")
```

### 9. Performance Analysis

```python
from finrl.plot import backtest_stats, backtest_plot

# Calculate performance metrics
perf_stats = backtest_stats(account_value, value_col_name="account_value")
print("📊 Performance Statistics:")
print(perf_stats)

# Calculate crypto-specific metrics
initial_value = account_value.iloc[0]["account_value"]
final_value = account_value.iloc[-1]["account_value"]
total_return = (final_value - initial_value) / initial_value
total_return_pct = total_return * 100

print(f"\n🎯 Trading Results:")
print(f"Initial Portfolio Value: ${initial_value:,.2f}")
print(f"Final Portfolio Value: ${final_value:,.2f}")
print(f"Total Return: {total_return_pct:.2f}%")

# Calculate daily returns for Sharpe ratio
daily_returns = account_value["account_value"].pct_change().dropna()
sharpe_ratio = np.sqrt(365) * daily_returns.mean() / daily_returns.std()  # Annualized
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")

# Max drawdown
portfolio_values = account_value["account_value"]
peak = portfolio_values.expanding(min_periods=1).max()
drawdown = (portfolio_values - peak) / peak
max_drawdown = drawdown.min()
print(f"Maximum Drawdown: {max_drawdown:.2%}")
```

### 10. Visualization

```python
import matplotlib.pyplot as plt

# Plot portfolio value over time
plt.figure(figsize=(15, 8))

# Portfolio performance
plt.subplot(2, 2, 1)
plt.plot(account_value.index, account_value["account_value"])
plt.title("Portfolio Value Over Time")
plt.xlabel("Days")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)

# Daily returns
plt.subplot(2, 2, 2)
plt.plot(daily_returns.index, daily_returns)
plt.title("Daily Returns")
plt.xlabel("Days") 
plt.ylabel("Daily Return")
plt.grid(True)

# Drawdown
plt.subplot(2, 2, 3)
plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
plt.plot(drawdown.index, drawdown, color='red')
plt.title("Drawdown Over Time")
plt.xlabel("Days")
plt.ylabel("Drawdown")
plt.grid(True)

# Action distribution
total_actions = np.sum(np.abs(actions.iloc[:, 1:].values), axis=0)  # Skip date column
crypto_names = [col for col in actions.columns if col != 'date']

plt.subplot(2, 2, 4)
plt.bar(crypto_names, total_actions)
plt.title("Total Trading Activity by Asset")
plt.xlabel("Cryptocurrency") 
plt.ylabel("Total Actions")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

## Advanced Crypto Features

### 1. Multi-Timeframe Analysis

```python
def create_multi_timeframe_features(df_1h, df_1d):
    """Combine multiple timeframes for better signals"""
    
    # Resample 1h to 1d for alignment
    df_1h_daily = df_1h.groupby(['tic', df_1h['date'].str[:10]]).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min', 
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    # Merge timeframes
    df_combined = df_1h.merge(
        df_1d[['date', 'tic', 'close_30_sma']].rename(columns={'close_30_sma': 'daily_sma_30'}),
        on=['date', 'tic'],
        how='left'
    )
    
    return df_combined
```

### 2. Custom Crypto Environment

```python
class AdvancedCryptoEnv(CryptoEnv):
    """Enhanced crypto environment with additional features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fear_greed_index = 50  # Market sentiment
        
    def calculate_reward(self, portfolio_return):
        """Custom reward function for crypto"""
        base_reward = portfolio_return
        
        # Bonus for consistent profits
        if hasattr(self, 'recent_returns'):
            if len(self.recent_returns) >= 7:  # Weekly consistency
                if all(r > 0 for r in self.recent_returns[-7:]):
                    base_reward *= 1.2  # 20% bonus
                    
        # Penalty for high volatility
        if hasattr(self, 'recent_returns'):
            volatility = np.std(self.recent_returns[-24:])  # 24h volatility
            if volatility > 0.05:  # >5% daily volatility
                base_reward *= 0.9  # 10% penalty
                
        return base_reward
```

### 3. Risk Management for Crypto

```python
def crypto_risk_management(actions, current_prices, portfolio_value):
    """Apply crypto-specific risk management"""
    
    # Position sizing - max 20% per crypto
    max_position_size = portfolio_value * 0.2
    
    for i, action in enumerate(actions):
        if action > 0:  # Buying
            max_shares = max_position_size / current_prices[i]
            actions[i] = min(action, max_shares)
    
    # Stop loss at portfolio level - liquidate if down 15%
    if portfolio_value < initial_portfolio_value * 0.85:
        actions = [-abs(action) for action in actions]  # Sell everything
        
    return actions
```

### 4. Live Trading Integration (Paper Trading)

```python
class CryptoPaperTrading:
    """Paper trading for crypto with real-time data"""
    
    def __init__(self, model, symbols, initial_balance=10000):
        self.model = model
        self.symbols = symbols
        self.balance = initial_balance
        self.positions = {symbol: 0 for symbol in symbols}
        self.exchange = ccxt.binance({'sandbox': True})  # Testnet
        
    def get_real_time_data(self):
        """Fetch real-time prices"""
        prices = {}
        for symbol in self.symbols:
            ticker = self.exchange.fetch_ticker(symbol)
            prices[symbol] = ticker['last']
        return prices
        
    def execute_trades(self, actions):
        """Execute paper trades based on model predictions"""
        current_prices = self.get_real_time_data()
        
        for i, (symbol, action) in enumerate(zip(self.symbols, actions)):
            if abs(action) > 0.01:  # Only trade if significant action
                if action > 0:  # Buy
                    cost = action * current_prices[symbol]
                    if cost <= self.balance:
                        self.positions[symbol] += action
                        self.balance -= cost
                        print(f"BUY {action:.4f} {symbol} at ${current_prices[symbol]:.2f}")
                        
                elif action < 0 and self.positions[symbol] > 0:  # Sell
                    sell_amount = min(-action, self.positions[symbol])
                    revenue = sell_amount * current_prices[symbol]
                    self.positions[symbol] -= sell_amount
                    self.balance += revenue
                    print(f"SELL {sell_amount:.4f} {symbol} at ${current_prices[symbol]:.2f}")
    
    def get_portfolio_value(self):
        """Calculate current portfolio value"""
        current_prices = self.get_real_time_data()
        total_value = self.balance
        
        for symbol, quantity in self.positions.items():
            total_value += quantity * current_prices[symbol]
            
        return total_value

# Initialize paper trader
paper_trader = CryptoPaperTrading(
    model=trained_sac,
    symbols=CRYPTO_SYMBOLS,
    initial_balance=10000
)

# Run paper trading loop (example)
"""
import time
for _ in range(100):  # Run for 100 iterations
    current_state = get_current_market_state()  # Implement this
    actions = trained_sac.predict(current_state, deterministic=True)[0]
    paper_trader.execute_trades(actions)
    
    portfolio_value = paper_trader.get_portfolio_value()
    print(f"Current portfolio value: ${portfolio_value:.2f}")
    
    time.sleep(300)  # Wait 5 minutes
"""
```

## Best Practices for Crypto Trading

### 1. Market Hours Consideration
```python
# Crypto markets are 24/7, but some times are more volatile
def get_market_session(timestamp):
    hour = timestamp.hour
    if 8 <= hour <= 16:
        return "asian"
    elif 13 <= hour <= 21: 
        return "european"
    elif 20 <= hour <= 4:
        return "american"
    else:
        return "quiet"
```

### 2. Volatility Management
```python
# Adjust position sizing based on volatility
def volatility_adjusted_sizing(base_action, volatility):
    if volatility > 0.1:  # High volatility (>10%)
        return base_action * 0.5  # Reduce position size
    elif volatility < 0.02:  # Low volatility (<2%)
        return base_action * 1.5  # Increase position size
    return base_action
```

### 3. News and Sentiment Integration
```python
# Incorporate external factors
def adjust_for_market_sentiment(actions, fear_greed_index):
    """Adjust actions based on market sentiment"""
    if fear_greed_index < 20:  # Extreme fear - reduce selling
        actions = [max(0, action) for action in actions]
    elif fear_greed_index > 80:  # Extreme greed - reduce buying
        actions = [min(0, action) for action in actions]
    return actions
```

This comprehensive guide provides everything needed to implement sophisticated cryptocurrency trading strategies using FinRL. The modular approach allows you to customize and extend the system based on your specific requirements.