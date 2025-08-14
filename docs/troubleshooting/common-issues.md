# Common Issues and Solutions

This guide addresses the most frequently encountered issues when using FinRL and provides practical solutions.

## Data-Related Issues

### 1. DataFrame Format Errors

#### Issue: "KeyError: 'date'"
```python
# ❌ Error: Missing required columns
df_bad = pd.DataFrame({
    'timestamp': ['2023-01-01', '2023-01-02'],
    'symbol': ['AAPL', 'AAPL'],
    'price': [150.0, 151.0]
})
```

**Solution**: Ensure correct column names
```python
# ✅ Correct format
df_correct = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02'],  # Must be 'date'
    'tic': ['AAPL', 'AAPL'],              # Must be 'tic'
    'close': [150.0, 151.0]               # Must be 'close'
})
```

#### Issue: "ValueError: all input array dimensions must match"

**Cause**: Inconsistent data structure across assets or dates.

**Solution**: Data alignment
```python
def fix_data_alignment(df):
    """Fix common data alignment issues"""
    
    # 1. Remove assets with insufficient data
    asset_counts = df.groupby('tic')['date'].nunique()
    min_required = asset_counts.quantile(0.8)  # Keep assets with 80%+ of dates
    valid_assets = asset_counts[asset_counts >= min_required].index
    df = df[df['tic'].isin(valid_assets)]
    
    # 2. Ensure all assets have same date range
    date_counts = df.groupby('date')['tic'].count()
    max_assets = date_counts.max()
    complete_dates = date_counts[date_counts == max_assets].index
    df = df[df['date'].isin(complete_dates)]
    
    # 3. Sort properly
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    # 4. Create proper index
    df.index = df['date'].factorize()[0]
    
    return df

# Apply fix
fixed_df = fix_data_alignment(problematic_df)
```

### 2. Missing Technical Indicators

#### Issue: "KeyError: 'macd'" or similar indicator errors

**Cause**: Technical indicators not calculated properly.

**Solution**: Proper feature engineering
```python
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.config import INDICATORS

# Method 1: Use FeatureEngineer
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_turbulence=True
)

# Always verify processed data
processed_df = fe.preprocess_data(raw_df)

# Check if indicators exist
missing_indicators = [ind for ind in INDICATORS if ind not in processed_df.columns]
if missing_indicators:
    print(f"⚠️ Missing indicators: {missing_indicators}")
else:
    print("✅ All indicators present")

# Method 2: Manual indicator calculation
import stockstats

def add_indicators_manually(df, indicators):
    """Manually add technical indicators"""
    df_result = pd.DataFrame()
    
    for tic in df['tic'].unique():
        tic_data = df[df['tic'] == tic].copy()
        
        # Convert to stockstats format
        stock_df = stockstats.StockDataFrame.retype(tic_data)
        
        # Add each indicator
        for indicator in indicators:
            try:
                tic_data[indicator] = stock_df[indicator]
            except Exception as e:
                print(f"Failed to add {indicator} for {tic}: {e}")
                tic_data[indicator] = 0  # Fill with zeros as fallback
        
        df_result = pd.concat([df_result, tic_data], ignore_index=True)
    
    return df_result

# Usage
df_with_indicators = add_indicators_manually(raw_df, INDICATORS)
```

## Environment Setup Issues

### 3. State Space Dimension Mismatch

#### Issue: "AssertionError: The observation returned by reset() method does not match the given observation space"

**Cause**: Calculated state_space doesn't match actual state dimensions.

**Solution**: Proper state space calculation
```python
def calculate_correct_state_space(df, tech_indicators):
    """Calculate exact state space dimensions"""
    
    stock_dim = len(df['tic'].unique())
    
    # State components:
    cash_dim = 1                                    # Cash balance
    price_dim = stock_dim                           # Current prices  
    holding_dim = stock_dim                         # Holdings per stock
    tech_dim = len(tech_indicators) * stock_dim     # Tech indicators per stock
    
    total_dim = cash_dim + price_dim + holding_dim + tech_dim
    
    print(f"State space calculation:")
    print(f"  Stock dimension: {stock_dim}")
    print(f"  Cash: {cash_dim}")
    print(f"  Prices: {price_dim}")
    print(f"  Holdings: {holding_dim}")
    print(f"  Technical indicators: {tech_dim} ({len(tech_indicators)} × {stock_dim})")
    print(f"  Total state space: {total_dim}")
    
    return total_dim

# Verify before creating environment
correct_state_space = calculate_correct_state_space(processed_df, INDICATORS)

env = StockTradingEnv(
    df=processed_df,
    stock_dim=len(processed_df['tic'].unique()),
    state_space=correct_state_space,  # Use calculated value
    **other_params
)
```

### 4. Action Space Issues

#### Issue: Actions not being executed properly

**Solution**: Verify action space configuration
```python
# Debug action processing
class DebuggingStockTradingEnv(StockTradingEnv):
    def step(self, actions):
        print(f"Raw actions: {actions}")
        print(f"Action shape: {actions.shape}")
        print(f"Action range: [{actions.min():.4f}, {actions.max():.4f}]")
        
        # Scale actions
        scaled_actions = actions * self.hmax
        print(f"Scaled actions: {scaled_actions}")
        
        # Continue with normal step
        return super().step(actions)

# Use debugging environment
debug_env = DebuggingStockTradingEnv(df=processed_df, **env_kwargs)
```

## Training Issues

### 5. Model Not Learning / Converging

#### Issue: Flat reward curves, no improvement

**Causes and Solutions**:

**A. Improper Reward Scaling**
```python
# Problem: Rewards too large or too small
env_kwargs = {
    "reward_scaling": 1e-4,  # Try different scales: 1e-3, 1e-4, 1e-5
    **other_params
}

# Monitor rewards during training
class RewardMonitoringEnv(StockTradingEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_history = []
    
    def step(self, actions):
        state, reward, done, truncated, info = super().step(actions)
        self.reward_history.append(reward)
        
        if len(self.reward_history) % 100 == 0:
            recent_rewards = self.reward_history[-100:]
            print(f"Last 100 rewards - Mean: {np.mean(recent_rewards):.6f}, "
                  f"Std: {np.std(recent_rewards):.6f}")
        
        return state, reward, done, truncated, info
```

**B. Learning Rate Issues**
```python
# Start with conservative learning rates
conservative_params = {
    "learning_rate": 3e-5,      # Lower learning rate
    "batch_size": 32,           # Smaller batches
}

# If too slow, gradually increase
aggressive_params = {
    "learning_rate": 1e-3,      # Higher learning rate
    "batch_size": 512,          # Larger batches
}

# Adaptive learning rate
def adaptive_learning_rate(model, performance_history, window=10):
    """Adjust learning rate based on performance"""
    if len(performance_history) < window:
        return
    
    recent_performance = np.mean(performance_history[-window:])
    older_performance = np.mean(performance_history[-2*window:-window])
    
    if recent_performance <= older_performance:  # Not improving
        new_lr = model.learning_rate * 0.5  # Reduce learning rate
        model.set_parameters({'learning_rate': new_lr})
        print(f"Reduced learning rate to {new_lr}")
```

**C. Exploration Issues**
```python
# Increase exploration for PPO
ppo_params = {
    "ent_coef": 0.1,            # Higher entropy (more exploration)
    "clip_range": 0.3,          # Larger clip range
}

# Add action noise for off-policy methods
ddpg_params = {
    "action_noise": "ornstein_uhlenbeck",
    **other_params
}

# Custom exploration schedule
from stable_baselines3.common.noise import NormalActionNoise

def create_exploration_noise(action_dim, sigma=0.1, decay=0.995):
    """Create decaying exploration noise"""
    return NormalActionNoise(
        mean=np.zeros(action_dim), 
        sigma=sigma * np.ones(action_dim)
    )
```

### 6. Memory Issues

#### Issue: "MemoryError" or "OOMKilled"

**Solutions**:

**A. Reduce Data Size**
```python
# Use fewer technical indicators
MINIMAL_INDICATORS = ["macd", "rsi_30", "close_30_sma"]

# Reduce time period
train_data = data_split(df, "2022-01-01", "2023-01-01")  # 1 year instead of 3

# Sample data
sampled_df = df.sample(frac=0.5, random_state=42)  # Use 50% of data
```

**B. Memory-Efficient Training**
```python
# Use smaller buffer sizes
memory_efficient_params = {
    "buffer_size": 10000,       # Smaller replay buffer
    "batch_size": 32,           # Smaller batches
    "n_steps": 512,             # Fewer steps per update (for PPO/A2C)
}

# Gradient checkpointing (for large networks)
policy_kwargs = {
    "net_arch": [64, 64],       # Smaller networks
    "activation_fn": torch.nn.ReLU
}
```

**C. Data Chunking**
```python
def train_in_chunks(df, chunk_size=50000, **env_kwargs):
    """Train on data chunks to manage memory"""
    
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    trained_models = []
    
    for i, chunk in enumerate(chunks):
        print(f"Training on chunk {i+1}/{len(chunks)}")
        
        # Create environment for this chunk
        env = StockTradingEnv(df=chunk, **env_kwargs)
        agent = DRLAgent(env=DummyVecEnv([lambda: env]))
        
        # Load previous model if exists
        if trained_models:
            model = trained_models[-1]
        else:
            model = agent.get_model("ppo")
        
        # Train on chunk
        model = agent.train_model(model, f"chunk_{i}", total_timesteps=10000)
        trained_models.append(model)
        
        # Clear memory
        del env, agent
        import gc; gc.collect()
    
    return trained_models[-1]  # Return final model
```

## Algorithm-Specific Issues

### 7. DDPG/TD3 Not Learning

**Common Issues**:
- No action noise
- Poor network initialization
- Learning rate too high

**Solutions**:
```python
# Proper DDPG configuration
ddpg_params = {
    "learning_rate": 1e-4,              # Lower learning rate
    "buffer_size": 100000,              # Larger buffer
    "action_noise": "ornstein_uhlenbeck", # Essential for exploration
    "batch_size": 256,                  # Larger batches
    "tau": 0.005,                       # Soft update rate
    "train_freq": (4, "step"),          # Train more frequently
    "gradient_steps": 4,                # Multiple gradient steps
    "learning_starts": 1000             # More initial exploration
}

# Custom action noise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

action_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(action_dim),
    sigma=0.1 * np.ones(action_dim),
    theta=0.15,
    dt=1e-2
)

model = agent.get_model("ddpg", model_kwargs={**ddpg_params, "action_noise": action_noise})
```

### 8. SAC Entropy Issues

**Issue**: "Entropy coefficient became nan"

**Solution**:
```python
# Proper SAC configuration
sac_params = {
    "ent_coef": "auto_0.1",      # Auto-tune with initial value
    "learning_rate": 3e-4,        # Standard learning rate
    "buffer_size": 100000,
    "batch_size": 256,
    "learning_starts": 1000,
    "train_freq": (1, "step"),
    "gradient_steps": 1,
    "target_entropy": "auto"      # Let SAC determine target entropy
}

# If still issues, use fixed entropy coefficient
sac_params_fixed = {
    "ent_coef": 0.01,            # Fixed value instead of auto
    **{k:v for k,v in sac_params.items() if k != "ent_coef"}
}
```

## Performance Issues

### 9. Poor Trading Performance

#### Issue: Model trained successfully but poor backtest results

**Diagnostic Steps**:

**A. Check Data Leakage**
```python
def check_data_leakage(train_data, test_data):
    """Ensure no overlap between train/test"""
    
    train_dates = set(train_data['date'].unique())
    test_dates = set(test_data['date'].unique())
    overlap = train_dates.intersection(test_dates)
    
    if overlap:
        print(f"❌ Data leakage detected: {len(overlap)} overlapping dates")
        return False
    
    # Check if test starts after train ends
    train_end = max(train_dates)
    test_start = min(test_dates)
    
    if test_start <= train_end:
        print(f"❌ Test period starts before training ends")
        return False
    
    print("✅ No data leakage detected")
    return True

# Usage
check_data_leakage(train_data, test_data)
```

**B. Validate Environment Consistency**
```python
def validate_environment_setup(train_env, test_env):
    """Ensure environments are set up consistently"""
    
    train_state_dim = train_env.observation_space.shape[0]
    test_state_dim = test_env.observation_space.shape[0]
    
    if train_state_dim != test_state_dim:
        print(f"❌ State dimension mismatch: train={train_state_dim}, test={test_state_dim}")
        return False
    
    train_action_dim = train_env.action_space.shape[0] 
    test_action_dim = test_env.action_space.shape[0]
    
    if train_action_dim != test_action_dim:
        print(f"❌ Action dimension mismatch: train={train_action_dim}, test={test_action_dim}")
        return False
    
    print("✅ Environment setup consistent")
    return True
```

**C. Analyze Actions**
```python
def analyze_trading_actions(actions_df, account_value_df):
    """Analyze if model is actually trading"""
    
    # Remove date column for analysis
    action_cols = [col for col in actions_df.columns if col != 'date']
    action_data = actions_df[action_cols].values
    
    # Trading statistics
    total_trades = np.sum(np.abs(action_data))
    non_zero_actions = np.sum(action_data != 0)
    action_percentage = (non_zero_actions / action_data.size) * 100
    
    print(f"Trading Activity Analysis:")
    print(f"  Total trades: {total_trades:.0f}")
    print(f"  Non-zero actions: {non_zero_actions} ({action_percentage:.1f}%)")
    print(f"  Average action magnitude: {np.mean(np.abs(action_data)):.4f}")
    
    # Check if stuck in cash
    final_portfolio = account_value_df.iloc[-1]['account_value']
    initial_portfolio = account_value_df.iloc[0]['account_value']
    
    if abs(final_portfolio - initial_portfolio) < 100:  # Very small change
        print("⚠️ Warning: Portfolio value barely changed - model may not be trading effectively")
    
    return {
        'total_trades': total_trades,
        'action_percentage': action_percentage,
        'portfolio_change': final_portfolio - initial_portfolio
    }

# Usage
trading_stats = analyze_trading_actions(actions_df, account_value_df)
```

### 10. Overfitting Issues

**Symptoms**: Great training performance, poor test performance

**Solutions**:

**A. Proper Validation**
```python
# Use proper train/validation/test split
def proper_data_split(df, train_ratio=0.6, val_ratio=0.2):
    """Split data with proper validation set"""
    
    dates = sorted(df['date'].unique())
    n_dates = len(dates)
    
    train_end_idx = int(n_dates * train_ratio)
    val_end_idx = int(n_dates * (train_ratio + val_ratio))
    
    train_dates = dates[:train_end_idx]
    val_dates = dates[train_end_idx:val_end_idx]
    test_dates = dates[val_end_idx:]
    
    train_data = df[df['date'].isin(train_dates)]
    val_data = df[df['date'].isin(val_dates)]
    test_data = df[df['date'].isin(test_dates)]
    
    return train_data, val_data, test_data

train_data, val_data, test_data = proper_data_split(processed_df)
```

**B. Early Stopping**
```python
from stable_baselines3.common.callbacks import EvalCallback

# Early stopping based on validation performance
eval_callback = EvalCallback(
    val_env,
    best_model_save_path="./best_model/",
    log_path="./eval_logs/",
    eval_freq=5000,
    n_eval_episodes=20,
    deterministic=True,
    render=False
)

# Train with early stopping
trained_model = agent.train_model(
    model,
    "model_with_early_stopping",
    total_timesteps=100000,
    callbacks=[eval_callback]
)
```

**C. Regularization**
```python
# Add dropout to policy networks
policy_kwargs = {
    "net_arch": [256, 256],
    "dropout": 0.1,              # Add dropout
    "batch_norm": True           # Batch normalization
}

# Reduce model complexity
simple_policy_kwargs = {
    "net_arch": [64, 64],        # Smaller network
    "activation_fn": torch.nn.Tanh  # Different activation
}
```

## Quick Debugging Checklist

When encountering issues, go through this checklist:

### Data Issues ✅
- [ ] Required columns present (`date`, `tic`, `close`)
- [ ] Date format is YYYY-MM-DD strings
- [ ] No missing values in critical columns
- [ ] All assets have same date range
- [ ] Technical indicators calculated correctly

### Environment Issues ✅
- [ ] State space calculation matches actual state
- [ ] Action space dimensions correct
- [ ] Reward scaling appropriate (try 1e-4)
- [ ] No data leakage between train/test

### Model Issues ✅
- [ ] Learning rate appropriate for algorithm
- [ ] Sufficient exploration (entropy coefficient, action noise)
- [ ] Proper validation setup
- [ ] Memory usage under control

### Performance Issues ✅
- [ ] Model actually trading (not stuck in cash)
- [ ] No overfitting (validate on separate data)
- [ ] Reasonable transaction costs
- [ ] Environment parameters realistic

By systematically working through these common issues and solutions, you can resolve most problems encountered when using FinRL for financial reinforcement learning.