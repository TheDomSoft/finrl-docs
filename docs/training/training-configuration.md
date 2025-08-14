# Training Configuration

This guide covers how to properly configure your FinRL training setup, including environment parameters, model configurations, and best practices.

## Environment Configuration

### Basic Environment Setup

```python
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# Calculate dimensions
stock_dim = len(df['tic'].unique())
state_space = 1 + 2 * stock_dim + len(tech_indicators) * stock_dim
action_space = stock_dim

# Environment parameters
env_kwargs = {
    "hmax": 100,                    # Maximum shares per trade
    "initial_amount": 1000000,      # Starting capital
    "num_stock_shares": [0] * stock_dim,  # Initial holdings
    "buy_cost_pct": [0.001] * stock_dim,  # 0.1% transaction cost
    "sell_cost_pct": [0.001] * stock_dim, # 0.1% transaction cost
    "reward_scaling": 1e-4,         # Reward normalization
    "turbulence_threshold": 140,    # Risk management threshold
    "print_verbosity": 10           # Logging frequency
}

def create_env(data):
    return StockTradingEnv(
        df=data,
        stock_dim=stock_dim,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=tech_indicators,
        **env_kwargs
    )

# Create vectorized environments
train_env = DummyVecEnv([lambda: create_env(train_data)])
val_env = DummyVecEnv([lambda: create_env(val_data)])
```

### Environment Parameter Guidelines

| Parameter | Description | Typical Values | Notes |
|-----------|-------------|----------------|-------|
| `hmax` | Max shares per trade | 100-1000 | Higher for more aggressive trading |
| `initial_amount` | Starting capital | 100,000-1,000,000 | Match your actual capital |
| `buy_cost_pct` | Buy transaction costs | 0.001-0.005 | 0.1%-0.5% realistic |
| `sell_cost_pct` | Sell transaction costs | 0.001-0.005 | Often same as buy costs |
| `reward_scaling` | Reward normalization | 1e-5 to 1e-3 | Adjust based on price levels |
| `turbulence_threshold` | Risk control threshold | 100-200 | Market-dependent |

### Advanced Environment Configuration

#### Custom Reward Functions

```python
class CustomRewardEnv(StockTradingEnv):
    def step(self, actions):
        state, reward, done, truncated, info = super().step(actions)
        
        # Add custom reward components
        portfolio_value = self.state[0] + np.sum(
            self.state[1:1+self.stock_dim] * 
            self.state[1+self.stock_dim:1+2*self.stock_dim]
        )
        
        # Risk-adjusted reward
        portfolio_returns = portfolio_value / self.initial_amount - 1
        portfolio_volatility = np.std(self.asset_memory[-30:]) if len(self.asset_memory) > 30 else 0.01
        sharpe_reward = portfolio_returns / (portfolio_volatility + 1e-6)
        
        # Combine rewards
        total_reward = reward + 0.1 * sharpe_reward
        
        return state, total_reward, done, truncated, info

# Use custom environment
def create_custom_env(data):
    return CustomRewardEnv(
        df=data,
        stock_dim=stock_dim,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=tech_indicators,
        **env_kwargs
    )
```

#### Dynamic Transaction Costs

```python
class DynamicCostEnv(StockTradingEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_buy_cost = kwargs.get('buy_cost_pct', [0.001] * self.stock_dim)
        self.base_sell_cost = kwargs.get('sell_cost_pct', [0.001] * self.stock_dim)
    
    def step(self, actions):
        # Adjust costs based on market conditions (e.g., volatility)
        turbulence = self.data[self.risk_indicator_col].iloc[0] if self.risk_indicator_col in self.data.columns else 0
        
        # Higher costs during high turbulence
        cost_multiplier = 1 + (turbulence / 100) * 0.5
        self.buy_cost_pct = [cost * cost_multiplier for cost in self.base_buy_cost]
        self.sell_cost_pct = [cost * cost_multiplier for cost in self.base_sell_cost]
        
        return super().step(actions)
```

## Model Configuration

### PPO Configuration

```python
# Conservative PPO for stable training
conservative_ppo = {
    "learning_rate": 3e-5,          # Lower learning rate
    "n_steps": 2048,                # Standard rollout length
    "batch_size": 64,               # Moderate batch size
    "ent_coef": 0.001,              # Low exploration
    "clip_range": 0.1,              # Conservative updates
    "n_epochs": 10,                 # Standard optimization epochs
    "gamma": 0.99,                  # Discount factor
    "gae_lambda": 0.95,             # GAE parameter
    "vf_coef": 0.25                 # Value function coefficient
}

# Aggressive PPO for faster learning
aggressive_ppo = {
    "learning_rate": 1e-3,          # Higher learning rate
    "n_steps": 4096,                # Longer rollouts
    "batch_size": 256,              # Larger batches
    "ent_coef": 0.01,               # More exploration
    "clip_range": 0.3,              # Less conservative
    "n_epochs": 20                  # More optimization
}

# Create PPO model
ppo_model = agent.get_model(
    "ppo", 
    model_kwargs=conservative_ppo,
    tensorboard_log="./ppo_logs/"
)
```

### SAC Configuration

```python
# Sample-efficient SAC
efficient_sac = {
    "learning_rate": 3e-4,          # Standard learning rate
    "buffer_size": 100000,          # Large replay buffer
    "batch_size": 256,              # Large batches
    "ent_coef": "auto",             # Automatic entropy tuning
    "learning_starts": 1000,        # Initial exploration
    "train_freq": (1, "step"),      # Train every step
    "gradient_steps": 1,            # Gradient steps per update
    "gamma": 0.99,                  # Discount factor
    "tau": 0.005                    # Soft update coefficient
}

# Memory-limited SAC
memory_limited_sac = {
    "learning_rate": 3e-4,
    "buffer_size": 10000,           # Smaller buffer
    "batch_size": 64,               # Smaller batches
    "ent_coef": "auto_0.1",         # Controlled entropy
    "learning_starts": 100,
    "train_freq": (4, "step"),      # Train less frequently
    "gradient_steps": 1
}

# Create SAC model
sac_model = agent.get_model(
    "sac",
    model_kwargs=efficient_sac,
    tensorboard_log="./sac_logs/"
)
```

### DDPG/TD3 Configuration

```python
# DDPG with action noise
ddpg_config = {
    "learning_rate": 1e-3,
    "buffer_size": 50000,
    "batch_size": 128,
    "tau": 0.005,
    "gamma": 0.99,
    "action_noise": "ornstein_uhlenbeck",
    "train_freq": (1, "episode"),
    "gradient_steps": -1,           # Same as batch size
    "learning_starts": 1000
}

# TD3 with improved stability
td3_config = {
    "learning_rate": 1e-3,
    "buffer_size": 1000000,
    "batch_size": 100,
    "tau": 0.005,
    "gamma": 0.99,
    "policy_delay": 2,              # Delayed policy updates
    "target_policy_noise": 0.2,     # Target policy smoothing
    "target_noise_clip": 0.5,       # Noise clipping
    "train_freq": (1, "step"),
    "gradient_steps": 1,
    "learning_starts": 1000
}

# Create models
ddpg_model = agent.get_model("ddpg", model_kwargs=ddpg_config)
td3_model = agent.get_model("td3", model_kwargs=td3_config)
```

## Policy Network Configuration

### Standard Networks

```python
# Simple dense networks
simple_policy = {
    "net_arch": [64, 64],           # Two hidden layers
    "activation_fn": torch.nn.ReLU
}

# Deeper networks for complex patterns
deep_policy = {
    "net_arch": [256, 256, 128],    # Three hidden layers
    "activation_fn": torch.nn.Tanh
}

# Use with any algorithm
model = agent.get_model(
    "ppo",
    policy_kwargs=simple_policy
)
```

### Custom Networks

```python
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TradingNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        return self.feature_extractor(observations)

# Use custom network
custom_policy = {
    "features_extractor_class": TradingNetwork,
    "features_extractor_kwargs": {"features_dim": 512},
    "net_arch": []  # Empty since we handle feature extraction
}

model = agent.get_model("ppo", policy_kwargs=custom_policy)
```

### Attention-Based Networks

```python
class AttentionTradingNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, num_assets=None):
        super().__init__(observation_space, features_dim)
        
        self.num_assets = num_assets or 10
        self.feature_dim = observation_space.shape[0] // self.num_assets
        
        # Multi-head attention for asset relationships
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=4,
            dropout=0.1
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )
    
    def forward(self, observations):
        batch_size = observations.shape[0]
        
        # Reshape for attention (sequence_length, batch_size, feature_dim)
        obs_reshaped = observations.view(batch_size, self.num_assets, -1)
        obs_reshaped = obs_reshaped.transpose(0, 1)
        
        # Apply attention
        attended_features, _ = self.attention(
            obs_reshaped, obs_reshaped, obs_reshaped
        )
        
        # Flatten back
        attended_features = attended_features.transpose(0, 1).flatten(1)
        
        return self.feature_extractor(attended_features)

# Use attention network
attention_policy = {
    "features_extractor_class": AttentionTradingNetwork,
    "features_extractor_kwargs": {
        "features_dim": 512,
        "num_assets": stock_dim
    }
}
```

## Training Schedules

### Learning Rate Schedules

```python
from stable_baselines3.common.schedules import linear_schedule

# Linear decay
linear_lr = linear_schedule(3e-4, 1e-5)

# Cosine annealing
def cosine_schedule(initial_value, final_value=None):
    if final_value is None:
        final_value = initial_value / 10
    
    def schedule(progress_remaining):
        return final_value + (initial_value - final_value) * (
            1 + math.cos(math.pi * (1 - progress_remaining))
        ) / 2
    
    return schedule

# Use scheduled learning rate
model = agent.get_model(
    "ppo",
    model_kwargs={
        "learning_rate": linear_lr,
        **other_params
    }
)
```

### Entropy Scheduling

```python
# Decay exploration over time
def entropy_schedule(initial_value=0.1, final_value=0.001):
    def schedule(progress_remaining):
        return final_value + (initial_value - final_value) * progress_remaining
    
    return schedule

# Apply to PPO
ppo_with_schedule = {
    "learning_rate": 3e-4,
    "ent_coef": entropy_schedule(0.1, 0.001),
    **other_ppo_params
}
```

## Data Configuration

### Data Splitting

```python
def robust_data_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create robust train/validation/test split"""
    
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio
    
    # Get unique dates and sort
    dates = sorted(df['date'].unique())
    n_dates = len(dates)
    
    # Calculate split indices
    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))
    
    # Split dates
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]
    
    # Create datasets
    train_data = df[df['date'].isin(train_dates)].reset_index(drop=True)
    val_data = df[df['date'].isin(val_dates)].reset_index(drop=True)
    test_data = df[df['date'].isin(test_dates)].reset_index(drop=True)
    
    # Fix indices for FinRL
    train_data.index = train_data['date'].factorize()[0]
    val_data.index = val_data['date'].factorize()[0]
    test_data.index = test_data['date'].factorize()[0]
    
    print(f"Data split:")
    print(f"  Train: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
    print(f"  Val: {len(val_dates)} days ({val_dates[0]} to {val_dates[-1]})")
    print(f"  Test: {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")
    
    return train_data, val_data, test_data

# Apply split
train_data, val_data, test_data = robust_data_split(processed_df)
```

### Feature Engineering Configuration

```python
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

# Basic feature engineering
basic_fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=[
        'macd', 'rsi_30', 'cci_30', 'dx_30',
        'close_30_sma', 'close_60_sma'
    ],
    use_turbulence=True,
    user_defined_feature=False
)

# Advanced feature engineering
advanced_fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=[
        'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'rsi_14',
        'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma',
        'close_5_sma', 'volume_sma', 'volume_delta'
    ],
    use_turbulence=True,
    user_defined_feature=True
)

# Process data
processed_df = advanced_fe.preprocess_data(raw_df)
```

## Configuration Templates

### Beginner Template

```python
# Simple, stable configuration for beginners
beginner_config = {
    "algorithm": "ppo",
    "env_kwargs": {
        "hmax": 100,
        "initial_amount": 100000,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4
    },
    "model_kwargs": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "ent_coef": 0.01
    },
    "policy_kwargs": {
        "net_arch": [64, 64]
    }
}
```

### Advanced Template

```python
# High-performance configuration for experienced users
advanced_config = {
    "algorithm": "sac",
    "env_kwargs": {
        "hmax": 1000,
        "initial_amount": 1000000,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "turbulence_threshold": 140
    },
    "model_kwargs": {
        "learning_rate": 3e-4,
        "buffer_size": 1000000,
        "batch_size": 256,
        "ent_coef": "auto",
        "learning_starts": 1000
    },
    "policy_kwargs": {
        "net_arch": [256, 256, 128]
    }
}
```

### Crypto Template

```python
# Optimized for cryptocurrency trading
crypto_config = {
    "algorithm": "sac",
    "env_kwargs": {
        "hmax": 1000,
        "initial_amount": 100000,
        "buy_cost_pct": [0.0025] * stock_dim,  # Higher crypto fees
        "sell_cost_pct": [0.0025] * stock_dim,
        "reward_scaling": 1e-5,  # Crypto prices often higher
        "turbulence_threshold": 200
    },
    "model_kwargs": {
        "learning_rate": 1e-4,
        "buffer_size": 500000,
        "batch_size": 512,
        "ent_coef": "auto_0.1",
        "train_freq": (1, "step")
    }
}
```

## Best Practices

### Configuration Validation

```python
def validate_config(config, data):
    """Validate configuration against data"""
    
    stock_dim = len(data['tic'].unique())
    
    # Check array lengths
    for param in ['buy_cost_pct', 'sell_cost_pct', 'num_stock_shares']:
        if param in config['env_kwargs']:
            if len(config['env_kwargs'][param]) != stock_dim:
                raise ValueError(f"{param} length ({len(config['env_kwargs'][param])}) != stock_dim ({stock_dim})")
    
    # Check state space calculation
    tech_indicators = config.get('tech_indicators', [])
    expected_state_space = 1 + 2 * stock_dim + len(tech_indicators) * stock_dim
    
    if 'state_space' in config and config['state_space'] != expected_state_space:
        print(f"Warning: state_space ({config['state_space']}) != expected ({expected_state_space})")
    
    print("✅ Configuration validation passed")
    return True

# Validate before training
validate_config(beginner_config, processed_df)
```

### Parameter Guidelines

| Parameter Type | Conservative | Moderate | Aggressive |
|---------------|--------------|----------|------------|
| Learning Rate | 1e-5 to 3e-5 | 3e-4 to 1e-3 | 1e-3 to 3e-3 |
| Batch Size | 32-64 | 128-256 | 512-1024 |
| Buffer Size | 10k-50k | 100k-500k | 1M+ |
| Exploration | Low (0.001) | Medium (0.01) | High (0.1) |

### Common Configuration Errors

```python
# ❌ Common mistakes
bad_config = {
    "reward_scaling": 1.0,          # Too large - rewards will dominate
    "learning_rate": 0.1,           # Too high - unstable training  
    "batch_size": 1,                # Too small - noisy gradients
    "ent_coef": 1.0,                # Too high - random exploration
    "hmax": 10000000               # Unrealistic position sizes
}

# ✅ Corrected version
good_config = {
    "reward_scaling": 1e-4,         # Appropriate scaling
    "learning_rate": 3e-4,          # Standard learning rate
    "batch_size": 64,               # Reasonable batch size
    "ent_coef": 0.01,               # Balanced exploration
    "hmax": 100                     # Realistic position size
}
```

## Next Steps

1. **Choose Configuration**: Select appropriate template
2. **Validate Setup**: Use validation functions
3. **Start Training**: Proceed to [Training Process](training-process.md)
4. **Monitor Performance**: Track metrics and adjust
5. **Tune Parameters**: Use [Hyperparameter Tuning](hyperparameter-tuning.md)

Remember to start with conservative configurations and gradually increase complexity as you gain experience and confidence in your setup.