# Agent Classes

## Overview

FinRL provides wrapper classes for deep reinforcement learning algorithms, making them easy to use for financial applications. The library supports both individual agents and ensemble methods.

## DRLAgent

The main agent class that provides a unified interface for various RL algorithms.

### Class Definition

```python
class DRLAgent:
    def __init__(self, env):
        self.env = env
```

### Supported Algorithms

| Algorithm | Full Name | Type | Best For |
|-----------|-----------|------|----------|
| `a2c` | Advantage Actor-Critic | On-policy | Fast training, simple problems |
| `ppo` | Proximal Policy Optimization | On-policy | Stable training, general purpose |
| `ddpg` | Deep Deterministic Policy Gradient | Off-policy | Continuous actions, deterministic |
| `sac` | Soft Actor-Critic | Off-policy | Sample efficient, stochastic |
| `td3` | Twin Delayed DDPG | Off-policy | Improved DDPG, reduced overestimation |

### Key Methods

#### get_model()

Creates and configures an RL model.

```python
def get_model(
    self,
    model_name: str,
    policy: str = "MlpPolicy",
    policy_kwargs: dict = None,
    model_kwargs: dict = None,
    verbose: int = 1,
    seed: int = None,
    tensorboard_log: str = None
) -> BaseAlgorithm
```

**Parameters:**

- `model_name`: Algorithm name ("a2c", "ppo", "ddpg", "sac", "td3")
- `policy`: Neural network policy ("MlpPolicy" for dense networks)
- `policy_kwargs`: Policy network configuration
- `model_kwargs`: Algorithm-specific parameters
- `verbose`: Logging level (0=silent, 1=info, 2=debug)
- `seed`: Random seed for reproducibility
- `tensorboard_log`: Directory for TensorBoard logs

**Example:**
```python
agent = DRLAgent(env=train_env)

# Create PPO model with custom parameters
model = agent.get_model(
    model_name="ppo",
    model_kwargs={
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "ent_coef": 0.01
    },
    tensorboard_log="./ppo_trading_logs/"
)
```

#### train_model()

Trains the RL model on the environment.

```python
@staticmethod
def train_model(
    model,
    tb_log_name: str,
    total_timesteps: int = 5000,
    callbacks = None
) -> BaseAlgorithm
```

**Parameters:**

- `model`: Initialized model from `get_model()`
- `tb_log_name`: Name for TensorBoard logs
- `total_timesteps`: Number of training steps
- `callbacks`: List of training callbacks

**Example:**
```python
trained_model = DRLAgent.train_model(
    model=model,
    tb_log_name="ppo_stock_trading",
    total_timesteps=100000,
    callbacks=[checkpoint_callback, early_stopping_callback]
)
```

**Note for Off-Policy Algorithms (SAC, DDPG, TD3):**
```python
# For SAC, DDPG, TD3 - you might see rollout_buffer errors in logs
# These are harmless and don't affect training
trained_sac_model = DRLAgent.train_model(
    model=sac_model,
    tb_log_name="sac_crypto_trading",
    total_timesteps=50000,
    callbacks=[checkpoint_callback, eval_callback]
)
# Errors like "Logging Error: 'rollout_buffer'" are expected and can be ignored
```

#### DRL_prediction()

Makes predictions using a trained model.

```python
@staticmethod
def DRL_prediction(
    model,
    environment,
    deterministic: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]
```

**Returns:**
- `account_memory`: Portfolio values over time
- `actions_memory`: Actions taken at each step

**Example:**
```python
account_values, actions = DRLAgent.DRL_prediction(
    model=trained_model,
    environment=test_env,
    deterministic=True
)
```

## Algorithm-Specific Configurations

### PPO (Proximal Policy Optimization)

**Default Parameters:**
```python
PPO_PARAMS = {
    "n_steps": 2048,        # Steps per rollout
    "ent_coef": 0.01,       # Entropy coefficient
    "learning_rate": 0.00025,  # Learning rate
    "batch_size": 64,       # Minibatch size
    "gamma": 0.99,          # Discount factor
    "gae_lambda": 0.95,     # GAE lambda
    "clip_range": 0.2,      # PPO clip range
    "n_epochs": 10          # Optimization epochs
}
```

**Best For:** General-purpose trading, stable training

**Custom Configuration:**
```python
ppo_model = agent.get_model(
    "ppo",
    model_kwargs={
        "learning_rate": 1e-4,      # Lower LR for more stable training
        "n_steps": 4096,            # Larger rollouts
        "batch_size": 128,          # Larger batches
        "ent_coef": 0.001,          # Less exploration
        "clip_range": 0.1           # More conservative updates
    }
)
```

### A2C (Advantage Actor-Critic)

**Default Parameters:**
```python
A2C_PARAMS = {
    "n_steps": 5,           # Steps per update
    "ent_coef": 0.01,       # Entropy coefficient
    "learning_rate": 0.0007,  # Learning rate
    "gamma": 0.99,          # Discount factor
    "gae_lambda": 1.0,      # GAE lambda
    "vf_coef": 0.25         # Value function coefficient
}
```

**Best For:** Fast training, simple trading strategies

### SAC (Soft Actor-Critic)

**Default Parameters:**
```python
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",     # Automatic entropy tuning
    "gamma": 0.99,
    "tau": 0.005                # Soft update coefficient
}
```

**Best For:** Sample-efficient learning, continuous markets (crypto)

**Custom Configuration:**
```python
sac_model = agent.get_model(
    "sac",
    model_kwargs={
        "buffer_size": 1000000,     # Larger replay buffer
        "learning_starts": 1000,    # More initial exploration
        "ent_coef": "auto",         # Full automatic entropy tuning
        "train_freq": (4, "step")   # Train every 4 steps
    }
)
```

!!! warning "SAC and TensorBoard Logging"
    SAC is an off-policy algorithm and doesn't have a `rollout_buffer` like on-policy algorithms (PPO, A2C). If you see `rollout_buffer` errors, they come from FinRL's default `TensorboardCallback`. The errors are harmless but indicate the callback can't access certain metrics for off-policy algorithms.

### DDPG (Deep Deterministic Policy Gradient)

**Default Parameters:**
```python
DDPG_PARAMS = {
    "batch_size": 128,
    "buffer_size": 50000,
    "learning_rate": 0.001,
    "tau": 0.005,               # Soft update coefficient
    "gamma": 0.99,
    "action_noise": None,       # Exploration noise
    "train_freq": (1, "episode")
}
```

**Best For:** Deterministic trading policies

**With Action Noise:**
```python
ddpg_model = agent.get_model(
    "ddpg", 
    model_kwargs={
        "action_noise": "ornstein_uhlenbeck",  # Add exploration noise
        "batch_size": 256,
        "buffer_size": 200000,
        "learning_rate": 1e-3
    }
)
```

### TD3 (Twin Delayed DDPG)

**Default Parameters:**
```python
TD3_PARAMS = {
    "batch_size": 100,
    "buffer_size": 1000000,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "tau": 0.005,
    "policy_delay": 2,          # Policy update delay
    "target_policy_noise": 0.2, # Target policy noise
    "target_noise_clip": 0.5    # Target noise clip
}
```

**Best For:** Improved DDPG with reduced overestimation bias

## DRLEnsembleAgent

Advanced ensemble method that combines multiple RL algorithms and selects the best performer dynamically.

### Class Definition

```python
class DRLEnsembleAgent:
    def __init__(
        self,
        df,
        train_period,
        val_test_period, 
        rebalance_window,
        validation_window,
        **env_kwargs
    )
```

### Key Features

- **Dynamic Model Selection**: Chooses best algorithm based on validation Sharpe ratio
- **Rolling Window Training**: Retrains models periodically
- **Risk Management**: Integrated turbulence-based risk control
- **Multiple Algorithms**: Trains A2C, PPO, DDPG, SAC, TD3 simultaneously

### Usage Example

```python
from finrl.agents.stablebaselines3.models import DRLEnsembleAgent

# Create ensemble agent
ensemble_agent = DRLEnsembleAgent(
    df=processed_data,
    train_period=("2020-01-01", "2021-01-01"),
    val_test_period=("2021-01-01", "2022-01-01"), 
    rebalance_window=63,        # Retrain every 63 days
    validation_window=63,       # Validate on 63 days
    **env_kwargs
)

# Define algorithm configurations
model_configs = {
    "A2C_model_kwargs": {"n_steps": 10, "ent_coef": 0.005},
    "PPO_model_kwargs": {"n_steps": 2048, "ent_coef": 0.01},
    "DDPG_model_kwargs": {"buffer_size": 50000, "batch_size": 128},
    "SAC_model_kwargs": {"ent_coef": "auto", "batch_size": 64},
    "TD3_model_kwargs": {"policy_delay": 2, "batch_size": 100}
}

timesteps_dict = {
    "a2c": 50000,
    "ppo": 50000,
    "ddpg": 50000,
    "sac": 50000,
    "td3": 50000
}

# Run ensemble strategy
summary = ensemble_agent.run_ensemble_strategy(
    **model_configs,
    timesteps_dict=timesteps_dict
)

print("Model Selection Summary:")
print(summary[['Iter', 'Model Used', 'A2C Sharpe', 'PPO Sharpe', 'DDPG Sharpe', 'SAC Sharpe', 'TD3 Sharpe']])
```

## Advanced Usage

### Custom Policy Networks

```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class CustomTradingNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, features_dim)
        )
    
    def forward(self, observations):
        return self.net(observations)

# Use custom network
policy_kwargs = {
    "features_extractor_class": CustomTradingNetwork,
    "features_extractor_kwargs": {"features_dim": 512}
}

model = agent.get_model("ppo", policy_kwargs=policy_kwargs)
```

### Training Callbacks

```python
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Save model periodically
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./trading_models/",
    name_prefix="ppo_trading"
)

# Evaluate on validation set
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./eval_logs/",
    eval_freq=5000,
    deterministic=True
)

trained_model = DRLAgent.train_model(
    model,
    "ppo_with_callbacks",
    total_timesteps=200000,
    callbacks=[checkpoint_callback, eval_callback]
)
```

### Model Loading and Saving

```python
# Save trained model
trained_model.save("./models/ppo_trading_final")

# Load pre-trained model
from stable_baselines3 import PPO
loaded_model = PPO.load("./models/ppo_trading_final")

# Continue training
loaded_model.learn(total_timesteps=50000)
```

## Best Practices

### Algorithm Selection Guidelines

!!! tip "Choose the Right Algorithm"
    - **PPO**: General purpose, stable training
    - **SAC**: High sample efficiency, good for continuous markets
    - **DDPG/TD3**: When you need deterministic policies
    - **A2C**: Quick prototyping, simple problems

### Hyperparameter Tuning

```python
# Start with conservative parameters
conservative_params = {
    "learning_rate": 3e-5,      # Lower learning rate
    "ent_coef": 0.001,          # Less exploration
    "batch_size": 32,           # Smaller batches
}

# Gradually increase complexity
aggressive_params = {
    "learning_rate": 1e-3,      # Higher learning rate
    "ent_coef": 0.1,            # More exploration
    "batch_size": 512,          # Larger batches
}
```

### Training Monitoring

```python
# Monitor training with TensorBoard
# Launch: tensorboard --logdir ./trading_logs

model = agent.get_model(
    "ppo",
    tensorboard_log="./trading_logs/",
    model_kwargs={"verbose": 2}  # Detailed logging
)
```