# Training Process and Callbacks

This guide covers the complete training process in FinRL, including setup, execution, monitoring, and the use of callbacks for advanced training control.

## Training Overview

The FinRL training process consists of several key phases:

1. **Data Preparation** - Split and preprocess data
2. **Environment Setup** - Create training and validation environments  
3. **Model Initialization** - Configure the RL algorithm
4. **Callback Configuration** - Set up monitoring and control
5. **Training Execution** - Run the training loop
6. **Model Evaluation** - Test the trained model

## Basic Training Setup

### Simple Training Example

```python
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.vec_env import DummyVecEnv

# 1. Create agent
agent = DRLAgent(env=train_env)

# 2. Get model
model = agent.get_model(
    model_name="ppo",
    tensorboard_log="./training_logs/"
)

# 3. Train model
trained_model = DRLAgent.train_model(
    model=model,
    tb_log_name="ppo_trading",
    total_timesteps=100000
)

print("✅ Training completed")
```

### Complete Training Pipeline

```python
import os
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

def complete_training_pipeline(
    train_data, 
    val_data, 
    algorithm="ppo",
    total_timesteps=100000,
    experiment_name="trading_experiment"
):
    """Complete training pipeline with logging and callbacks"""
    
    # Create experiment directory
    experiment_dir = f"./experiments/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"🚀 Starting experiment: {experiment_name}")
    print(f"📁 Experiment directory: {experiment_dir}")
    
    # 1. Environment setup
    print("🔧 Setting up environments...")
    train_env = DummyVecEnv([lambda: create_env(train_data)])
    val_env = DummyVecEnv([lambda: create_env(val_data)])
    
    # 2. Model initialization
    print(f"🤖 Initializing {algorithm.upper()} model...")
    agent = DRLAgent(env=train_env)
    model = agent.get_model(
        model_name=algorithm,
        tensorboard_log=f"{experiment_dir}/tensorboard/"
    )
    
    # 3. Callback setup
    print("📊 Setting up callbacks...")
    callbacks = setup_callbacks(experiment_dir, val_env)
    
    # 4. Training execution
    print("🎯 Starting training...")
    start_time = datetime.now()
    
    trained_model = DRLAgent.train_model(
        model=model,
        tb_log_name=f"{algorithm}_{experiment_name}",
        total_timesteps=total_timesteps,
        callbacks=callbacks
    )
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    # 5. Save final model
    model_path = f"{experiment_dir}/final_model"
    trained_model.save(model_path)
    
    print(f"✅ Training completed in {training_duration}")
    print(f"💾 Model saved to: {model_path}")
    
    return trained_model, experiment_dir

# Run pipeline
trained_model, experiment_dir = complete_training_pipeline(
    train_data=train_data,
    val_data=val_data,
    algorithm="sac",
    total_timesteps=50000,
    experiment_name="crypto_sac_experiment"
)
```

## Training Callbacks

Callbacks provide powerful control over the training process. They allow you to save models, evaluate performance, implement early stopping, and more.

### Essential Callbacks

#### 1. Checkpoint Callback

Saves model periodically during training to prevent loss of progress.

```python
from stable_baselines3.common.callbacks import CheckpointCallback

# Basic checkpoint
checkpoint_callback = CheckpointCallback(
    save_freq=5000,                     # Save every 5000 steps
    save_path="./checkpoints/",         # Directory to save
    name_prefix="trading_model",        # Filename prefix
    save_replay_buffer=True,            # Save replay buffer for off-policy algorithms
    save_vecnormalize=True              # Save normalization stats
)

# Advanced checkpoint with custom naming
class TimestampedCheckpointCallback(CheckpointCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = datetime.now()
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Custom filename with timestamp and performance
            elapsed = datetime.now() - self.start_time
            model_path = f"{self.save_path}/model_{self.num_timesteps}_{elapsed.total_seconds():.0f}s"
            self.model.save(model_path)
            
            if self.verbose > 1:
                print(f"💾 Checkpoint saved: {model_path}")
        
        return True

# Use timestamped checkpoint
timestamped_checkpoint = TimestampedCheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/",
    verbose=2
)
```

#### 2. Evaluation Callback

Evaluates model performance on validation data during training.

```python
from stable_baselines3.common.callbacks import EvalCallback

# Basic evaluation
eval_callback = EvalCallback(
    eval_env=val_env,                   # Validation environment
    best_model_save_path="./best_model/", # Save best model here
    log_path="./eval_logs/",            # Evaluation logs
    eval_freq=2000,                     # Evaluate every 2000 steps
    n_eval_episodes=5,                  # Episodes per evaluation
    deterministic=True,                 # Use deterministic policy
    render=False,                       # Don't render during eval
    verbose=1
)

# Advanced evaluation with custom metrics
class TradingEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sharpe_history = []
        self.max_drawdown_history = []
    
    def _on_step(self) -> bool:
        # Run standard evaluation
        super()._on_step()
        
        if self.n_calls % self.eval_freq == 0:
            # Calculate custom trading metrics
            if hasattr(self.eval_env.envs[0], 'asset_memory'):
                returns = np.array(self.eval_env.envs[0].asset_memory)
                
                # Calculate Sharpe ratio
                daily_returns = np.diff(returns) / returns[:-1]
                sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)
                self.sharpe_history.append(sharpe)
                
                # Calculate max drawdown
                peak = np.maximum.accumulate(returns)
                drawdown = (returns - peak) / peak
                max_drawdown = np.min(drawdown)
                self.max_drawdown_history.append(max_drawdown)
                
                # Log custom metrics
                self.logger.record("eval/sharpe_ratio", sharpe)
                self.logger.record("eval/max_drawdown", max_drawdown)
                
                print(f"📈 Eval Sharpe: {sharpe:.3f}, Max DD: {max_drawdown:.3f}")
        
        return True

# Use trading evaluation
trading_eval = TradingEvalCallback(
    eval_env=val_env,
    best_model_save_path="./best_models/",
    log_path="./eval_logs/",
    eval_freq=5000,
    n_eval_episodes=10,
    deterministic=True
)
```

#### 3. Early Stopping Callback

Stops training when performance stops improving.

```python
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement

# Stop if no improvement for 10 evaluations
early_stopping = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=10,        # Stop after 10 evaluations without improvement
    min_evals=5,                        # Minimum evaluations before stopping
    verbose=1
)

# Combine with evaluation callback
eval_with_early_stopping = EvalCallback(
    eval_env=val_env,
    best_model_save_path="./best_model/",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
    callback_after_eval=early_stopping,  # Apply early stopping after eval
    verbose=1
)
```

### Custom Callbacks

#### Performance Monitoring Callback

```python
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

class PerformanceMonitorCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []
        
    def _init_callback(self) -> None:
        # Create plots directory
        os.makedirs("./plots", exist_ok=True)
    
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            # Get episode info
            info = self.locals.get('infos', [{}])[0]
            
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
            
            # Get portfolio value if available
            if hasattr(self.training_env.envs[0], 'asset_memory'):
                current_value = self.training_env.envs[0].asset_memory[-1]
                self.portfolio_values.append(current_value)
        
        # Plot progress periodically
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            self._plot_progress()
        
        return True
    
    def _plot_progress(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        if self.episode_rewards:
            axes[0, 0].plot(self.episode_rewards)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
        
        # Episode lengths
        if self.episode_lengths:
            axes[0, 1].plot(self.episode_lengths)
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
        
        # Portfolio values
        if self.portfolio_values:
            axes[1, 0].plot(self.portfolio_values)
            axes[1, 0].set_title('Portfolio Value')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Value')
        
        # Moving average of rewards
        if len(self.episode_rewards) > 10:
            window = min(50, len(self.episode_rewards) // 4)
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(moving_avg)
            axes[1, 1].set_title(f'Moving Average Rewards (window={window})')
        
        plt.tight_layout()
        plt.savefig(f"./plots/training_progress_step_{self.n_calls}.png")
        plt.close()
        
        if self.verbose > 0:
            print(f"📊 Progress plot saved: step {self.n_calls}")

# Use performance monitor
performance_monitor = PerformanceMonitorCallback(
    check_freq=5000,
    verbose=1
)
```

#### Learning Rate Scheduler Callback

```python
class LearningRateSchedulerCallback(BaseCallback):
    def __init__(self, initial_lr=3e-4, decay_rate=0.95, decay_freq=10000):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_freq = decay_freq
        self.current_lr = initial_lr
    
    def _on_step(self) -> bool:
        # Decay learning rate
        if self.n_calls % self.decay_freq == 0 and self.n_calls > 0:
            self.current_lr *= self.decay_rate
            
            # Update model learning rate
            if hasattr(self.model, 'learning_rate'):
                self.model.learning_rate = self.current_lr
            elif hasattr(self.model, 'lr_schedule'):
                # For models with learning rate schedules
                self.model.lr_schedule = lambda _: self.current_lr
            
            if self.verbose > 0:
                print(f"📉 Learning rate updated: {self.current_lr:.2e}")
            
            # Log learning rate
            self.logger.record("train/learning_rate", self.current_lr)
        
        return True

# Use LR scheduler
lr_scheduler = LearningRateSchedulerCallback(
    initial_lr=3e-4,
    decay_rate=0.95,
    decay_freq=20000
)
```

#### Action Analysis Callback

```python
class ActionAnalysisCallback(BaseCallback):
    def __init__(self, log_freq=5000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.actions_history = []
    
    def _on_step(self) -> bool:
        # Collect actions
        if 'actions' in self.locals:
            actions = self.locals['actions'].flatten()
            self.actions_history.append(actions.copy())
        
        # Analyze actions periodically
        if self.n_calls % self.log_freq == 0 and len(self.actions_history) > 100:
            self._analyze_actions()
        
        return True
    
    def _analyze_actions(self):
        """Analyze action statistics"""
        recent_actions = np.array(self.actions_history[-1000:])  # Last 1000 actions
        
        # Calculate statistics
        mean_actions = np.mean(recent_actions, axis=0)
        std_actions = np.std(recent_actions, axis=0)
        action_magnitude = np.mean(np.abs(recent_actions))
        zero_action_pct = np.mean(recent_actions == 0) * 100
        
        # Log statistics
        self.logger.record("actions/mean_magnitude", action_magnitude)
        self.logger.record("actions/zero_percentage", zero_action_pct)
        
        for i, (mean_val, std_val) in enumerate(zip(mean_actions, std_actions)):
            self.logger.record(f"actions/asset_{i}_mean", mean_val)
            self.logger.record(f"actions/asset_{i}_std", std_val)
        
        if self.verbose > 0:
            print(f"📊 Action Analysis - Magnitude: {action_magnitude:.4f}, "
                  f"Zero %: {zero_action_pct:.1f}%")

# Use action analysis
action_analysis = ActionAnalysisCallback(
    log_freq=5000,
    verbose=1
)
```

### Callback Combinations

#### Complete Callback Setup

```python
def setup_callbacks(experiment_dir, val_env, algorithm="ppo"):
    """Setup comprehensive callback suite"""
    
    callbacks = []
    
    # 1. Checkpoint callback
    checkpoint_cb = CheckpointCallback(
        save_freq=10000,
        save_path=f"{experiment_dir}/checkpoints/",
        name_prefix=f"{algorithm}_model",
        save_replay_buffer=True,
        verbose=1
    )
    callbacks.append(checkpoint_cb)
    
    # 2. Evaluation with early stopping
    early_stopping = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=15,
        min_evals=5,
        verbose=1
    )
    
    eval_cb = TradingEvalCallback(
        eval_env=val_env,
        best_model_save_path=f"{experiment_dir}/best_model/",
        log_path=f"{experiment_dir}/eval_logs/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        callback_after_eval=early_stopping,
        verbose=1
    )
    callbacks.append(eval_cb)
    
    # 3. Performance monitoring
    performance_cb = PerformanceMonitorCallback(
        check_freq=5000,
        verbose=1
    )
    callbacks.append(performance_cb)
    
    # 4. Action analysis
    action_cb = ActionAnalysisCallback(
        log_freq=5000,
        verbose=1
    )
    callbacks.append(action_cb)
    
    # 5. Learning rate scheduling (for certain algorithms)
    if algorithm in ['ppo', 'a2c']:
        lr_cb = LearningRateSchedulerCallback(
            initial_lr=3e-4,
            decay_rate=0.98,
            decay_freq=25000
        )
        callbacks.append(lr_cb)
    
    return callbacks

# Use in training
callbacks = setup_callbacks(experiment_dir, val_env, "sac")

trained_model = DRLAgent.train_model(
    model=model,
    tb_log_name="sac_trading_full",
    total_timesteps=100000,
    callbacks=callbacks
)
```

## Training Monitoring

### TensorBoard Integration

```python
# Launch TensorBoard in separate terminal:
# tensorboard --logdir ./training_logs/

# Or programmatically
import subprocess
import threading

def launch_tensorboard(log_dir="./training_logs/", port=6006):
    """Launch TensorBoard in background"""
    def run_tensorboard():
        cmd = f"tensorboard --logdir {log_dir} --port {port}"
        subprocess.run(cmd, shell=True)
    
    thread = threading.Thread(target=run_tensorboard, daemon=True)
    thread.start()
    print(f"🖥️ TensorBoard launched at http://localhost:{port}")

# Launch before training
launch_tensorboard()

# Train with tensorboard logging
model = agent.get_model(
    "ppo",
    tensorboard_log="./training_logs/",
    model_kwargs={"verbose": 2}
)
```

### Real-time Monitoring

```python
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt

class RealTimeMonitor:
    def __init__(self, refresh_interval=30):
        self.refresh_interval = refresh_interval
        self.metrics = {
            'episode_rewards': [],
            'portfolio_values': [],
            'steps': []
        }
    
    def update_metrics(self, env):
        """Update metrics from environment"""
        if hasattr(env.envs[0], 'asset_memory'):
            portfolio_value = env.envs[0].asset_memory[-1]
            self.metrics['portfolio_values'].append(portfolio_value)
            self.metrics['steps'].append(len(env.envs[0].asset_memory))
    
    def plot_live_metrics(self):
        """Plot live training metrics"""
        if len(self.metrics['portfolio_values']) < 2:
            return
        
        clear_output(wait=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Portfolio value
        ax1.plot(self.metrics['steps'], self.metrics['portfolio_values'])
        ax1.set_title('Portfolio Value')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Value')
        ax1.grid(True)
        
        # Returns
        values = np.array(self.metrics['portfolio_values'])
        returns = (values[1:] - values[:-1]) / values[:-1] * 100
        ax2.plot(returns)
        ax2.set_title('Episode Returns (%)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Return %')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def monitor_training(self, env):
        """Monitor training in real-time"""
        while True:
            time.sleep(self.refresh_interval)
            self.update_metrics(env)
            self.plot_live_metrics()

# Use real-time monitor (in Jupyter notebook)
monitor = RealTimeMonitor(refresh_interval=60)
# Start monitoring in background thread if needed
```

## Training Best Practices

### 1. Incremental Training

```python
def incremental_training(
    initial_data, 
    new_data_chunks, 
    base_model=None,
    timesteps_per_chunk=50000
):
    """Train incrementally on new data"""
    
    current_model = base_model
    
    for i, chunk in enumerate(new_data_chunks):
        print(f"📈 Training on chunk {i+1}/{len(new_data_chunks)}")
        
        # Create environment for this chunk
        chunk_env = DummyVecEnv([lambda: create_env(chunk)])
        
        if current_model is None:
            # Initial training
            agent = DRLAgent(env=chunk_env)
            current_model = agent.get_model("ppo")
        else:
            # Continue training existing model
            current_model.set_env(chunk_env)
        
        # Train on chunk
        current_model = DRLAgent.train_model(
            model=current_model,
            tb_log_name=f"incremental_chunk_{i}",
            total_timesteps=timesteps_per_chunk
        )
        
        # Save checkpoint
        current_model.save(f"./incremental_models/chunk_{i}_model")
    
    return current_model
```

### 2. Hyperparameter Validation

```python
def validate_training_setup(model, env, n_test_steps=1000):
    """Validate training setup before full training"""
    
    print("🔍 Validating training setup...")
    
    # Test environment
    obs = env.reset()
    for _ in range(10):
        action = model.predict(obs, deterministic=False)[0]
        obs, rewards, dones, infos = env.step(action)
        
        if np.isnan(rewards).any():
            raise ValueError("❌ NaN rewards detected")
        
        if np.isnan(obs).any():
            raise ValueError("❌ NaN observations detected")
    
    # Quick training test
    try:
        model.learn(total_timesteps=n_test_steps, tb_log_name="validation_test")
        print("✅ Training setup validation passed")
    except Exception as e:
        print(f"❌ Training setup validation failed: {e}")
        raise
    
    return True

# Validate before full training
validate_training_setup(model, train_env, n_test_steps=500)
```

### 3. Resource Management

```python
import psutil
import gc

class ResourceMonitor:
    def __init__(self, memory_limit_gb=8):
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        self.initial_memory = psutil.virtual_memory().used
    
    def check_resources(self):
        """Check system resources"""
        memory = psutil.virtual_memory()
        
        if memory.used > self.memory_limit_bytes:
            print(f"⚠️ Memory usage high: {memory.used / 1024**3:.1f} GB")
            self.cleanup_memory()
        
        if memory.percent > 90:
            print(f"⚠️ System memory critical: {memory.percent:.1f}%")
    
    def cleanup_memory(self):
        """Clean up memory"""
        gc.collect()
        print("🧹 Memory cleanup performed")
    
    def log_resources(self):
        """Log current resource usage"""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        print(f"💾 Memory: {memory.percent:.1f}% ({memory.used / 1024**3:.1f} GB)")
        print(f"🖥️ CPU: {cpu:.1f}%")

# Monitor during training
resource_monitor = ResourceMonitor(memory_limit_gb=8)

# Add to callback
class ResourceMonitorCallback(BaseCallback):
    def __init__(self, monitor, check_freq=10000):
        super().__init__()
        self.monitor = monitor
        self.check_freq = check_freq
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            self.monitor.check_resources()
        return True

# Use in training
resource_cb = ResourceMonitorCallback(resource_monitor, check_freq=5000)
```

## Troubleshooting Training Issues

### Common Problems and Solutions

#### 1. Training Not Progressing

```python
# Check if model is actually learning
def diagnose_learning(model, env, n_episodes=5):
    """Diagnose learning issues"""
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.4f}")
    
    avg_reward = np.mean(episode_rewards)
    print(f"Average reward: {avg_reward:.4f}")
    
    if avg_reward == episode_rewards[0]:
        print("⚠️ All episodes have same reward - model may not be learning")
    
    return episode_rewards

# Run diagnosis
diagnose_learning(model, train_env)
```

#### 2. Memory Issues

```python
# Reduce memory usage
def create_memory_efficient_config():
    """Create memory-efficient training configuration"""
    
    return {
        "buffer_size": 10000,        # Smaller replay buffer
        "batch_size": 32,            # Smaller batches
        "n_steps": 512,              # Fewer steps per update (PPO)
        "learning_starts": 100,      # Start learning earlier
        "train_freq": (4, "step"),   # Train less frequently
    }

# Use smaller network
memory_efficient_policy = {
    "net_arch": [64, 64],           # Smaller network
    "activation_fn": torch.nn.ReLU
}
```

#### 3. Unstable Training

```python
# Stabilize training
def create_stable_config():
    """Create configuration for stable training"""
    
    return {
        "learning_rate": 3e-5,       # Lower learning rate
        "clip_range": 0.1,           # Smaller clip range (PPO)
        "ent_coef": 0.001,           # Less exploration
        "max_grad_norm": 0.5,        # Gradient clipping
        "vf_coef": 0.25,             # Value function coefficient
    }
```

## Next Steps

1. **Implement Training**: Use the complete pipeline
2. **Set Up Monitoring**: Configure TensorBoard and callbacks
3. **Monitor Progress**: Watch metrics and logs
4. **Adjust Parameters**: Use [Hyperparameter Tuning](hyperparameter-tuning.md)
5. **Evaluate Results**: Test trained models thoroughly

Remember to start with simple configurations and gradually add complexity as you gain confidence in your setup. Always monitor resource usage and be prepared to adjust parameters based on training behavior.