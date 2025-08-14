# Hyperparameter Tuning

Hyperparameter tuning is crucial for achieving optimal performance in reinforcement learning for trading. This guide covers systematic approaches to find the best parameters for your FinRL models.

## Overview of Hyperparameter Tuning

### Why Tune Hyperparameters?

- **Performance**: Can improve returns by 20-50%
- **Stability**: Reduces training instability and variance
- **Efficiency**: Faster convergence and better sample efficiency
- **Robustness**: Better generalization to new market conditions

### Key Parameters to Tune

| Parameter Category | High Impact | Medium Impact | Low Impact |
|-------------------|-------------|---------------|------------|
| **Learning** | learning_rate | batch_size | n_epochs |
| **Exploration** | ent_coef | ent_coef_decay | target_entropy |
| **Network** | net_arch | activation_fn | optimizer |
| **Environment** | reward_scaling | transaction_costs | hmax |
| **Algorithm** | buffer_size | tau | gamma |

## Manual Tuning Strategies

### 1. Grid Search Approach

```python
import itertools
from typing import Dict, List, Any

def grid_search_tuning(
    param_grid: Dict[str, List[Any]],
    train_data,
    val_data,
    algorithm="ppo",
    base_timesteps=50000
):
    """Manual grid search for hyperparameter tuning"""
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    results = []
    best_performance = -float('inf')
    best_params = None
    
    print(f"🔍 Testing {len(param_combinations)} parameter combinations...")
    
    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_names, params))
        print(f"\n📊 Combination {i+1}/{len(param_combinations)}: {param_dict}")
        
        try:
            # Train model with these parameters
            performance = train_and_evaluate(
                train_data, val_data, algorithm, param_dict, base_timesteps
            )
            
            results.append({
                'params': param_dict.copy(),
                'performance': performance,
                'sharpe_ratio': performance.get('sharpe_ratio', 0),
                'total_return': performance.get('total_return', 0)
            })
            
            # Track best performance
            if performance['sharpe_ratio'] > best_performance:
                best_performance = performance['sharpe_ratio']
                best_params = param_dict.copy()
                print(f"🎯 New best Sharpe ratio: {best_performance:.3f}")
        
        except Exception as e:
            print(f"❌ Training failed: {e}")
            results.append({
                'params': param_dict.copy(),
                'performance': None,
                'error': str(e)
            })
    
    return results, best_params

# Example usage
param_grid = {
    'learning_rate': [1e-5, 3e-5, 1e-4, 3e-4],
    'ent_coef': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'reward_scaling': [1e-5, 1e-4, 1e-3]
}

results, best_params = grid_search_tuning(param_grid, train_data, val_data)
print(f"🏆 Best parameters: {best_params}")
```

### 2. Random Search

```python
import random
import numpy as np

def random_search_tuning(
    param_ranges: Dict[str, tuple],
    train_data,
    val_data,
    n_trials=20,
    algorithm="sac"
):
    """Random search for hyperparameter tuning"""
    
    results = []
    best_performance = -float('inf')
    best_params = None
    
    print(f"🎲 Starting random search with {n_trials} trials...")
    
    for trial in range(n_trials):
        # Sample random parameters
        params = {}
        for param_name, (min_val, max_val, param_type) in param_ranges.items():
            if param_type == 'log':
                # Log-uniform sampling for learning rates
                params[param_name] = 10 ** random.uniform(np.log10(min_val), np.log10(max_val))
            elif param_type == 'int':
                params[param_name] = random.randint(min_val, max_val)
            elif param_type == 'choice':
                params[param_name] = random.choice(min_val)  # min_val is actually the choices list
            else:
                params[param_name] = random.uniform(min_val, max_val)
        
        print(f"\n🔄 Trial {trial+1}/{n_trials}: {params}")
        
        try:
            performance = train_and_evaluate(
                train_data, val_data, algorithm, params, 30000
            )
            
            results.append({
                'trial': trial + 1,
                'params': params.copy(),
                'performance': performance
            })
            
            if performance['sharpe_ratio'] > best_performance:
                best_performance = performance['sharpe_ratio']
                best_params = params.copy()
                print(f"🎯 New best Sharpe: {best_performance:.3f}")
        
        except Exception as e:
            print(f"❌ Trial failed: {e}")
    
    return results, best_params

# Define parameter ranges
param_ranges = {
    'learning_rate': (1e-5, 1e-3, 'log'),
    'ent_coef': (0.001, 0.1, 'log'),
    'batch_size': ([32, 64, 128, 256], None, 'choice'),
    'buffer_size': (10000, 1000000, 'int'),
    'reward_scaling': (1e-6, 1e-2, 'log')
}

results, best_params = random_search_tuning(param_ranges, train_data, val_data, n_trials=15)
```

### 3. Bayesian Optimization

```python
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    
    def bayesian_optimization_tuning(train_data, val_data, algorithm="ppo", n_calls=25):
        """Bayesian optimization for hyperparameter tuning"""
        
        # Define search space
        space = [
            Real(1e-5, 1e-3, "log-uniform", name='learning_rate'),
            Real(0.001, 0.1, "log-uniform", name='ent_coef'),
            Categorical([32, 64, 128, 256], name='batch_size'),
            Real(1e-6, 1e-2, "log-uniform", name='reward_scaling'),
            Integer(1000, 100000, name='buffer_size') if algorithm in ['sac', 'ddpg', 'td3'] else 
            Integer(512, 4096, name='n_steps')  # For on-policy algorithms
        ]
        
        @use_named_args(space)
        def objective(**params):
            """Objective function to minimize (negative Sharpe ratio)"""
            print(f"🔍 Testing: {params}")
            
            try:
                performance = train_and_evaluate(
                    train_data, val_data, algorithm, params, 25000
                )
                sharpe_ratio = performance.get('sharpe_ratio', 0)
                
                # Return negative because we want to maximize Sharpe ratio
                return -sharpe_ratio
            
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
                return 10  # High penalty for failed runs
        
        # Run Bayesian optimization
        print(f"🚀 Starting Bayesian optimization with {n_calls} evaluations...")
        
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=5,
            acq_func='EI',  # Expected Improvement
            random_state=42
        )
        
        # Extract best parameters
        best_params = {}
        for i, param_name in enumerate(['learning_rate', 'ent_coef', 'batch_size', 
                                      'reward_scaling', 'buffer_size' if algorithm in ['sac', 'ddpg', 'td3'] else 'n_steps']):
            best_params[param_name] = result.x[i]
        
        print(f"🏆 Best parameters found: {best_params}")
        print(f"🎯 Best Sharpe ratio: {-result.fun:.3f}")
        
        return best_params, result

except ImportError:
    print("⚠️ scikit-optimize not installed. Use: pip install scikit-optimize")
    
    def bayesian_optimization_tuning(*args, **kwargs):
        raise ImportError("Please install scikit-optimize for Bayesian optimization")
```

## Automated Tuning with Optuna

```python
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    
    def optuna_tuning(
        train_data, 
        val_data, 
        algorithm="sac", 
        n_trials=50,
        timeout=3600  # 1 hour timeout
    ):
        """Advanced hyperparameter tuning with Optuna"""
        
        def objective(trial):
            """Objective function for Optuna optimization"""
            
            # Suggest hyperparameters based on algorithm
            if algorithm == "ppo":
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                    'n_steps': trial.suggest_int('n_steps', 512, 4096, step=512),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                    'ent_coef': trial.suggest_float('ent_coef', 0.001, 0.1, log=True),
                    'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
                    'n_epochs': trial.suggest_int('n_epochs', 5, 20),
                    'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                    'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99)
                }
            
            elif algorithm == "sac":
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
                    'buffer_size': trial.suggest_int('buffer_size', 10000, 1000000, log=True),
                    'learning_starts': trial.suggest_int('learning_starts', 100, 5000),
                    'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8]),
                    'gradient_steps': trial.suggest_int('gradient_steps', 1, 8),
                    'ent_coef': trial.suggest_categorical('ent_coef', ['auto', 'auto_0.1', 0.01, 0.1]),
                    'tau': trial.suggest_float('tau', 0.001, 0.1, log=True),
                    'gamma': trial.suggest_float('gamma', 0.9, 0.999)
                }
            
            elif algorithm == "ddpg":
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
                    'buffer_size': trial.suggest_int('buffer_size', 10000, 500000, log=True),
                    'learning_starts': trial.suggest_int('learning_starts', 100, 2000),
                    'tau': trial.suggest_float('tau', 0.001, 0.1, log=True),
                    'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                    'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8]),
                    'gradient_steps': trial.suggest_int('gradient_steps', -1, 8)
                }
            
            # Environment parameters
            env_params = {
                'reward_scaling': trial.suggest_float('reward_scaling', 1e-6, 1e-2, log=True),
                'hmax': trial.suggest_int('hmax', 50, 500),
                'transaction_cost': trial.suggest_float('transaction_cost', 0.0001, 0.01, log=True)
            }
            
            # Network architecture
            net_arch_choice = trial.suggest_categorical('net_arch', [
                [64, 64], [128, 128], [256, 256], 
                [64, 64, 64], [128, 128, 128],
                [256, 128], [512, 256]
            ])
            
            activation_fn = trial.suggest_categorical('activation_fn', ['relu', 'tanh'])
            
            try:
                # Train and evaluate with suggested parameters
                performance = train_and_evaluate_with_pruning(
                    train_data, val_data, algorithm, params, env_params, 
                    net_arch_choice, activation_fn, trial, 30000
                )
                
                return performance['sharpe_ratio']
            
            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"❌ Trial failed: {e}")
                return -10  # Large penalty for failed trials
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        print(f"🚀 Starting Optuna optimization for {algorithm.upper()}")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        print(f"🏆 Best trial: {study.best_trial.number}")
        print(f"🎯 Best Sharpe ratio: {study.best_value:.3f}")
        print(f"📊 Best parameters: {study.best_params}")
        
        return study.best_params, study
    
    def train_and_evaluate_with_pruning(
        train_data, val_data, algorithm, params, env_params, 
        net_arch, activation_fn, trial, timesteps
    ):
        """Train with intermediate pruning based on performance"""
        
        # Create environments with custom parameters
        env_kwargs = {
            'hmax': env_params['hmax'],
            'initial_amount': 1000000,
            'buy_cost_pct': [env_params['transaction_cost']] * len(train_data['tic'].unique()),
            'sell_cost_pct': [env_params['transaction_cost']] * len(train_data['tic'].unique()),
            'reward_scaling': env_params['reward_scaling'],
            'num_stock_shares': [0] * len(train_data['tic'].unique())
        }
        
        train_env = DummyVecEnv([lambda: create_env(train_data, **env_kwargs)])
        val_env = DummyVecEnv([lambda: create_env(val_data, **env_kwargs)])
        
        # Setup model
        agent = DRLAgent(env=train_env)
        
        policy_kwargs = {
            'net_arch': net_arch,
            'activation_fn': torch.nn.ReLU if activation_fn == 'relu' else torch.nn.Tanh
        }
        
        model = agent.get_model(
            algorithm,
            model_kwargs=params,
            policy_kwargs=policy_kwargs
        )
        
        # Training with intermediate evaluation for pruning
        checkpoint_freq = timesteps // 5  # Evaluate 5 times during training
        
        for checkpoint in range(1, 6):
            current_timesteps = checkpoint * checkpoint_freq
            
            # Train for this checkpoint
            model.learn(total_timesteps=checkpoint_freq)
            
            # Evaluate intermediate performance
            test_performance = evaluate_model(model, val_env, n_episodes=3)
            intermediate_sharpe = test_performance.get('sharpe_ratio', 0)
            
            # Report intermediate value for pruning
            trial.report(intermediate_sharpe, checkpoint)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Final evaluation
        final_performance = evaluate_model(model, val_env, n_episodes=10)
        
        return final_performance

except ImportError:
    print("⚠️ Optuna not installed. Use: pip install optuna")
    
    def optuna_tuning(*args, **kwargs):
        raise ImportError("Please install optuna for advanced hyperparameter tuning")
```

## Algorithm-Specific Tuning Guidelines

### PPO Tuning

```python
def tune_ppo_hyperparameters(train_data, val_data):
    """Specific tuning guidelines for PPO"""
    
    # Priority order for PPO tuning
    tuning_stages = [
        # Stage 1: Core learning parameters
        {
            'learning_rate': [1e-5, 3e-5, 1e-4, 3e-4],
            'n_steps': [1024, 2048, 4096],
            'ent_coef': [0.001, 0.01, 0.1]
        },
        # Stage 2: Training dynamics
        {
            'batch_size': [32, 64, 128],
            'n_epochs': [5, 10, 20],
            'clip_range': [0.1, 0.2, 0.3]
        },
        # Stage 3: Fine-tuning
        {
            'gamma': [0.99, 0.995, 0.999],
            'gae_lambda': [0.9, 0.95, 0.99],
            'vf_coef': [0.25, 0.5, 1.0]
        }
    ]
    
    best_params = {}
    
    for stage_num, stage_params in enumerate(tuning_stages, 1):
        print(f"🔍 PPO Tuning Stage {stage_num}: {list(stage_params.keys())}")
        
        stage_results, stage_best = grid_search_tuning(
            stage_params, train_data, val_data, "ppo", 25000
        )
        
        best_params.update(stage_best)
        print(f"✅ Stage {stage_num} complete. Best so far: {best_params}")
    
    return best_params

# PPO-specific tips
ppo_tips = """
PPO Hyperparameter Tips:
1. Start with learning_rate=3e-4, n_steps=2048
2. Increase n_steps for more stable gradients
3. Higher ent_coef for more exploration
4. clip_range=0.2 is usually good starting point
5. batch_size should be <= n_steps
"""
print(ppo_tips)
```

### SAC Tuning

```python
def tune_sac_hyperparameters(train_data, val_data):
    """Specific tuning guidelines for SAC"""
    
    # SAC parameter sensitivity (high to low)
    sac_priorities = {
        'high_impact': ['learning_rate', 'ent_coef', 'batch_size'],
        'medium_impact': ['buffer_size', 'learning_starts', 'train_freq'],
        'low_impact': ['tau', 'gamma', 'gradient_steps']
    }
    
    # Stage 1: High impact parameters
    high_impact_grid = {
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'ent_coef': ['auto', 'auto_0.1', 0.01],
        'batch_size': [128, 256, 512]
    }
    
    print("🔍 SAC Stage 1: High impact parameters")
    stage1_results, stage1_best = grid_search_tuning(
        high_impact_grid, train_data, val_data, "sac", 30000
    )
    
    # Stage 2: Medium impact parameters
    medium_impact_grid = {
        'buffer_size': [50000, 100000, 500000],
        'learning_starts': [1000, 5000, 10000],
        'train_freq': [1, 4, 8]
    }
    
    print("🔍 SAC Stage 2: Medium impact parameters")
    stage2_results, stage2_best = grid_search_tuning(
        {**stage1_best, **medium_impact_grid}, 
        train_data, val_data, "sac", 30000
    )
    
    return {**stage1_best, **stage2_best}

# SAC-specific tips
sac_tips = """
SAC Hyperparameter Tips:
1. ent_coef='auto' usually works well
2. Larger buffer_size improves stability
3. batch_size=256 is good starting point
4. learning_starts should be > batch_size
5. train_freq=1 for sample efficiency
"""
print(sac_tips)
```

## Environment Parameter Tuning

### Reward Scaling Optimization

```python
def optimize_reward_scaling(train_data, val_data, algorithm="ppo"):
    """Find optimal reward scaling through systematic testing"""
    
    # Test different reward scaling values
    scaling_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    results = []
    
    for scaling in scaling_values:
        print(f"🔍 Testing reward_scaling: {scaling}")
        
        env_kwargs = {
            'reward_scaling': scaling,
            'hmax': 100,
            'initial_amount': 1000000,
            'buy_cost_pct': [0.001] * len(train_data['tic'].unique()),
            'sell_cost_pct': [0.001] * len(train_data['tic'].unique()),
            'num_stock_shares': [0] * len(train_data['tic'].unique())
        }
        
        # Quick training to test reward scaling
        train_env = DummyVecEnv([lambda: create_env(train_data, **env_kwargs)])
        val_env = DummyVecEnv([lambda: create_env(val_data, **env_kwargs)])
        
        agent = DRLAgent(env=train_env)
        model = agent.get_model(algorithm)
        
        # Short training to evaluate scaling
        model = DRLAgent.train_model(
            model, f"reward_scaling_test_{scaling}", 10000
        )
        
        performance = evaluate_model(model, val_env)
        results.append({
            'reward_scaling': scaling,
            'sharpe_ratio': performance['sharpe_ratio'],
            'total_return': performance['total_return'],
            'max_drawdown': performance['max_drawdown']
        })
        
        print(f"📊 Scaling {scaling}: Sharpe={performance['sharpe_ratio']:.3f}")
    
    # Find best scaling
    best_result = max(results, key=lambda x: x['sharpe_ratio'])
    print(f"🏆 Best reward scaling: {best_result['reward_scaling']}")
    
    return best_result['reward_scaling'], results
```

### Transaction Cost Impact

```python
def analyze_transaction_cost_impact(train_data, val_data):
    """Analyze impact of different transaction costs"""
    
    cost_levels = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01]
    results = []
    
    for cost in cost_levels:
        print(f"💰 Testing transaction cost: {cost*100:.2f}%")
        
        env_kwargs = {
            'buy_cost_pct': [cost] * len(train_data['tic'].unique()),
            'sell_cost_pct': [cost] * len(train_data['tic'].unique()),
            'reward_scaling': 1e-4,
            'hmax': 100,
            'initial_amount': 1000000,
            'num_stock_shares': [0] * len(train_data['tic'].unique())
        }
        
        # Test with different costs
        performance = quick_backtest(train_data, val_data, env_kwargs)
        
        results.append({
            'transaction_cost': cost,
            'cost_percentage': cost * 100,
            'net_return': performance['total_return'],
            'sharpe_ratio': performance['sharpe_ratio'],
            'total_trades': performance.get('total_trades', 0)
        })
    
    # Plot results
    import matplotlib.pyplot as plt
    
    costs = [r['cost_percentage'] for r in results]
    returns = [r['net_return'] for r in results]
    sharpes = [r['sharpe_ratio'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(costs, returns, 'b-o')
    ax1.set_xlabel('Transaction Cost (%)')
    ax1.set_ylabel('Net Return (%)')
    ax1.set_title('Return vs Transaction Cost')
    ax1.grid(True)
    
    ax2.plot(costs, sharpes, 'r-o')
    ax2.set_xlabel('Transaction Cost (%)')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratio vs Transaction Cost')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('./transaction_cost_analysis.png')
    plt.show()
    
    return results
```

## Tuning Evaluation and Validation

### Cross-Validation for RL

```python
def time_series_cross_validation(
    data, 
    n_splits=5, 
    test_size_ratio=0.2,
    algorithm="ppo",
    params=None
):
    """Time series cross-validation for hyperparameter validation"""
    
    dates = sorted(data['date'].unique())
    n_dates = len(dates)
    test_size = int(n_dates * test_size_ratio)
    
    cv_results = []
    
    for split in range(n_splits):
        # Calculate split indices
        start_idx = split * (n_dates - test_size) // (n_splits - 1) if n_splits > 1 else 0
        train_end_idx = start_idx + (n_dates - test_size)
        test_end_idx = train_end_idx + test_size
        
        # Create train/test splits
        train_dates = dates[start_idx:train_end_idx]
        test_dates = dates[train_end_idx:test_end_idx]
        
        train_split = data[data['date'].isin(train_dates)]
        test_split = data[data['date'].isin(test_dates)]
        
        print(f"🔄 CV Split {split+1}/{n_splits}: "
              f"Train {train_dates[0]} to {train_dates[-1]}, "
              f"Test {test_dates[0]} to {test_dates[-1]}")
        
        # Train and evaluate
        try:
            performance = train_and_evaluate(
                train_split, test_split, algorithm, params or {}, 25000
            )
            cv_results.append(performance)
        except Exception as e:
            print(f"❌ CV split failed: {e}")
            cv_results.append(None)
    
    # Calculate cross-validation statistics
    valid_results = [r for r in cv_results if r is not None]
    
    if valid_results:
        cv_stats = {
            'mean_sharpe': np.mean([r['sharpe_ratio'] for r in valid_results]),
            'std_sharpe': np.std([r['sharpe_ratio'] for r in valid_results]),
            'mean_return': np.mean([r['total_return'] for r in valid_results]),
            'std_return': np.std([r['total_return'] for r in valid_results]),
            'success_rate': len(valid_results) / n_splits
        }
        
        print(f"📊 CV Results:")
        print(f"  Mean Sharpe: {cv_stats['mean_sharpe']:.3f} ± {cv_stats['std_sharpe']:.3f}")
        print(f"  Mean Return: {cv_stats['mean_return']:.3f} ± {cv_stats['std_return']:.3f}")
        print(f"  Success Rate: {cv_stats['success_rate']:.1%}")
        
        return cv_stats
    else:
        print("❌ All CV splits failed")
        return None
```

### Parameter Sensitivity Analysis

```python
def parameter_sensitivity_analysis(train_data, val_data, base_params, algorithm="ppo"):
    """Analyze sensitivity to individual parameters"""
    
    # Parameters to test sensitivity
    sensitivity_params = {
        'learning_rate': [base_params['learning_rate'] * f for f in [0.1, 0.5, 2.0, 5.0]],
        'ent_coef': [base_params.get('ent_coef', 0.01) * f for f in [0.1, 0.5, 2.0, 5.0]],
        'batch_size': [32, 64, 128, 256] if 'batch_size' in base_params else None
    }
    
    sensitivity_results = {}
    
    for param_name, param_values in sensitivity_params.items():
        if param_values is None:
            continue
            
        print(f"🔍 Sensitivity analysis for {param_name}")
        param_results = []
        
        for value in param_values:
            test_params = base_params.copy()
            test_params[param_name] = value
            
            try:
                performance = train_and_evaluate(
                    train_data, val_data, algorithm, test_params, 20000
                )
                param_results.append({
                    'value': value,
                    'sharpe_ratio': performance['sharpe_ratio'],
                    'total_return': performance['total_return']
                })
                print(f"  {param_name}={value}: Sharpe={performance['sharpe_ratio']:.3f}")
            except Exception as e:
                print(f"  {param_name}={value}: FAILED ({e})")
        
        sensitivity_results[param_name] = param_results
    
    # Plot sensitivity
    import matplotlib.pyplot as plt
    
    n_params = len(sensitivity_results)
    fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 4))
    if n_params == 1:
        axes = [axes]
    
    for i, (param_name, results) in enumerate(sensitivity_results.items()):
        values = [r['value'] for r in results]
        sharpes = [r['sharpe_ratio'] for r in results]
        
        axes[i].plot(values, sharpes, 'b-o')
        axes[i].set_xlabel(param_name)
        axes[i].set_ylabel('Sharpe Ratio')
        axes[i].set_title(f'Sensitivity: {param_name}')
        axes[i].grid(True)
        
        # Highlight base value
        base_value = base_params.get(param_name)
        if base_value in values:
            base_idx = values.index(base_value)
            axes[i].plot(base_value, sharpes[base_idx], 'ro', markersize=10, label='Base')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('./parameter_sensitivity.png')
    plt.show()
    
    return sensitivity_results
```

## Tuning Best Practices

### 1. Tuning Workflow

```python
def complete_hyperparameter_tuning_workflow(train_data, val_data, test_data, algorithm="sac"):
    """Complete hyperparameter tuning workflow"""
    
    print("🚀 Starting Complete Hyperparameter Tuning Workflow")
    
    # Stage 1: Quick parameter screening
    print("\n📊 Stage 1: Quick Parameter Screening")
    quick_params = {
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'reward_scaling': [1e-5, 1e-4, 1e-3]
    }
    
    quick_results, best_quick = grid_search_tuning(
        quick_params, train_data, val_data, algorithm, 10000
    )
    
    # Stage 2: Focused optimization
    print("\n🎯 Stage 2: Focused Optimization")
    if 'optuna' in globals():
        best_params, study = optuna_tuning(
            train_data, val_data, algorithm, n_trials=30
        )
    else:
        # Fallback to random search
        param_ranges = {
            'learning_rate': (1e-5, 1e-3, 'log'),
            'ent_coef': (0.001, 0.1, 'log') if algorithm in ['ppo', 'a2c'] else ('auto', None, 'choice'),
            'batch_size': ([64, 128, 256], None, 'choice'),
            'reward_scaling': (1e-6, 1e-2, 'log')
        }
        _, best_params = random_search_tuning(
            param_ranges, train_data, val_data, 20, algorithm
        )
    
    # Stage 3: Validation
    print("\n✅ Stage 3: Cross-Validation")
    cv_stats = time_series_cross_validation(
        train_data, n_splits=3, algorithm=algorithm, params=best_params
    )
    
    # Stage 4: Sensitivity analysis
    print("\n🔍 Stage 4: Sensitivity Analysis")
    sensitivity = parameter_sensitivity_analysis(
        train_data, val_data, best_params, algorithm
    )
    
    # Stage 5: Final test
    print("\n🏁 Stage 5: Final Test on Hold-out Set")
    final_performance = train_and_evaluate(
        train_data, test_data, algorithm, best_params, 50000
    )
    
    # Summary
    print("\n📋 TUNING SUMMARY")
    print("=" * 50)
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Best Parameters: {best_params}")
    print(f"CV Sharpe: {cv_stats['mean_sharpe']:.3f} ± {cv_stats['std_sharpe']:.3f}")
    print(f"Test Sharpe: {final_performance['sharpe_ratio']:.3f}")
    print(f"Test Return: {final_performance['total_return']:.3f}%")
    
    return {
        'best_params': best_params,
        'cv_stats': cv_stats,
        'final_performance': final_performance,
        'sensitivity': sensitivity
    }
```

### 2. Common Tuning Mistakes

```python
# ❌ Common mistakes to avoid
tuning_mistakes = """
Common Hyperparameter Tuning Mistakes:

1. Data Leakage:
   ❌ Using future data in training
   ✅ Strict chronological splits

2. Overfitting to Validation Set:
   ❌ Too many tuning iterations
   ✅ Use hold-out test set

3. Insufficient Training:
   ❌ Too few timesteps for evaluation
   ✅ Adequate training time per trial

4. Parameter Range Issues:
   ❌ Too narrow/wide search ranges
   ✅ Start with literature values

5. Ignoring Computational Constraints:
   ❌ Unrealistic parameter combinations
   ✅ Consider training time/memory

6. Single Metric Optimization:
   ❌ Only optimizing Sharpe ratio
   ✅ Consider multiple metrics
"""

print(tuning_mistakes)
```

### 3. Quick Reference Guidelines

```python
# Quick hyperparameter reference
hyperparameter_quick_reference = {
    'PPO': {
        'learning_rate': '3e-4 (start here)',
        'n_steps': '2048 (increase for stability)',
        'batch_size': '64 (≤ n_steps)',
        'ent_coef': '0.01 (increase for exploration)',
        'clip_range': '0.2 (standard)',
        'n_epochs': '10 (increase carefully)'
    },
    
    'SAC': {
        'learning_rate': '3e-4 (robust default)',
        'batch_size': '256 (larger is often better)',
        'buffer_size': '100000 (increase if memory allows)',
        'ent_coef': 'auto (let SAC tune it)',
        'learning_starts': '1000 (> batch_size)',
        'train_freq': '1 (train every step)'
    },
    
    'Environment': {
        'reward_scaling': '1e-4 (adjust based on price scale)',
        'transaction_cost': '0.001 (0.1% realistic)',
        'hmax': '100 (reasonable position sizes)',
        'initial_amount': '1000000 (match real capital)'
    }
}

# Print quick reference
for category, params in hyperparameter_quick_reference.items():
    print(f"\n{category} Parameters:")
    for param, guideline in params.items():
        print(f"  {param}: {guideline}")
```

## Next Steps

1. **Start Simple**: Begin with literature defaults
2. **Screen Quickly**: Use fast, small-scale screening
3. **Focus Optimization**: Deep dive on promising regions
4. **Validate Thoroughly**: Use proper cross-validation
5. **Test Finally**: Evaluate on true hold-out set

Remember that hyperparameter tuning is iterative - start with reasonable defaults, validate systematically, and be prepared to revisit your choices as you gain more data and experience.