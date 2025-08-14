# Data Format Requirements

## Overview

FinRL expects data in a specific pandas DataFrame format. Understanding these requirements is crucial for successful implementation of trading strategies.

## Required DataFrame Structure

### Core Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | `str` | Trading date in YYYY-MM-DD format | `"2023-01-15"` |
| `tic` | `str` | Asset ticker/symbol | `"AAPL"`, `"BTC-USD"` |
| `close` | `float` | Closing price | `150.25` |

### Optional Columns

| Column | Type | Description | Usage |
|--------|------|-------------|-------|
| `open` | `float` | Opening price | OHLCV data, additional features |
| `high` | `float` | Highest price | OHLCV data, volatility calculation |
| `low` | `float` | Lowest price | OHLCV data, support/resistance |
| `volume` | `float` | Trading volume | Liquidity analysis, volume indicators |
| `vix` | `float` | Volatility index | Market sentiment, risk assessment |
| `turbulence` | `float` | Market turbulence metric | Risk management |

## Data Structure Examples

### Single Asset Format

```python
import pandas as pd

# Single asset data (e.g., Bitcoin)
df_single = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'tic': ['BTC-USD', 'BTC-USD', 'BTC-USD'], 
    'open': [16500.0, 16800.0, 17100.0],
    'high': [16900.0, 17200.0, 17500.0],
    'low': [16400.0, 16700.0, 17000.0],
    'close': [16800.0, 17100.0, 17400.0],
    'volume': [25000000, 28000000, 32000000]
})

print(df_single)
```

```
        date       tic     open     high      low    close     volume
0  2023-01-01   BTC-USD  16500.0  16900.0  16400.0  16800.0   25000000
1  2023-01-02   BTC-USD  16800.0  17200.0  16700.0  17100.0   28000000
2  2023-01-03   BTC-USD  17100.0  17500.0  17000.0  17400.0   32000000
```

### Multi-Asset Format

```python
# Multi-asset data (stocks + crypto)
df_multi = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
    'tic': ['AAPL', 'MSFT', 'AAPL', 'MSFT'],
    'close': [130.50, 240.80, 132.20, 243.10],
    'volume': [89000000, 45000000, 92000000, 47000000]
})

print(df_multi)
```

```
        date   tic   close     volume
0  2023-01-01  AAPL  130.50   89000000
1  2023-01-01  MSFT  240.80   45000000  
2  2023-01-02  AAPL  132.20   92000000
3  2023-01-02  MSFT  243.10   47000000
```

## Technical Indicators

### Built-in Indicators

FinRL includes these technical indicators by default:

```python
from finrl.config import INDICATORS

print("Default indicators:", INDICATORS)
# Output: ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']
```

### Indicator Descriptions

| Indicator | Full Name | Description |
|-----------|-----------|-------------|
| `macd` | Moving Average Convergence Divergence | Trend-following momentum indicator |
| `boll_ub` | Bollinger Bands Upper | Upper boundary of price volatility |
| `boll_lb` | Bollinger Bands Lower | Lower boundary of price volatility |
| `rsi_30` | Relative Strength Index (30-day) | Overbought/oversold oscillator |
| `cci_30` | Commodity Channel Index (30-day) | Momentum oscillator |
| `dx_30` | Directional Movement Index (30-day) | Trend strength indicator |
| `close_30_sma` | Simple Moving Average (30-day) | Price trend smoothing |
| `close_60_sma` | Simple Moving Average (60-day) | Long-term price trend |

### Adding Technical Indicators

```python
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

# Define custom indicators
CUSTOM_INDICATORS = [
    "macd", "rsi_14", "rsi_30",
    "cci_14", "cci_30", 
    "dx_14", "dx_30",
    "boll_ub", "boll_lb",
    "close_14_sma", "close_30_sma", "close_60_sma",
    "volume_14_sma", "volume_30_sma"
]

# Initialize feature engineer
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=CUSTOM_INDICATORS,
    use_turbulence=True,
    user_defined_feature=False
)

# Process data with technical indicators
processed_df = fe.preprocess_data(raw_df)
```

## Data Validation and Preprocessing

### Required Data Checks

```python
def validate_data(df):
    """Validate DataFrame format for FinRL"""
    
    # Check required columns
    required_cols = ['date', 'tic', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check data types
    if df['date'].dtype != 'object':
        raise ValueError("Date column must be string type")
    
    if not pd.api.types.is_numeric_dtype(df['close']):
        raise ValueError("Close column must be numeric")
    
    # Check for missing values in critical columns
    if df[required_cols].isnull().any().any():
        raise ValueError("Found missing values in required columns")
    
    # Check date format
    try:
        pd.to_datetime(df['date'])
    except:
        raise ValueError("Date column must be in YYYY-MM-DD format")
    
    print("✅ Data validation passed!")
    return True

# Validate your data
validate_data(df)
```

### Data Cleaning Pipeline

```python
def clean_financial_data(df):
    """Clean and prepare financial data for FinRL"""
    
    # 1. Sort data properly
    df = df.sort_values(['date', 'tic'], ignore_index=True)
    
    # 2. Remove duplicates
    df = df.drop_duplicates(['date', 'tic'])
    
    # 3. Handle missing values
    # Forward fill then backward fill
    df = df.groupby('tic').fillna(method='ffill').fillna(method='bfill')
    
    # 4. Remove assets with insufficient data
    asset_counts = df.groupby('tic')['date'].count()
    valid_assets = asset_counts[asset_counts >= 252].index  # At least 1 year of data
    df = df[df['tic'].isin(valid_assets)]
    
    # 5. Remove outliers (optional)
    def remove_price_outliers(group, column='close', z_threshold=3):
        z_scores = np.abs((group[column] - group[column].mean()) / group[column].std())
        return group[z_scores <= z_threshold]
    
    df = df.groupby('tic').apply(remove_price_outliers).reset_index(drop=True)
    
    # 6. Ensure consistent date range across all assets
    date_counts = df.groupby('date')['tic'].count()
    complete_dates = date_counts[date_counts == date_counts.max()].index
    df = df[df['date'].isin(complete_dates)]
    
    return df

# Clean your data
clean_df = clean_financial_data(raw_df)
```

## DateTime Handling

### Date Format Requirements

FinRL expects dates in **YYYY-MM-DD** string format:

```python
# ✅ Correct format
correct_dates = ["2023-01-15", "2023-02-28", "2023-12-31"]

# ❌ Incorrect formats
incorrect_dates = [
    "01/15/2023",      # MM/DD/YYYY
    "15-01-2023",      # DD-MM-YYYY  
    "2023.01.15",      # Dots as separators
    datetime.date(2023, 1, 15)  # datetime object
]

# Convert to correct format
from datetime import datetime

def standardize_dates(date_series):
    """Convert various date formats to YYYY-MM-DD strings"""
    return pd.to_datetime(date_series).dt.strftime('%Y-%m-%d')

# Usage
df['date'] = standardize_dates(df['date'])
```

### Timezone Handling

For international markets, consider timezone differences:

```python
from finrl.config import TIME_ZONE_USEASTERN, TIME_ZONE_SHANGHAI

# Set appropriate timezone based on market
TIME_ZONES = {
    'US': TIME_ZONE_USEASTERN,    # "US/Eastern"
    'CN': TIME_ZONE_SHANGHAI,     # "Asia/Shanghai"  
    'EU': 'Europe/London'
}

def localize_dates(df, market='US'):
    """Localize dates to market timezone"""
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.tz_localize(TIME_ZONES[market])
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df
```

## Data Frequency Considerations

### Daily Data (Recommended)
- Most stable and reliable
- Best for long-term strategies
- Lower computational requirements

```python
# Daily data example
daily_data = {
    'date': pd.date_range('2023-01-01', '2023-12-31', freq='D'),
    'close': np.random.normal(100, 10, 365)
}
```

### Intraday Data (Advanced)
- Higher frequency (hourly, minute)
- More noise, requires careful preprocessing
- Higher computational requirements

```python
# Hourly data example
hourly_data = {
    'date': pd.date_range('2023-01-01', '2023-01-31', freq='H'),
    'close': np.random.normal(100, 5, 744)  # 24*31 hours
}

# Convert to daily for FinRL
def resample_to_daily(df, price_col='close'):
    """Resample intraday data to daily"""
    df['date'] = pd.to_datetime(df['date'])
    
    daily_df = df.groupby(['tic', df['date'].dt.date]).agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    daily_df['date'] = daily_df['date'].astype(str)
    return daily_df
```

## State Space Calculation

The environment's state space dimensions depend on your data structure:

```python
def calculate_state_space(stock_dim, tech_indicators):
    """Calculate state space dimensions"""
    
    # Components of state vector:
    cash_dim = 1                           # Cash balance
    price_dim = stock_dim                  # Current prices
    holdings_dim = stock_dim               # Current holdings
    tech_dim = len(tech_indicators) * stock_dim  # Technical indicators
    
    total_dim = cash_dim + price_dim + holdings_dim + tech_dim
    
    print(f"State space breakdown:")
    print(f"  Cash: {cash_dim}")
    print(f"  Prices: {price_dim}")
    print(f"  Holdings: {holdings_dim}")
    print(f"  Technical indicators: {tech_dim}")
    print(f"  Total: {total_dim}")
    
    return total_dim

# Example calculation
stock_dim = 5  # 5 assets
tech_indicators = ["macd", "rsi_30", "boll_ub", "boll_lb", "close_30_sma"]
state_space = calculate_state_space(stock_dim, tech_indicators)
# Output: Total state space = 1 + 5 + 5 + (5 * 5) = 36
```

## Common Data Issues and Solutions

### Issue 1: Inconsistent Date Ranges

**Problem**: Different assets have different date ranges
```python
# Asset A: 2020-01-01 to 2023-12-31  
# Asset B: 2021-06-01 to 2023-12-31
```

**Solution**: Align date ranges
```python
def align_date_ranges(df):
    """Ensure all assets have the same date range"""
    # Find common date range
    date_counts = df.groupby('date')['tic'].count()
    max_assets = date_counts.max()
    complete_dates = date_counts[date_counts == max_assets].index
    
    return df[df['date'].isin(complete_dates)]
```

### Issue 2: Missing Price Data

**Problem**: Some dates have missing price data
```python
# 2023-01-15: AAPL=150.0, MSFT=NaN
```

**Solution**: Fill missing values appropriately
```python
def handle_missing_prices(df):
    """Handle missing price data"""
    # Group by ticker and forward fill
    df = df.groupby('tic').fillna(method='ffill')
    
    # If still missing, backward fill
    df = df.groupby('tic').fillna(method='bfill')
    
    # Remove assets with too much missing data
    missing_pct = df.groupby('tic')['close'].isnull().mean()
    valid_assets = missing_pct[missing_pct < 0.05].index  # < 5% missing
    
    return df[df['tic'].isin(valid_assets)]
```

### Issue 3: Price Scale Differences

**Problem**: Assets with very different price scales
```python
# BTC-USD: ~$50,000
# AAPL: ~$150  
# DOGE-USD: ~$0.10
```

**Solution**: Use percentage returns or normalization
```python
def normalize_prices(df):
    """Normalize prices to handle scale differences"""
    df = df.copy()
    
    # Calculate percentage returns
    df['pct_return'] = df.groupby('tic')['close'].pct_change()
    
    # Or normalize to first price
    df['normalized_close'] = df.groupby('tic')['close'].transform(
        lambda x: x / x.iloc[0]
    )
    
    return df
```

## Best Practices

!!! tip "Data Quality First"
    Always validate and clean your data before training. Poor data quality leads to poor model performance.

!!! warning "Avoid Lookahead Bias"
    Ensure technical indicators only use historical data. Never use future information.

!!! info "Memory Considerations"
    For large datasets (>1M rows), consider data sampling or chunking strategies.

!!! success "Consistent Formatting"
    Use consistent date formats, column names, and data types across all datasets.