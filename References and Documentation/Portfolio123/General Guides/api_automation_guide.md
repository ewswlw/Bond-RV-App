# Portfolio123 API and Automation Guide

## When to Use

- Use this guide when you need to script interactions with Portfolio123—pulling data, triggering simulations, or automating workflows through the REST API.
- Apply it before provisioning credentials for agents so they follow the recommended authentication patterns, credit cost awareness, and error handling.
- Reference it during integration projects to confirm endpoint capabilities, parameter options, and Python client usage examples.
- Consult it when troubleshooting automation; the sections on rate limits, retries, and common exceptions provide remediation steps.
- If you operate exclusively within the Portfolio123 UI, lighter docs may suffice; once programmatic access is required, rely on this guide.

## Overview

Portfolio123 provides a comprehensive REST API that enables programmatic access to all platform features. The `p123api` Python package wraps these endpoints for easy integration.

## Getting Started

### Installation

```bash
pip install --upgrade p123api
```

### Authentication

```python
from p123api import Client

# Initialize client with API credentials
client = Client(api_id='YOUR_API_ID', api_key='YOUR_API_KEY')
```

**Obtaining API Credentials**:
1. Log in to Portfolio123
2. Navigate to Account Settings
3. Generate API ID and API Key
4. Store securely (treat like passwords)

### Basic Usage Pattern

```python
import p123api

try:
    client = p123api.Client(api_id='YOUR_API_ID', api_key='YOUR_API_KEY')
    
    # Call API functions
    result = client.some_function(parameters)
    
except p123api.ClientException as e:
    print(f"API Error: {e}")
```

## API Endpoint Categories

### 1. Data Retrieval

#### data() - Point-in-Time Data Retrieval

Retrieve large sets of historical data from the point-in-time engine.

**Cost**: 1 credit per 100K data points

**Identifiers Supported**:
- Tickers
- FIGI
- P123 UIDs (p123Uids)
- CIKs
- GVKeys (requires Compustat license)

**Example**:
```python
result = client.data({
    'tickers': ['AAPL:USA', 'MSFT:USA', 'GOOGL:USA'],
    'formulas': ['Close(0)', 'Volume(0)', 'PEExclXorTTM', 'ROE'],
    'startDt': '2020-01-01',
    'endDt': '2024-12-31',
    # Optional parameters
    'currency': 'USD',
    'precision': 2,
    'frequency': 'Every Week',
    'pitMethod': 'Complete',  # or 'Prelim'
    'includeNames': True,
    'region': 'United States',
    'ignoreErrors': True
}, to_pandas=True)
```

**Frequency Options**:
- 'Every Week'
- 'Every N Weeks' (2, 3, 4, 6, 8, 13, 26, 52)

**PIT Method**:
- `Prelim`: Uses preliminary financial data (faster availability)
- `Complete`: Uses only complete/restated data (more accurate)

**Data License Note**: 
- Free trial: IBM, MSFT, INTC for 5 years
- Full access requires Factset or Compustat license

---

#### data_prices() - EOD Price Data

Retrieve end-of-day prices for stocks and ETFs.

**Cost**: 1 credit per call

**Example**:
```python
prices = client.data_prices(
    identifier='AAPL:USA',  # or ticker, or P123 ID (int)
    start='2024-01-01',
    end='2024-12-31',  # or None for current date
    to_pandas=True
)
```

---

#### data_universe() - Universe Data Retrieval

Retrieve point-in-time data for a selected universe with optional preprocessing.

**Cost**: 1 credit per 100K data points

**Example**:
```python
result = client.data_universe({
    'universe': 'SP500',  # or custom universe name, or 'APIUniverse'
    'asOfDts': ['2024-03-16', '2024-03-09'],  # Use weekend dates
    'formulas': ['Close(0)/close(5)', 'PEExclXorTTM', 'SalesGr%TTM'],
    'names': ['1wk%', 'PE', 'SalesGr'],  # Optional column names
    'pitMethod': 'Complete',
    'precision': 2,
    'type': 'stock',  # or 'etf'
    'includeNames': True,
    'figi': 'Country Composite',  # or 'Share Class'
    'currency': 'USD',
    
    # Preprocessing/Normalization
    'preproc': {
        'scaling': 'rank',  # or 'minmax', 'normal'
        'scope': 'dataset',  # or 'date'
        'trimPct': 5.0,
        'outliers': True,
        'naFill': False,
        'outlierLimit': 5,
        'excludedFormulas': ['Close(0)/close(5)'],  # Technical factors
        'mlTrainingEnd': '2023-12-31'  # For dataset scope
    }
}, to_pandas=True)
```

**Scaling Methods**:
- `minmax`: Scale to [0, 1] range
- `rank`: Convert to percentile ranks
- `normal`: Z-score normalization (mean=0, std=1)

**Scope**:
- `dataset`: Compute scaling parameters across entire period
- `date`: Compute scaling separately for each date

**Use Case**: Perfect for ML model training data preparation

---

### 2. AI Factor Operations

#### AI Factor Training

Train machine learning models on historical data.

**Example** (conceptual - check latest API docs):
```python
# AI Factor training via API
# Parameters would include:
# - Universe
# - Features (formulas)
# - Algorithm (XGBoost, LightGBM, etc.)
# - Preprocessing settings
# - Training period
# - Hyperparameters
```

**Note**: Check latest API documentation for exact AI Factor training endpoint syntax.

---

### 3. Universe Operations

#### universe_list() - List Universes

Get list of available universes.

```python
universes = client.universe_list()
```

#### universe_get() - Get Universe Stocks

Retrieve stocks in a universe as of a specific date.

```python
stocks = client.universe_get(
    universe_name='Russell2000',
    as_of_date='2024-01-01'
)
```

#### universe_update() - Update API Universe

Update the special 'APIUniverse' with custom stock list.

```python
client.universe_update(
    tickers=['AAPL:USA', 'MSFT:USA', 'GOOGL:USA']
)
```

**Use Case**: Dynamically update universe for backtesting or screening

---

### 4. Ranking Operations

#### rank_ranks() - Get Ranking System Results

Retrieve stocks ranked by a ranking system.

**Example**:
```python
ranks = client.rank_ranks(
    rank_name='MyValueRanking',
    as_of_date='2024-01-01',
    universe='SP500',
    top_n=50,  # Top 50 stocks
    additional_data=['FIGI', 'MktCap', 'PEExclXorTTM']
)
```

**Additional Data**: Can include FIGI, formulas, or other factors

**Use Case**: Export ranking results for external analysis or trading

---

### 5. Screen Operations

#### screen_run() - Execute Screen

Run a screen and get results.

**Example**:
```python
results = client.screen_run(
    screen_name='MyValueScreen',
    as_of_date='2024-01-01',
    max_results=100
)
```

---

### 6. Strategy Operations

#### strategy_backtest() - Run Strategy Backtest

Execute backtest for a strategy.

**Example**:
```python
backtest = client.strategy_backtest(
    strategy_name='MyStrategy',
    start_date='2020-01-01',
    end_date='2024-12-31'
)
```

#### strategy_rebalance() - Get Rebalance Trades

Get recommended trades for next rebalance.

```python
trades = client.strategy_rebalance(
    strategy_name='MyLiveStrategy',
    account_id='MyAccount'
)
```

---

### 7. Data Series Operations

#### series_create() - Create Custom Data Series

Upload custom time series data.

**Example**:
```python
client.series_create(
    name='MyMacroIndicator',
    data=pandas_dataframe,
    frequency='monthly'
)
```

---

### 8. Stock Factor Operations

#### stock_factor_create() - Upload Custom Stock Factors

Upload custom factor data for stocks.

**Example**:
```python
client.stock_factor_create(
    name='MyCustomFactor',
    data=pandas_dataframe,  # Must have key column (ticker, FIGI, etc.)
    key_type='FIGI'  # or 'ticker', 'p123uid', 'cusip', 'cik', 'gvkey'
)
```

**Key Column**: First column must be identifier (ticker, FIGI, CUSIP, CIK, GVKey, or P123 StockID)

**Use Case**: Import alternative data, custom calculations, or third-party signals

---

## Advanced API Patterns

### Bulk Data Download

```python
import pandas as pd

# Download data for large universe
universes = ['SP500', 'Russell2000', 'NASDAQ100']
all_data = []

for univ in universes:
    data = client.data_universe({
        'universe': univ,
        'asOfDts': ['2024-01-01'],
        'formulas': ['PEExclXorTTM', 'ROE', 'SalesGr%TTM'],
        'names': ['PE', 'ROE', 'SalesGr']
    }, to_pandas=True)
    
    data['Universe'] = univ
    all_data.append(data)

combined = pd.concat(all_data, ignore_index=True)
combined.to_csv('universe_data.csv', index=False)
```

### Time Series Analysis

```python
# Get historical data for analysis
data = client.data({
    'tickers': ['AAPL:USA'],
    'formulas': ['Close(0)', 'Volume(0)', 'PEExclXorTTM'],
    'startDt': '2015-01-01',
    'endDt': '2024-12-31',
    'frequency': 'Every Week'
}, to_pandas=True)

# Analyze with pandas
data['Returns'] = data['Close(0)'].pct_change()
data['MA50'] = data['Close(0)'].rolling(50).mean()
```

### Automated Screening Pipeline

```python
from datetime import datetime, timedelta

# Weekly screening automation
def run_weekly_screen():
    # Get last Friday
    today = datetime.now()
    days_since_friday = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=days_since_friday)
    as_of = last_friday.strftime('%Y-%m-%d')
    
    # Run screen
    results = client.screen_run(
        screen_name='WeeklyValueScreen',
        as_of_date=as_of,
        max_results=50
    )
    
    # Export results
    results.to_csv(f'screen_results_{as_of}.csv')
    
    return results

# Schedule this to run weekly
```

### Factor Research Workflow

```python
# Research custom factor
def test_custom_factor(formula, universe='SP500', years=10):
    import datetime
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=years*365)
    
    # Get factor data
    data = client.data_universe({
        'universe': universe,
        'asOfDts': [start_date.strftime('%Y-%m-%d')],
        'formulas': [formula, 'Close(0)', 'FRank(' + formula + ')'],
        'names': ['Factor', 'Price', 'FactorRank']
    }, to_pandas=True)
    
    # Analyze factor distribution
    print(data['Factor'].describe())
    
    # Check correlation with future returns
    # (Would need forward returns data)
    
    return data

# Test a custom factor
test_custom_factor('ROE * SalesGr%TTM')
```

### ML Model Training Data Preparation

```python
# Prepare training data for ML model
def prepare_ml_data(universe, features, start, end):
    data = client.data_universe({
        'universe': universe,
        'asOfDts': pd.date_range(start, end, freq='W-FRI').strftime('%Y-%m-%d').tolist(),
        'formulas': features + ['Close(0)'],
        'preproc': {
            'scaling': 'rank',
            'scope': 'date',
            'trimPct': 5.0,
            'outliers': True,
            'naFill': True,
            'mlTrainingEnd': end
        }
    }, to_pandas=True)
    
    # Calculate forward returns as target
    data = data.sort_values(['Ticker', 'Date'])
    data['Target'] = data.groupby('Ticker')['Close(0)'].pct_change(periods=4).shift(-4)
    
    # Remove NAs
    data = data.dropna()
    
    return data

features = ['PEExclXorTTM', 'ROE', 'SalesGr%TTM', 'DebtToEquity', 'GrMarginTTM']
train_data = prepare_ml_data('Russell2000', features, '2015-01-01', '2023-12-31')
```

---

## API Credits and Costs

### Credit System

- **Data Retrieval**: 1 credit per 100K data points
- **Price Data**: 1 credit per call
- **Other Operations**: Varies by endpoint

### Optimizing Credit Usage

1. **Batch Requests**: Combine multiple tickers in single call
2. **Use Universes**: `data_universe()` more efficient than individual `data()` calls
3. **Limit Frequency**: Use weekly/monthly instead of daily when possible
4. **Cache Results**: Store downloaded data locally
5. **Precision**: Lower precision = smaller data = fewer credits

### Example Credit Calculation

```python
# Example: Download 500 stocks, 10 factors, 5 years weekly
stocks = 500
factors = 10
weeks = 5 * 52  # 260 weeks
data_points = stocks * factors * weeks  # 1,300,000

credits = data_points / 100000  # 13 credits
```

---

## Error Handling

### Common Errors

#### Invalid Ticker
```python
try:
    data = client.data({
        'tickers': ['INVALID_TICKER'],
        'formulas': ['Close(0)'],
        'startDt': '2024-01-01',
        'endDt': '2024-12-31',
        'ignoreErrors': False  # Will raise error
    })
except p123api.ClientException as e:
    print(f"Error: {e}")
```

**Solution**: Set `ignoreErrors: True` to skip invalid tickers

#### Data License Required
```python
# Some factors require data license
try:
    data = client.data({
        'tickers': ['AAPL:USA'],
        'formulas': ['SalesQTR'],  # Requires license
        'startDt': '2024-01-01',
        'endDt': '2024-12-31'
    })
except p123api.ClientException as e:
    print("Data license required")
```

**Solution**: Use normalized data via `data_universe()` with preprocessing

#### Rate Limiting
```python
import time

def api_call_with_retry(func, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except p123api.ClientException as e:
            if 'rate limit' in str(e).lower() and attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

---

## Integration Examples

### QuantRocket Integration

```python
# Export P123 rankings to QuantRocket
def export_to_quantrocket(rank_name, universe):
    # Get rankings with FIGI
    ranks = client.rank_ranks(
        rank_name=rank_name,
        universe=universe,
        additional_data=['FIGI']
    )
    
    # Map to QuantRocket format
    qr_format = ranks[['FIGI', 'Rank']].rename(columns={'Rank': 'Signal'})
    
    # Upload to QuantRocket
    # (QuantRocket-specific code here)
    
    return qr_format
```

### Interactive Brokers Integration

```python
# Generate order list for IB
def generate_ib_orders(strategy_name, account_id):
    trades = client.strategy_rebalance(
        strategy_name=strategy_name,
        account_id=account_id
    )
    
    # Convert to IB order format
    ib_orders = []
    for _, trade in trades.iterrows():
        order = {
            'symbol': trade['Ticker'].split(':')[0],
            'action': 'BUY' if trade['Shares'] > 0 else 'SELL',
            'quantity': abs(trade['Shares']),
            'order_type': 'MKT'
        }
        ib_orders.append(order)
    
    return ib_orders
```

### Pandas/Jupyter Workflow

```python
import pandas as pd
import matplotlib.pyplot as plt

# Complete research workflow
def research_workflow(universe, factors, start, end):
    # Download data
    data = client.data_universe({
        'universe': universe,
        'asOfDts': [end],
        'formulas': factors
    }, to_pandas=True)
    
    # Analyze
    print(data.describe())
    
    # Visualize
    data[factors].hist(bins=50, figsize=(15, 10))
    plt.tight_layout()
    plt.show()
    
    # Correlation matrix
    corr = data[factors].corr()
    print(corr)
    
    return data
```

---

## Best Practices

### 1. Use Point-in-Time Data
Always use point-in-time data to avoid look-ahead bias:
```python
# Good - Point-in-time
data = client.data_universe({
    'universe': 'SP500',
    'asOfDts': ['2020-01-01'],
    'pitMethod': 'Complete'
})

# Bad - Current data for historical date
# (Don't query current values for past dates)
```

### 2. Handle Missing Data
```python
# Explicitly handle NAs
data = client.data_universe({
    'universe': 'SP500',
    'asOfDts': ['2024-01-01'],
    'formulas': ['PEExclXorTTM', 'ROE'],
    'preproc': {
        'naFill': True  # Fill NAs with neutral values
    }
}, to_pandas=True)
```

### 3. Optimize Frequency
```python
# For long-term backtests, use lower frequency
data = client.data({
    'tickers': ['AAPL:USA'],
    'formulas': ['Close(0)'],
    'startDt': '2000-01-01',
    'endDt': '2024-12-31',
    'frequency': 'Every 4 Weeks'  # Monthly, saves credits
})
```

### 4. Cache Expensive Calls
```python
import pickle
import os

def get_data_cached(cache_file, **api_params):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    data = client.data(**api_params, to_pandas=True)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    return data
```

### 5. Validate Identifiers
```python
# Use FIGI for robust mapping
data = client.data({
    'figi': ['BBG000B9XRY4'],  # Apple FIGI
    'formulas': ['Close(0)'],
    'startDt': '2024-01-01',
    'endDt': '2024-12-31'
})
```

---

## API Reference Links

- **API Wrapper Documentation**: https://portfolio123.customerly.help/en/articles/13765-the-api-wrapper-p123api
- **PyPI Package**: https://pypi.org/project/p123api/
- **API Specification**: https://api.portfolio123.com/docs
- **Community Forum**: https://community.portfolio123.com/

---

## Summary

The Portfolio123 API provides comprehensive programmatic access to:
- **Data Retrieval**: Historical point-in-time data with preprocessing
- **AI Factors**: Machine learning model training and deployment
- **Universes**: Dynamic universe management
- **Rankings**: Automated ranking system execution
- **Screens**: Programmatic screening
- **Strategies**: Backtest and rebalance automation
- **Custom Data**: Import alternative data and custom factors

**Key Advantages**:
- Python-native with pandas integration
- Point-in-time data prevents look-ahead bias
- Built-in preprocessing for ML workflows
- FIGI support for robust identifier mapping
- Credit-based pricing scales with usage

**Common Use Cases**:
- Automated trading system development
- Factor research and backtesting
- ML model training data preparation
- Integration with external platforms (QuantRocket, IB)
- Custom reporting and analytics

