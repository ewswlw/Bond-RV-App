# Portfolio123 Quick Reference Card

## When to Use

- Use this card when you need a fast lookup of common Portfolio123 factor codes, syntax patterns, and sample screens without digging into the full knowledge base.
- Apply it during live sessions or calls where quick recall of factor names, composites, or logic operators is critical.
- Reference it as a companion while drafting rules to ensure you use the right code direction (higher vs lower is better).
- Consult it for template screens and ranking structures before customizing deeper solutions.
- For exhaustive coverage, move to the complete factor reference; this card is optimized for rapid access.

## Essential Factor Codes

### Valuation
| Concept | Factor Code | Direction |
|---------|-------------|-----------|
| P/E Ratio | `PEExclXorTTM` | Lower is better |
| Price/Book | `PBExclXor` | Lower is better |
| Price/Sales | `PSExclXor` | Lower is better |
| EV/EBITDA | `EVToEBITDA` | Lower is better |
| Dividend Yield | `DivYield` | Higher is better |
| FCF Yield | `FCFYield` | Higher is better |

### Growth
| Concept | Factor Code | Direction |
|---------|-------------|-----------|
| Revenue Growth | `SalesGr%TTM` | Higher is better |
| EPS Growth | `EPSGr%TTM` | Higher is better |
| FCF Growth | `FCFGr%TTM` | Higher is better |
| LT Growth Est | `LTGrthRtMean` | Higher is better |

### Quality
| Concept | Factor Code | Direction |
|---------|-------------|-----------|
| Return on Equity | `ROE` | Higher is better |
| Return on Assets | `ROATTM` | Higher is better |
| Gross Margin | `GrMarginTTM` | Higher is better |
| Operating Margin | `OpMarginTTM` | Higher is better |
| Net Margin | `NetMarginTTM` | Higher is better |

### Financial Health
| Concept | Factor Code | Direction |
|---------|-------------|-----------|
| Debt/Equity | `DebtToEquity` | Lower is better |
| Current Ratio | `CurrentRatio` | Higher is better |
| Interest Coverage | `InterestCoverage` | Higher is better |

### Size & Classification
| Concept | Factor Code | Notes |
|---------|-------------|-------|
| Market Cap | `MktCap` | In millions USD |
| Sector | `Sector` | Use `Sector = TECH` |
| Industry | `Industry` | Use `Industry = SOFTWARE` |

### Technical
| Concept | Factor Code | Notes |
|---------|-------------|-------|
| Current Price | `Close(0)` | Today's close |
| Price N days ago | `Close(N)` | N trading days back |
| 50-day MA | `SMA(50)` | Simple moving average |
| Volume | `Volume(0)` | Today's volume |

## Common Patterns

### Screen: Value Stocks
```
PEExclXorTTM < 15
PBExclXor < 2
ROE > 0.15
DivYield > 0.03
```

### Screen: Growth Stocks
```
SalesGr%TTM > 0.20
EPSGr%TTM > 0.25
ROE > 0.20
GrMarginTTM > 0.30
```

### Screen: Small-Cap Value
```
MktCap > 300 AND MktCap < 2000
PEExclXorTTM < 15
PBExclXor < 1.5
ROE > 0.10
```

### Ranking: Composite Value
```
Value (100%)
├── P/E (40%) - lower is better
├── P/B (30%) - lower is better
├── P/S (20%) - lower is better
└── Div Yield (10%) - higher is better
```

### Ranking: Multi-Factor
```
Overall (100%)
├── Value (40%)
│   ├── P/E (50%)
│   └── P/B (50%)
├── Growth (30%)
│   ├── Sales Growth (60%)
│   └── EPS Growth (40%)
└── Quality (30%)
    ├── ROE (50%)
    └── Gross Margin (50%)
```

## Operators & Logic

### Comparison
- `<` Less than
- `>` Greater than
- `<=` Less than or equal
- `>=` Greater than or equal
- `=` Equal to
- `!=` Not equal to

### Logical
- `AND` Both conditions must be true
- `OR` At least one condition must be true
- `()` Parentheses for grouping

### Example
```
(PEExclXorTTM < 15 AND ROE > 0.15) OR (DivYield > 0.05)
```

## Functions

### Relative Comparisons
```
PEExclXorTTM < Industry(PEExclXorTTM)    # Below industry average
ROE > Sector(ROE)                        # Above sector average
```

### Percentile Ranking
```
FRank(PEExclXorTTM) < 20    # P/E in bottom 20%
FRank(ROE) > 80             # ROE in top 20%
```

### Moving Averages
```
SMA(50) > SMA(200)          # Golden cross
Close(0) > SMA(20)          # Price above 20-day MA
```

### Price Changes
```
Close(0)/Close(252) > 1.30  # Up 30% over 1 year
Close(0)/Close(21) > 1.05   # Up 5% over 1 month
```

## Time Periods

| Abbreviation | Meaning | Example |
|--------------|---------|---------|
| TTM | Trailing Twelve Months | Last 12 months |
| MRQ | Most Recent Quarter | Latest quarter |
| MRY | Most Recent Year | Latest annual report |
| PTM | Prior Twelve Months | Previous 12 months |
| FQ1 | Fiscal Quarter 1 | Next quarter |
| FY1 | Fiscal Year 1 | Next year |

**Best Practice**: Use TTM for income statement items (freshest data)

## Sectors (RBICS)

| Mnemonic | Sector Name |
|----------|-------------|
| TECH | Technology |
| HEALTHCARE | Healthcare |
| ENERGY | Energy |
| FINANCIAL | Financial Services |
| CONSUMER | Consumer Discretionary |
| STAPLES | Consumer Staples |
| INDUSTRIAL | Industrials |
| MATERIALS | Materials |
| UTILITIES | Utilities |
| REALESTATE | Real Estate |
| TELECOM | Telecommunications |

## ML Algorithms Quick Guide

| Algorithm | Best For | Scales Well? |
|-----------|----------|--------------|
| XGBoost | General purpose, complex patterns | ✅ Yes |
| LightGBM | Large datasets, speed | ✅ Yes |
| Random Forest | Robust baseline, interpretability | ✅ Yes |
| Linear Regression | Simple relationships, speed | ✅ Yes |
| Neural Networks | Very complex patterns | ✅ Yes (GPU) |
| SVM | Small-medium datasets | ❌ No |
| GAM | Small-medium datasets | ❌ No |

## API Quick Start

```python
from p123api import Client

client = Client(api_id='YOUR_ID', api_key='YOUR_KEY')

# Get data
data = client.data({
    'tickers': ['AAPL:USA', 'MSFT:USA'],
    'formulas': ['Close(0)', 'PEExclXorTTM', 'ROE'],
    'startDt': '2024-01-01',
    'endDt': '2024-12-31'
}, to_pandas=True)

# Run screen
results = client.screen_run(
    screen_name='MyScreen',
    as_of_date='2024-10-11'
)

# Get rankings
ranks = client.rank_ranks(
    rank_name='MyRanking',
    universe='SP500',
    top_n=50
)
```

## Common Pitfalls

❌ **Don't**: Use ticker-only identification for historical data  
✅ **Do**: Use FIGI or P123 StockID

❌ **Don't**: Ignore delisted stocks in backtests  
✅ **Do**: Include delisted stocks to avoid survivorship bias

❌ **Don't**: Use current data for past dates  
✅ **Do**: Use point-in-time data via API

❌ **Don't**: Overfit ML models with too many features  
✅ **Do**: Start with 10-30 features, validate out-of-sample

❌ **Don't**: Use MRY for income statement items  
✅ **Do**: Use TTM for freshest data

## Resources

- **Full Knowledge Base**: See `README.md`
- **Factor Reference**: `factors_complete.md`
- **Screening Guide**: `screening_ranking_guide.md`
- **ML Guide**: `ml_ai_capabilities_guide.md`
- **API Guide**: `api_automation_guide.md`
