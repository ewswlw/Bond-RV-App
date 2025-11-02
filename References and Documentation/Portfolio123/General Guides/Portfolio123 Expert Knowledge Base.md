# Portfolio123 Expert Knowledge Base

## When to Use

- Use this overview when you need a roadmap of all Portfolio123 documentation modules and how they interconnect.
- Apply it before onboarding new agents or humans so they know which file to consult for factors, rules, AI, APIs, or identifiers.
- Reference it when organizing training materials or ensuring coverage for a support workflow.
- Consult it if you lose track of which guide contains a particular topic; this index provides quick links and context.
- Once you know the precise module you need, follow the link to dive deeper.

**Version**: 1.0  
**Created**: October 11, 2025  
**Purpose**: Comprehensive reference for AI agents to provide expert assistance with Portfolio123

---

## Overview

This knowledge base provides complete documentation of the Portfolio123 platform, enabling an AI agent to translate plain English investment requests into precise Portfolio123 screens, ranking systems, and machine learning models.

## Contents

### 📘 Core Documentation

1. **[portfolio123_knowledge_base.md](./portfolio123_knowledge_base.md)** - **START HERE**
   - Master index and usage guide
   - How to use this knowledge base effectively
   - Example workflows for AI agents

### 📊 Data Field Reference

2. **[factors_complete.md](./factors_complete.md)** - **1,126 Factors Documented**
   - Complete dictionary of all available data fields
   - Categorized by type: Macro, Fundamental, Technical, Estimates, etc.
   - Factor codes, names, and descriptions
   - **Use this to**: Map plain English concepts to Portfolio123 factor codes

### 🔍 Screening & Ranking

3. **[screening_ranking_guide.md](./screening_ranking_guide.md)** - **Complete Rule Syntax Guide**
   - How to create stock screens (filtering)
   - How to build ranking systems (scoring)
   - Operators, functions, and syntax
   - 50+ real-world examples
   - **Use this to**: Construct actual Portfolio123 rules from user requests

### 🤖 Machine Learning

4. **[ml_ai_capabilities_guide.md](./ml_ai_capabilities_guide.md)** - **AI Factor Deep Dive**
   - 9 supported ML algorithms (XGBoost, Neural Networks, etc.)
   - Data preprocessing and normalization
   - Model training workflows
   - Scalability and performance optimization
   - **Use this to**: Help users create AI-powered stock selection models

### 🔌 API & Automation

5. **[api_automation_guide.md](./api_automation_guide.md)** - **Complete API Reference**
   - Python `p123api` package documentation
   - Data retrieval endpoints
   - Automated backtesting
   - Integration examples (QuantRocket, Interactive Brokers)
   - **Use this to**: Generate Python code for automation and data export

### 🏷️ Identifiers & Mapping

6. **[ticker_identifier_system.md](./ticker_identifier_system.md)** - **Stock Identification Guide**
   - Ticker symbols, FIGI, CUSIP, CIK, P123 StockID
   - Handling delisted stocks
   - Cross-listing management
   - **Use this to**: Ensure accurate stock identification and data mapping

### 🏭 Industry Classification

7. **[rbics_classification.md](./rbics_classification.md)** - **RBICS System Overview**
   - Sector and industry codes
   - Hierarchical structure
   - Mnemonic codes (e.g., TECH, ENERGY)
   - **Use this to**: Filter by sector/industry correctly

---

## Quick Start for AI Agents

### Step 1: Understand the Request
Parse the user's plain English request to identify key concepts:
- Valuation (cheap, expensive, P/E, P/B)
- Growth (revenue growth, earnings growth)
- Quality (profitability, margins, ROE)
- Size (large-cap, small-cap, market cap)
- Sector/Industry (technology, healthcare, energy)
- Momentum (price trends, moving averages)
- Dividends (yield, payout ratio)

### Step 2: Map to Factors
Use **`factors_complete.md`** to find the exact Portfolio123 factor codes:
- "Cheap" → `PEExclXorTTM`, `PBExclXor`, `PSExclXor`
- "High growth" → `SalesGr%TTM`, `EPSGr%TTM`
- "Profitable" → `ROE`, `ROATTM`, `NetIncTTM`
- "Small-cap" → `MktCap < 2000`
- "Technology" → `Sector = TECH`

### Step 3: Construct Rules
Use **`screening_ranking_guide.md`** to build the syntax:

**For Screening (pass/fail filtering)**:
```
PEExclXorTTM < 15
SalesGr%TTM > 0.20
ROE > 0.15
MktCap > 300 AND MktCap < 2000
Sector = TECH
```

**For Ranking (scoring 0-100)**:
```
Value Composite (40%)
├── P/E Ratio (50%) - lower is better
└── P/B Ratio (50%) - lower is better

Growth Composite (30%)
├── Revenue Growth (60%) - higher is better
└── EPS Growth (40%) - higher is better

Quality Composite (30%)
├── ROE (100%) - higher is better
```

### Step 4: Provide Context
Explain to the user:
- What each rule does
- Why it matches their request
- Any assumptions made
- Potential refinements

### Step 5: Offer Automation (Optional)
If appropriate, use **`api_automation_guide.md`** to generate Python code:
```python
from p123api import Client

client = Client(api_id='YOUR_ID', api_key='YOUR_KEY')

results = client.screen_run(
    screen_name='TechValueScreen',
    as_of_date='2024-10-11',
    max_results=50
)
```

---

## Key Concepts

### Time Periods
- **TTM**: Trailing Twelve Months (most recent 12 months)
- **MRQ**: Most Recent Quarter
- **MRY**: Most Recent Year
- **PTM**: Prior Twelve Months

**Best Practice**: Use TTM for income statement items for freshest data.

### Operators
- `<`, `>`, `<=`, `>=`, `=`, `!=`
- `AND`, `OR`
- Parentheses for complex logic: `(A AND B) OR (C AND D)`

### Functions
- `SMA(50)` - 50-day moving average
- `Close(0)` - Current price
- `Close(252)` - Price 252 trading days ago
- `Industry(PEExclXorTTM)` - Industry average P/E
- `FRank(ROE)` - Percentile rank of ROE

### Ranking Direction
- **Lower is better**: P/E, P/B, Debt/Equity, Volatility
- **Higher is better**: ROE, Growth rates, Margins, Yields

### NA Handling
- **In Screens**: NA values cause stock to fail the rule
- **In Rankings**: NA values penalized by default (can be set to neutral)

---

## Example Use Cases

### Use Case 1: Value Screening
**User**: "*Find undervalued large-cap stocks*"

**Agent Response**:
```
# Large-cap value screen
MktCap > 10000              # Large-cap (>$10B)
PEExclXorTTM < 15           # P/E below 15
PBExclXor < 2               # P/B below 2
ROE > 0.10                  # Profitable
```

### Use Case 2: Growth Ranking
**User**: "*Rank stocks by growth potential*"

**Agent Response**:
```
Growth Ranking System:

Revenue Growth (40%) - higher is better
├── SalesGr%TTM

Earnings Growth (35%) - higher is better
├── EPSGr%TTM

Analyst Estimates (25%) - higher is better
├── LTGrthRtMean (Long-term growth estimate)
```

### Use Case 3: AI-Powered Selection
**User**: "*Use machine learning to find stocks that will outperform*"

**Agent Response**:
"I'll help you create an AI Factor using Portfolio123's ML capabilities:

1. **Select Features**: 
   - Valuation: PEExclXorTTM, PBExclXor
   - Quality: ROE, GrMarginTTM
   - Momentum: Close(0)/Close(252)
   - Growth: SalesGr%TTM

2. **Choose Algorithm**: XGBoost (best for tabular data)

3. **Preprocessing**:
   - Scaling: Rank (handles outliers well)
   - Scope: Dataset (prevents look-ahead bias)
   - NA Fill: True

4. **Training Period**: 2010-2023 (captures multiple market cycles)

5. **Validation**: Backtest on 2024 data

See `ml_ai_capabilities_guide.md` for complete implementation details."

### Use Case 4: API Automation
**User**: "*Export my ranking results to CSV every week*"

**Agent Response**:
```python
from p123api import Client
from datetime import datetime, timedelta

def weekly_export():
    client = Client(api_id='YOUR_ID', api_key='YOUR_KEY')
    
    # Get last Friday
    today = datetime.now()
    last_friday = today - timedelta(days=(today.weekday() - 4) % 7)
    as_of = last_friday.strftime('%Y-%m-%d')
    
    # Get rankings
    ranks = client.rank_ranks(
        rank_name='MyRankingSystem',
        as_of_date=as_of,
        universe='SP500',
        top_n=50,
        additional_data=['FIGI', 'MktCap', 'PEExclXorTTM']
    )
    
    # Export to CSV
    ranks.to_csv(f'rankings_{as_of}.csv', index=False)
    print(f"Exported rankings for {as_of}")

# Run weekly (use scheduler like cron or Windows Task Scheduler)
weekly_export()
```

---

## Statistics

### Data Coverage
- **Total Factors**: 1,126+
- **Macro Economic Data**: 94 factors
- **Fundamental Data**: 36 factors
- **Analyst Estimates**: 127 factors
- **Functions**: 251 functions
- **Ratios & Metrics**: 103 factors
- **Institutional Data**: 51 factors

### ML Algorithms
- 9 algorithms supported
- 5 scale well to large datasets
- 3 preprocessing methods
- 2 NA handling approaches

### Industry Classification
- 11 major sectors
- 100+ industries
- 300+ sub-industries
- Hierarchical RBICS system

---

## Best Practices

1. **Always use TTM** for income statement and cash flow items
2. **Use FIGI** for external data mapping (not tickers)
3. **Include delisted stocks** in backtests to avoid survivorship bias
4. **Test out-of-sample** when creating ML models
5. **Start simple** with screens and rankings, then add complexity
6. **Validate assumptions** with backtests before live trading
7. **Monitor performance** and retrain ML models regularly
8. **Use point-in-time data** via API to prevent look-ahead bias

---

## Support Resources

- **Portfolio123 Help Center**: https://portfolio123.customerly.help/
- **Community Forum**: https://community.portfolio123.com/
- **Factor Reference**: https://www.portfolio123.com/doc/doc_index.jsp
- **API Documentation**: https://api.portfolio123.com/docs
- **PyPI Package**: https://pypi.org/project/p123api/

---

## Version History

- **v1.0** (October 11, 2025): Initial comprehensive knowledge base
  - Complete factor reference (1,126 factors)
  - Screening and ranking guide with 50+ examples
  - ML/AI capabilities deep dive
  - Full API reference and automation guide
  - Identifier system documentation
  - RBICS classification guide

---

## License

This knowledge base is created for educational and reference purposes. Portfolio123 is a registered trademark of FactSet Research Systems Inc.

---

**For AI Agents**: This knowledge base is designed to be your complete reference for Portfolio123. Always start with `portfolio123_knowledge_base.md` for the master index, then dive into specific guides as needed. When in doubt, consult the factor reference and screening guide for exact syntax.

