# Portfolio123 Screening and Ranking Rules Guide

## When to Use

- Use this guide when you need detailed instructions on building Portfolio123 screens or ranking systems, including syntax, operators, and examples.
- Apply it prior to translating user requirements into formulas to ensure accuracy and consistency.
- Reference it during reviews or troubleshooting to verify logic structure and rule behavior.
- Consult it when teaching Portfolio123 formula basics to new analysts.
- For quick factor lookups, pair with the quick reference card; rely on this document for comprehensive rule construction.

## Overview

Portfolio123 provides powerful tools for creating stock screens and ranking systems using a combination of visual wizards and a flexible formula language. This guide explains how to translate plain English requirements into Portfolio123 rules.

## Part 1: Stock Screening

### What is a Screen?

A screen filters a universe of stocks based on specific criteria. Stocks either pass (included) or fail (excluded) each rule.

### Screen Creation Methods

#### 1. Rules Wizard (No Code)
- Point-and-click interface
- Select factor from dropdown
- Choose operator (>, <, =, !=)
- Enter value
- **Best For**: Beginners, simple screens

#### 2. Free-Form Editor (Code-Based)
- Type formulas directly
- Access to all 1,000+ factors
- Complex logic with AND/OR
- **Best For**: Advanced users, complex conditions

### Basic Screen Syntax

#### Simple Comparisons
```
PEExclXorTTM < 20          # P/E ratio less than 20
MktCap > 1000              # Market cap greater than $1B
DivYield > 0.03            # Dividend yield above 3%
```

#### Operators
- `<` - Less than
- `>` - Greater than
- `<=` - Less than or equal
- `>=` - Greater than or equal
- `=` - Equal to
- `!=` - Not equal to

#### Combining Rules with AND
```
PEExclXorTTM < 20 AND ROE > 0.15
```
Both conditions must be true.

#### Combining Rules with OR
```
DivYield > 0.05 OR FCFYield > 0.05
```
At least one condition must be true.

#### Complex Logic with Parentheses
```
(PEExclXorTTM < 15 AND ROE > 0.15) OR (PBExclXor < 1 AND ROATTM > 0.10)
```

### Common Screening Patterns

#### Value Screening
```
# Low P/E, high dividend yield
PEExclXorTTM < 15
DivYield > 0.03
PBExclXor < 2
```

#### Growth Screening
```
# Revenue and earnings growth
SalesGr%TTM > 0.15
EPSGr%TTM > 0.20
ROE > 0.15
```

#### Quality Screening
```
# High margins, low debt
GrMarginTTM > 0.30
DebtToEquity < 0.5
ROE > 0.20
```

#### Momentum Screening
```
# Price momentum
Close(0)/Close(252) > 1.20   # Up 20% over 1 year
SMA(50) > SMA(200)           # Golden cross
```

### Industry and Sector Filtering

#### Include Specific Sector
```
Sector = TECH
```

#### Exclude Specific Industries
```
Industry != WOODPAPER
Industry != COAL
```

#### Multiple Sector Selection
```
Sector = TECH OR Sector = HEALTHCARE
```

### Market Cap Filters

```
MktCap > 10000        # Large cap (>$10B)
MktCap > 2000         # Mid cap (>$2B)
MktCap < 2000         # Small cap (<$2B)
MktCap < 300          # Micro cap (<$300M)
```

### Relative Comparisons

#### Compare to Industry Average
```
PEExclXorTTM < Industry(PEExclXorTTM)
ROE > Industry(ROE)
```

#### Compare to Sector Average
```
GrMarginTTM > Sector(GrMarginTTM)
```

### Time Period Abbreviations

Understanding time periods is crucial for Portfolio123:

- **TTM** - Trailing Twelve Months (most recent 12 months)
- **PTM** - Prior Twelve Months (previous 12 months)
- **MRQ** - Most Recent Quarter
- **MRY** - Most Recent Year (annual report)
- **FY** - Fiscal Year
- **FQ** - Fiscal Quarter

**Example**: If latest quarter ended 9/30/2024:
- TTM = Four quarters ending 9/30/2024
- PTM = Four quarters ending 9/30/2023
- MRQ = Quarter ending 9/30/2024
- MRY = Annual report (likely 12/31/2023)

**Best Practice**: Use TTM for income statement and cash flow items for freshest data.

### Functions in Screens

#### Moving Averages
```
SMA(50) > SMA(200)           # 50-day MA above 200-day MA
Close(0) > SMA(20)           # Price above 20-day MA
```

#### Price Changes
```
Close(0)/Close(5) > 1.05     # Up 5% in 5 days
Close(0)/Close(252) > 1.30   # Up 30% in 1 year
```

#### Statistical Functions
```
StdDev(Close(0),20) < 0.30   # Low volatility
Avg(Volume(0),20) > 1000000  # Average volume > 1M
```

#### Ranking Functions
```
FRank(PEExclXorTTM) < 20     # P/E in bottom 20th percentile
FRank(ROE) > 80              # ROE in top 20th percentile
```

### Advanced Screening Techniques

#### Growth Rate Calculations
```
# 3-year revenue CAGR > 15%
(Sales#TTM / Sales#PTM(3))^(1/3) - 1 > 0.15
```

#### Quality Score Composite
```
# Create composite quality score
SetVar("Quality", (FRank(ROE) + FRank(ROATTM) + FRank(GrMarginTTM))/3)
ShowVar("Quality") > 70
```

#### Conditional Logic
```
# Different criteria for different sectors
(Sector=TECH AND PEExclXorTTM<30) OR (Sector=UTILITIES AND DivYield>0.04)
```

### Special Functions

#### Ticker Lists
```
Ticker("AAPL,MSFT,GOOGL")    # Include only these tickers
```

#### FIGI Lists
```
FIGI("BBG000B9XRY4,BBG000BPH459")   # Filter by FIGI
```

#### Exclude Delisted
```
IsActive = 1                  # Only currently listed stocks
```

### NA (Not Available) Handling

In screens, if a factor is NA for a stock, the stock **fails** that rule.

**Example**:
```
PEExclXorTTM < 20
```
Stocks with negative earnings (NA P/E) are excluded.

**Workaround** for including NA:
```
PEExclXorTTM < 20 OR IsNA(PEExclXorTTM)
```

---

## Part 2: Ranking Systems

### What is a Ranking System?

A ranking system assigns each stock a percentile score (0-100) based on multiple factors. Higher scores indicate better stocks according to your criteria.

### Ranking System Structure

#### Hierarchical Node System

Ranking systems use a tree structure:

```
Overall Rank (100%)
├── Value Composite (40%)
│   ├── P/E Ratio (50%)
│   └── P/B Ratio (50%)
├── Growth Composite (30%)
│   ├── Revenue Growth (60%)
│   └── Earnings Growth (40%)
└── Momentum Composite (30%)
    ├── 6-Month Return (50%)
    └── 12-Month Return (50%)
```

### Node Types

#### 1. Factor Rank Node
- Ranks stocks based on single factor or formula
- Converts values to percentile ranks (0-100)
- Specify if higher or lower values are better

**Example**: P/E Ratio (lower is better)

#### 2. Composite Node
- Combines multiple child nodes
- Applies weights to each child
- Produces weighted average rank

**Example**: Value Composite combining P/E, P/B, P/S

#### 3. Conditional Node
- Applies different ranking logic based on conditions
- Example: Different factors for different sectors

### How Ranking Works

#### Step 1: Sort and Percentile Conversion

For each factor:
1. Sort all stocks from best to worst
2. Assign percentile ranks 0-100
3. Best stock = 100, Worst stock = 0

**Example with 5 stocks**:

| Stock | P/E Ratio | Rank |
|-------|-----------|------|
| A | 8 | 100 |
| B | 12 | 80 |
| C | 15 | 60 |
| D | 18 | 40 |
| E | 25 | 20 |

#### Step 2: Apply Weights

If you have two factors with weights:
- P/E (75%)
- P/B (25%)

For Stock A:
- P/E Rank: 80
- P/B Rank: 60
- Weighted Score: (80 × 0.75) + (60 × 0.25) = 60 + 15 = 75

#### Step 3: Re-normalize

After computing weighted averages, re-normalize to 0-100 percentile scale.

**Example**:

| Stock | Weighted Avg | Final Rank |
|-------|--------------|------------|
| A | 75 | 100 |
| B | 65 | 80 |
| C | 55 | 60 |
| D | 45 | 40 |
| E | 35 | 20 |

### Ranking Direction

For each factor, specify which direction is better:

**Lower Values Better**:
- P/E Ratio
- P/B Ratio
- Debt/Equity
- Volatility

**Higher Values Better**:
- ROE
- Revenue Growth
- Dividend Yield
- Profit Margin

### Ranking Scope

#### Universe Ranking (Default)
Rank against all stocks in universe.

#### Industry Ranking
Rank against stocks in same industry.
- **Use Case**: Sector-neutral strategies
- **Example**: Find best value stocks within each industry

#### Sector Ranking
Rank against stocks in same sector.

### NA Handling in Ranking

#### Default: NAs Penalized
- NA values sorted to bottom
- Assigned rank just below worst valid value
- **Impact**: Companies with many NAs rank poorly

**Example** (5 stocks, 2 with NA):

| Stock | P/E | Rank |
|-------|-----|------|
| A | 10 | 100 |
| B | 15 | 80 |
| C | 20 | 60 |
| D | NA | 40 |
| E | NA | 40 |

#### Alternative: NAs Neutral
- NA values excluded from initial ranking
- After ranking valid values, NAs assigned middle rank (~50)
- **Use Case**: Long/short strategies

**Example** (same 5 stocks, NAs neutral):

| Stock | P/E | Rank |
|-------|-----|------|
| A | 10 | 100 |
| B | 15 | 66.7 |
| C | 20 | 33.3 |
| D | NA | 50 |
| E | NA | 50 |

### Weighting Best Practices

#### Equal Weighting
Simple and often effective:
```
Value (33.3%)
Growth (33.3%)
Momentum (33.3%)
```

#### Factor-Importance Weighting
Based on research or backtesting:
```
Value (50%)
Quality (30%)
Momentum (20%)
```

#### Optimization
Use Portfolio123's optimizer to find optimal weights:
- Test different weight combinations
- Maximize Sharpe ratio or other metrics
- **Tip**: Keep weights divisible by 4 for cleaner optimization

### Creating Custom Formulas in Ranking

You can rank on formulas, not just single factors:

#### Relative Valuation
```
PEExclXorTTM / Industry(PEExclXorTTM)
```
Ranks stocks by P/E relative to industry.

#### Quality-Adjusted Growth
```
SalesGr%TTM * GrMarginTTM
```
Growth weighted by margin quality.

#### Risk-Adjusted Return
```
(Close(0)/Close(252) - 1) / StdDev(Close(0),252)
```
Sharpe-like ratio for momentum.

### Composite Node Example

**Value Composite**:
- P/E Ratio (30%) - Lower is better
- P/B Ratio (25%) - Lower is better
- P/S Ratio (20%) - Lower is better
- Dividend Yield (15%) - Higher is better
- FCF Yield (10%) - Higher is better

This creates a comprehensive value score combining multiple valuation metrics.

### Conditional Ranking Example

**Sector-Specific Ranking**:

```
IF Sector = TECH
  THEN Rank on: PEG, Revenue Growth, R&D/Sales
ELSE IF Sector = UTILITIES
  THEN Rank on: Dividend Yield, Payout Ratio, Debt/Equity
ELSE
  THEN Rank on: P/E, ROE, Debt/Equity
```

### Using AI Factors in Ranking

AI Factors can be used like any other factor:

```
AI Factor Composite (40%)
├── AIFactor("MyMLModel") (100%)

Traditional Factors (60%)
├── Value (50%)
└── Momentum (50%)
```

This combines machine learning predictions with traditional factors.

### Rank Performance Testing

After creating a ranking system, test it:

1. **Bucketized Backtest**: Divide universe into deciles by rank
2. **Compare Performance**: Top decile vs. bottom decile
3. **Evaluate Metrics**:
   - Sharpe Ratio
   - Information Coefficient (IC)
   - Turnover
   - Decile Spread

**Good Ranking System Characteristics**:
- Top decile significantly outperforms bottom decile
- Monotonic relationship (higher ranks = better returns)
- Stable IC over time
- Reasonable turnover

---

## Part 3: Translating Plain English to P123 Rules

### Example 1: "Find cheap, growing companies"

**Plain English**:
- Low P/E ratio
- High revenue growth
- Positive earnings

**P123 Screen**:
```
PEExclXorTTM < 15
SalesGr%TTM > 0.15
EPSExclXorTTM > 0
```

**P123 Ranking**:
```
Value (50%)
├── PEExclXorTTM (lower is better)

Growth (50%)
├── SalesGr%TTM (higher is better)
```

### Example 2: "High-quality dividend stocks"

**Plain English**:
- Dividend yield above 3%
- Payout ratio sustainable (<80%)
- Strong return on equity
- Low debt

**P123 Screen**:
```
DivYield > 0.03
PayoutTTM < 0.80
ROE > 0.15
DebtToEquity < 0.5
```

**P123 Ranking**:
```
Income Quality (100%)
├── Dividend Yield (40%) - higher is better
├── ROE (30%) - higher is better
├── Payout Ratio (15%) - lower is better
└── Debt/Equity (15%) - lower is better
```

### Example 3: "Momentum stocks with improving fundamentals"

**Plain English**:
- Stock up 20%+ over past year
- Earnings estimates being revised upward
- Positive earnings surprise
- Above-average volume

**P123 Screen**:
```
Close(0)/Close(252) > 1.20
EPSMeanEst#FQ1 > EPSMeanEst#FQ1(4)
EPSSurpriseTTM > 0
Volume(0) > Avg(Volume(0),50)
```

**P123 Ranking**:
```
Momentum (60%)
├── 12-Month Return (50%) - higher is better
├── 6-Month Return (30%) - higher is better
└── Volume Trend (20%) - higher is better

Estimate Revisions (40%)
├── EPS Estimate Change (60%) - higher is better
└── EPS Surprise (40%) - higher is better
```

### Example 4: "Undervalued stocks in technology sector"

**Plain English**:
- Technology sector only
- P/E below sector average
- P/B below 3
- Market cap above $1B

**P123 Screen**:
```
Sector = TECH
PEExclXorTTM < Sector(PEExclXorTTM)
PBExclXor < 3
MktCap > 1000
```

**P123 Ranking**:
```
Relative Value (100%)
├── P/E vs Sector (40%) - lower is better
│   Formula: PEExclXorTTM / Sector(PEExclXorTTM)
├── P/B (30%) - lower is better
└── P/S vs Sector (30%) - lower is better
    Formula: PSExclXor / Sector(PSExclXor)
```

### Example 5: "Small-cap value with insider buying"

**Plain English**:
- Market cap $300M - $2B
- Low P/B ratio
- Insider purchases in last 6 months
- Profitable

**P123 Screen**:
```
MktCap > 300 AND MktCap < 2000
PBExclXor < 1.5
InsiderBuying6M > 0
NetIncTTM > 0
```

**P123 Ranking**:
```
Small Cap Value (100%)
├── Valuation (60%)
│   ├── P/B (50%) - lower is better
│   └── P/E (50%) - lower is better
├── Insider Activity (25%) - higher is better
│   Formula: InsiderBuying6M
└── Profitability (15%) - higher is better
    Formula: ROE
```

---

## Part 4: Advanced Techniques

### SetVar and ShowVar

Create intermediate variables for complex calculations:

```
SetVar("QualityScore", (FRank(ROE) + FRank(ROATTM) + FRank(GrMarginTTM))/3)
SetVar("ValueScore", (FRank(PEExclXorTTM) + FRank(PBExclXor))/2)
ShowVar("ComboScore", QualityScore * ValueScore)
ComboScore > 2500
```

### Industry-Relative Screening

```
# Find stocks with P/E in bottom 20% of their industry
FRank(PEExclXorTTM, Industry) < 20
```

### Multi-Factor Screens with Scoring

```
SetVar("ValuePoints", 
  (PEExclXorTTM < 15) + 
  (PBExclXor < 2) + 
  (PSExclXor < 1) + 
  (DivYield > 0.03))

ShowVar("ValuePoints") >= 3
```
This requires at least 3 out of 4 value criteria to pass.

### Dynamic Thresholds

```
# P/E below 80th percentile of universe
PEExclXorTTM < Percentile(PEExclXorTTM, 80)
```

---

## Part 5: Common Factors Reference

### Valuation
- `PEExclXorTTM` - P/E ratio (excluding extraordinary items)
- `PBExclXor` - Price to Book
- `PSExclXor` - Price to Sales
- `PFCFTTM` - Price to Free Cash Flow
- `EVToEBITDA` - Enterprise Value to EBITDA
- `DivYield` - Dividend Yield
- `FCFYield` - Free Cash Flow Yield

### Growth
- `SalesGr%TTM` - Revenue growth rate (TTM)
- `EPSGr%TTM` - EPS growth rate (TTM)
- `FCFGr%TTM` - Free cash flow growth rate
- `LTGrthRtMean` - Long-term growth estimate

### Quality/Profitability
- `ROE` - Return on Equity
- `ROATTM` - Return on Assets
- `ROICTTM` - Return on Invested Capital
- `GrMarginTTM` - Gross Margin
- `OpMarginTTM` - Operating Margin
- `NetMarginTTM` - Net Margin

### Financial Health
- `DebtToEquity` - Debt to Equity ratio
- `CurrentRatio` - Current Ratio
- `QuickRatio` - Quick Ratio
- `InterestCoverage` - Interest Coverage

### Momentum/Technical
- `Close(0)` - Current price
- `SMA(50)` - 50-day moving average
- `RSI(14)` - Relative Strength Index
- `Volume(0)` - Current volume

### Analyst Estimates
- `EPSMeanEst#FQ1` - Mean EPS estimate for next quarter
- `EPSSurpriseTTM` - EPS surprise (actual vs estimate)
- `NumOfAnalysts` - Number of analysts covering stock

---

## Summary

**Screening**: Binary pass/fail filtering
- Use simple comparisons and logical operators
- Combine with AND/OR for complex logic
- NA values cause stocks to fail rules

**Ranking**: Percentile scoring (0-100)
- Hierarchical node structure with weights
- Factors converted to percentile ranks
- Weighted averages re-normalized
- NA handling configurable (penalized or neutral)

**Best Practices**:
1. Start simple, add complexity gradually
2. Use TTM for freshest fundamental data
3. Combine value, growth, quality, and momentum
4. Test thoroughly with backtests
5. Monitor live performance vs. backtest
6. Rebalance and retrain regularly

**Resources**:
- Factor Reference: https://www.portfolio123.com/doc/doc_index.jsp
- P123 Language Guide: https://www.portfolio123.com/doc/P123AcronymsAndFunctions.pdf
- Community Forum: https://community.portfolio123.com/

