# Business Logic Reference Guide

**Version**: 1.0  
**Last Updated**: 2025-01-21 16:30 ET  
**Purpose**: Quick reference for all business logic calculations and formulas

---

## Table of Contents

1. [Spread Calculations](#spread-calculations)
2. [CR01 Calculations](#cr01-calculations)
3. [Change Metrics](#change-metrics)
4. [Aggregation Logic](#aggregation-logic)
5. [Filtering Logic](#filtering-logic)
6. [Pair Analytics](#pair-analytics)

---

## Spread Calculations

### Tight Bid >3mm

**Definition**: Best bid spread available with size > 3mm

**Formula**:
```
Tight Bid >3mm = MIN(Bid Spread) WHERE Bid Size > 3,000,000 AND Bid Spread >= 0
```

**Associated Data**:
- `Dealer @ Tight Bid >3mm`: Dealer providing tightest bid
- `Size @ Tight Bid >3mm`: Size available at tightest bid

**Business Rule**: Negative spreads filtered out (data quality issue)

### Wide Offer >3mm

**Definition**: Worst offer spread available with size > 3mm

**Formula**:
```
Wide Offer >3mm = MAX(Ask Spread) WHERE Ask Size > 3,000,000 AND Ask Spread >= 0
```

**Associated Data**:
- `Dealer @ Wide Offer >3mm`: Dealer providing widest offer
- `Size @ Wide Offer >3mm`: Size available at widest offer

**Business Rule**: Negative spreads filtered out (data quality issue)

### Tight Bid (Overall)

**Definition**: Best bid spread regardless of size

**Formula**:
```
Tight Bid = MIN(Bid Spread) WHERE Bid Spread >= 0
```

**Business Rule**: Negative spreads filtered out

### Wide Offer (Overall)

**Definition**: Worst offer spread regardless of size

**Formula**:
```
Wide Offer = MAX(Ask Spread) WHERE Ask Spread >= 0
```

**Business Rule**: Negative spreads filtered out

### Bid/Offer Spread

**Definition**: Difference between tight bid and wide offer

**Formulas**:
```
Bid/Offer>3mm = Tight Bid >3mm - Wide Offer >3mm
Bid/Offer = Tight Bid - Wide Offer
```

**Business Rules**:
- Only calculated if BOTH values exist (not NaN)
- Negative value = bid above offer (inverted market)
- Positive value = normal bid-offer spread

---

## CR01 Calculations

### CR01 @ Tight Bid

**Definition**: Credit risk exposure at tightest bid >3mm

**Formula**:
```
CR01 @ Tight Bid = Bid Workout Risk (avg) × (Size @ Tight Bid >3mm / 10,000)
```

**Where**:
- `Bid Workout Risk (avg)` = Average Bid Workout Risk across all dealers for Date+CUSIP+Benchmark group
- `Size @ Tight Bid >3mm` = Size available at tightest bid >3mm

**Units**: CR01 per $10k face value

**Business Rule**: Only calculated if both Bid Workout Risk and Size exist

### CR01 @ Wide Offer

**Definition**: Credit risk exposure at widest offer >3mm

**Formula**:
```
CR01 @ Wide Offer = Bid Workout Risk (avg) × (Size @ Wide Offer >3mm / 10,000)
```

**Where**:
- `Bid Workout Risk (avg)` = Average Bid Workout Risk across all dealers for Date+CUSIP+Benchmark group
- `Size @ Wide Offer >3mm` = Size available at widest offer >3mm

**Units**: CR01 per $10k face value

**Business Rule**: Only calculated if both Bid Workout Risk and Size exist

---

## Change Metrics

### Day-over-Day (DoD) Change

**Definition**: Change from previous trading day

**Formula**:
```
DoD Chg [Metric] = Last Date Value - Second Last Date Value
```

**Business Rules**:
1. Only calculated if CUSIP+Benchmark pair exists on both dates
2. Only calculated if BOTH values exist (not NaN/blank)
3. If either value is blank/NaN, DoD = blank (pd.NA)
4. Matching by (CUSIP, Benchmark) tuple ensures correct pairing

**Applied To** (17 metrics):
- Tight Bid >3mm
- Wide Offer >3mm
- Tight Bid
- Wide Offer
- Size @ Tight Bid >3mm
- Size @ Wide Offer >3mm
- CR01 @ Tight Bid
- CR01 @ Wide Offer
- Cumm. Bid Size
- Cumm. Offer Size
- # of Bids >3mm
- # of Offers >3mm
- Bid RBC
- Ask RBC
- Bid Size RBC
- Offer Size RBC

### Month-to-Date (MTD) Change

**Definition**: Change from first trading day of current month

**Formula**:
```
MTD Chg [Metric] = Last Date Value - MTD Reference Date Value

MTD Reference Date = First date of current month (or closest available date)
```

**Business Rules**:
1. Only calculated if CUSIP+Benchmark pair exists on both dates
2. Only calculated if BOTH values exist (not NaN)
3. If either value is blank/NaN, MTD = blank (pd.NA)

**Applied To**: Same 17 metrics as DoD

### Year-to-Date (YTD) Change

**Definition**: Change from first trading day of current year

**Formula**:
```
YTD Chg [Metric] = Last Date Value - YTD Reference Date Value

YTD Reference Date = January 1st of current year (or closest available date)
```

**Business Rules**:
1. Only calculated if CUSIP+Benchmark pair exists on both dates
2. Only calculated if BOTH values exist (not NaN)
3. If either value is blank/NaN, YTD = blank (pd.NA)

**Applied To**: Same 17 metrics as DoD

### Custom Date Change

**Definition**: Change from approximately 1 year ago

**Formula**:
```
Custom Date Chg [Metric] = Last Date Value - Custom Date Reference Value

Custom Date Reference = Closest available date ~1 year ago
```

**Business Rules**:
1. Only calculated if CUSIP+Benchmark pair exists on both dates
2. Only calculated if BOTH values exist (not NaN)
3. If either value is blank/NaN, Custom Date = blank (pd.NA)

**Applied To**: Same 17 metrics as DoD

---

## Aggregation Logic

### Group Aggregation (Date+CUSIP+Benchmark)

**Purpose**: Aggregate multiple dealer quotes per bond per day

**Grouping**: `Date`, `CUSIP`, `Benchmark`

**Aggregations**:

1. **Tight Bid >3mm**:
   ```
   Filter: Bid Size > 3,000,000 AND Bid Spread >= 0
   Result: MIN(Bid Spread)
   ```

2. **Wide Offer >3mm**:
   ```
   Filter: Ask Size > 3,000,000 AND Ask Spread >= 0
   Result: MAX(Ask Spread)
   ```

3. **Tight Bid**:
   ```
   Filter: Bid Spread >= 0
   Result: MIN(Bid Spread)
   ```

4. **Wide Offer**:
   ```
   Filter: Ask Spread >= 0
   Result: MAX(Ask Spread)
   ```

5. **Cumm. Bid Size**:
   ```
   Result: SUM(Bid Size) (ignoring NaN)
   ```

6. **Cumm. Offer Size**:
   ```
   Result: SUM(Ask Size) (ignoring NaN)
   ```

7. **# of Bids >3mm**:
   ```
   Filter: Bid Size > 3,000,000
   Result: COUNT(DISTINCT Dealer)
   ```

8. **# of Offers >3mm**:
   ```
   Filter: Ask Size > 3,000,000
   Result: COUNT(DISTINCT Dealer)
   ```

9. **# Quotes**:
   ```
   Result: COUNT(DISTINCT Dealer) per CUSIP (aggregated across all Benchmarks)
   ```

10. **Bid Workout Risk**:
    ```
    Result: AVG(Bid Workout Risk)
    ```

11. **Time**:
    ```
    Result: MAX(Time) (latest time in group)
    ```

12. **RBC Columns**:
    ```
    Filter: Dealer = 'RBC'
    Result: First occurrence (or latest time if multiple)
    ```

### Portfolio Aggregation

**Purpose**: Aggregate portfolio holdings across multiple accounts/portfolios

**Grouping**: `CUSIP` (normalized)

**Aggregations**:

1. **QUANTITY**:
   ```
   Result: SUM(QUANTITY) across all ACCOUNT+PORTFOLIO combinations
   ```

2. **POSITION CR01**:
   ```
   Result: SUM(POSITION CR01) across all ACCOUNT+PORTFOLIO combinations
   ```

**Business Rule**: Sum values when same CUSIP appears in multiple accounts/portfolios

---

## Filtering Logic

### Dealer Filtering (Runs Pipeline)

**Purpose**: Include only relevant dealers

**Filter**:
```
Dealer IN ('BMO', 'BNS', 'NBF', 'RBC', 'TD')
```

**Applied**: During load step (before writing to parquet)

**Business Rule**: Other dealers filtered out (not stored in parquet)

### Negative Spread Filtering

**Purpose**: Remove data quality issues

**Filter**:
```
Bid Spread < 0 → Set to NaN
Ask Spread < 0 → Set to NaN
```

**Applied**: 
- Before aggregation (in `runs_today.py`)
- During load step (in `runs_pipeline/load.py`)

**Business Rule**: Negative spreads are data quality issues, not valid quotes

### Outlier Spread Filtering

**Purpose**: Remove extreme outliers based on Date+CUSIP group statistics

**Method**:
```
1. Group by Date+CUSIP
2. For each group:
   - Calculate high = MAX(Bid Spread, Ask Spread) from valid positive values
   - Calculate low = MIN(Bid Spread, Ask Spread) from valid positive values
   - Exclude current row from statistics (prevent self-skewing)
   - Drop rows where Bid Spread OR Ask Spread outside [low - 20, high + 20]
3. Exclude CUSIPs with Custom_Sector IN ('Non Financial Hybrid', 'Non Financial Hybrids', 'Financial Hybrid', 'HY')
```

**Applied**: After deduplication, before writing to parquet

**Business Rule**: Filters both positive and negative outliers based on group statistics

### Size Threshold Filtering

**Purpose**: Identify quotes with meaningful size

**Threshold**: 3,000,000 (3mm)

**Applied To**:
- Tight Bid >3mm / Wide Offer >3mm calculations
- # of Bids >3mm / # of Offers >3mm counts
- Size @ Tight Bid >3mm / Size @ Wide Offer >3mm

**Business Rule**: Only quotes with size > 3mm considered for ">3mm" metrics

### Sector Filtering (Universe Views)

**Purpose**: Exclude non-trading sectors from universe analysis

**Excluded Sectors**:
```
'Asset Backed Subs', 'Auto ABS', 'Bail In', 'CAD Govt', 'CASH CAD', 
'CASH USD', 'CDX', 'CP', 'Covered', 'Dep Note', 'Financial Hybrid', 
'HY', 'Non Financial Hybrid', 'Non Financial Hybrids', 'USD Govt', 
'University', 'Utility'
```

**Applied**: In `runs_views.py` for universe tables

**Business Rule**: These sectors not relevant for relative value trading

---

## Pair Analytics

### Spread Calculation

**Purpose**: Compare two bonds' spread values over time

**Formula**:
```
For each Date:
  Spread = Bond_1 Value - Bond_2 Value

Time series: [Spread_1, Spread_2, ..., Spread_N]
```

**Business Rule**: Spreads aligned by Date (both bonds must have data on same date)

### Statistics Calculation

**Input**: Time series of spreads `[S_1, S_2, ..., S_N]`

**Formulas**:

1. **Last Value**:
   ```
   Last = S_N (most recent spread)
   ```

2. **Average Value**:
   ```
   Avg = MEAN([S_1, S_2, ..., S_N])
   ```

3. **vs Average**:
   ```
   vs Avg = Last - Avg
   ```

4. **Z Score**:
   ```
   IF StdDev > 0:
     Z Score = vs Avg / StdDev
   ELSE:
     Z Score = NULL
   ```

5. **Percentile**:
   ```
   Percentile = (COUNT(S_i <= Last) / N) × 100
   ```

**Business Rules**:
- Requires at least 2 data points
- Z Score only calculated if standard deviation > 0
- Percentile ranges from 0-100 (0 = lowest, 100 = highest)

### Pair Filtering

**Recent Date Filter**:
```
Keep CUSIPs present in most recent 75% of dates
```

**Term Filter** (Term Combinations):
```
ABS(Yrs (Cvn)_1 - Yrs (Cvn)_2) <= 0.8
```

**Ticker Filter** (Ticker Combinations):
```
Ticker_1 = Ticker_2
```

**Sector Filter** (Custom Sector):
```
Custom_Sector_1 = Custom_Sector_2
```

**CAD/USD Filter**:
```
Ticker_1 = Ticker_2 AND
Custom_Sector_1 = Custom_Sector_2 AND
ABS(Yrs (Cvn)_1 - Yrs (Cvn)_2) <= 2.0
```

**CR01 Filter** (Executable CR01 analyses):
```
CR01 @ Tight Bid > 2000 (for holdings)
CR01 @ Wide Offer > 2000 (for universe)
Bid/Offer>3mm < 3 (for decent bid offer filter)
```

---

## Deduplication Logic

### Bond Pipeline Deduplication

**Primary Key**: `Date + CUSIP`

**Strategy**: Keep last occurrence

**Applied**: Within each file and across files

**Business Rule**: One record per Date+CUSIP (most recent data wins)

### Runs Pipeline Deduplication

**Primary Key**: `Date + Dealer + CUSIP`

**Strategy**: End-of-day snapshot

**Method**:
```
1. Group by Date+Dealer+CUSIP
2. Sort by Time descending (latest first)
3. If tie on Time, keep last row by position
4. Keep first row per group (latest Time)
```

**Business Rule**: Capture end-of-day quote snapshot per dealer per bond

### Portfolio Pipeline Deduplication

**Primary Key**: `Date + CUSIP + ACCOUNT + PORTFOLIO`

**Strategy**: Keep last occurrence

**Business Rule**: Preserve account/portfolio detail while removing duplicates

---

## CUSIP Normalization

**Purpose**: Standardize CUSIP identifiers for matching

**Process**:
```
1. Convert to uppercase: "89678zab2" → "89678ZAB2"
2. Remove trailing "Corp" or " CORP": "06418GAD9 Corp" → "06418GAD9"
3. Remove whitespace: " 06418GAD9 " → "06418GAD9"
4. Take first 9 characters: "06418GAD9EXTRA" → "06418GAD9"
5. Validate length: Must be exactly 9 characters
```

**Applied In**:
- Bond Pipeline: All CUSIPs normalized
- Portfolio Pipeline: All CUSIPs normalized
- Runs Pipeline: CUSIPs kept as-is, but normalized for matching

**Business Rule**: Invalid CUSIPs logged but kept in data

---

## Reference Date Calculation

### MTD Reference Date

**Purpose**: Find first trading day of current month

**Method**:
```
1. Get last date from data
2. Calculate first of month: pd.Timestamp(last_date.year, last_date.month, 1)
3. Find closest available date >= first of month AND < last date
4. Use that date as MTD reference
```

### YTD Reference Date

**Purpose**: Find first trading day of current year

**Method**:
```
1. Get last date from data
2. Calculate first of year: pd.Timestamp(last_date.year, 1, 1)
3. Find closest available date >= first of year AND < last date
4. Use that date as YTD reference
```

### Custom Date Reference

**Purpose**: Find date approximately 1 year ago

**Method**:
```
1. Get last date from data
2. Calculate one year ago: last_date - pd.DateOffset(years=1)
3. Find closest available date <= one year ago
4. Use that date as Custom Date reference
```

---

**End of Reference Guide**

