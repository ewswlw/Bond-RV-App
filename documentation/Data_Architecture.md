# Bond RV App - Complete Data Architecture Documentation

**Version**: 1.0  
**Last Updated**: 2025-01-21 16:30 ET  
**Author**: Data Engineering Team  
**Purpose**: Comprehensive documentation of data architecture, business logic, and system design for developers and AI agents

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Data Sources & Inputs](#data-sources--inputs)
5. [Data Processing Pipelines](#data-processing-pipelines)
6. [Data Storage (Parquet Files)](#data-storage-parquet-files)
7. [Analytics & Calculations](#analytics--calculations)
8. [Output Tables & Views](#output-tables--views)
9. [Business Logic Documentation](#business-logic-documentation)
10. [Database Schema Design](#database-schema-design)
11. [Developer Guide](#developer-guide)

---

## Executive Summary

The Bond RV App is a comprehensive data engineering system for processing bond market data for relative value trading applications. The system processes three primary data sources:

1. **Bond Details** - Historical bond characteristics and metrics (75 columns)
2. **Runs Data** - Dealer quote runs with bid/offer spreads and sizes (30 columns)
3. **Portfolio Holdings** - Portfolio positions with account/portfolio detail (82 columns)

The system transforms raw Excel files into optimized Parquet time-series tables, then generates analytics including:
- Daily runs analytics with Day-over-Day (DoD), Month-to-Date (MTD), Year-to-Date (YTD), and Custom Date changes
- Pair analytics comparing bond spreads across different criteria
- Formatted portfolio and universe views for trading decisions

**Key Metrics Calculated:**
- Tight Bid/Wide Offer spreads (>3mm size threshold)
- CR01 (credit risk) calculations
- Bid/Offer spreads
- Cumulative bid/offer sizes
- Dealer quote counts
- Historical change metrics (DoD, MTD, YTD, Custom Date)

---

## System Overview

### Architecture Pattern

The system follows a **modular ETL (Extract, Transform, Load) architecture** with three independent pipelines:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Bond Pipeline  │     │  Runs Pipeline  │     │Portfolio Pipeline│
│                 │     │                 │     │                 │
│ Extract →       │     │ Extract →       │     │ Extract →       │
│ Transform →     │     │ Transform →     │     │ Transform →     │
│ Load            │     │ Load            │     │ Load            │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Parquet Storage       │
                    │  (5 parquet files)      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Analytics Layer       │
                    │  (runs_today.py,        │
                    │   runs_views.py,        │
                    │   comb.py)              │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Output Tables         │
                    │  (CSV, Excel, TXT)     │
                    └─────────────────────────┘
```

### Technology Stack

- **Language**: Python 3.11+
- **Data Processing**: pandas 2.0+, pyarrow 12.0+
- **File I/O**: openpyxl 3.1+ (Excel), pyarrow (Parquet)
- **Environment**: Poetry (dependency management)
- **Storage Format**: Parquet (columnar, compressed, time-series optimized)

### Key Design Principles

1. **Modular Design**: Each pipeline is independent with clear separation of concerns
2. **Incremental Loading**: Append mode for daily updates, override mode for rebuilds
3. **Data Integrity**: Primary key enforcement, deduplication, validation
4. **Schema Evolution**: Master schema detection from latest files, backward compatibility
5. **Performance**: Vectorized operations, optimized Parquet storage

---

## Data Flow Architecture

### High-Level Data Flow

```
Raw Excel Files
    ↓
[Extract Layer]
    ├─ Excel file discovery
    ├─ Date extraction (filename or columns)
    ├─ Header row detection
    └─ Raw DataFrame creation
    ↓
[Transform Layer]
    ├─ CUSIP normalization
    ├─ Schema alignment
    ├─ NA value cleaning
    ├─ Deduplication
    ├─ Data type conversion
    └─ Validation
    ↓
[Load Layer]
    ├─ Date checking (append mode)
    ├─ Primary key enforcement
    ├─ Parquet writing
    └─ Summary statistics
    ↓
Parquet Files
    ├─ historical_bond_details.parquet
    ├─ universe.parquet
    ├─ bql.parquet
    ├─ runs_timeseries.parquet
    └─ historical_portfolio.parquet
    ↓
[Analytics Layer]
    ├─ runs_today.py (aggregation & change calculations)
    ├─ runs_views.py (formatted tables)
    └─ comb.py (pair analytics)
    ↓
Output Files
    ├─ runs_today.csv
    ├─ portfolio_runs_view.txt/xlsx
    ├─ uni_runs_view.txt/xlsx
    └─ comb.xlsx/txt/csv
```

### Detailed Pipeline Flow

#### Bond Pipeline Flow

```
1. File Discovery
   └─ Pattern: "API MM.DD.YY.xlsx"
   └─ Header row: Row 3 (index 2)
   └─ Date extraction: From filename

2. Extraction
   └─ Read Excel with header row
   └─ Extract date from filename
   └─ Create DataFrame with Date column first

3. Transformation
   └─ Normalize CUSIPs (uppercase, 9-char, remove "Corp")
   └─ Validate CUSIPs (log invalid, keep in data)
   └─ Remove duplicates (Date+CUSIP, keep last)
   └─ Clean NA values (#N/A patterns → NaN)
   └─ Convert numeric columns (Years, Spreads, Metrics)
   └─ Align to master schema (75 columns)

4. Loading
   └─ Append mode: Skip existing dates, append new
   └─ Override mode: Delete existing, rebuild all
   └─ Create universe.parquet (most recent per CUSIP)
   └─ Optional: Process BQL workbook → bql.parquet
```

#### Runs Pipeline Flow

```
1. File Discovery
   └─ Pattern: "RUNS MM.DD.YY.xlsx"
   └─ Header row: Row 1 (index 0)
   └─ Date extraction: From Date column (not filename)

2. Extraction
   └─ Read Excel with header row
   └─ Parse Date column (MM/DD/YY format)
   └─ Parse Time column (HH:MM format)
   └─ Combine Date + Time → datetime objects

3. Transformation
   └─ Normalize dealer names (SCM → BNS)
   └─ End-of-day deduplication (Date+Dealer+CUSIP, keep latest Time)
   └─ Validate CUSIPs (check length, log invalid)
   └─ Track orphan CUSIPs (not in universe.parquet)
   └─ Clean NA values
   └─ Align to master schema (30 columns)

4. Loading
   └─ Filter dealers (BMO, BNS, NBF, RBC, TD only)
   └─ Filter negative spreads (set to NaN)
   └─ Filter outlier spreads ([low-20, high+20] range)
   └─ Append/Override mode
   └─ Primary key: Date+Dealer+CUSIP
```

#### Portfolio Pipeline Flow

```
1. File Discovery
   └─ Pattern: "Aggies MM.DD.YY.xlsx"
   └─ Header row: Row 1 (index 0)
   └─ Date extraction: From filename

2. Extraction
   └─ Read Excel with header row
   └─ Remove "Unnamed:" columns
   └─ Extract date from filename
   └─ Create DataFrame with Date column first

3. Transformation
   └─ Filter rows (drop blank SECURITY or CUSIP)
   └─ Normalize CUSIPs (same as bond pipeline)
   └─ Clean NA values
   └─ Remove duplicates (Date+CUSIP+ACCOUNT+PORTFOLIO, keep last)
   └─ Align to master schema (82 columns)

4. Loading
   └─ Append/Override mode
   └─ Primary key: Date+CUSIP+ACCOUNT+PORTFOLIO
```

---

## Data Sources & Inputs

### Bond Details (API Historical)

**File Pattern**: `API MM.DD.YY.xlsx` (e.g., `API 10.20.25.xlsx`)

**Structure**:
- Header row: Row 3 (index 2, 0-based)
- Columns: 75 columns (evolved from 59 in older files)
- Date: Extracted from filename pattern

**Key Columns**:
- `CUSIP` - 9-character identifier
- `Security` - Bond name/description
- `Benchmark` - Reference benchmark
- `G Sprd` - Spread to government
- `Yrs (Cvn)` - Years to conversion
- `vs BI`, `vs BCE` - Spreads vs benchmarks
- `MTD Equity`, `YTD Equity` - Equity performance metrics
- `Retracement`, `Z Score` - Statistical metrics
- `Custom_Sector` - Sector classification
- `Currency` - CAD/USD
- `Ticker` - Bloomberg ticker

**NA Values**: `#N/A Field Not Applicable`, `#N/A Invalid Security`, `N/A`, etc.

### Runs Data (Historical Runs)

**File Pattern**: `RUNS MM.DD.YY.xlsx` (e.g., `RUNS 10.31.25.xlsx`)

**Structure**:
- Header row: Row 1 (index 0)
- Columns: 30 columns (evolved from 28 in one old file)
- Date: From `Date` column (MM/DD/YY format)
- Time: From `Time` column (HH:MM format)

**Key Columns**:
- `Date` - Trading date
- `Time` - Quote time
- `CUSIP` - 9-character identifier (not normalized)
- `Dealer` - Dealer name (BMO, BNS, NBF, RBC, TD, etc.)
- `Security` - Bond name
- `Benchmark` - Reference benchmark
- `Bid Spread` - Bid spread to benchmark
- `Ask Spread` - Ask spread to benchmark
- `Bid Size` - Bid size in face value
- `Ask Size` - Ask size in face value
- `Bid Workout Risk` - CR01 per $10k face value
- `Bid Price`, `Ask Price` - Quote prices
- `Ticker` - Bloomberg ticker

**Dealers**: BMO, BNS, NBF, RBC, TD (filtered during load), plus others filtered out

**Data Quality Issues**:
- Negative spreads (filtered to NaN)
- Outlier spreads (filtered by Date+CUSIP group statistics)

### Portfolio Holdings (AD History)

**File Pattern**: `Aggies MM.DD.YY.xlsx` (e.g., `Aggies 11.04.25.xlsx`)

**Structure**:
- Header row: Row 1 (index 0)
- Columns: 82 columns (detected dynamically)
- Date: Extracted from filename
- Unnamed columns: Automatically removed

**Key Columns**:
- `CUSIP` - 9-character identifier
- `SECURITY` - Bond name
- `ACCOUNT` - Account identifier
- `PORTFOLIO` - Portfolio identifier
- `QUANTITY` - Position quantity
- `POSITION CR01` - Position CR01 risk
- Plus 77 other columns (prices, yields, spreads, etc.)

**Row Filtering**: Rows with blank SECURITY or CUSIP are dropped

### BQL Workbook

**File**: `bql.xlsx` (fixed location in Support Files)

**Structure**:
- Sheet name: `bql`
- Multi-index header: 4 rows
  - Row 0: Ignored
  - Row 1: Security names (1st level)
  - Row 2: CUSIPs (2nd level)
  - Row 3: Ignored
  - Row 4: Data starts
- Column 0: Daily timestamps (`CUSIPs` label)

**Output Format**: Long-form DataFrame
- Columns: `Date`, `Name`, `CUSIP`, `Value`
- Value: Spread value from Bloomberg query

---

## Data Processing Pipelines

### Bond Pipeline (`bond_pipeline/`)

**Modules**:
- `config.py` - Configuration, paths, constants
- `utils.py` - Helper functions (date parsing, CUSIP validation, logging)
- `extract.py` - Excel file reading and date extraction
- `transform.py` - Data cleaning, normalization, deduplication
- `load.py` - Parquet writing (append/override modes)
- `pipeline.py` - Main orchestration script with CLI

**Processing Steps**:

1. **Extraction** (`ExcelExtractor`):
   ```python
   - Discover Excel files matching pattern
   - Extract date from filename (MM.DD.YY → datetime)
   - Read Excel with header row 3
   - Add Date column as first column
   ```

2. **Transformation** (`DataTransformer`):
   ```python
   - Normalize CUSIPs: uppercase, remove "Corp", enforce 9-char
   - Validate CUSIPs: check length, log invalid (keep in data)
   - Remove duplicates: Date+CUSIP, keep last occurrence
   - Clean NA values: #N/A patterns → NaN
   - Convert numeric: Years, Spreads, Metrics → float64
   - Align schema: Fill missing columns with NaN, reorder
   ```

3. **Loading** (`ParquetLoader`):
   ```python
   - Append mode:
     * Read existing parquet dates
     * Skip dates already present
     * Append new dates only
   - Override mode:
     * Delete existing parquet files
     * Rebuild from all Excel files
   - Create universe.parquet:
     * Group by CUSIP
     * Keep most recent date per CUSIP
     * Select 13 key columns only
   ```

**Output Files**:
- `historical_bond_details.parquet` - Time series (Date+CUSIP primary key)
- `universe.parquet` - Current universe (CUSIP primary key)
- `bql.parquet` - BQL spreads (Date+CUSIP primary key, optional)

### Runs Pipeline (`runs_pipeline/`)

**Modules**:
- `extract.py` - RUNS Excel reading, Date/Time parsing
- `transform.py` - End-of-day deduplication, CUSIP validation/orphan tracking
- `load.py` - Runs parquet writing, dealer filtering, outlier filtering
- `pipeline.py` - Main orchestration script with CLI

**Processing Steps**:

1. **Extraction** (`RunsExtractor`):
   ```python
   - Discover Excel files matching pattern
   - Read Excel with header row 1
   - Parse Date column: MM/DD/YY → datetime
   - Parse Time column: HH:MM → time object
   - Combine Date + Time → datetime
   ```

2. **Transformation** (`RunsTransformer`):
   ```python
   - Normalize dealers: SCM → BNS
   - End-of-day deduplication:
     * Group by Date+Dealer+CUSIP
     * Keep latest Time per group
     * If tie, keep last row by position
   - Validate CUSIPs: check length (vectorized), log invalid
   - Track orphans: Compare with universe.parquet
   - Clean NA values
   - Align schema: 30 columns
   ```

3. **Loading** (`RunsLoader`):
   ```python
   - Filter dealers: BMO, BNS, NBF, RBC, TD only
   - Filter negative spreads: Set to NaN
   - Filter outlier spreads:
     * Group by Date+CUSIP
     * Calculate high/low from valid positive values
     * Drop rows outside [low-20, high+20] range
     * Exclude hybrid sectors from outlier calc
   - Append/Override mode
   - Primary key: Date+Dealer+CUSIP
   ```

**Output Files**:
- `runs_timeseries.parquet` - Time series (Date+Dealer+CUSIP primary key)

### Portfolio Pipeline (`portfolio_pipeline/`)

**Modules**:
- `extract.py` - Portfolio Excel reading, date extraction from filename
- `transform.py` - CUSIP normalization, row filtering, deduplication
- `load.py` - Portfolio parquet writing
- `pipeline.py` - Main orchestration script with CLI

**Processing Steps**:

1. **Extraction** (`PortfolioExtractor`):
   ```python
   - Discover Excel files matching pattern
   - Read Excel with header row 1
   - Remove "Unnamed:" columns
   - Extract date from filename (MM.DD.YY → datetime)
   - Add Date column as first column
   ```

2. **Transformation** (`PortfolioTransformer`):
   ```python
   - Filter rows: Drop blank SECURITY or CUSIP
   - Normalize CUSIPs: Same as bond pipeline
   - Clean NA values
   - Remove duplicates: Date+CUSIP+ACCOUNT+PORTFOLIO, keep last
   - Align schema: 82 columns
   ```

3. **Loading** (`PortfolioLoader`):
   ```python
   - Append/Override mode
   - Primary key: Date+CUSIP+ACCOUNT+PORTFOLIO
   ```

**Output Files**:
- `historical_portfolio.parquet` - Time series (Date+CUSIP+ACCOUNT+PORTFOLIO primary key)

---

## Data Storage (Parquet Files)

### File Locations

All parquet files stored in: `bond_data/parquet/`

### File Specifications

#### 1. `historical_bond_details.parquet`

**Purpose**: Time series of all bonds over time

**Primary Key**: `Date + CUSIP` (unique combination)

**Schema**: 75 columns + `Date` column (first position)

**Row Count**: ~25,000+ rows spanning 2023-2025

**Key Columns**:
- `Date` (datetime64) - First column
- `CUSIP` (string) - 9-character identifier
- `Security` (string) - Bond name
- `Benchmark` (string) - Reference benchmark
- `G Sprd` (float64) - Spread to government
- `Yrs (Cvn)` (float64) - Years to conversion
- `vs BI`, `vs BCE` (float64) - Spreads vs benchmarks
- `MTD Equity`, `YTD Equity` (float64) - Equity metrics
- `Retracement` (float64) - Retracement percentage
- `Z Score` (float64) - Statistical z-score
- `Custom_Sector` (string) - Sector classification
- `Currency` (string) - CAD/USD
- `Ticker` (string) - Bloomberg ticker
- Plus 60+ other columns

**Update Strategy**: Append mode (skip existing dates) or Override mode (rebuild all)

**Deduplication**: One record per Date+CUSIP (keeps last occurrence)

#### 2. `universe.parquet`

**Purpose**: Current universe of all unique CUSIPs ever seen

**Primary Key**: `CUSIP` (unique)

**Schema**: 13 key columns only

**Columns** (in order):
1. `CUSIP`
2. `Benchmark Cusip`
3. `Custom_Sector`
4. `Bloomberg Cusip`
5. `Security`
6. `Benchmark`
7. `Pricing Date`
8. `Pricing Date (Bench)`
9. `Worst Date`
10. `Yrs (Worst)`
11. `Ticker`
12. `Currency`
13. `Equity Ticker`

**Update Strategy**: Always rebuild from `historical_bond_details` (most recent date per CUSIP)

**Row Count**: ~1,000+ unique CUSIPs

#### 3. `bql.parquet`

**Purpose**: Long-form Bloomberg query spreads dataset

**Primary Key**: `Date + CUSIP`

**Schema**: 4 columns
- `Date` (datetime64)
- `Name` (string) - Security name from workbook header
- `CUSIP` (string) - 9-character identifier
- `Value` (float64) - Spread value

**Update Strategy**: Override mode (always rebuild from `bql.xlsx`)

**Row Count**: Varies by date range and CUSIP coverage

**Orphan Tracking**: Logs CUSIPs not in `universe.parquet`

#### 4. `runs_timeseries.parquet`

**Purpose**: Time series of dealer quotes over time (end-of-day snapshots)

**Primary Key**: `Date + Dealer + CUSIP` (unique combination, enforced after deduplication)

**Schema**: 30 columns with Date and Time as first columns

**Dealer Filtering**: Only includes dealers: BMO, BNS, NBF, RBC, TD

**Data Quality Filters**:
- Negative Bid Spread and Ask Spread values filtered out (set to NaN)
- Outlier Spread Filtering: Rows with Bid Spread or Ask Spread outside `[low - 20, high + 20]` range (based on Date+CUSIP group statistics) are dropped
- Outlier filtering excludes CUSIPs with Custom_Sector: "Non Financial Hybrid", "Non Financial Hybrids", "Financial Hybrid", "HY"

**Row Count**: ~19,000+ rows (after deduplication, dealer filtering, and outlier filtering) spanning 2022-2025

**Key Columns**:
- `Date` (datetime64) - First column
- `Time` (time) - Second column
- `CUSIP` (string) - Not normalized (kept as-is)
- `Dealer` (string) - BMO, BNS, NBF, RBC, TD only
- `Security` (string) - Bond name
- `Benchmark` (string) - Reference benchmark
- `Bid Spread` (float64) - Bid spread to benchmark (negative values filtered)
- `Ask Spread` (float64) - Ask spread to benchmark (negative values filtered)
- `Bid Size` (float64) - Bid size in face value
- `Ask Size` (float64) - Ask size in face value
- `Bid Workout Risk` (float64) - CR01 per $10k face value
- `Bid Price`, `Ask Price` (float64) - Quote prices
- Plus 18 other columns

**Deduplication**: Keep latest Time per Date+Dealer+CUSIP (end-of-day snapshot)

#### 5. `historical_portfolio.parquet`

**Purpose**: Time series of portfolio holdings over time

**Primary Key**: `Date + CUSIP + ACCOUNT + PORTFOLIO` (unique combination, preserves account/portfolio detail)

**Schema**: 82 columns (excluding Unnamed columns) with Date column first

**Column Filtering**: Unnamed columns (Unnamed: 0, Unnamed: 1, etc.) automatically removed

**Row Filtering**: Rows with blank SECURITY or CUSIP automatically dropped

**Row Count**: Varies based on portfolio holdings across dates

**Key Columns**:
- `Date` (datetime64) - First column
- `CUSIP` (string) - Normalized 9-character identifier
- `SECURITY` (string) - Bond name
- `ACCOUNT` (string) - Account identifier
- `PORTFOLIO` (string) - Portfolio identifier
- `QUANTITY` (float64) - Position quantity
- `POSITION CR01` (float64) - Position CR01 risk
- Plus 75 other columns

**Deduplication**: Keep last occurrence per Date+CUSIP+ACCOUNT+PORTFOLIO

---

## Analytics & Calculations

### Runs Today Analytics (`analytics/runs/runs_today.py`)

**Purpose**: Aggregate runs data by Date+CUSIP+Benchmark, compute change metrics, merge external data

**Input**: `runs_timeseries.parquet`

**Output**: `runs_today.csv`

**Processing Steps**:

1. **Load Data**:
   ```python
   - Read runs_timeseries.parquet
   - Find last date and second-to-last date
   - Find reference dates: MTD (first of month), YTD (first of year), Custom Date (~1 year ago)
   ```

2. **Filter to Required Dates**:
   ```python
   - Only process dates needed for calculations:
     * Last date (today)
     * Second-to-last date (for DoD)
     * MTD reference date
     * YTD reference date
     * Custom Date reference date
   - Optimized: Process only 5 dates instead of all dates
   ```

3. **Aggregate by Date+CUSIP+Benchmark**:
   ```python
   For each Date+CUSIP+Benchmark group:
   - Tight Bid >3mm: Smallest Bid Spread with Bid Size > 3mm
   - Wide Offer >3mm: Largest Ask Spread with Ask Size > 3mm
   - Tight Bid: Smallest Bid Spread overall (excluding negatives)
   - Wide Offer: Largest Ask Spread overall (excluding negatives)
   - Dealer @ Tight Bid >3mm: Dealer with tightest bid >3mm
   - Dealer @ Wide Offer >3mm: Dealer with widest offer >3mm
   - Size @ Tight Bid >3mm: Size at tightest bid >3mm
   - Size @ Wide Offer >3mm: Size at widest offer >3mm
   - CR01 @ Tight Bid: Bid Workout Risk * (Size @ Tight Bid >3mm / 10000)
   - CR01 @ Wide Offer: Bid Workout Risk * (Size @ Wide Offer >3mm / 10000)
   - Cumm. Bid Size: Sum of all Bid Sizes
   - Cumm. Offer Size: Sum of all Ask Sizes
   - # of Bids >3mm: Count unique Dealers with Bid Size > 3mm
   - # of Offers >3mm: Count unique Dealers with Ask Size > 3mm
   - Bid RBC, Ask RBC: RBC dealer quotes (first occurrence)
   - Bid Size RBC, Offer Size RBC: RBC dealer sizes
   - Bid Workout Risk: Average across group
   - Time: Latest Time in group
   ```

4. **Filter to Last Date**:
   ```python
   - Keep only rows from last date
   - Match by (CUSIP, Benchmark) tuple to ensure correct pairing
   ```

5. **Calculate Change Metrics**:
   ```python
   For each metric column:
   - DoD Chg: Last Date value - Second Last Date value
   - MTD Chg: Last Date value - MTD Reference Date value
   - YTD Chg: Last Date value - YTD Reference Date value
   - Custom Date Chg: Last Date value - Custom Date Reference Date value
   
   Only calculate if BOTH values exist and are not blank/NaN
   If either value is blank, set change to blank (pd.NA)
   ```

6. **Calculate Derived Columns**:
   ```python
   - Bid/Offer>3mm: Tight Bid >3mm - Wide Offer >3mm
   - Bid/Offer: Tight Bid - Wide Offer
   - # Quotes: Count unique Dealers per CUSIP on last date
   ```

7. **Merge External Data**:
   ```python
   - From historical_portfolio.parquet (last date):
     * QUANTITY (sum if multiple ACCOUNT/PORTFOLIO rows)
     * POSITION CR01 (sum if multiple rows)
   - From historical_bond_details.parquet (last date):
     * G Sprd, Yrs (Cvn), vs BI, vs BCE
     * MTD Equity, YTD Equity, Retracement
     * Yrs Since Issue, Z Score, Retracement2
     * Rating, Custom_Sector
   - Match by normalized CUSIP
   ```

8. **Export to CSV**:
   ```python
   - Sort by CUSIP
   - Write to runs_today.csv
   ```

**Key Business Logic**:

- **Size Threshold**: 3,000,000 (3mm) for ">3mm" metrics
- **Negative Spread Filtering**: Negative Bid Spread and Ask Spread values filtered out (set to NaN) before aggregation
- **Matching Logic**: Uses (CUSIP, Benchmark) tuple to ensure correct pairing across dates
- **Change Calculations**: Only calculated if both values exist (not blank/NaN)

### Runs Views (`analytics/runs/runs_views.py`)

**Purpose**: Create formatted tables from `runs_today.csv` for portfolio and universe monitoring

**Input**: `runs_today.csv`

**Output**: 
- `portfolio_runs_view.txt` / `portfolio_runs_view.xlsx` (16 portfolio tables)
- `uni_runs_view.txt` / `uni_runs_view.xlsx` (16 universe tables)

**Portfolio Tables** (16 tables):

1. **Portfolio Sorted By CR01 Risk** - All portfolio holdings sorted by POSITION CR01 descending
2. **Portfolio Sorted By CR01 Risk (DoD TB>3mm)** - Filtered to DoD TB>3mm > 0, sorted by POSITION CR01
3. **Portfolio Sorted By CR01 Risk (DoD WO>3mm)** - Filtered to DoD WO>3mm > 0, sorted by POSITION CR01
4. **Portfolio DoD Offer Chg** - Filtered to DoD WO>3mm > 0, sorted by DoD WO>3mm descending
5. **Portfolio MTD** - Excludes DoD columns, sorted by MTD Chg Tight Bid descending
6. **Portfolio YTD** - Excludes DoD and MTD columns, sorted by YTD Chg Tight Bid descending
7. **Portfolio Custom Date** - Excludes DoD, MTD, YTD columns, sorted by Custom Date Chg Tight Bid descending
8-16. Additional filtered/sorted variations

**Universe Tables** (16 tables):

1. **Universe Sorted By DoD Moves** - All rows sorted by DoD WO>3mm descending (50 rows in .txt, all rows in Excel)
2. **Universe Sorted By DoD Moves With Size On Offer >3mm** - Top 20 and bottom 20 by DoD WO>3mm
3. **Universe Sorted By DoD Moves (Wide Offer)** - Top 20 and bottom 20 by DoD WO
4. **Universe Sorted By MTD Moves** - Top 20 and bottom 20 by MTD Chg Tight Bid
5-16. Additional filtered/sorted variations

**Filters Applied**:
- Excluded Custom_Sector values: Asset Backed Subs, Auto ABS, Bail In, CAD Govt, CASH CAD, CASH USD, CDX, CP, Covered, Dep Note, Financial Hybrid, HY, Non Financial Hybrid, Non Financial Hybrids, USD Govt, University, Utility
- Various filters for DoD/MTD/YTD changes being non-zero
- Size filters (>3mm thresholds)

**Column Formatting**:
- Whole numbers: Thousand separators (e.g., 5,000,000)
- Yrs (Cvn): 1 decimal place (e.g., 3.4)
- Retracement: Percentage with 2 decimals and "%" suffix (e.g., 42.87%)
- MTD Equity, YTD Equity: 1 decimal place

### Pair Analytics (`analytics/comb/comb.py`)

**Purpose**: Compare bond spreads across different pairing criteria

**Input**: `bql.parquet`, `historical_bond_details.parquet`, `runs_today.csv`, `historical_portfolio.parquet`

**Output**: `comb.xlsx`, `comb.txt`, `comb_validation.txt`, `all_combinations.csv`

**Analysis Types** (11 analyses):

1. **All Combinations** - All CAD CUSIP pairs
2. **Term Combinations** - Pairs with Yrs (Cvn) difference <= 0.8
3. **Ticker Combinations** - Pairs with matching Ticker
4. **Custom Sector** - Pairs with matching Custom_Sector
5. **Custom Bond Combinations** - Target bond vs all CAD CUSIPs
6. **Custom Bond vs Holdings** - Target bond vs CR01 holdings
7. **CAD Cheap vs USD** - CAD/USD pairs (matching Ticker, Sector, Yrs Cvn diff <= 2.0)
8. **CAD Rich vs USD** - USD/CAD pairs (matching Ticker, Sector, Yrs Cvn diff <= 2.0)
9. **Executable CR01 vs Holdings** - CR01 universe vs CR01 holdings
10. **Executable CR01 Decent Bid Offer vs Holdings** - CR01 universe/holdings with Bid/Offer < 3
11. **All Combos vs Holdings** - All portfolio CUSIP pairs

**Pair Statistics Calculated**:

For each pair (Bond_1, Bond_2):
```python
- Spreads = Bond_1 Values - Bond_2 Values (aligned by Date)
- Last: Last spread value
- Avg: Average spread value
- vs Avg: Last - Avg (deviation from average)
- Z Score: vs Avg / Standard Deviation (if std > 0)
- Percentile: Rank of last value (0-100)
```

**Filters**:
- Recent date percent: 0.75 (75% of most recent dates)
- Currency filter: CAD (for most analyses)
- Max Yrs Cvn diff: 0.8 (term combinations)
- Max Yrs Cvn diff CAD/USD: 2.0
- CR01 thresholds: 2000 (tight bid), 2000 (wide offer)
- Bid/Offer threshold: 3 (for decent bid offer filter)

**Output Format**:
- Excel: One sheet per analysis type with formatted tables
- Text: Top 80 pairs per analysis
- CSV: All pairs from All Combinations analysis (sorted by Z Score)

---

## Business Logic Documentation

### CUSIP Normalization

**Purpose**: Standardize CUSIP identifiers for matching across datasets

**Process**:
```python
1. Convert to uppercase: "89678zab2" → "89678ZAB2"
2. Remove trailing "Corp" or " CORP": "06418GAD9 Corp" → "06418GAD9"
3. Remove whitespace: " 06418GAD9 " → "06418GAD9"
4. Take first 9 characters: "06418GAD9EXTRA" → "06418GAD9"
5. Validate length: Must be exactly 9 characters
```

**Applied In**:
- Bond Pipeline: All CUSIPs normalized
- Portfolio Pipeline: All CUSIPs normalized
- Runs Pipeline: CUSIPs kept as-is (not normalized), but normalized for matching with other datasets

**Validation**:
- Invalid CUSIPs logged but kept in data
- Orphan CUSIPs (not in universe.parquet) tracked and logged

### Deduplication Logic

#### Bond Pipeline Deduplication

**Primary Key**: `Date + CUSIP`

**Strategy**: Keep last occurrence per Date+CUSIP

**Applied**: Within each file and across files

**Logging**: All duplicates logged to `duplicates.log` with bond names

#### Runs Pipeline Deduplication

**Primary Key**: `Date + Dealer + CUSIP`

**Strategy**: End-of-day snapshot
1. Group by Date+Dealer+CUSIP
2. Sort by Time descending (latest first)
3. If tie on Time, keep last row by position
4. Keep first row per group (latest Time)

**Purpose**: Capture end-of-day quote snapshot per dealer per bond

**Logging**: Summary of duplicates logged (not row-by-row for performance)

#### Portfolio Pipeline Deduplication

**Primary Key**: `Date + CUSIP + ACCOUNT + PORTFOLIO`

**Strategy**: Keep last occurrence per Date+CUSIP+ACCOUNT+PORTFOLIO

**Purpose**: Preserve account/portfolio detail while removing duplicates

### Spread Calculations

#### Tight Bid >3mm

**Definition**: Smallest Bid Spread among quotes with Bid Size > 3,000,000

**Calculation**:
```python
1. Filter rows: Bid Size > 3,000,000
2. Filter out negative Bid Spreads (set to NaN)
3. Find minimum Bid Spread
4. Return spread value, dealer, and size
```

**Business Logic**: Represents best bid available with meaningful size (>3mm)

#### Wide Offer >3mm

**Definition**: Largest Ask Spread among quotes with Ask Size > 3,000,000

**Calculation**:
```python
1. Filter rows: Ask Size > 3,000,000
2. Filter out negative Ask Spreads (set to NaN)
3. Find maximum Ask Spread
4. Return spread value, dealer, and size
```

**Business Logic**: Represents worst offer available with meaningful size (>3mm)

#### Tight Bid / Wide Offer (Overall)

**Definition**: Smallest Bid Spread / Largest Ask Spread overall (any size)

**Calculation**:
```python
1. Filter out negative spreads (set to NaN)
2. Tight Bid: Minimum Bid Spread
3. Wide Offer: Maximum Ask Spread
```

**Business Logic**: Best bid/worst offer regardless of size

#### Bid/Offer Spread

**Definition**: Difference between Tight Bid and Wide Offer

**Calculation**:
```python
Bid/Offer>3mm = Tight Bid >3mm - Wide Offer >3mm
Bid/Offer = Tight Bid - Wide Offer

Only calculated if BOTH values exist (not NaN)
```

**Business Logic**: Represents bid-offer spread (negative = bid above offer, positive = normal spread)

### CR01 Calculations

**Definition**: Credit Risk 01 - Risk per $10,000 face value

**Formula**:
```python
CR01 @ Tight Bid = Bid Workout Risk * (Size @ Tight Bid >3mm / 10000)
CR01 @ Wide Offer = Bid Workout Risk * (Size @ Wide Offer >3mm / 10000)
```

**Bid Workout Risk**: Average Bid Workout Risk across all dealers for Date+CUSIP+Benchmark group

**Units**: CR01 per $10k face value

**Business Logic**: Represents credit risk exposure at executable bid/offer sizes

### Change Metrics (DoD, MTD, YTD, Custom Date)

#### Day-over-Day (DoD) Change

**Definition**: Change from second-to-last date to last date

**Calculation**:
```python
DoD Chg [Metric] = Last Date Value - Second Last Date Value

Only calculated if:
- Both values exist (not NaN/blank)
- Both values are not empty strings
- CUSIP+Benchmark pair exists on both dates
```

**Applied To**: All spread, size, and count metrics

**Business Logic**: Daily change in market conditions

#### Month-to-Date (MTD) Change

**Definition**: Change from first date of current month to last date

**Calculation**:
```python
MTD Chg [Metric] = Last Date Value - MTD Reference Date Value

MTD Reference Date = First date of current month (or closest available)
```

**Business Logic**: Monthly performance metric

#### Year-to-Date (YTD) Change

**Definition**: Change from first date of current year to last date

**Calculation**:
```python
YTD Chg [Metric] = Last Date Value - YTD Reference Date Value

YTD Reference Date = January 1st of current year (or closest available)
```

**Business Logic**: Yearly performance metric

#### Custom Date Change

**Definition**: Change from approximately 1 year ago to last date

**Calculation**:
```python
Custom Date Chg [Metric] = Last Date Value - Custom Date Reference Value

Custom Date Reference = Closest available date ~1 year ago
```

**Business Logic**: Year-over-year comparison

### Outlier Spread Filtering (Runs Pipeline)

**Purpose**: Remove data quality issues (outlier spreads)

**Method**:
```python
1. Group by Date+CUSIP
2. For each group:
   - Calculate high/low of Bid Spread and Ask Spread from valid positive values
   - Exclude current row from statistics (prevent self-skewing)
   - Drop rows where Bid Spread or Ask Spread outside [low - 20, high + 20] range
3. Exclude CUSIPs with Custom_Sector: "Non Financial Hybrid", "Non Financial Hybrids", "Financial Hybrid", "HY"
```

**Business Logic**: Filters both positive and negative outliers based on Date+CUSIP group statistics

**Applied**: After deduplication, before writing to parquet

### Pair Analytics Statistics

**Purpose**: Compare bond spreads to identify relative value opportunities

**Spread Calculation**:
```python
For each Date:
  Spread = Bond_1 Value - Bond_2 Value

Time series: [Spread_1, Spread_2, ..., Spread_N]
```

**Statistics**:
```python
Last = Spread_N (most recent spread)
Avg = Mean(Spreads)
vs Avg = Last - Avg (deviation from average)
Z Score = vs Avg / StdDev(Spreads) (if std > 0)
Percentile = Rank(Last) / N * 100 (0-100)
```

**Business Logic**:
- Positive Z Score: Spread wider than average (Bond_1 rich vs Bond_2)
- Negative Z Score: Spread tighter than average (Bond_1 cheap vs Bond_2)
- High Percentile: Spread near historical highs
- Low Percentile: Spread near historical lows

---

## Output Tables & Views

### runs_today.csv

**Source**: `analytics/runs/runs_today.py`

**Purpose**: Daily runs analytics with change metrics

**Columns** (100+ columns):

**Core Columns**:
- `CUSIP`, `Security`, `Benchmark`
- `Bid Workout Risk`
- `Tight Bid >3mm`, `Wide Offer >3mm`
- `Tight Bid`, `Wide Offer`
- `Bid/Offer>3mm`, `Bid/Offer`
- `Dealer @ Tight Bid >3mm`, `Dealer @ Wide Offer >3mm`
- `Dealer @ Tight T-1`, `Dealer @ Wide T-1` (previous day dealers)
- `Size @ Tight Bid >3mm`, `Size @ Wide Offer >3mm`
- `CR01 @ Tight Bid`, `CR01 @ Wide Offer`
- `Cumm. Bid Size`, `Cumm. Offer Size`
- `# of Bids >3mm`, `# of Offers >3mm`
- `# Quotes` (unique dealers per CUSIP)
- `Bid RBC`, `Ask RBC`, `Bid Size RBC`, `Offer Size RBC`

**Portfolio Columns** (merged):
- `QUANTITY`
- `POSITION CR01`

**Bond Details Columns** (merged):
- `G Sprd`, `Yrs (Cvn)`, `vs BI`, `vs BCE`
- `MTD Equity`, `YTD Equity`, `Retracement`
- `Yrs Since Issue`, `Z Score`, `Retracement2`
- `Rating`, `Custom_Sector`

**Change Columns** (for each metric):
- `DoD Chg [Metric]` (17 metrics)
- `MTD Chg [Metric]` (17 metrics)
- `YTD Chg [Metric]` (17 metrics)
- `Custom Date Chg [Metric]` (17 metrics)

**Row Count**: Varies by date (typically 500-1000 rows per date)

**Sorting**: By CUSIP ascending

### portfolio_runs_view.txt / .xlsx

**Source**: `analytics/runs/runs_views.py`

**Purpose**: Formatted portfolio monitoring tables

**Tables**: 16 tables with different filters and sorting

**Key Tables**:
1. Portfolio Sorted By CR01 Risk
2. Portfolio DoD Offer Chg (sorted by DoD WO>3mm)
3. Portfolio MTD (sorted by MTD Chg Tight Bid)
4. Portfolio YTD (sorted by YTD Chg Tight Bid)
5. Portfolio Custom Date (sorted by Custom Date Chg Tight Bid)
6-16. Additional filtered variations

**Filters**: Portfolio holdings only (QUANTITY > 0 or POSITION CR01 > 0)

**Formatting**: Thousand separators, decimal places, percentages

### uni_runs_view.txt / .xlsx

**Source**: `analytics/runs/runs_views.py`

**Purpose**: Formatted universe-wide analysis tables

**Tables**: 16 tables with different filters and sorting

**Key Tables**:
1. Universe Sorted By DoD Moves (all rows, sorted by DoD WO>3mm)
2. Universe Sorted By DoD Moves With Size On Offer >3mm (top/bottom 20)
3. Universe Sorted By DoD Moves (Wide Offer) (top/bottom 20)
4. Universe Sorted By MTD Moves (top/bottom 20)
5-16. Additional filtered variations

**Filters**: Excludes certain Custom_Sector values (govt, cash, hybrids, etc.)

**Formatting**: Thousand separators, decimal places, percentages

### comb.xlsx / .txt / .csv

**Source**: `analytics/comb/comb.py`

**Purpose**: Pair analytics results

**Excel**: 11 sheets (one per analysis type)

**Text**: Top 80 pairs per analysis

**CSV**: All pairs from All Combinations analysis

**Columns**:
- `Bond_1`, `Bond_2` (or `universe_name`, `holdings_name` for CR01 analyses)
- `Last` - Last spread value
- `Avg` - Average spread value
- `vs Avg` - Deviation from average
- `Z Score` - Standardized deviation
- `Percentile` - Rank percentile (0-100)
- `cusip_1`, `cusip_2` (or `universe_cusip`, `holdings_cusip`)
- `xxcy_diff` (for CAD/USD analyses) - CAD Equiv Swap difference

**Sorting**: By Z Score descending (highest first)

---

## Database Schema Design

### Proposed SQLite Database Schema

For potential migration to `.db` format, here is the proposed schema:

#### Table: `historical_bond_details`

```sql
CREATE TABLE historical_bond_details (
    date DATE NOT NULL,
    cusip VARCHAR(9) NOT NULL,
    security TEXT,
    benchmark TEXT,
    g_sprd REAL,
    yrs_cvn REAL,
    vs_bi REAL,
    vs_bce REAL,
    mtd_equity REAL,
    ytd_equity REAL,
    retracement REAL,
    yrs_since_issue REAL,
    z_score REAL,
    retracement2 REAL,
    rating TEXT,
    custom_sector TEXT,
    currency VARCHAR(3),
    ticker TEXT,
    -- Plus 60+ other columns
    PRIMARY KEY (date, cusip)
);

CREATE INDEX idx_bond_details_date ON historical_bond_details(date);
CREATE INDEX idx_bond_details_cusip ON historical_bond_details(cusip);
CREATE INDEX idx_bond_details_ticker ON historical_bond_details(ticker);
CREATE INDEX idx_bond_details_sector ON historical_bond_details(custom_sector);
```

#### Table: `universe`

```sql
CREATE TABLE universe (
    cusip VARCHAR(9) PRIMARY KEY,
    benchmark_cusip VARCHAR(9),
    custom_sector TEXT,
    bloomberg_cusip VARCHAR(9),
    security TEXT,
    benchmark TEXT,
    pricing_date DATE,
    pricing_date_bench DATE,
    worst_date DATE,
    yrs_worst REAL,
    ticker TEXT,
    currency VARCHAR(3),
    equity_ticker TEXT
);

CREATE INDEX idx_universe_ticker ON universe(ticker);
CREATE INDEX idx_universe_sector ON universe(custom_sector);
```

#### Table: `bql`

```sql
CREATE TABLE bql (
    date DATE NOT NULL,
    name TEXT,
    cusip VARCHAR(9) NOT NULL,
    value REAL,
    PRIMARY KEY (date, cusip)
);

CREATE INDEX idx_bql_date ON bql(date);
CREATE INDEX idx_bql_cusip ON bql(cusip);
```

#### Table: `runs_timeseries`

```sql
CREATE TABLE runs_timeseries (
    date DATE NOT NULL,
    time TIME NOT NULL,
    dealer VARCHAR(10) NOT NULL,
    cusip VARCHAR(9) NOT NULL,
    security TEXT,
    benchmark TEXT,
    bid_spread REAL,
    ask_spread REAL,
    bid_size REAL,
    ask_size REAL,
    bid_workout_risk REAL,
    bid_price REAL,
    ask_price REAL,
    ticker TEXT,
    -- Plus 18 other columns
    PRIMARY KEY (date, dealer, cusip)
);

CREATE INDEX idx_runs_date ON runs_timeseries(date);
CREATE INDEX idx_runs_cusip ON runs_timeseries(cusip);
CREATE INDEX idx_runs_dealer ON runs_timeseries(dealer);
CREATE INDEX idx_runs_date_cusip ON runs_timeseries(date, cusip);
```

#### Table: `historical_portfolio`

```sql
CREATE TABLE historical_portfolio (
    date DATE NOT NULL,
    cusip VARCHAR(9) NOT NULL,
    account TEXT NOT NULL,
    portfolio TEXT NOT NULL,
    security TEXT,
    quantity REAL,
    position_cr01 REAL,
    -- Plus 75 other columns
    PRIMARY KEY (date, cusip, account, portfolio)
);

CREATE INDEX idx_portfolio_date ON historical_portfolio(date);
CREATE INDEX idx_portfolio_cusip ON historical_portfolio(cusip);
CREATE INDEX idx_portfolio_account ON historical_portfolio(account);
```

#### Table: `runs_today` (Materialized View / Table)

```sql
CREATE TABLE runs_today (
    cusip VARCHAR(9),
    security TEXT,
    benchmark TEXT,
    bid_workout_risk REAL,
    tight_bid_3mm REAL,
    wide_offer_3mm REAL,
    tight_bid REAL,
    wide_offer REAL,
    bid_offer_3mm REAL,
    bid_offer REAL,
    dealer_tight_bid_3mm VARCHAR(10),
    dealer_wide_offer_3mm VARCHAR(10),
    size_tight_bid_3mm REAL,
    size_wide_offer_3mm REAL,
    cr01_tight_bid REAL,
    cr01_wide_offer REAL,
    cumm_bid_size REAL,
    cumm_offer_size REAL,
    num_bids_3mm INTEGER,
    num_offers_3mm INTEGER,
    num_quotes INTEGER,
    bid_rbc REAL,
    ask_rbc REAL,
    bid_size_rbc REAL,
    offer_size_rbc REAL,
    quantity REAL,
    position_cr01 REAL,
    g_sprd REAL,
    yrs_cvn REAL,
    vs_bi REAL,
    vs_bce REAL,
    mtd_equity REAL,
    ytd_equity REAL,
    retracement REAL,
    yrs_since_issue REAL,
    z_score REAL,
    retracement2 REAL,
    rating TEXT,
    custom_sector TEXT,
    -- Plus 68 DoD/MTD/YTD/Custom Date change columns
    PRIMARY KEY (cusip, benchmark)
);

CREATE INDEX idx_runs_today_cusip ON runs_today(cusip);
CREATE INDEX idx_runs_today_sector ON runs_today(custom_sector);
```

### Views for Common Queries

```sql
-- Latest bond details per CUSIP
CREATE VIEW latest_bond_details AS
SELECT DISTINCT ON (cusip) *
FROM historical_bond_details
ORDER BY cusip, date DESC;

-- Runs today aggregated view
CREATE VIEW runs_today_agg AS
SELECT 
    cusip,
    benchmark,
    MIN(bid_spread) FILTER (WHERE bid_size > 3000000) AS tight_bid_3mm,
    MAX(ask_spread) FILTER (WHERE ask_size > 3000000) AS wide_offer_3mm,
    COUNT(DISTINCT dealer) FILTER (WHERE bid_size > 3000000) AS num_bids_3mm,
    COUNT(DISTINCT dealer) FILTER (WHERE ask_size > 3000000) AS num_offers_3mm
FROM runs_timeseries
WHERE date = (SELECT MAX(date) FROM runs_timeseries)
GROUP BY cusip, benchmark;
```

---

## Developer Guide

### Running the Pipelines

#### Unified Pipeline Orchestrator (Recommended)

```bash
poetry run python run_pipeline.py
```

**Options**:
1. Bond Pipeline only
2. Runs Pipeline only
3. Portfolio Pipeline only
4. Both Bond and Runs Pipelines (default)
5. Individual Parquet Files
6. All Pipelines (Bond, Runs, Portfolio)

**Modes**:
- Append: Add only new dates (default, daily use)
- Override: Rebuild everything from scratch

#### Individual Pipelines

```bash
# Bond Pipeline
poetry run python -m bond_pipeline.pipeline -i "Raw Data/" -m append --process-bql

# Runs Pipeline
poetry run python -m runs_pipeline.pipeline -i "Historical Runs/" -m append

# Portfolio Pipeline
poetry run python -m portfolio_pipeline.pipeline -i "AD History/" -m append
```

### Running Analytics

```bash
# Runs Today Analytics
poetry run python analytics/runs/runs_today.py

# Runs Views (Portfolio and Universe)
poetry run python analytics/runs/runs_views.py

# Pair Analytics
poetry run python analytics/comb/comb.py
```

### File Paths

**Input Directories** (configured in `bond_pipeline/config.py`):
- Bond: `C:\Users\Eddy\...\API Historical\`
- Runs: `C:\Users\Eddy\...\Historical Runs\`
- Portfolio: `C:\Users\Eddy\...\AD History\`
- BQL: `C:\Users\Eddy\...\bql.xlsx`

**Output Directories**:
- Parquet: `bond_data/parquet/`
- Logs: `bond_data/logs/`
- Analytics: `analytics/processed_data/`

### Logging

**Log Files**:
- `processing.log` - File extraction and loading operations
- `duplicates.log` - All duplicate CUSIPs detected
- `validation.log` - CUSIP validation warnings
- `summary.log` - Pipeline execution summary
- `parquet_stats.log` - Dataset diagnostics (df.info/head/tail)
- `runs_today.log` - Runs today analytics execution log

**Log Levels**:
- DEBUG: Detailed diagnostic info
- INFO: General informational messages
- WARNING: Validation issues, schema mismatches
- ERROR: File read failures, data corruption
- CRITICAL: System failures

### Testing

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/unit/test_utils.py

# Run with coverage
poetry run pytest --cov=bond_pipeline --cov-report=html
```

### Common Tasks

#### Adding a New Column

1. **Bond Pipeline**:
   - Column will be detected automatically from latest Excel file
   - Add to master schema in `transform.py` if needed
   - Update `universe.parquet` columns if needed (13 columns only)

2. **Runs Pipeline**:
   - Add to `RUNS_COLUMNS` in `config.py`
   - Update schema alignment in `transform.py`

3. **Portfolio Pipeline**:
   - Column will be detected automatically from latest Excel file
   - Add to master schema in `transform.py` if needed

#### Modifying Business Logic

1. **Spread Calculations**: Modify `compute_group_metrics()` in `runs_today.py`
2. **Change Metrics**: Modify DoD/MTD/YTD calculation loops in `runs_today.py`
3. **Filters**: Update filter logic in `runs_views.py` or `comb.py`
4. **Deduplication**: Modify deduplication methods in transform modules

#### Debugging

1. **Check Logs**: Review `bond_data/logs/` for detailed execution logs
2. **Parquet Stats**: Check `parquet_stats.log` for dataset diagnostics
3. **CSV Output**: Inspect `runs_today.csv` for intermediate results
4. **Python Debugger**: Add breakpoints in pipeline code

### Performance Optimization

**Current Optimizations**:
- Vectorized operations (pandas operations instead of loops)
- Date filtering before aggregation (only process required dates)
- Parquet columnar storage (efficient compression)
- Incremental loading (append mode)

**Future Optimizations**:
- Parallel file processing
- Caching of reference data
- Database indexing (if migrated to .db)
- Incremental view updates

---

## Appendix

### Column Reference

See individual pipeline documentation for complete column lists:
- Bond Pipeline: 75 columns (see `bond_pipeline/config.py`)
- Runs Pipeline: 30 columns (see `bond_pipeline/config.py` - `RUNS_COLUMNS`)
- Portfolio Pipeline: 82 columns (detected dynamically)

### Configuration Reference

See `bond_pipeline/config.py` for all configuration constants:
- File patterns
- Header rows
- Column lists
- Paths
- Thresholds

### Error Handling

**Common Errors**:
- Missing files: Check input directory paths
- Schema mismatches: Check master schema alignment
- Date parsing errors: Verify filename patterns
- CUSIP validation: Check normalization logic
- Memory errors: Process files in batches

**Recovery**:
- Use override mode to rebuild corrupted parquet files
- Check logs for specific error messages
- Validate input Excel files manually

---

**End of Documentation**

