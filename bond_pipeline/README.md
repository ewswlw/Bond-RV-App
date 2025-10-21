# Bond Data Pipeline

A modular data engineering pipeline for processing bond data from Excel files into Parquet tables for a relative value bond trading application.

## Overview

This pipeline processes Excel files containing bond data and creates two parquet tables:

1. **`historical_bond_details.parquet`**: Time series data with unique `Date + CUSIP` combinations
2. **`universe.parquet`**: Current universe of all unique CUSIPs ever seen (13 key columns)

## Features

- ✅ **Modular Architecture**: Separate modules for extraction, transformation, and loading
- ✅ **Incremental Loading**: Append mode only processes new dates
- ✅ **Data Validation**: CUSIP normalization and validation
- ✅ **Deduplication**: Handles duplicate CUSIPs within same date
- ✅ **Schema Evolution**: Handles files with different column counts
- ✅ **Comprehensive Logging**: Tracks processing, duplicates, and validation issues

## Installation

### Requirements
- Python 3.11+
- pandas
- pyarrow
- openpyxl

### Install Dependencies
```bash
pip install pandas pyarrow openpyxl
```

## Usage

### Basic Usage

```bash
# Append new data (default mode)
python pipeline.py --input "Universe Historical/" --mode append

# Override all data (rebuild from scratch)
python pipeline.py --input "Universe Historical/" --mode override
```

### Short Form
```bash
python pipeline.py -i "Universe Historical/" -m append
python pipeline.py -i "Universe Historical/" -m override
```

### Modes

#### Append Mode (Default)
- Checks existing parquet for dates already processed
- Processes only files with new dates
- Appends to existing `historical_bond_details.parquet`
- Rebuilds `universe.parquet` from complete historical data

#### Override Mode
- Deletes existing parquet files
- Processes all Excel files from scratch
- Creates new `historical_bond_details.parquet`
- Creates new `universe.parquet`

## File Structure

```
bond_pipeline/
├── config.py              # Configuration and constants
├── utils.py               # Utility functions
├── extract.py             # Excel file reading
├── transform.py           # Data cleaning and transformation
├── load.py                # Parquet writing
├── pipeline.py            # Main orchestration script
└── README.md              # This file

bond_data/
├── parquet/
│   ├── historical_bond_details.parquet
│   └── universe.parquet
└── logs/
    ├── processing.log
    ├── duplicates.log
    ├── validation.log
    └── summary.log
```

## Data Processing Rules

### Date Extraction
- Extracts date from filename pattern: `API MM.DD.YY.xlsx`
- Example: `API 10.20.25.xlsx` → `2025-10-20`
- Stored as datetime64 in parquet

### CUSIP Handling
- **Normalization**: Converts to uppercase (e.g., `89678zab2` → `89678ZAB2`)
- **Validation**: Checks for 9-character length
- **Invalid CUSIPs**: Logged but included in data with validation flag

### NA/Null Handling
Converts these strings to NULL/NaN:
- `#N/A Field Not Applicable`
- `#N/A Invalid Security`
- `#N/A`
- `N/A`
- Empty strings

### Deduplication
- **Within File**: Keeps last occurrence of duplicate CUSIPs
- **Historical Table**: Ensures unique `Date + CUSIP` combinations
- **Universe Table**: Keeps most recent date's data for each CUSIP

### Schema Evolution
- Uses latest file schema (75 columns) as master
- Older files with fewer columns are filled with NAs
- All files aligned to master schema

## Output Tables

### Historical Bond Details
- **Primary Key**: `Date + CUSIP`
- **Columns**: All 75 columns from source files
- **First Column**: `Date` (extracted from filename)
- **Format**: Parquet

### Universe
- **Primary Key**: `CUSIP`
- **Columns**: 13 key columns
  1. CUSIP
  2. Benchmark Cusip
  3. Custom_Sector
  4. Bloomberg Cusip
  5. Security
  6. Benchmark
  7. Pricing Date
  8. Pricing Date (Bench)
  9. Worst Date
  10. Yrs (Worst)
  11. Ticker
  12. Currency
  13. Equity Ticker
- **Format**: Parquet

## Logging

The pipeline generates four log files:

1. **`processing.log`**: File extraction and loading operations
2. **`duplicates.log`**: Duplicate CUSIP detection and removal
3. **`validation.log`**: CUSIP validation and data quality issues
4. **`summary.log`**: Pipeline execution summary and statistics

## Examples

### First Time Setup (Override Mode)
```bash
python pipeline.py -i "/path/to/Universe Historical/" -m override
```

### Daily Updates (Append Mode)
```bash
# Add new file to input directory
# Run pipeline in append mode
python pipeline.py -i "/path/to/Universe Historical/" -m append
```

### Check Results
```python
import pandas as pd

# Read historical data
df_hist = pd.read_parquet('bond_data/parquet/historical_bond_details.parquet')
print(f"Historical: {len(df_hist)} rows, {df_hist['Date'].nunique()} dates")

# Read universe
df_univ = pd.read_parquet('bond_data/parquet/universe.parquet')
print(f"Universe: {len(df_univ)} unique CUSIPs")
```

## Error Handling

- **File Read Errors**: Logged and skipped, pipeline continues
- **Invalid Dates**: Logged and file skipped
- **Invalid CUSIPs**: Logged but included with validation flag
- **Duplicate Date+CUSIP**: Prevented, pipeline fails if detected in output

## Performance

- Processes ~2,600 bonds per file
- Typical runtime: 10-30 seconds for 11 files
- Memory usage: ~100-200 MB for full dataset

## Troubleshooting

### No files found
- Check input directory path
- Ensure files match pattern `*.xlsx`

### Duplicate Date+CUSIP errors
- Check source files for data quality issues
- Review duplicates.log for details

### Missing columns
- Older files may have fewer columns (filled with NAs)
- Check validation.log for details

## Future Enhancements

Potential improvements:
- Partitioning by year for large datasets
- Incremental universe updates (avoid full rebuild)
- Data quality dashboard
- Automated testing suite
- Integration with data sources (APIs, databases)

## Contact

For questions or issues, refer to the documentation in `bond_pipeline_documentation.md`.

