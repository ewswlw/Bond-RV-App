# Runs Data Pipeline - Documentation

**Project**: Relative Value Bond App - Historical Runs Processing  
**Date Started**: January 2025  
**Last Updated**: January 2, 2025  
**Approach**: Modular data engineering pipeline for dealer quote data (runs)

---

## Overview

The Runs Pipeline processes Historical Runs Excel files into `runs_timeseries.parquet`. This pipeline handles bond pricing quotes from dealers, with the key requirement that for every Date, Dealer, CUSIP there should be only one entry (end-of-day snapshot with latest Time).

### Purpose

- Process RUNS Excel files (dealer quote data) into optimized Parquet time-series table
- Deduplicate to end-of-day snapshots (latest Time per Date+Dealer+CUSIP)
- Track orphan CUSIPs (CUSIPs in runs but not in universe.parquet)
- Validate data quality and log issues
- Support incremental data loading (append mode) and full rebuilds (override mode)

### Architecture

- **Modular ETL**: Separate modules for Extract → Transform → Load
- **Data Pipeline**: Excel files → Parquet table (runs_timeseries.parquet)
- **Incremental Loading**: Append mode (new dates only) vs Override mode (rebuild all)
- **Data Integrity**: Primary key enforcement (Date+Dealer+CUSIP), deduplication, validation

### Integration with Bond Pipeline

- **Shared Utilities**: Reuses logging, validation, and formatting utilities from `bond_pipeline.utils`
- **Combined Logs**: Both pipelines write to same log files in `bond_data/logs/`
- **CUSIP Orphan Tracking**: Compares runs CUSIPs with `universe.parquet` from bond_pipeline
- **Configuration**: Runs-specific config added to `bond_pipeline.config`

---

## Data Discovery & Analysis

### File Structure

- **Location**: `Historical Runs/` folder
- **File Pattern**: `RUNS MM.DD.YY.xlsx` (e.g., `RUNS 10.31.25.xlsx`)
- **Date Range**: December 2022 to October 2025 (15 files analyzed)
- **File Format**: Excel (.xlsx) with header on row 1 (0-indexed)

### Data Characteristics

#### File Statistics

Based on analysis of 15 RUNS files:
- **Total Rows**: 182,231 rows across all files
- **Average Rows per File**: ~12,149 rows
- **Unique CUSIPs**: 1,702 unique CUSIPs across all files
- **Unique Dealers**: 11 dealers (RBC, TD, CIBC, BMO, NBF, SCM, MS, YTMC, BBLP, IAS, TDS)
- **Duplicate Groups**: 53,048 groups with duplicate Date+Dealer+CUSIP combinations
- **Files with Duplicates**: 14 out of 15 files (only RUNS 12.31.22.xlsx has zero duplicates)

#### Column Structure (30 columns)

1. Reference Security
2. **Date** ⭐ (MM/DD/YY format, parsed to datetime)
3. **Time** ⭐ (HH:MM format, parsed to datetime.time)
4. Bid Workout Risk
5. Ticker
6. **Dealer** ⭐
7. Source
8. Security
9. Bid Price
10. Ask Price
11. Bid Spread
12. Ask Spread
13. Benchmark
14. Reference Benchmark
15. Bid Size
16. Ask Size
17. Sector
18. Bid Yield To Convention
19. Ask Yield To Convention
20. Bid Discount Margin
21. Ask Discount Margin
22. **CUSIP** ⭐
23. Sender Name
24. Currency
25. Subject
26. Keyword
27. Bid Interpolated Spread to Government
28. Ask Interpolated Spread to Government
29. Bid Contributed Yield
30. Bid Z-spread

**Note**: One old file (RUNS 12.31.22.xlsx) has 28 columns vs 30 in others. The pipeline aligns all files to the 30-column master schema, filling missing columns with NaN.

### Duplicate Pattern Analysis

**Finding**: 95% of duplicate groups have different times AND different data values (intraday price updates).

#### Pattern 1: Same Time, Different Data (True Duplicates)
- **Count**: 2,420 groups (5% of duplicates)
- **Pattern**: Same Date+Dealer+CUSIP+Time but different pricing data
- **Handling**: Keep last row by position (tiebreaker)

#### Pattern 2: Different Times, Identical Data
- **Count**: 3,729 groups (7% of duplicates)
- **Pattern**: Same Date+Dealer+CUSIP with different times but identical pricing
- **Handling**: Keep row with latest Time (end-of-day snapshot)

#### Pattern 3: Different Times, Different Data (Price Updates) ⭐ MOST COMMON
- **Count**: 49,319 groups (93% of duplicates)
- **Pattern**: Same Date+Dealer+CUSIP with different times AND different pricing (intraday updates)
- **Handling**: Keep row with latest Time (end-of-day snapshot)

---

## Output Data Format

### `runs_timeseries.parquet`

- **Purpose**: Time series of all dealer quotes over time
- **Primary Key**: `Date + Dealer + CUSIP` (unique combination, enforced after deduplication)
- **Schema**: All 30 columns with Date and Time as first columns
- **Row count**: ~130,000+ rows (after deduplication) spanning 2022-2025
- **Storage**: `bond_data/parquet/runs_timeseries.parquet`

### Column Order

1. **Date** (datetime object, mm/dd/yyyy format)
2. **Time** (datetime.time object, hh:mm format)
3. All other columns in master schema order

---

## Processing Rules

### Date/Time Handling

- **Date Column**: Parse from MM/DD/YY string (e.g., "10/31/25") → datetime object (preserves mm/dd/yyyy format)
- **Time Column**: Parse from HH:MM string (e.g., "15:45") → datetime.time object
- **Storage**: Both stored as datetime objects in parquet (Date as datetime64, Time as object containing time objects)
- **Column Position**: Date and Time are always first columns in final parquet file

### End-of-Day Deduplication

- **Primary Key**: Date + Dealer + CUSIP
- **Deduplication Logic**: Keep row with latest Time per Date+Dealer+CUSIP group
- **Same-Time Tiebreaker**: If multiple rows have same latest Time, keep last row by DataFrame index position
- **Logging**: All removed duplicates logged to `bond_data/logs/duplicates.log` with full details

### CUSIP Validation & Orphan Tracking

- **Validation**: Check CUSIP length (should be 9, but don't normalize - keep as-is)
- **Invalid CUSIPs**: Log warnings to `bond_data/logs/validation.log` but keep in data
- **Orphan Tracking**: Compare CUSIPs from runs data with `universe.parquet` from bond_pipeline
- **Orphan Logging**: Log all orphan CUSIPs (in runs but not in universe) to `validation.log`
- **No Normalization**: CUSIPs kept as-is from Excel (no uppercase conversion, no text removal)

### Schema Alignment

- **Master Schema**: 30 columns from latest files (defined in `bond_pipeline.config.RUNS_COLUMNS`)
- **Older Files**: Fill missing columns (e.g., old 28-column file) with NaN
- **Column Order**: Preserve order from master schema, with Date and Time first

### Data Cleaning

- **NA Values**: Convert NA strings to NULL: `#N/A Field Not Applicable`, `#N/A Invalid Security`, `N/A`, etc.
- **Empty Cells**: Convert to NULL/NaN
- **Preserve Other Values**: All other values kept as-is

### Data Quality Validation

- **Date Range**: Check dates are within reasonable range (1900-2100)
- **Time Format**: Validate time is valid HH:MM format
- **Prices**: Check prices are positive (if not NaN)
- **Spreads**: Check spreads are reasonable (< 1000 bps if not NaN)
- **Dealers**: Check dealers are in known list
- **Logging**: All validation issues logged to `bond_data/logs/validation.log`

---

## Module Reference

### `runs_pipeline.extract.RunsExtractor`

Extract data from RUNS Excel files.

#### Methods

- `__init__(log_file: Path)`: Initialize with logging
- `read_excel_file(file_path: Path) -> Optional[pd.DataFrame]`: 
  - Read Excel with header=0 (row 1 in Excel)
  - Parse Date column: MM/DD/YY string → datetime object
  - Parse Time column: HH:MM string → datetime.time object
  - Reorder columns: Date, Time first (preserve other column order)
  - Return DataFrame with parsed Date/Time or None if failed
- `extract_all_files(file_paths: List[Path]) -> List[pd.DataFrame]`: 
  - Extract from all files
  - Return list of DataFrames (not dict like bond_pipeline since no date from filename)

---

### `runs_pipeline.transform.RunsTransformer`

Transform and clean RUNS data with specialized deduplication logic.

#### Methods

- `__init__(log_file_dupes: Path, log_file_valid: Path)`: Initialize with duplicate and validation loggers
- `deduplicate_end_of_day(df: pd.DataFrame) -> pd.DataFrame`:
  - **Optimized**: Uses vectorized pandas operations (sort_values + drop_duplicates)
  - Group by Date+Dealer+CUSIP
  - Sort by Date+Dealer+CUSIP (ascending), then Time descending (latest first), then index descending
  - Keep first row per group using drop_duplicates (latest time, tiebreaker: last row position)
  - Log summary of duplicates (first 10 detailed, then summary) to duplicates.log
  - **Performance**: ~100x faster than previous row-by-row iteration (O(n log n) vs O(n*groups))
  - Return deduplicated DataFrame
- `validate_cusips(df: pd.DataFrame, universe_parquet: Optional[Path] = None) -> pd.DataFrame`:
  - **Optimized**: Uses vectorized pandas operations for validation (no row-by-row iteration)
  - Check CUSIP length (should be 9, but don't normalize) using vectorized `.str.len()`
  - Log invalid CUSIPs summary (counts of empty vs wrong length) to validation.log
  - Track orphan CUSIPs vs universe.parquet using `check_cusip_orphans` utility
  - **Enhanced Orphan Logging**: Orphan CUSIPs logged with context:
    - Security (bond name)
    - Date (formatted MM/DD/YYYY)
    - Dealer
    - Time (HH:MM format)
    - Ticker (if available)
  - Log orphan CUSIPs with full context to validation.log
  - **Performance**: ~100x faster than previous row-by-row iteration (vectorized operations)
  - Keep invalid CUSIPs in data (just log warnings)
- `align_to_master_schema(df: pd.DataFrame, master_schema: Optional[list] = None) -> pd.DataFrame`:
  - Align to 30-column master schema
  - Fill missing columns (e.g., old 28-column file) with NaN
  - Reorder to match master schema
- `clean_data(df: pd.DataFrame) -> pd.DataFrame`: Use `clean_na_values` from bond_pipeline.utils
- `transform(df: pd.DataFrame, universe_parquet: Optional[Path] = None) -> pd.DataFrame`:
  - Apply all transformations in order:
    1. Deduplicate (end-of-day snapshots)
    2. Clean NA values
    3. Validate CUSIPs and track orphans
    4. Align to master schema

---

### `runs_pipeline.load.RunsLoader`

Load RUNS data to runs_timeseries.parquet with primary key enforcement.

#### Methods

- `__init__(log_file: Path)`: Initialize with logging
- `get_existing_dates() -> Set[datetime]`: 
  - Read Date column from runs_timeseries.parquet
  - Return set of unique dates (for append mode filtering)
- `validate_primary_key(df: pd.DataFrame) -> bool`:
  - Check Date+Dealer+CUSIP uniqueness
  - Log errors if duplicates found
  - Return True if valid, False if duplicates found
- `load_append(data: pd.DataFrame) -> bool`:
  - Get existing dates
  - Filter out rows with dates already in parquet
  - Validate primary key on filtered data
  - Append to existing parquet file
  - Log summary statistics
- `load_override(data: pd.DataFrame) -> bool`:
  - Delete existing runs_timeseries.parquet if exists
  - Validate primary key
  - Write all data to new parquet file
  - Log summary statistics
- `get_summary_stats() -> dict`: 
  - Return statistics:
    - runs_rows: Total rows
    - runs_dates: Unique dates
    - runs_cusips: Unique CUSIPs
    - runs_dealers: Unique dealers
    - date_range: (min_date, max_date)

---

### `runs_pipeline.pipeline.RunsDataPipeline`

Main orchestrator for RUNS data pipeline.

#### Methods

- `__init__(input_dir: Path, mode: str = 'append')`:
  - Initialize with run_id, metadata, log rotation
  - Initialize RunsExtractor, RunsTransformer, RunsLoader
  - Set up file logger and console logger
  - Log run header to summary.log
- `run() -> bool`: Execute complete pipeline:
  1. Get file list from input_dir matching RUNS pattern
  2. Sort files chronologically by date in filename (earliest first)
  3. Extract all files (combine into single DataFrame)
  4. Transform: deduplicate end-of-day, validate CUSIPs/orphans, align schema, clean data, validate data quality
  5. Sort combined DataFrame by Date ascending (earliest to latest)
  6. Load to parquet (append or override mode based on self.mode)
  7. Generate and log summary statistics
  8. Return True if successful

---

## Configuration

All RUNS pipeline configuration is in `bond_pipeline.config`:

- `RUNS_INPUT_DIR`: Path to Historical Runs folder
- `RUNS_FILE_PATTERN`: Regex pattern for RUNS files
- `RUNS_HEADER_ROW = 0`: Header row (0-indexed)
- `RUNS_PARQUET`: Path to runs_timeseries.parquet output
- `RUNS_PRIMARY_KEY = ['Date', 'Dealer', 'CUSIP']`: Primary key columns
- `RUNS_DATE_FORMAT = '%m/%d/%Y'`: Date format string
- `RUNS_TIME_FORMAT = '%H:%M'`: Time format string
- `RUNS_COLUMNS`: List of 30 column names (master schema)
- `RUNS_VALIDATE_*`: Boolean flags for validation settings
- `RUNS_KNOWN_DEALERS`: List of known dealer names
- `RUNS_TRACK_ORPHAN_CUSIPS = True`: Enable orphan tracking
- `RUNS_LOG_INVALID_CUSIPS = True`: Enable invalid CUSIP logging

---

## Usage Examples

### CLI Usage

```bash
# Activate virtual environment
Bond-RV-App\Scripts\activate  # Windows

# Append new data (default mode)
python -m runs_pipeline.pipeline -i "Historical Runs/" -m append

# Override all data (rebuild everything)
python -m runs_pipeline.pipeline -i "Historical Runs/" -m override

# Use default input directory
python -m runs_pipeline.pipeline -m override

# Short form
python -m runs_pipeline.pipeline -i "Historical Runs/" -m append
```

### Python API Usage

```python
from pathlib import Path
from runs_pipeline.pipeline import RunsDataPipeline

# Create pipeline instance
pipeline = RunsDataPipeline(
    input_dir=Path("Historical Runs/"),
    mode='append'  # or 'override'
)

# Run pipeline
success = pipeline.run()

if success:
    print("Pipeline completed successfully!")
else:
    print("Pipeline failed. Check logs for details.")
```

### Module-Level Usage

```python
from runs_pipeline.extract import RunsExtractor
from runs_pipeline.transform import RunsTransformer
from runs_pipeline.load import RunsLoader
from pathlib import Path

# Extract
extractor = RunsExtractor(LOG_FILE_PROCESSING)
dataframes = extractor.extract_all_files(file_paths)

# Transform
transformer = RunsTransformer(LOG_FILE_DUPLICATES, LOG_FILE_VALIDATION)
combined_df = pd.concat(dataframes, ignore_index=True)
transformed_df = transformer.transform(combined_df)

# Load
loader = RunsLoader(LOG_FILE_PROCESSING)
success = loader.load_append(transformed_df)  # or load_override()
```

---

## Pipeline Modes

### Append Mode (Default, Daily Use)

- **Use case**: Add new RUNS Excel files to existing parquet table
- **Behavior**: Skip dates already in parquet, append only new dates
- **Output**: Updated `runs_timeseries.parquet` (with new dates appended)
- **Performance**: Faster (only processes new dates)

### Override Mode (Rebuild Everything)

- **Use case**: First-time setup, schema changes, data corruption recovery
- **Behavior**: Delete existing parquet file, process all Excel files from scratch
- **Output**: New `runs_timeseries.parquet`
- **Performance**: Slower but ensures clean data

---

## Integration Points

### Shared Utilities

The runs_pipeline reuses utilities from `bond_pipeline.utils`:

- `setup_logging()`: Logging setup with file and console handlers
- `setup_console_logger()`: Console-only logger for user messages
- `format_date_string()`: Format datetime objects for display/logging
- `format_run_header()`: Format pipeline run headers
- `get_run_id()`: Get sequential run ID
- `save_run_metadata()`: Save run metadata for tracking
- `check_and_rotate_logs()`: Log rotation management
- `clean_na_values()`: Clean NA values in DataFrames
- `align_to_master_schema()`: Align DataFrames to master schema
- `check_cusip_orphans()`: Compare CUSIPs with universe.parquet
- `validate_runs_data()`: Validate data quality

### Combined Logging

Both pipelines write to the same log files in `bond_data/logs/`:

- `processing.log`: File extraction and loading operations (combined)
- `duplicates.log`: All duplicate Date+Dealer+CUSIP detected (runs) and Date+CUSIP (bonds)
- `validation.log`: CUSIP validation warnings and orphan tracking (combined)
- `summary.log`: Pipeline execution summary (combined, with run headers)

### CUSIP Orphan Tracking

- `RunsTransformer.validate_cusips()` reads `universe.parquet` to check for orphan CUSIPs
- Orphan CUSIPs (in runs but not in universe) are logged to `validation.log` with detailed context
- **Enhanced Logging**: Each orphan CUSIP includes:
  - Security (bond name) for quick identification
  - Date (when the quote was made)
  - Dealer (which dealer provided the quote)
  - Time (quote timestamp)
  - Ticker (if available)
- **Example Log Format**: `Orphan CUSIP: 06368MYR9 (Security=Some Bond Name, Date=10/31/2025, Dealer=RBC, Time=15:45, Ticker=ABC)`
- This helps identify data quality issues and missing bonds quickly

---

## Logging Standards

### Log File Organization

```
bond_data/logs/
├── processing.log      # File extraction and loading operations (combined)
├── duplicates.log      # All duplicate Date+Dealer+CUSIP detected (runs)
├── validation.log      # CUSIP validation warnings and orphan tracking (combined)
└── summary.log         # Pipeline execution summary (run headers, stats)
```

### Logging Levels

- **DEBUG**: Detailed diagnostic info (file-by-file progress)
- **INFO**: General informational messages
- **WARNING**: CUSIP validation issues, schema mismatches, orphan CUSIPs
- **ERROR**: File read failures, primary key violations, data corruption
- **CRITICAL**: System failures

### Logging Best Practices

1. **Dual logging**: File logging (detailed) + Console logging (essential only)
2. **No console spam**: Suppress detailed logs on console (console_level=logging.CRITICAL)
3. **Run metadata**: Track run_id, timestamp, mode for each execution
4. **Log rotation**: Archive logs after N runs (currently 10)
5. **Structured info**: Include Date, Time, Dealer, CUSIP, file path in log messages

### Example Logging Pattern

```python
# Setup logger (file only, no console)
self.logger = setup_logging(LOG_FILE_PROCESSING, 'runs_extract', console_level=logging.CRITICAL)

# Log file processing
self.logger.info(f"Processing RUNS file: {filename}")
self.logger.info(f"Extracted {len(df)} rows")

# Log deduplication
self.logger_dupes.info(f"Duplicate Date+Dealer+CUSIP: {date} - {dealer} - {cusip} (kept latest Time)")

# Log validation issues
self.logger_valid.warning(f"Found {count} invalid CUSIPs (logged summary)")

# Orphan CUSIP logging (with context)
# Example output: "Orphan CUSIP: 06368MYR9 (Security=Some Bond, Date=10/31/2025, Dealer=RBC, Time=15:45, Ticker=ABC)"
self.logger_valid.info(f"  Orphan CUSIP: {cusip} ({context_details})")
```

---

## Troubleshooting Common Issues

### Import Errors

```bash
# Make sure you're in project root and virtual environment is activated
cd Bond-RV-App
Bond-RV-App\Scripts\activate

# Use module syntax for imports
python -m runs_pipeline.pipeline
```

### Date/Time Parsing Errors

- **Issue**: Date or Time column cannot be parsed
- **Solution**: Check Excel file format matches expected MM/DD/YY and HH:MM formats
- **Logging**: Parsing errors logged to processing.log with row numbers

### Primary Key Violations

- **Issue**: Duplicate Date+Dealer+CUSIP combinations found after deduplication
- **Solution**: Check deduplication logic or data quality issues
- **Logging**: Primary key violations logged to processing.log with sample duplicates

### Orphan CUSIP Warnings

- **Issue**: CUSIPs in runs data not found in universe.parquet
- **Solution**: Normal - may indicate new bonds or data quality issues
- **Logging**: All orphan CUSIPs logged to validation.log

### Schema Mismatches

- **Issue**: Old file (28 columns) vs new files (30 columns)
- **Solution**: Pipeline automatically aligns to master schema (fills missing columns with NaN)
- **Logging**: Schema mismatches logged to validation.log

### File Processing Order

- **Issue**: Files processed in wrong order
- **Solution**: Pipeline automatically sorts files chronologically by date in filename
- **Verification**: Check processing.log for file processing order

---

## Checking Results

```bash
# View output parquet file
ls bond_data/parquet/runs_timeseries.parquet

# Check logs
cat bond_data/logs/summary.log       # Pipeline execution summary
cat bond_data/logs/processing.log    # File-by-file details
cat bond_data/logs/duplicates.log    # Duplicate Date+Dealer+CUSIP
cat bond_data/logs/validation.log    # CUSIP validation and orphan tracking

# Quick stats in Python
import pandas as pd
df = pd.read_parquet('bond_data/parquet/runs_timeseries.parquet')
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Unique CUSIPs: {df['CUSIP'].nunique()}")
print(f"Unique Dealers: {df['Dealer'].nunique()}")
```

---

## Common Patterns & Examples

### Date/Time Parsing

```python
from bond_pipeline.utils import parse_runs_date, parse_runs_time

# Parse date
date_str = "10/31/25"
date_obj = parse_runs_date(date_str)  # Returns: datetime(2025, 10, 31)

# Parse time
time_str = "15:45"
time_obj = parse_runs_time(time_str)  # Returns: time(15, 45)
```

### End-of-Day Deduplication

```python
from runs_pipeline.transform import RunsTransformer

transformer = RunsTransformer(log_dupes, log_valid)
df_cleaned = transformer.deduplicate_end_of_day(df)

# Group by Date+Dealer+CUSIP
# Sort by Time descending (latest first)
# Keep first row per group (latest time)
```

### CUSIP Orphan Tracking

```python
from bond_pipeline.utils import check_cusip_orphans

runs_cusips = set(df['CUSIP'].dropna().unique())
orphans = check_cusip_orphans(runs_cusips, UNIVERSE_PARQUET, logger)

# Logs all orphan CUSIPs to validation.log
# Returns set of orphan CUSIPs
```

### Reading/Writing Parquet

```python
import pandas as pd
from bond_pipeline.config import RUNS_PARQUET, RUNS_PRIMARY_KEY

# Read existing dates (for append mode)
df = pd.read_parquet(RUNS_PARQUET, columns=['Date'])
existing_dates = set(df['Date'].dropna().unique())

# Write new data (append mode)
df_new.to_parquet(RUNS_PARQUET, mode='append', index=False)

# Write new data (override mode)
df_new.to_parquet(RUNS_PARQUET, mode='overwrite', index=False)

# Validate primary key
duplicates = df.duplicated(subset=RUNS_PRIMARY_KEY)
if duplicates.any():
    print(f"Found {duplicates.sum()} duplicate Date+Dealer+CUSIP combinations!")
```

---

## Key Takeaways

### Essential Rules

1. **End-of-Day Snapshots**: Keep latest Time per Date+Dealer+CUSIP (most recent quote of the day)
2. **Primary Key**: Enforce Date+Dealer+CUSIP uniqueness after deduplication
3. **No CUSIP Normalization**: Keep CUSIPs as-is from Excel (log invalid but don't normalize)
4. **Orphan Tracking**: Track CUSIPs in runs but not in universe.parquet
5. **Schema Alignment**: Align all files to 30-column master schema (fill missing with NaN)
6. **Combined Logs**: Both pipelines write to same log files
7. **Shared Utilities**: Reuse bond_pipeline utilities where possible

### Workflow Rules

1. **Chronological Processing**: Sort files by date in filename (earliest first)
2. **Deduplication First**: Deduplicate before schema alignment and validation
3. **Validation Logging**: Log all validation issues but keep data (don't exclude)
4. **Date Filtering**: Append mode skips existing dates, override mode rebuilds all

---

---

## Performance Optimizations

### Vectorized Operations (January 2, 2025)

The runs pipeline has been optimized for performance using vectorized pandas operations:

#### Deduplication Optimization
- **Before**: Row-by-row iteration through groups (O(n*groups))
- **After**: Vectorized sort + drop_duplicates (O(n log n))
- **Speed Improvement**: ~100x faster for large datasets (130k+ rows)
- **Method**: `sort_values()` with primary key + Time descending, then `drop_duplicates(keep='first')`

#### CUSIP Validation Optimization
- **Before**: Row-by-row iteration with `for idx, cusip in df['CUSIP'].items()`
- **After**: Vectorized string operations (`.str.len()`, `.isna()`, `.str.strip()`)
- **Speed Improvement**: ~100x faster for 130k+ rows
- **Method**: Vectorized masks for empty CUSIPs and invalid length checks

#### Expected Performance
- **Before**: 5-10 minutes for ~130,000 rows
- **After**: 5-30 seconds for ~130,000 rows (depending on dataset size and duplicate count)
- **Improvement**: 10-120x faster depending on data characteristics

### Enhanced Orphan Logging (January 2, 2025)

Orphan CUSIP logging now includes full context for quick identification:

- **Security**: Bond name for easy identification
- **Date**: Quote date (formatted MM/DD/YYYY)
- **Dealer**: Which dealer provided the quote
- **Time**: Quote timestamp (HH:MM format)
- **Ticker**: Security ticker (if available)

**Example Log Entry**:
```
Orphan CUSIP: 06368MYR9 (Security=Some Bond Name, Date=10/31/2025, Dealer=RBC, Time=15:45, Ticker=ABC)
```

This eliminates the need to cross-reference parquet files to identify orphan bonds.

---

**File Created**: January 2025  
**Last Major Update**: January 2, 2025  
**Project**: Bond RV App - Runs Data Pipeline  
**Version**: 1.1

