# CLAUDE.md

**Last Updated**: October 26, 2025 6:30 PM

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bond RV App is a modular ETL pipeline that processes bond market data from two sources:

1. **Excel Pipeline** - Processes Excel files containing historical bond market data and transforms them into optimized Parquet tables for relative value trading analysis. Handles schema evolution (59-75 columns across different file vintages), CUSIP validation, deduplication, and produces two core outputs: a historical time series table and a universe table of unique bonds.

2. **Outlook Email Pipeline** - Archives and parses bond trading emails from Outlook RUNS folder, extracting dealer quotes (bid/ask spreads, sizes, benchmarks) into a clean time series Parquet file.

## Architecture

### Excel Pipeline Flow (ETL Pattern)
1. **Extract** (`extract.py`) - Reads Excel files with regex pattern matching on filenames to extract dates
2. **Transform** (`transform.py`) - Validates CUSIPs, removes duplicates, normalizes data, aligns schemas
3. **Load** (`load.py`) - Writes to Parquet with append/override modes

### Excel Pipeline Components
- **config.py** - Central configuration with hardcoded Dropbox paths, schema definitions, NA value mappings
- **utils.py** - Date parsing, CUSIP validation, logging setup
- **pipeline.py** - Main orchestrator that coordinates E-T-L flow
- **run_pipeline.py** - Simple CLI wrapper at project root

### Outlook Email Pipeline Flow
1. **Archive** (`monitor_outlook.py`) - Archives Outlook RUNS folder emails to CSV files (one CSV per date)
2. **Parse** (`runs_miner.py`) - Parses email bodies from CSVs, extracts bond quotes, writes to Parquet

### Outlook Pipeline Components
- **monitor_outlook.py** - CLI wrapper for OutlookMonitor class
- **utils/outlook_monitor.py** - Outlook COM automation, email archiving, incremental sync
- **runs_miner.py** - Standalone parser with dynamic column detection, CUSIP validation, data cleaning

### Critical Design Patterns
- **Schema Evolution**: Master schema is dynamically set from the latest file, older files are padded with missing columns
- **Deduplication**: Keeps last occurrence of Date+CUSIP duplicates (not first)
- **Append Mode**: Checks existing Parquet for dates, skips already-processed dates
- **Override Mode**: Deletes existing Parquet and rebuilds from scratch

### Data Flow
```
Excel files (Raw Data/ or Dropbox) → extract.py → raw_data dict[datetime, DataFrame]
→ transform.py → transformed_data dict[datetime, DataFrame]
→ load.py → historical_bond_details.parquet + universe.parquet
```

**Input Flexibility**: The pipeline supports two workflows:
- **Local Workflow**: Files in `Raw Data/` folder (recommended - simple drag & drop)
- **Dropbox Workflow**: Files in Dropbox folder (optional - for automatic syncing across computers)

## Virtual Environment Setup

**CRITICAL**: This project uses a virtual environment named `Bond-RV-App` (matching the project name).

### First-Time Setup
```bash
# Create virtual environment
python -m venv Bond-RV-App

# Activate virtual environment
Bond-RV-App\Scripts\activate       # Windows
source Bond-RV-App/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Daily Use
```bash
# Always activate before running any scripts
Bond-RV-App\Scripts\activate       # Windows
source Bond-RV-App/bin/activate    # Mac/Linux
```

## Running the Pipeline

**Note**: All commands below assume the virtual environment is activated.

### Quick Run (Interactive)
```bash
# From project root
python run_pipeline.py
# Select mode: 1=override (first run), 2=append (daily updates)
```

### Using Virtual Environment Without Activation
```bash
# Windows - use full path to venv Python
Bond-RV-App\Scripts\python.exe run_pipeline.py
```

### Direct Pipeline Execution
```bash
# From bond_pipeline directory
python pipeline.py -i "../Raw Data/" -m override    # First run
python pipeline.py -i "../Raw Data/" -m append      # Daily updates

# With custom input directory
python pipeline.py -i "path/to/excel/files" -m append
```

### Non-Interactive (for automation)
```bash
echo 2 | python run_pipeline.py  # Runs in append mode
```

## Running the Outlook Email Pipeline

**Purpose**: Archive and parse bond trading emails from Outlook RUNS folder into a clean time series dataset.

**Two-Step Workflow**:
1. **Step 1**: Archive emails from Outlook to CSV files (one CSV per date)
2. **Step 2**: Parse email bodies from CSVs into clean Parquet format

### Step 1: Archive Emails from Outlook

```bash
# From project root (virtual environment activated)

# Incremental sync - archive new emails only (recommended for daily use)
python monitor_outlook.py

# Full rebuild - delete all CSV files, sync index, and logs, then re-archive everything
python monitor_outlook.py --rebuild

# Archive last N days to today (e.g., last 2 days)
python monitor_outlook.py --days 2
```

**Output**: CSV files in `C:\Users\Eddy\YTM Capital Dropbox\...\Support Files\Outlook Runs\`
- One CSV per date: `Outlook Data 10.26.2025.csv`
- Sync index: `sync_index.csv` (tracks processed emails)
- **Log file**: `bond_data/logs/outlook_monitor.log` (detailed execution log)

**Requirements**:
- Outlook must be installed and configured
- RUNS folder must exist in Outlook Inbox
- Email address: `eddy.winiarz@ytmcapital.com`

**Logging**: All operations are logged to `bond_data/logs/outlook_monitor.log` with:
- Summary statistics (emails processed, files created, performance metrics)
- Error details (when processing fails, includes email subject/ID)
- Performance metrics (emails/sec, total duration)

### Step 2: Parse Emails into Clean Parquet

```bash
# From project root (virtual environment activated)

# Incremental mode - process new CSV files only (recommended for daily use)
python runs_miner.py

# Full rebuild - delete output and reprocess all CSVs from scratch
python runs_miner.py --rebuild
```

**Input**: CSV files from Step 1 (`Outlook Data *.csv`)
**Output**: `bond_timeseries_clean.parquet` (in `bond_data/parquet/`)
**Processing Index**: `runs_processing_index.csv` (tracks which CSVs have been processed)

**Incremental Mode** (default):
- Only processes CSV files not yet in the processing index
- Appends new data to existing Parquet file
- Deduplicates combined data (Date + CUSIP + Dealer)
- Fast for daily updates

**Rebuild Mode** (`--rebuild`):
- Deletes existing Parquet output and processing index
- Reprocesses ALL CSV files from scratch
- Use for first run or when data needs to be rebuilt

**What it does**:
- Parses email bodies with dynamic column detection (handles 48+ different email formats)
- Validates CUSIPs (9-character alphanumeric)
- Extracts bid/ask spreads, sizes, benchmarks
- Deduplicates by Date + CUSIP + Dealer (keeps most recent)
- Normalizes security names, converts Unicode fractions (¼→0.25)
- Splits dealer names into Bank + Sender
- Extracts ticker, coupon, maturity from security names
- Comprehensive validation (misalignment, inverted spreads, etc.)

**Output Schema** (15 columns):
- **Date** - When email was received (mm/dd/yyyy string format)
- **Time** - Time email was received (hh:mm string format, no seconds)
- **Dealer** - Bank code (BMO, NBF, RBC)
- **Sender** - Trader name
- **Ticker** - Issuer symbol extracted from security name
- **Security** - Normalized security name
- **CUSIP** - 9-character bond identifier
- **Coupon** - Coupon rate extracted from security name
- **Maturity Date** - Maturity date (mm/dd/yyyy string format)
- **B_Spd** - Bid spread over benchmark in basis points
- **A_Spd** - Ask spread over benchmark in basis points
- **B_Sz_MM** - Bid size in millions (standardized: NBF thousands converted to millions)
- **A_Sz_MM** - Ask size in millions (standardized: NBF thousands converted to millions)
- **Bench** - Benchmark bond (e.g., "CAN 1.5 06/01/26", fractions converted to decimals)
- **B_GSpd** - Bid G-spread (validated: within ±10 bps of B_Spd, else NA)

**Configuration** (edit at top of `runs_miner.py`):
```python
INPUT_DIR = r"C:\...\Support Files\Outlook Runs"  # Where monitor_outlook.py saves CSVs
OUTPUT_DIR = r"."  # Project root (current directory)
OUTPUT_FILENAME = "bond_timeseries_clean.parquet"
```

### Daily Workflow Example

```bash
# 1. Archive new emails from today
python monitor_outlook.py

# 2. Parse new CSV files (incremental - only processes new files)
python runs_miner.py
```

This workflow:
- Step 1 creates new CSV files for today's emails (or updates existing ones)
- Step 2 processes only the new CSV files and appends to the Parquet file
- Total time: ~1-2 seconds for typical daily volume

### First-Time Setup / Full Rebuild

```bash
# When setting up for the first time or need to rebuild everything:

# 1. Archive all emails from Outlook
python monitor_outlook.py --rebuild

# 2. Process all CSVs from scratch
python runs_miner.py --rebuild
```

## Testing

### Run All Tests
```bash
pytest                           # All tests
pytest -v                        # Verbose
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests only
```

### Run Specific Test
```bash
pytest tests/unit/test_utils.py                    # Single file
pytest tests/unit/test_utils.py::test_function_name # Single test
pytest -k "cusip"                                   # Pattern match
```

### Test Coverage
```bash
pytest --cov=bond_pipeline --cov-report=html
```

## Configuration & Paths

### Excel Pipeline Paths
The DEFAULT_INPUT_DIR in `config.py` is hardcoded to a Dropbox path:
```python
DEFAULT_INPUT_DIR = Path(r"C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Support Files\API Historical")
```

This path is machine-specific and assumes Dropbox sync. When working on different machines, this path must be updated.

**File Naming Convention**: Excel files must match pattern `API MM.DD.YY.xlsx`
- Example: `API 10.20.25.xlsx` (October 20, 2025)
- Regex: `r'API\s+(\d{2})\.(\d{2})\.(\d{2})\.xlsx$'`
- Years 00-49 → 2000-2049, years 50-99 → 1950-1999

### Outlook Pipeline Paths
Configured in `monitor_outlook.py` and `runs_miner.py`:

**monitor_outlook.py** (line 26):
```python
OUTPUT_DIR = r"C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Support Files\Outlook Runs"
```

**runs_miner.py** (lines 23-25):
```python
INPUT_DIR = r"C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Support Files\Outlook Runs"
OUTPUT_DIR = r"."  # Project root
OUTPUT_FILENAME = "bond_timeseries_clean.parquet"
```

**File Naming Convention**: CSV files created by monitor_outlook.py
- Pattern: `Outlook Data YYYY-MM-DD.csv`
- Example: `Outlook Data 2025-10-26.csv`
- Sync index: `sync_index.csv` (tracks processed email IDs)

### Output Structure
```
bond_data/
├── parquet/
│   ├── historical_bond_details.parquet  # Excel pipeline output (Date + CUSIP)
│   ├── universe.parquet                  # Excel pipeline output (CUSIP only, 13 cols)
│   └── bond_timeseries_clean.parquet     # Outlook pipeline output (runs_miner.py)
└── logs/
    ├── processing.log       # Excel pipeline: Main operations
    ├── duplicates.log       # Excel pipeline: Duplicate detection
    ├── validation.log       # Excel pipeline: CUSIP validation
    ├── summary.log          # Excel pipeline: High-level summary
    └── outlook_monitor.log  # Outlook pipeline: Email archiving log

C:\...\Support Files\Outlook Runs\         # Outlook email archives
├── Outlook Data 10.20.2025.csv
├── Outlook Data 10.21.2025.csv
├── Outlook Data 10.26.2025.csv
├── sync_index.csv                         # Tracks processed emails (monitor_outlook.py)
└── runs_processing_index.csv              # Tracks processed CSVs (runs_miner.py)
```

## Data Quality Rules

### Excel Pipeline Data Quality

**CUSIP Validation**:
- Must be exactly 9 alphanumeric characters
- Automatically uppercased
- Invalid CUSIPs are logged but NOT removed (kept in dataset with validation warnings)
- Common issues: wrong length (8, 10, 12 chars), special characters, Bloomberg IDs

**Duplicate Handling**:
- Duplicates defined as: same Date + CUSIP combination within a single file
- Strategy: Keep LAST occurrence (not first)
- Logged to `duplicates.log` with sample CUSIPs

**NA Value Normalization**:
These strings are converted to pandas NA:
- `#N/A Field Not Applicable`
- `#N/A Invalid Security`
- `#N/A`, `N/A`, empty strings

### Outlook Pipeline Data Quality (runs_miner.py)

**Dynamic Column Detection**:
- Handles 48+ different email header formats
- Uses separator anchoring (`/` for spreads, `x` for sizes)
- CUSIP position detected by validation (search backwards from end)
- Prevents column misalignment errors

**CUSIP Validation**:
- Must be exactly 9 alphanumeric characters
- Automatically uppercased
- Invalid CUSIPs logged to console with line numbers
- Rows with invalid CUSIPs are kept but flagged

**Duplicate Handling**:
- Duplicates defined as: same Date + CUSIP + Dealer combination
- Strategy: Keep most recent by ReceivedDateTime (intraday updates)
- Allows same bond quotes from different dealers on same date

**Data Cleaning**:
- Security names normalized (extra whitespace removed)
- CUSIP-Security name conflicts resolved (canonical name = shortest among most common)
- Unicode fraction conversion: ¼→0.25, ½→0.5, ¾→0.75, ⅛→0.125, ⅜→0.375, ⅝→0.625, ⅞→0.875
- Size field cleaning: "2MM" → 2
- Size standardization: NBF dealer uses thousands (M), values ≥1000 divided by 1000 to convert to millions (MM)
- Dealer code mapping: "BMO CAPITAL MARKETS"→"BMO", "NATIONAL BANK FINANC"→"NBF", etc.

**Data Validation & Quality Checks**:
- **Spread Range Validation**: Deletes rows where both B_Spd and A_Spd are outside 10-2000 bps range
- **B_GSpd Validation**: Sets B_GSpd to NA if not within ±10 bps of B_Spd (prints examples)
- **Column misalignment detection**: Compares parsed vs expected column count
- **Invalid CUSIP detection**: Length and character validation
- **Inverted spreads check**: Flags if bid_spread < ask_spread
- **Non-numeric spreads and sizes**: Validates numeric conversion
- **Unicode fraction detection**: Ensures all fractions converted to decimals in benchmarks

## Coding Standards

### Documentation Requirements
- All new documentation must include **date and timestamp**
- When creating documentation, review and update related existing docs for consistency

### Configuration
- All config must be in `.py` files (NOT `.json`, `.yaml`, `.toml`)
- Use `config.py` for all constants and paths

### Dependencies
- Project uses pip with `requirements.txt` (not Poetry)
- All dependencies (production + testing) are in a single `requirements.txt` file
- Core dependencies: pandas>=2.0.0, pyarrow>=12.0.0, openpyxl>=3.1.0
- Test dependencies: pytest>=7.0.0, pytest-cov>=4.0.0, pytest-mock>=3.10.0

## Common Modifications

### Adding New Validation Rules
Edit `transform.py` → `DataTransformer` class → Add validation logic in `transform_single_file()`

### Changing Output Schema
Edit `config.py` → Modify `UNIVERSE_COLUMNS` for universe table

### Adjusting Duplicate Logic
Edit `transform.py` → `remove_duplicates()` method → Change `keep='last'` parameter

### Adding New CLI Options
Edit `pipeline.py` → `main()` function → Add argparse arguments

## Troubleshooting

### Excel Pipeline Issues

**Pipeline Fails with "No files found"**:
- Check Dropbox sync status (must be fully synced)
- Verify DEFAULT_INPUT_DIR path in `config.py` matches your machine
- Confirm Excel files match naming pattern `API MM.DD.YY.xlsx`

**"Could not convert with type str" Error**:
- Object columns must be converted to strings before Parquet write
- Fix is in `load.py` lines 92-95 (already implemented for append mode)

**Different Results on Different Computers**:
- Run in override mode to rebuild from scratch
- Ensure same Excel files are present
- Check Dropbox sync completion

### Outlook Pipeline Issues

**monitor_outlook.py fails to connect**:
- Ensure Outlook is installed and running
- Verify email is configured: `eddy.winiarz@ytmcapital.com`
- Check that RUNS folder exists in Outlook Inbox (not a subfolder)
- Windows only: Requires `pywin32` for COM automation

**Multiple CSV files for the same date**:
- This should NOT happen - each date should have ONE CSV file
- If you see duplicates, run `python monitor_outlook.py --rebuild` to clear and rebuild
- Check `bond_data/logs/outlook_monitor.log` for errors during file writing

**runs_miner.py finds no CSV files**:
- Run `monitor_outlook.py` first to archive emails
- Verify INPUT_DIR path in `runs_miner.py` (line 28) matches OUTPUT_DIR in `monitor_outlook.py` (line 28)
- Check that CSV files exist and match pattern: `Outlook Data *.csv`

**runs_miner.py says "No new files to process"**:
- This is normal if all CSV files have been processed already
- Run `python monitor_outlook.py` first to archive new emails
- Or use `python runs_miner.py --rebuild` to reprocess everything

**Data looks corrupted or incomplete**:
- Full rebuild Outlook CSVs: `python monitor_outlook.py --rebuild`
- Full rebuild Parquet output: `python runs_miner.py --rebuild`
- Check `bond_data/logs/outlook_monitor.log` for email archiving errors
- Delete `runs_processing_index.csv` manually if index is corrupted

**High column misalignment warnings**:
- This is expected for some email formats
- Check validation output for specific issues
- If >30% misaligned, verify email format hasn't changed significantly

**Invalid CUSIP warnings**:
- Some emails may contain non-bond identifiers
- Review logged line numbers to identify problematic emails
- CUSIPs must be exactly 9 alphanumeric characters

**Unicode fraction display issues (�)**:
- Ensure CSV files are UTF-8 encoded
- Check console encoding supports Unicode
- Fractions are automatically converted to decimals in output

**Empty output Parquet file**:
- Check that input CSVs have "Body of Email" column
- Verify email bodies contain bond data (not just text)
- Look for parsing errors in console output

### General Issues

**Missing Dependencies**:
```bash
# All dependencies are in requirements.txt
pip install -r requirements.txt
```

**Permission denied when saving Parquet**:
- Close any programs that might have the file open (Excel, Power BI, etc.)
- Check file permissions in output directory
- For runs_miner.py, ensure project root is writable

## Documentation Structure

The project has comprehensive documentation in the `Documentation/` folder:
- **Setup/** - Installation and virtual environment guides
- **Workflows/** - Local and Dropbox workflows
- **Architecture/** - Technical documentation
- **Reference/** - Testing, deliverables, and decision documents

See `Documentation/README.md` for complete documentation index.
