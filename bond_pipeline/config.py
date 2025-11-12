"""
Configuration module for bond data pipeline.
Contains constants, paths, and schema definitions.
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "bond_data"
PARQUET_DIR = DATA_DIR / "parquet"
LOGS_DIR = DATA_DIR / "logs"

# Default input directory for Excel files (Dropbox)
DEFAULT_INPUT_DIR = Path(r"C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Support Files\API Historical")

# Parquet file names
HISTORICAL_PARQUET = PARQUET_DIR / "historical_bond_details.parquet"
UNIVERSE_PARQUET = PARQUET_DIR / "universe.parquet"
BQL_PARQUET = PARQUET_DIR / "bql.parquet"

# BQL input configuration
BQL_EXCEL_PATH = Path(
    r"C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Support Files\bql.xlsx"
)
BQL_SHEET_NAME = "bql"
BQL_HEADER_LABEL = "CUSIPs"
BQL_NAME_ROW_INDEX = 1  # Name is on level 1 (row 1) of multi-index header
BQL_CUSIP_ROW_INDEX = 2  # CUSIP is on level 2 (row 2) of multi-index header
BQL_DATA_START_ROW = 4  # Data starts at row 4 (0-indexed: row 4 = index 4)
BQL_DATE_COLUMN_NAME = "Date"
BQL_VALUE_COLUMN_NAME = "Value"
BQL_NAME_COLUMN_NAME = "Name"
BQL_CUSIP_COLUMN_NAME = "CUSIP"
BQL_OUTPUT_COLUMNS = [
    BQL_DATE_COLUMN_NAME,
    BQL_NAME_COLUMN_NAME,
    BQL_CUSIP_COLUMN_NAME,
    BQL_VALUE_COLUMN_NAME,
]

# Excel file patterns
FILE_PATTERN = r'API\s+(\d{2})\.(\d{2})\.(\d{2})\.xlsx$'
HEADER_ROW = 2  # 0-indexed, row 3 in Excel (rows 1-2 are empty/metadata)

# Universe table columns (13 columns as specified)
UNIVERSE_COLUMNS = [
    'CUSIP',
    'Benchmark Cusip',
    'Custom_Sector',
    'Bloomberg Cusip',
    'Security',
    'Benchmark',
    'Pricing Date',
    'Pricing Date (Bench)',
    'Worst Date',
    'Yrs (Worst)',
    'Ticker',
    'Currency',
    'Equity Ticker'
]

# Master schema - 75 columns from latest files
# This will be populated dynamically from the latest file schema
MASTER_SCHEMA_COLUMNS = None  # Will be set during first file read

# NA values to convert to NULL
NA_VALUES = [
    '#N/A Field Not Applicable',
    '#N/A Invalid Security',
    '#N/A',
    'N/A',
    ''
]

# CUSIP validation
CUSIP_LENGTH = 9

# Date column name
DATE_COLUMN = 'Date'

# Logging
LOG_FILE_PROCESSING = LOGS_DIR / "processing.log"
LOG_FILE_DUPLICATES = LOGS_DIR / "duplicates.log"
LOG_FILE_VALIDATION = LOGS_DIR / "validation.log"
LOG_FILE_SUMMARY = LOGS_DIR / "summary.log"
LOG_FILE_PARQUET_STATS = LOGS_DIR / "parquet_stats.log"

# Enhanced logging configuration
LOG_ROTATION_RUNS = 10  # Keep last N runs in active log files
LOG_ARCHIVE_DIR = LOGS_DIR / "archive"
LOG_METADATA_FILE = LOGS_DIR / ".run_metadata.json"
BOND_NAME_COLUMN = "Security"  # Column to use for bond names in logs

# ===== RUNS PIPELINE CONFIGURATION =====

# Default input directory for RUNS Excel files
RUNS_INPUT_DIR = Path(r"C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Support Files\Historical Runs")

# RUNS file pattern
RUNS_FILE_PATTERN = r'RUNS\s+(\d{2})\.(\d{2})\.(\d{2})\.xlsx$'
RUNS_HEADER_ROW = 0  # 0-indexed, row 1 in Excel (header is in first row)

# RUNS parquet output file
RUNS_PARQUET = PARQUET_DIR / "runs_timeseries.parquet"

# RUNS primary key columns
RUNS_PRIMARY_KEY = ['Date', 'Dealer', 'CUSIP']

# RUNS datetime column formats
RUNS_DATE_FORMAT = '%m/%d/%Y'  # mm/dd/yyyy format
RUNS_TIME_FORMAT = '%H:%M'      # hh:mm format

# RUNS schema (30 columns from analysis)
RUNS_COLUMNS = [
    'Reference Security',
    'Date',
    'Time',
    'Bid Workout Risk',
    'Ticker',
    'Dealer',
    'Source',
    'Security',
    'Bid Price',
    'Ask Price',
    'Bid Spread',
    'Ask Spread',
    'Benchmark',
    'Reference Benchmark',
    'Bid Size',
    'Ask Size',
    'Sector',
    'Bid Yield To Convention',
    'Ask Yield To Convention',
    'Bid Discount Margin',
    'Ask Discount Margin',
    'CUSIP',
    'Sender Name',
    'Currency',
    'Subject',
    'Keyword',
    'Bid Interpolated Spread to Government',
    'Ask Interpolated Spread to Government',
    'Bid Contributed Yield',
    'Bid Z-spread'
]

# RUNS validation settings
RUNS_VALIDATE_DATE_RANGE = True
RUNS_VALIDATE_TIME_FORMAT = True
RUNS_VALIDATE_PRICES_POSITIVE = True
RUNS_VALIDATE_SPREADS_REASONABLE = True
RUNS_VALIDATE_DEALERS = True

# RUNS known dealers list
RUNS_KNOWN_DEALERS = [
    'RBC',
    'TD',
    'CIBC',
    'BMO',
    'NBF',
    'BNS',
    'MS',
    'YTMC',
    'BBLP',
    'IAS',
    'TDS'
]

# CUSIP orphan tracking
RUNS_TRACK_ORPHAN_CUSIPS = True  # Track CUSIPs not in universe.parquet
RUNS_LOG_INVALID_CUSIPS = True   # Log invalid CUSIPs (wrong length, etc.)

# Ensure directories exist
PARQUET_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

