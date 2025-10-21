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

# Parquet file names
HISTORICAL_PARQUET = PARQUET_DIR / "historical_bond_details.parquet"
UNIVERSE_PARQUET = PARQUET_DIR / "universe.parquet"

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

# Ensure directories exist
PARQUET_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

