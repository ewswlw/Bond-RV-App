"""
Utility functions for bond data pipeline.
Includes date parsing, CUSIP validation, and logging helpers.
"""

import re
import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from io import StringIO
import pandas as pd

from .config import (
    FILE_PATTERN,
    CUSIP_LENGTH,
    NA_VALUES,
    LOG_ROTATION_RUNS,
    LOG_ARCHIVE_DIR,
    LOG_METADATA_FILE
)


def setup_logging(log_file: Path, name: str = None, console_level: int = logging.WARNING) -> logging.Logger:
    """
    Set up logging to both file and console with separate levels.

    Args:
        log_file: Path to log file
        name: Logger name (optional)
        console_level: Logging level for console (default: WARNING to suppress most output)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name or log_file.stem)
    logger.setLevel(logging.DEBUG)  # Capture everything at logger level

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # File handler - captures ALL detail
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(file_formatter)

    # Console handler - only essential messages (WARNING and above)
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    console_formatter = logging.Formatter('%(message)s')  # Minimal format for console
    ch.setFormatter(console_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def setup_console_logger(name: str = 'console') -> logging.Logger:
    """
    Set up a console-only logger for high-level user messages.

    Args:
        name: Logger name

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler only
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(console_formatter)

    logger.addHandler(ch)

    return logger


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract date from filename pattern: API MM.DD.YY.xlsx
    
    Args:
        filename: Name of the Excel file
    
    Returns:
        datetime object or None if pattern doesn't match
    
    Examples:
        'API 10.20.25.xlsx' -> datetime(2025, 10, 20)
        'API 08.04.23.xlsx' -> datetime(2023, 8, 4)
    """
    match = re.search(FILE_PATTERN, filename, re.IGNORECASE)
    
    if not match:
        return None
    
    month, day, year = match.groups()
    
    # Convert 2-digit year to 4-digit (assume 20xx for < 50, else 19xx)
    full_year = int(f"20{year}") if int(year) < 50 else int(f"19{year}")
    
    try:
        date_obj = datetime(full_year, int(month), int(day))
        return date_obj
    except ValueError:
        return None


def validate_cusip(cusip: str) -> Tuple[str, bool, str]:
    """
    Validate and normalize CUSIP.
    
    Args:
        cusip: CUSIP string to validate
    
    Returns:
        Tuple of (normalized_cusip, is_valid, error_message)
    
    Examples:
        '89678zab2' -> ('89678ZAB2', True, '')
        '12345' -> ('12345', False, 'Invalid length: 5')
    """
    if pd.isna(cusip) or cusip == '':
        return (cusip, False, 'Empty CUSIP')
    
    # Convert to string and normalize
    cusip_str = str(cusip).strip().upper()
    
    # Check length
    if len(cusip_str) != CUSIP_LENGTH:
        return (cusip_str, False, f'Invalid length: {len(cusip_str)} (expected {CUSIP_LENGTH})')
    
    # Check alphanumeric
    if not cusip_str.isalnum():
        return (cusip_str, False, 'Contains non-alphanumeric characters')
    
    return (cusip_str, True, '')


def clean_na_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NA string values with proper NaN and convert object columns to string.
    
    Args:
        df: DataFrame to clean
    
    Returns:
        Cleaned DataFrame
    """
    # Replace NA strings with NaN
    df = df.replace(NA_VALUES, pd.NA)
    
    # Replace additional NA patterns
    df = df.replace(r'^#N/A.*', pd.NA, regex=True)
    
    # Convert object columns to string type to handle mixed types
    for col in df.columns:
        if df[col].dtype == 'object':
            # Convert to string, handling NaN properly
            df[col] = df[col].astype(str)
            # Replace 'nan' string back to actual NaN
            df[col] = df[col].replace(['nan', '<NA>'], pd.NA)
    
    return df


def get_master_schema(df: pd.DataFrame) -> list:
    """
    Extract master schema from a DataFrame.
    
    Args:
        df: DataFrame with the schema
    
    Returns:
        List of column names
    """
    return df.columns.tolist()


def align_to_master_schema(df: pd.DataFrame, master_columns: list) -> pd.DataFrame:
    """
    Align DataFrame to master schema by adding missing columns with NaN.
    
    Args:
        df: DataFrame to align
        master_columns: List of master column names
    
    Returns:
        Aligned DataFrame with all master columns
    """
    # Add missing columns with NaN
    for col in master_columns:
        if col not in df.columns:
            df[col] = pd.NA
    
    # Reorder columns to match master schema
    df = df[master_columns]
    
    return df


def format_date_string(date_obj: datetime) -> str:
    """
    Format datetime object to string for display.
    
    Args:
        date_obj: datetime object
    
    Returns:
        Formatted date string (YYYY-MM-DD)
    """
    return date_obj.strftime('%Y-%m-%d')


def get_file_list(input_dir: Path, pattern: str = '*.xlsx') -> list:
    """
    Get list of Excel files in directory.

    Args:
        input_dir: Directory to search
        pattern: File pattern to match

    Returns:
        List of Path objects
    """
    return sorted(list(input_dir.glob(pattern)))


# ============================================================================
# Enhanced Logging Functions
# ============================================================================

def get_run_id() -> int:
    """
    Get next run ID from metadata file.

    Returns:
        Sequential run ID
    """
    if LOG_METADATA_FILE.exists():
        try:
            with open(LOG_METADATA_FILE, 'r') as f:
                metadata = json.load(f)
                run_id = metadata.get('last_run_id', 0) + 1
        except (json.JSONDecodeError, KeyError):
            run_id = 1
    else:
        run_id = 1

    return run_id


def save_run_metadata(run_id: int, start_time: datetime, mode: str):
    """
    Save run metadata to tracking file.

    Args:
        run_id: Run ID
        start_time: Run start timestamp
        mode: Pipeline mode (append/override)
    """
    metadata = {
        'last_run_id': run_id,
        'last_run_time': start_time.isoformat(),
        'last_run_mode': mode
    }

    if LOG_METADATA_FILE.exists():
        try:
            with open(LOG_METADATA_FILE, 'r') as f:
                existing = json.load(f)
                if 'run_history' in existing:
                    metadata['run_history'] = existing['run_history'][-9:]  # Keep last 9
                else:
                    metadata['run_history'] = []
        except (json.JSONDecodeError, KeyError):
            metadata['run_history'] = []
    else:
        metadata['run_history'] = []

    # Add current run to history
    metadata['run_history'].append({
        'run_id': run_id,
        'timestamp': start_time.isoformat(),
        'mode': mode
    })

    with open(LOG_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)


def format_run_header(run_id: int, start_time: datetime, mode: str, input_dir: Path) -> str:
    """
    Format enhanced pipeline run header.

    Args:
        run_id: Run ID
        start_time: Run start timestamp
        mode: Pipeline mode
        input_dir: Input directory path

    Returns:
        Formatted header string
    """
    border = "=" * 80

    date_str = start_time.strftime("%Y-%m-%d")
    time_str = start_time.strftime("%H:%M:%S")
    mode_upper = mode.upper()

    header_lines = [
        "",
        border,
        f"  PIPELINE RUN #{run_id}",
        f"  Date: {date_str}  |  Start Time: {time_str}  |  Mode: {mode_upper}",
        border,
        ""
    ]

    return "\n".join(header_lines)


def format_section_header(title: str) -> str:
    """
    Format section header for logs.

    Args:
        title: Section title

    Returns:
        Formatted header string
    """
    border = "-" * 80
    return f"\n{border}\n{title}\n{border}"


def format_dataframe_info(df: pd.DataFrame, filename: str = "", date_str: str = "") -> str:
    """
    Format DataFrame info for logging.

    Args:
        df: DataFrame to describe
        filename: Optional filename
        date_str: Optional date string

    Returns:
        Formatted info string
    """
    buffer = StringIO()
    df.info(buf=buffer, memory_usage='deep')
    info_str = buffer.getvalue()

    # Calculate memory in MB
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 * 1024)

    title = f"DataFrame Info"
    if filename:
        title += f" - {filename}"
    if date_str:
        title += f" - {date_str}"

    header = format_section_header(title)

    result = [
        header,
        f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
        f"Memory: {memory_mb:.2f} MB",
        "",
        "Column Details:",
    ]

    # Add column info in a cleaner format
    for col in df.columns[:20]:  # Show first 20 columns
        non_null = df[col].count()
        total = len(df)
        dtype = df[col].dtype
        result.append(f"  {col:<30} {non_null:>6}/{total:<6} non-null  ({dtype})")

    if len(df.columns) > 20:
        result.append(f"  ... and {len(df.columns) - 20} more columns")

    return "\n".join(result)


def check_and_rotate_logs(log_file: Path):
    """
    Check if log rotation is needed and rotate if necessary.
    Keeps last N runs based on LOG_ROTATION_RUNS.

    Args:
        log_file: Path to log file to check
    """
    if not log_file.exists():
        return

    # Count run markers in log file
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            run_count = content.count('PIPELINE RUN #')

        # If we have more than LOG_ROTATION_RUNS, archive oldest runs
        if run_count > LOG_ROTATION_RUNS:
            # Split content by run markers
            runs = content.split('╔═')

            # Keep header (before first run) + last N runs
            if len(runs) > LOG_ROTATION_RUNS + 1:
                # Archive old runs
                archive_name = f"{log_file.name}.{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
                archive_path = LOG_ARCHIVE_DIR / archive_name

                # Write old runs to archive
                old_runs = '╔═'.join(runs[:-(LOG_ROTATION_RUNS)])
                with open(archive_path, 'w', encoding='utf-8') as f:
                    f.write(old_runs)

                # Keep only recent runs in main log
                recent_runs = '╔═'.join(runs[-(LOG_ROTATION_RUNS):])
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(recent_runs)

    except Exception as e:
        # If rotation fails, log to stderr but don't stop pipeline
        print(f"Warning: Log rotation failed for {log_file}: {e}", file=sys.stderr)


def log_with_timestamp(logger: logging.Logger, level: str, message: str):
    """
    Log message with clean timestamp format.

    Args:
        logger: Logger instance
        level: Log level (INFO, WARNING, ERROR)
        message: Message to log
    """
    timestamp = datetime.now().strftime("[%H:%M:%S]")

    level_icons = {
        'INFO': 'ℹ️',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'SUCCESS': '✅'
    }

    icon = level_icons.get(level, '')
    formatted_msg = f"{timestamp} {icon} {message}"

    if level == 'ERROR':
        logger.error(formatted_msg)
    elif level == 'WARNING':
        logger.warning(formatted_msg)
    else:
        logger.info(formatted_msg)

