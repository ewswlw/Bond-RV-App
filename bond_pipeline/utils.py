"""
Utility functions for bond data pipeline.
Includes date parsing, CUSIP validation, and logging helpers.
"""

import logging
import json
import sys
import re
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import StringIO
import pandas as pd

from .config import (
    BQL_CUSIP_COLUMN_NAME,
    BQL_DATE_COLUMN_NAME,
    BQL_HEADER_LABEL,
    BQL_NAME_COLUMN_NAME,
    BQL_OUTPUT_COLUMNS,
    BQL_VALUE_COLUMN_NAME,
    CUSIP_LENGTH,
    FILE_PATTERN,
    HISTORICAL_PARQUET,
    LOG_ARCHIVE_DIR,
    LOG_METADATA_FILE,
    LOG_ROTATION_RUNS,
    LOG_FILE_PARQUET_STATS,
    NA_VALUES,
    RUNS_DATE_FORMAT,
    RUNS_KNOWN_DEALERS,
    RUNS_TIME_FORMAT,
    RUNS_VALIDATE_DATE_RANGE,
    RUNS_VALIDATE_DEALERS,
    RUNS_VALIDATE_PRICES_POSITIVE,
    RUNS_VALIDATE_SPREADS_REASONABLE,
    RUNS_VALIDATE_TIME_FORMAT,
    UNIVERSE_PARQUET,
    BQL_PARQUET,
    RUNS_PARQUET,
)


@dataclass
class BQLTransformArtifacts:
    """
    Container for BQL transformation outputs.

    Attributes:
        dataframe: Long-form DataFrame containing BQL results.
        cusip_to_name: Mapping of normalized CUSIP to security name.
        issues: Collection of warnings encountered during transformation.
    """

    dataframe: pd.DataFrame
    cusip_to_name: Dict[str, str]
    issues: List[str]


def normalize_bql_cusip(header_value: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalize a BQL column header into a 9-character CUSIP string.

    Args:
        header_value: Raw header string from the workbook.

    Returns:
        Tuple of (cleaned_cusip, error_message). cleaned_cusip is None when invalid.
    """
    if header_value is None or (isinstance(header_value, float) and pd.isna(header_value)):
        return (None, "Missing header value")

    raw_value = str(header_value).strip().upper()

    if not raw_value:
        return (None, "Empty header value")

    cleaned = re.sub(r"\s+CORP$", "", raw_value, flags=re.IGNORECASE)
    cleaned = cleaned.replace("CORP", "")
    cleaned = re.sub(r"\s+", "", cleaned)

    if len(cleaned) != CUSIP_LENGTH:
        return (None, f"Invalid CUSIP length after cleaning: {len(cleaned)}")

    if not cleaned.isalnum():
        return (None, "CUSIP contains non-alphanumeric characters after cleaning")

    return (cleaned, None)


def reshape_bql_to_long(raw_df: pd.DataFrame) -> BQLTransformArtifacts:
    """
    Convert the wide BQL workbook DataFrame into long-form layout.

    Args:
        raw_df: DataFrame loaded directly from the BQL workbook.

    Returns:
        BQLTransformArtifacts with long-form data, CUSIP-name mapping, and issues.
    """
    empty_result = BQLTransformArtifacts(
        dataframe=pd.DataFrame(columns=BQL_OUTPUT_COLUMNS),
        cusip_to_name={},
        issues=[],
    )

    if raw_df is None or raw_df.empty:
        return empty_result

    issues: List[str] = []

    header_label = str(raw_df.columns[0]).strip()
    if header_label != BQL_HEADER_LABEL:
        issues.append(
            f"Unexpected first column label: expected '{BQL_HEADER_LABEL}', found '{header_label}'",
        )

    if raw_df.shape[0] <= 1 or raw_df.shape[1] <= 1:
        issues.append("BQL DataFrame missing required rows or columns.")
        return BQLTransformArtifacts(
            dataframe=pd.DataFrame(columns=BQL_OUTPUT_COLUMNS),
            cusip_to_name={},
            issues=issues,
        )

    name_row = raw_df.iloc[0].copy()
    data_df = raw_df.iloc[1:].copy()
    data_df = data_df.reset_index(drop=True)

    original_date_column = raw_df.columns[0]
    data_df[BQL_DATE_COLUMN_NAME] = pd.to_datetime(data_df[original_date_column], errors="coerce")
    data_df = data_df.drop(columns=[original_date_column])
    data_df = data_df.dropna(subset=[BQL_DATE_COLUMN_NAME])
    data_df[BQL_DATE_COLUMN_NAME] = data_df[BQL_DATE_COLUMN_NAME].dt.normalize()

    if data_df.empty:
        issues.append("No valid dates found in BQL data after parsing.")
        return BQLTransformArtifacts(
            dataframe=pd.DataFrame(columns=BQL_OUTPUT_COLUMNS),
            cusip_to_name={},
            issues=issues,
        )

    long_frames: List[pd.DataFrame] = []
    cusip_to_name: Dict[str, str] = {}
    seen_cusips: set[str] = set()

    for column_label in raw_df.columns[1:]:
        cleaned_cusip, error = normalize_bql_cusip(column_label)
        if error:
            issues.append(f"Skipped column '{column_label}': {error}")
            continue

        if cleaned_cusip in seen_cusips:
            issues.append(f"Duplicate CUSIP column encountered: {cleaned_cusip}. Skipping duplicate.")
            continue

        seen_cusips.add(cleaned_cusip)

        security_name_value = name_row[column_label] if column_label in name_row.index else ""
        security_name = str(security_name_value).strip() if pd.notna(security_name_value) else ""

        if column_label not in data_df.columns:
            issues.append(f"BQL data missing expected column '{column_label}'. Skipping.")
            continue

        value_series = pd.to_numeric(data_df[column_label], errors="coerce")
        if value_series.notna().sum() == 0:
            issues.append(f"Column '{column_label}' contains no numeric values. Skipping.")
            continue

        cusip_to_name[cleaned_cusip] = security_name

        temp_df = pd.DataFrame({
            BQL_DATE_COLUMN_NAME: data_df[BQL_DATE_COLUMN_NAME],
            BQL_NAME_COLUMN_NAME: security_name,
            BQL_CUSIP_COLUMN_NAME: cleaned_cusip,
            BQL_VALUE_COLUMN_NAME: value_series,
        })

        temp_df = temp_df.dropna(subset=[BQL_VALUE_COLUMN_NAME])

        if not temp_df.empty:
            long_frames.append(temp_df)

    if not long_frames:
        issues.append("No usable value columns found in BQL workbook.")
        return BQLTransformArtifacts(
            dataframe=pd.DataFrame(columns=BQL_OUTPUT_COLUMNS),
            cusip_to_name=cusip_to_name,
            issues=issues,
        )

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df = long_df[BQL_OUTPUT_COLUMNS]
    long_df.sort_values([BQL_DATE_COLUMN_NAME, BQL_CUSIP_COLUMN_NAME], inplace=True)
    long_df.reset_index(drop=True, inplace=True)

    return BQLTransformArtifacts(
        dataframe=long_df,
        cusip_to_name=cusip_to_name,
        issues=issues,
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


def parse_mmddyyyy(date_str: str) -> Optional[datetime]:
    """
    Parse MMDDYYYY string format to datetime object.
    
    Args:
        date_str: Date string in MMDDYYYY format
    
    Returns:
        datetime object or None if invalid
    
    Examples:
        '08242025' -> datetime(2025, 8, 24)
        '01052023' -> datetime(2023, 1, 5)
    """
    try:
        if len(date_str) != 8:
            return None
        month = int(date_str[0:2])
        day = int(date_str[2:4])
        year = int(date_str[4:8])
        return datetime(year, month, day)
    except (ValueError, IndexError):
        return None


def format_date_string(date_obj) -> str:
    """
    Format datetime object to display string (MM/DD/YYYY).
    
    Args:
        date_obj: datetime object
    
    Returns:
        Formatted date string (MM/DD/YYYY) for display/logging
    
    Examples:
        datetime(2025, 8, 24) -> '08/24/2025'
        datetime(2023, 1, 5) -> '01/05/2023'
    """
    if isinstance(date_obj, datetime):
        return date_obj.strftime('%m/%d/%Y')
    elif isinstance(date_obj, str):
        # Try to parse if it's a string (for backward compatibility)
        try:
            # Try parsing as datetime first
            if hasattr(date_obj, 'strftime'):
                return date_obj.strftime('%m/%d/%Y')
            # Try parsing MMDDYYYY format
            parsed = parse_mmddyyyy(date_obj)
            if parsed:
                return parsed.strftime('%m/%d/%Y')
        except:
            pass
        return str(date_obj)
    else:
        return str(date_obj)


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
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
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


def log_parquet_diagnostics(log_file: Optional[Path] = None) -> None:
    """
    Capture diagnostics for core parquet datasets and append them to a log file.

    Args:
        log_file: Optional path override for the parquet diagnostics log.
    """
    target_log = log_file or LOG_FILE_PARQUET_STATS
    logger = setup_logging(
        target_log,
        name='parquet_stats',
        console_level=logging.CRITICAL,
    )

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info("=" * 80)
    logger.info(f"PARQUET DATASET DIAGNOSTICS @ {timestamp}")

    datasets = [
        ("historical_bond_details.parquet", HISTORICAL_PARQUET),
        ("universe.parquet", UNIVERSE_PARQUET),
        ("bql.parquet", BQL_PARQUET),
        ("runs_timeseries.parquet", RUNS_PARQUET),
    ]

    def _sanitize_text(text: str) -> str:
        """Ensure log output stays ASCII-friendly on Windows consoles."""
        if isinstance(text, str):
            return text.encode('ascii', 'replace').decode('ascii')
        return str(text).encode('ascii', 'replace').decode('ascii')

    def _log_info(message: str) -> None:
        """Log ASCII-sanitized info messages."""
        logger.info(_sanitize_text(message))

    def _log_multiline(section_title: str, content: str) -> None:
        """Helper to log multi-line content with a section header."""
        _log_info(section_title)
        for line in content.strip().splitlines():
            _log_info(f"    {line}")

    for friendly_name, parquet_path in datasets:
        _log_info("-" * 80)
        _log_info(f"Dataset: {friendly_name}")
        _log_info(f"Path: {parquet_path}")

        if not parquet_path.exists():
            logger.warning(
                _sanitize_text(
                    f"File not found; skipping diagnostics for {friendly_name}"
                )
            )
            continue

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as exc:
            logger.error(
                _sanitize_text(
                    f"Failed to read parquet '{friendly_name}': {exc}"
                )
            )
            continue

        row_count = len(df)
        column_count = len(df.columns)
        _log_info(
            f"Shape: {row_count:,} rows x {column_count} columns"
        )

        info_buffer = StringIO()
        df.info(buf=info_buffer)
        _log_multiline("DataFrame info():", info_buffer.getvalue())

        try:
            describe_df = df.describe(include='all', datetime_is_numeric=True)
        except TypeError:
            describe_df = df.describe(include='all')

        if not describe_df.empty:
            _log_multiline(
                "DataFrame describe():",
                describe_df.to_string(),
            )
        else:
            _log_info("DataFrame describe(): <empty>")

        head_df = df.head(3)
        tail_df = df.tail(3)

        if not head_df.empty:
            _log_multiline(
                "Head (3 rows):",
                head_df.to_string(index=False),
            )
        else:
            _log_info("Head (3 rows): <empty>")

        if not tail_df.empty:
            _log_multiline(
                "Tail (3 rows):",
                tail_df.to_string(index=False),
            )
        else:
            _log_info("Tail (3 rows): <empty>")

    _log_info("=" * 80)


# ============================================================================
# RUNS Pipeline Utility Functions
# ============================================================================

def parse_runs_date(date_str) -> Optional[datetime]:
    """
    Parse MM/DD/YY string to datetime object (preserves mm/dd/yyyy format).
    
    Args:
        date_str: Date string in MM/DD/YY format (e.g., "10/31/25")
    
    Returns:
        datetime object or None if invalid
    
    Examples:
        '10/31/25' -> datetime(2025, 10, 31)
        '01/02/24' -> datetime(2024, 1, 2)
    """
    if pd.isna(date_str) or date_str == '':
        return None
    
    date_str = str(date_str).strip()
    
    try:
        # Try parsing MM/DD/YY format
        parts = date_str.split('/')
        if len(parts) != 3:
            return None
        
        month = int(parts[0])
        day = int(parts[1])
        year_str = parts[2]
        
        # Convert 2-digit year to 4-digit
        # Assume 20xx for years < 50, else 19xx
        year_int = int(year_str)
        if len(year_str) == 2:
            full_year = int(f"20{year_str}") if year_int < 50 else int(f"19{year_str}")
        else:
            full_year = year_int
        
        date_obj = datetime(full_year, month, day)
        return date_obj
    
    except (ValueError, IndexError, AttributeError):
        return None


def parse_runs_time(time_str) -> Optional[time]:
    """
    Parse HH:MM string to datetime.time object.
    
    Args:
        time_str: Time string in HH:MM format (e.g., "15:45")
    
    Returns:
        datetime.time object or None if invalid
    
    Examples:
        '15:45' -> time(15, 45)
        '08:12' -> time(8, 12)
    """
    if pd.isna(time_str) or time_str == '':
        return None
    
    time_str = str(time_str).strip()
    
    try:
        # Try parsing HH:MM format
        parts = time_str.split(':')
        if len(parts) != 2:
            return None
        
        hour = int(parts[0])
        minute = int(parts[1])
        
        # Validate hour and minute ranges
        if hour < 0 or hour > 23:
            return None
        if minute < 0 or minute > 59:
            return None
        
        time_obj = time(hour, minute)
        return time_obj
    
    except (ValueError, IndexError, AttributeError):
        return None


def check_cusip_orphans(
    runs_cusips: set,
    universe_parquet: Path,
    logger: logging.Logger,
    runs_df: Optional[pd.DataFrame] = None
) -> set:
    """
    Compare CUSIPs from runs data with universe.parquet.
    Find CUSIPs in runs but not in universe (orphans).
    
    Args:
        runs_cusips: Set of CUSIPs from runs data
        universe_parquet: Path to universe.parquet file
        logger: Logger instance for validation logging
        runs_df: Optional DataFrame with runs data (for detailed logging)
    
    Returns:
        Set of orphan CUSIPs (in runs but not in universe)
    """
    orphans = set()
    
    if not universe_parquet.exists():
        logger.warning(
            f"Universe parquet file not found: {universe_parquet}. "
            "Cannot check for orphan CUSIPs."
        )
        return orphans
    
    try:
        # Read CUSIPs from universe parquet
        universe_df = pd.read_parquet(universe_parquet, columns=['CUSIP'])
        universe_cusips = set(universe_df['CUSIP'].dropna().unique())
        
        # Find orphans (in runs but not in universe)
        orphans = runs_cusips - universe_cusips
        
        if orphans:
            logger.warning(
                f"Found {len(orphans)} orphan CUSIPs not in universe.parquet"
            )
            
            # Log orphan CUSIPs with context if DataFrame provided
            if runs_df is not None and 'CUSIP' in runs_df.columns:
                # Get sample rows for each orphan CUSIP (first occurrence per CUSIP)
                orphan_samples = []
                log_columns = ['CUSIP']
                
                # Add available columns for context
                available_cols = []
                if 'Security' in runs_df.columns:
                    available_cols.append('Security')
                if 'Date' in runs_df.columns:
                    available_cols.append('Date')
                if 'Dealer' in runs_df.columns:
                    available_cols.append('Dealer')
                if 'Time' in runs_df.columns:
                    available_cols.append('Time')
                if 'Ticker' in runs_df.columns:
                    available_cols.append('Ticker')
                
                log_columns.extend(available_cols)
                
                # Get first occurrence of each orphan CUSIP
                for cusip in sorted(list(orphans))[:20]:
                    orphan_rows = runs_df[runs_df['CUSIP'] == cusip]
                    if len(orphan_rows) > 0:
                        # Take first row for this CUSIP
                        sample_row = orphan_rows.iloc[0]
                        orphan_samples.append({
                            col: sample_row[col] if col in sample_row.index else None
                            for col in log_columns
                        })
                
                # Log orphan details
                for sample in orphan_samples:
                    cusip = sample['CUSIP']
                    details = []
                    
                    # Security (bond name)
                    security = sample.get('Security')
                    if security and pd.notna(security) and str(security).strip():
                        details.append(f"Security={str(security).strip()}")
                    
                    # Date
                    date_val = sample.get('Date')
                    if date_val and pd.notna(date_val):
                        date_str = format_date_string(date_val)
                        if date_str:
                            details.append(f"Date={date_str}")
                    
                    # Dealer
                    dealer = sample.get('Dealer')
                    if dealer and pd.notna(dealer) and str(dealer).strip():
                        details.append(f"Dealer={str(dealer).strip()}")
                    
                    # Time
                    time_val = sample.get('Time')
                    if time_val and pd.notna(time_val):
                        if hasattr(time_val, 'strftime'):
                            time_str = time_val.strftime('%H:%M')
                        else:
                            time_str = str(time_val)
                        details.append(f"Time={time_str}")
                    
                    # Ticker
                    ticker = sample.get('Ticker')
                    if ticker and pd.notna(ticker) and str(ticker).strip():
                        details.append(f"Ticker={str(ticker).strip()}")
                    
                    details_str = ", ".join(details) if details else ""
                    logger.info(f"  Orphan CUSIP: {cusip}" + (f" ({details_str})" if details_str else ""))
            else:
                # Fallback: just log CUSIPs without context
                for cusip in sorted(list(orphans))[:20]:
                    logger.info(f"  Orphan CUSIP: {cusip}")
            
            if len(orphans) > 20:
                logger.info(f"  ... and {len(orphans) - 20} more orphan CUSIPs")
        else:
            logger.info(
                f"All {len(runs_cusips)} CUSIPs from runs data found in universe.parquet"
            )
    
    except Exception as e:
        logger.error(
            f"Error checking orphan CUSIPs against universe.parquet: {str(e)}"
        )
    
    return orphans


def validate_runs_data(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    Validate data quality for runs data.
    
    Checks:
    - Date within reasonable range (1900-2100)
    - Time in valid format (HH:MM)
    - Prices positive (if not NaN)
    - Spreads reasonable (if not NaN)
    - Dealers in known list
    
    Args:
        df: DataFrame to validate
        logger: Logger instance for validation logging
    
    Returns:
        True if validation passes (with warnings), False if critical errors
    """
    is_valid = True
    warnings_count = 0
    
    # Validate dates
    if RUNS_VALIDATE_DATE_RANGE and 'Date' in df.columns:
        invalid_dates = df[df['Date'].isna()].index
        if len(invalid_dates) > 0:
            warnings_count += len(invalid_dates)
            logger.warning(f"Found {len(invalid_dates)} rows with missing Date")
        
        # Check date range
        valid_dates = df[df['Date'].notna()]['Date']
        if len(valid_dates) > 0:
            try:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                
                # Check reasonable date range (1900-2100)
                min_reasonable = datetime(1900, 1, 1)
                max_reasonable = datetime(2100, 12, 31)
                
                if min_date < min_reasonable or max_date > max_reasonable:
                    logger.warning(
                        f"Date range outside reasonable bounds: "
                        f"{min_date} to {max_date}"
                    )
                    warnings_count += 1
            except Exception as e:
                logger.warning(f"Error validating date range: {str(e)}")
                warnings_count += 1
    
    # Validate times
    if RUNS_VALIDATE_TIME_FORMAT and 'Time' in df.columns:
        invalid_times = df[df['Time'].isna()].index
        if len(invalid_times) > 0:
            warnings_count += len(invalid_times)
            logger.warning(f"Found {len(invalid_times)} rows with missing Time")
    
    # Validate prices (positive if not NaN)
    if RUNS_VALIDATE_PRICES_POSITIVE:
        price_cols = ['Bid Price', 'Ask Price']
        for col in price_cols:
            if col in df.columns:
                negative_prices = df[
                    (df[col].notna()) & (df[col] < 0)
                ].index
                
                if len(negative_prices) > 0:
                    warnings_count += len(negative_prices)
                    logger.warning(
                        f"Found {len(negative_prices)} rows with negative {col}"
                    )
    
    # Validate spreads (reasonable range if not NaN)
    if RUNS_VALIDATE_SPREADS_REASONABLE:
        spread_cols = ['Bid Spread', 'Ask Spread']
        for col in spread_cols:
            if col in df.columns:
                # Check for extremely large spreads (> 1000 bps)
                extreme_spreads = df[
                    (df[col].notna()) & (df[col] > 1000)
                ].index
                
                if len(extreme_spreads) > 0:
                    warnings_count += len(extreme_spreads)
                    logger.warning(
                        f"Found {len(extreme_spreads)} rows with extreme {col} (> 1000 bps)"
                    )
    
    # Validate dealers
    if RUNS_VALIDATE_DEALERS and 'Dealer' in df.columns:
        unknown_dealers = df[
            (df['Dealer'].notna()) & 
            (~df['Dealer'].isin(RUNS_KNOWN_DEALERS))
        ]
        
        if len(unknown_dealers) > 0:
            unique_unknown = unknown_dealers['Dealer'].unique()
            warnings_count += len(unique_unknown)
            logger.warning(
                f"Found {len(unique_unknown)} unknown dealers: "
                f"{', '.join(unique_unknown)}"
            )
    
    if warnings_count > 0:
        logger.warning(f"Data validation completed with {warnings_count} warnings")
    else:
        logger.info("Data validation passed with no warnings")
    
    return is_valid

