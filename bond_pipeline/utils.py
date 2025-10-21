"""
Utility functions for bond data pipeline.
Includes date parsing, CUSIP validation, and logging helpers.
"""

import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from config import FILE_PATTERN, CUSIP_LENGTH, NA_VALUES


def setup_logging(log_file: Path, name: str = None) -> logging.Logger:
    """
    Set up logging to both file and console.
    
    Args:
        log_file: Path to log file
        name: Logger name (optional)
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name or log_file.stem)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
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

