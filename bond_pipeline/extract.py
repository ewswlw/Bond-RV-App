"""
Extract module for reading Excel files.
Handles file reading, date extraction, and initial data loading.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .config import (
    BQL_EXCEL_PATH,
    BQL_HEADER_LABEL,
    BQL_SHEET_NAME,
    DATE_COLUMN,
    HEADER_ROW,
)
from .utils import extract_date_from_filename, format_dataframe_info, format_date_string, setup_logging


class ExcelExtractor:
    """Extract data from Excel files."""
    
    def __init__(self, log_file: Path):
        """
        Initialize extractor with logging.

        Args:
            log_file: Path to log file
        """
        # Suppress console output - only write to file
        self.logger = setup_logging(log_file, 'extract', console_level=logging.CRITICAL)
    
    def read_excel_file(self, file_path: Path) -> Optional[Tuple[pd.DataFrame, datetime]]:
        """
        Read Excel file and extract date from filename.
        
        Args:
            file_path: Path to Excel file
        
        Returns:
            Tuple of (DataFrame, date) or None if failed
        """
        filename = file_path.name
        
        # Extract date from filename
        date_obj = extract_date_from_filename(filename)
        
        if date_obj is None:
            self.logger.error(f"Could not extract date from filename: {filename}")
            return None
        
        self.logger.info(f"Processing file: {filename} | Date: {format_date_string(date_obj)}")
        
        # Read Excel file
        try:
            df = pd.read_excel(file_path, header=HEADER_ROW)
            self.logger.info(f"  Read {len(df)} rows, {len(df.columns)} columns")

            # Log DataFrame info
            df_info = format_dataframe_info(df, filename, format_date_string(date_obj))
            self.logger.info(df_info)

            # Add Date column as first column (as datetime object)
            df.insert(0, DATE_COLUMN, date_obj)

            return (df, date_obj)

        except Exception as e:
            self.logger.error(f"  Error reading file: {str(e)}")
            return None
    
    def extract_all_files(self, file_paths: list) -> dict:
        """
        Extract data from all Excel files.
        
        Args:
            file_paths: List of Path objects to Excel files
        
        Returns:
            Dictionary mapping date to DataFrame
        """
        results = {}
        
        self.logger.info(f"Starting extraction of {len(file_paths)} files")
        self.logger.info("=" * 80)
        
        for file_path in file_paths:
            result = self.read_excel_file(file_path)
            
            if result is not None:
                df, date_obj = result
                results[date_obj] = df
        
        self.logger.info("=" * 80)
        self.logger.info(f"Successfully extracted {len(results)} files")
        
        return results

    def read_bql_workbook(self, workbook_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """
        Read the BQL Excel workbook and return the raw wide DataFrame.
        
        The workbook has a 4-row multi-index header:
        - Row 0 (index 0): Ignored
        - Row 1 (index 1): Name (1st level)
        - Row 2 (index 2): CUSIP (2nd level)
        - Row 3 (index 3): Ignored
        - Row 4 (index 4): Data starts here

        Args:
            workbook_path: Optional explicit workbook path. Defaults to config path.

        Returns:
            DataFrame containing the raw BQL data with multi-index columns, or None if read fails.
        """
        path = workbook_path or BQL_EXCEL_PATH
        self.logger.info(f"Processing BQL workbook: {path}")

        if not path.exists():
            self.logger.error(f"BQL workbook does not exist: {path}")
            return None

        try:
            # Read with 4-row multi-index header (rows 0-3)
            df = pd.read_excel(path, sheet_name=BQL_SHEET_NAME, header=[0, 1, 2, 3])
        except Exception as exc:
            self.logger.error(f"Failed to read BQL workbook: {exc}")
            return None

        if df.empty:
            self.logger.warning("BQL workbook is empty.")
            return df

        # Check first column label (should be "CUSIPs" at some level)
        first_column_label = df.columns[0]
        first_col_str = str(first_column_label[0] if isinstance(first_column_label, tuple) else first_column_label).strip()
        if first_col_str != BQL_HEADER_LABEL:
            self.logger.warning(
                "Unexpected BQL header label. Expected '%s', found '%s' (path=%s)",
                BQL_HEADER_LABEL,
                first_col_str,
                path,
            )

        self.logger.info(f"  Read BQL sheet with {len(df)} rows, {len(df.columns)} columns")
        self.logger.debug(
            "  BQL columns preview: %s",
            ", ".join(map(str, df.columns[: min(10, len(df.columns))])),
        )

        return df

