"""
Extract module for reading Portfolio Excel files.
Handles file reading, date extraction from filename, and initial data loading.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from bond_pipeline.config import (
    PORTFOLIO_HEADER_ROW,
    DATE_COLUMN,
)
from bond_pipeline.utils import (
    extract_portfolio_date_from_filename,
    format_dataframe_info,
    format_date_string,
    setup_logging
)


class PortfolioExtractor:
    """
    Extract data from Portfolio Excel files.
    
    Similar to ExcelExtractor but:
    - Uses Aggies MM.DD.YY.xlsx filename pattern
    - Header row is row 1 (index 0)
    - Extracts date from filename
    """
    
    def __init__(self, log_file: Path):
        """
        Initialize extractor with logging.

        Args:
            log_file: Path to log file
        """
        # Suppress console output - only write to file
        self.logger = setup_logging(
            log_file, 'portfolio_extract', console_level=logging.CRITICAL
        )
    
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
        date_obj = extract_portfolio_date_from_filename(filename)
        
        if date_obj is None:
            self.logger.error(f"Could not extract date from filename: {filename}")
            return None
        
        self.logger.info(f"Processing file: {filename} | Date: {format_date_string(date_obj)}")
        
        # Read Excel file
        try:
            df = pd.read_excel(file_path, header=PORTFOLIO_HEADER_ROW)
            self.logger.info(f"  Read {len(df)} rows, {len(df.columns)} columns")

            # Remove Unnamed columns (Unnamed: 0, Unnamed: 1, etc.)
            unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)
                self.logger.info(f"  Removed {len(unnamed_cols)} Unnamed columns: {unnamed_cols}")

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

