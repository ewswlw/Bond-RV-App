"""
Extract module for reading Excel files.
Handles file reading, date extraction, and initial data loading.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import logging

from config import HEADER_ROW, DATE_COLUMN
from utils import extract_date_from_filename, setup_logging, format_date_string


class ExcelExtractor:
    """Extract data from Excel files."""
    
    def __init__(self, log_file: Path):
        """
        Initialize extractor with logging.
        
        Args:
            log_file: Path to log file
        """
        self.logger = setup_logging(log_file, 'extract')
    
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
            
            # Add Date column as first column
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

