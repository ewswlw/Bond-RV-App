"""
Extract module for reading RUNS Excel files.
Handles file reading, date/time parsing, and initial data loading.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import logging
import re

from bond_pipeline.config import (
    RUNS_HEADER_ROW,
    RUNS_FILE_PATTERN
)
from bond_pipeline.utils import (
    setup_logging,
    parse_runs_date,
    parse_runs_time,
    format_date_string,
    format_dataframe_info
)


class RunsExtractor:
    """
    Extract data from RUNS Excel files.
    
    Similar to ExcelExtractor but:
    - Reads Date/Time from columns (not filename)
    - No header row offset needed (header is row 1)
    - Handles 30-column schema
    """
    
    def __init__(self, log_file: Path):
        """
        Initialize extractor with logging.

        Args:
            log_file: Path to log file
        """
        # Suppress console output - only write to file
        self.logger = setup_logging(
            log_file, 'runs_extract', console_level=logging.CRITICAL
        )
    
    def read_excel_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Read RUNS Excel file and parse Date/Time columns.
        
        Date and Time come from columns, not filename.
        Parses Date (MM/DD/YY) and Time (HH:MM) to datetime objects.
        
        Args:
            file_path: Path to Excel file
        
        Returns:
            DataFrame with Date and Time as datetime objects, or None if failed
        """
        filename = file_path.name
        
        # Check if file matches RUNS pattern (for logging)
        match = re.search(RUNS_FILE_PATTERN, filename, re.IGNORECASE)
        if match:
            self.logger.info(f"Processing RUNS file: {filename}")
        else:
            self.logger.warning(
                f"File {filename} does not match RUNS pattern, proceeding anyway"
            )
        
        try:
            # Read Excel file (header is row 1, 0-indexed)
            df = pd.read_excel(file_path, header=RUNS_HEADER_ROW)
            self.logger.info(
                f"  Read {len(df)} rows, {len(df.columns)} columns"
            )
            
            if len(df) == 0:
                self.logger.warning(f"  File is empty: {filename}")
                return None
            
            # Check if Date and Time columns exist
            if 'Date' not in df.columns:
                self.logger.error(f"  Date column not found in {filename}")
                return None
            
            if 'Time' not in df.columns:
                self.logger.error(f"  Time column not found in {filename}")
                return None
            
            # Parse Date column: MM/DD/YY string → datetime object
            self.logger.debug("  Parsing Date column...")
            date_parse_errors = 0
            parsed_dates = []
            
            for idx, date_val in df['Date'].items():
                parsed_date = parse_runs_date(date_val)
                if parsed_date is None:
                    date_parse_errors += 1
                    if date_parse_errors <= 5:  # Log first 5 errors
                        self.logger.warning(
                            f"  Could not parse Date: {date_val} (row {idx})"
                        )
                parsed_dates.append(parsed_date)
            
            df['Date'] = parsed_dates
            
            if date_parse_errors > 0:
                self.logger.warning(
                    f"  Failed to parse {date_parse_errors} dates"
                )
            
            # Parse Time column: HH:MM string → datetime.time object
            self.logger.debug("  Parsing Time column...")
            time_parse_errors = 0
            parsed_times = []
            
            for idx, time_val in df['Time'].items():
                parsed_time = parse_runs_time(time_val)
                if parsed_time is None:
                    time_parse_errors += 1
                    if time_parse_errors <= 5:  # Log first 5 errors
                        self.logger.warning(
                            f"  Could not parse Time: {time_val} (row {idx})"
                        )
                parsed_times.append(parsed_time)
            
            df['Time'] = parsed_times
            
            if time_parse_errors > 0:
                self.logger.warning(
                    f"  Failed to parse {time_parse_errors} times"
                )
            
            # Reorder columns: Date, Time first (preserve other column order)
            cols = df.columns.tolist()
            if 'Date' in cols:
                cols.remove('Date')
            if 'Time' in cols:
                cols.remove('Time')
            
            # Ensure Date and Time are first columns
            df = df[['Date', 'Time'] + cols]
            
            # Log DataFrame info
            df_info = format_dataframe_info(df, filename, "")
            self.logger.info(df_info)
            
            return df
            
        except Exception as e:
            self.logger.error(f"  Error reading file {filename}: {str(e)}", exc_info=True)
            return None

    def peek_file_dates(self, file_path: Path) -> set[datetime]:
        """
        Quickly inspect a RUNS Excel file to determine the set of dates it contains.

        Args:
            file_path: Path to the Excel file.

        Returns:
            Set of datetime objects representing dates present in the file.
        """
        try:
            date_series = pd.read_excel(
                file_path,
                header=RUNS_HEADER_ROW,
                usecols=['Date'],
            )['Date']
        except ValueError:
            self.logger.warning(
                f"  Unable to isolate Date column in {file_path.name}; "
                "will process file fully."
            )
            return set()
        except Exception as exc:
            self.logger.warning(
                f"  Failed to peek dates for {file_path.name}: {exc}"
            )
            return set()

        dates: set[datetime] = set()
        for value in date_series.dropna():
            parsed = parse_runs_date(value)
            if parsed:
                dates.add(parsed)

        return dates
    
    def extract_all_files(self, file_paths: List[Path]) -> List[pd.DataFrame]:
        """
        Extract data from all RUNS Excel files.
        
        Args:
            file_paths: List of Path objects to Excel files
        
        Returns:
            List of DataFrames (one per file)
        """
        extracted_data = []
        
        for file_path in file_paths:
            df = self.read_excel_file(file_path)
            if df is not None:
                extracted_data.append(df)
            else:
                self.logger.warning(
                    f"Skipping file due to extraction error: {file_path.name}"
                )
        
        self.logger.info(
            f"Successfully extracted {len(extracted_data)} out of "
            f"{len(file_paths)} files"
        )
        
        return extracted_data

