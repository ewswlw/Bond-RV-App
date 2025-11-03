"""
Load module for writing RUNS data to parquet files.
Handles append and override modes with Date+Dealer+CUSIP primary key.
"""

import pandas as pd
from pathlib import Path
from typing import Set, Optional
from datetime import datetime
import logging

from bond_pipeline import config as bond_config
from bond_pipeline.config import RUNS_PRIMARY_KEY
from bond_pipeline.utils import (
    setup_logging,
    format_date_string
)


class RunsLoader:
    """
    Load RUNS data to runs_timeseries.parquet.
    
    Primary key: Date + Dealer + CUSIP (enforced after deduplication)
    Supports append mode (skip existing dates) and override mode (rebuild all)
    """
    
    def __init__(self, log_file: Path):
        """
        Initialize loader with logging.

        Args:
            log_file: Path to log file
        """
        # Suppress console output - only write to file
        self.logger = setup_logging(
            log_file, 'runs_load', console_level=logging.CRITICAL
        )
    
    def get_existing_dates(self) -> Set[datetime]:
        """
        Get set of dates already in runs_timeseries.parquet.
        
        Returns:
            Set of datetime objects (unique dates)
        """
        if not bond_config.RUNS_PARQUET.exists():
            self.logger.info("No existing runs parquet file found")
            return set()
        
        try:
            # Read only Date column for efficiency
            df = pd.read_parquet(bond_config.RUNS_PARQUET, columns=['Date'])
            existing_dates = set(df['Date'].dropna().unique())
            
            # Convert any non-datetime dates if needed
            # Parquet dates come as numpy datetime64, convert to Python datetime
            converted_dates = set()
            for date_val in existing_dates:
                if pd.isna(date_val):
                    continue
                # Convert numpy datetime64 to Python datetime
                if hasattr(date_val, 'to_pydatetime'):
                    # numpy datetime64
                    converted_dates.add(date_val.to_pydatetime())
                elif isinstance(date_val, datetime):
                    converted_dates.add(date_val)
                elif isinstance(date_val, str):
                    # Try to parse if string
                    from bond_pipeline.utils import parse_runs_date
                    parsed = parse_runs_date(date_val)
                    if parsed:
                        converted_dates.add(parsed)
                    else:
                        # Keep as-is if can't parse
                        converted_dates.add(date_val)
                else:
                    # Try converting pandas Timestamp
                    if pd.api.types.is_datetime64_any_dtype(type(date_val)):
                        if hasattr(date_val, 'to_pydatetime'):
                            converted_dates.add(date_val.to_pydatetime())
                        else:
                            converted_dates.add(pd.Timestamp(date_val).to_pydatetime())
                    else:
                        converted_dates.add(date_val)
            
            self.logger.info(
                f"Found {len(converted_dates)} existing dates in runs parquet"
            )
            
            return converted_dates
            
        except Exception as e:
            self.logger.error(f"Error reading existing parquet: {str(e)}")
            return set()
    
    def validate_primary_key(self, df: pd.DataFrame) -> bool:
        """
        Validate Date+Dealer+CUSIP uniqueness (primary key).
        
        Args:
            df: DataFrame to validate
        
        Returns:
            True if valid (no duplicates), False if duplicates found
        """
        if not all(col in df.columns for col in RUNS_PRIMARY_KEY):
            missing = [col for col in RUNS_PRIMARY_KEY if col not in df.columns]
            self.logger.error(
                f"Primary key columns missing: {missing}"
            )
            return False
        
        # Check for duplicates
        duplicates = df.duplicated(subset=RUNS_PRIMARY_KEY, keep=False)
        
        if duplicates.any():
            dup_count = duplicates.sum()
            self.logger.error(
                f"Primary key violation: Found {dup_count} duplicate "
                f"Date+Dealer+CUSIP combinations!"
            )
            
            # Log sample duplicates
            dup_df = df[duplicates].copy()
            sample_dupes = dup_df.groupby(RUNS_PRIMARY_KEY).size().head(10)
            
            self.logger.error("Sample duplicate combinations:")
            for (date, dealer, cusip), count in sample_dupes.items():
                self.logger.error(
                    f"  Date={format_date_string(date)}, Dealer={dealer}, "
                    f"CUSIP={cusip}: {count} occurrences"
                )
            
            return False
        
        self.logger.info(
            f"Primary key validation passed: All {len(df)} rows are unique "
            f"(Date+Dealer+CUSIP)"
        )
        
        return True
    
    def load_append(self, data: pd.DataFrame) -> bool:
        """
        Append new data to runs_timeseries.parquet.
        Skip dates already in parquet file.
        
        Args:
            data: DataFrame with new data to append
        
        Returns:
            True if successful
        """
        if data is None or len(data) == 0:
            self.logger.warning("No data to append")
            return True
        
        # Get existing dates
        existing_dates = self.get_existing_dates()
        
        # Filter out rows with dates already in parquet
        if len(existing_dates) > 0 and 'Date' in data.columns:
            data_filtered = data[~data['Date'].isin(existing_dates)].copy()
            
            skipped_count = len(data) - len(data_filtered)
            if skipped_count > 0:
                self.logger.info(
                    f"Filtered out {skipped_count} rows with existing dates"
                )
            
            data = data_filtered
        
        if len(data) == 0:
            self.logger.info(
                "All dates already exist in parquet, nothing to append"
            )
            return True
        
        # Validate primary key on filtered data
        if not self.validate_primary_key(data):
            return False
        
        # Append to existing or create new
        try:
            if bond_config.RUNS_PARQUET.exists():
                # Read existing
                existing_df = pd.read_parquet(bond_config.RUNS_PARQUET)
                self.logger.info(
                    f"Loaded existing parquet: {len(existing_df)} rows"
                )
                
                # Validate existing data has no duplicates
                existing_dupes = existing_df.duplicated(
                    subset=RUNS_PRIMARY_KEY, keep=False
                )
                if existing_dupes.any():
                    self.logger.error(
                        f"Existing parquet has duplicate Date+Dealer+CUSIP "
                        f"combinations! Cannot append."
                    )
                    return False
                
                # Combine
                combined_df = pd.concat([existing_df, data], ignore_index=True)
                
                # Validate combined data
                if not self.validate_primary_key(combined_df):
                    return False
                
                self.logger.info(
                    f"Combined total: {len(combined_df)} rows "
                    f"({len(existing_df)} existing + {len(data)} new)"
                )
                
                # Convert Time column to string for parquet compatibility
                combined_df_write = combined_df.copy()
                if 'Time' in combined_df_write.columns:
                    combined_df_write['Time'] = combined_df_write['Time'].apply(
                        lambda x: x.strftime('%H:%M') if x is not None and hasattr(x, 'strftime') else x
                    )
                
                # Write combined
                combined_df_write.to_parquet(
                    bond_config.RUNS_PARQUET, index=False, engine='pyarrow'
                )
                self.logger.info(
                    f"Successfully appended to {bond_config.RUNS_PARQUET}"
                )
                
            else:
                # Convert Time column to string for parquet compatibility
                data_write = data.copy()
                if 'Time' in data_write.columns:
                    data_write['Time'] = data_write['Time'].apply(
                        lambda x: x.strftime('%H:%M') if x is not None and hasattr(x, 'strftime') else x
                    )
                
                # Create new
                data_write.to_parquet(bond_config.RUNS_PARQUET, index=False, engine='pyarrow')
                self.logger.info(
                    f"Created new parquet: {bond_config.RUNS_PARQUET} ({len(data)} rows)"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing parquet: {str(e)}", exc_info=True)
            return False
    
    def load_override(self, data: pd.DataFrame) -> bool:
        """
        Override runs_timeseries.parquet with all data.
        Delete existing file and write new one.
        
        Args:
            data: DataFrame with all data to write
        
        Returns:
            True if successful
        """
        if data is None or len(data) == 0:
            self.logger.error("No data to write")
            return False
        
        # Validate primary key
        if not self.validate_primary_key(data):
            return False
        
        self.logger.info(
            f"Override mode: Writing {len(data)} rows to runs_timeseries.parquet"
        )
        
        try:
            # Delete existing if present
            if bond_config.RUNS_PARQUET.exists():
                bond_config.RUNS_PARQUET.unlink()
                self.logger.info("Deleted existing runs parquet")
            
            # Convert Time column to string for parquet compatibility
            data_write = data.copy()
            if 'Time' in data_write.columns:
                data_write['Time'] = data_write['Time'].apply(
                    lambda x: x.strftime('%H:%M') if x is not None and hasattr(x, 'strftime') else x
                )
            
            # Write new
            data_write.to_parquet(bond_config.RUNS_PARQUET, index=False, engine='pyarrow')
            self.logger.info(
                f"Successfully created {bond_config.RUNS_PARQUET} ({len(data)} rows)"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing parquet: {str(e)}", exc_info=True)
            return False
    
    def get_summary_stats(self) -> dict:
        """
        Get summary statistics from runs_timeseries.parquet.
        
        Returns:
            Dictionary with statistics:
            - runs_rows: Total rows
            - runs_dates: Unique dates
            - runs_cusips: Unique CUSIPs
            - runs_dealers: Unique dealers
            - date_range: (min_date, max_date)
        """
        stats = {}
        
        if not bond_config.RUNS_PARQUET.exists():
            self.logger.warning("Runs parquet file not found, no stats available")
            return stats
        
        try:
            df = pd.read_parquet(bond_config.RUNS_PARQUET)
            
            stats['runs_rows'] = len(df)
            stats['runs_dates'] = df['Date'].nunique() if 'Date' in df.columns else 0
            stats['runs_cusips'] = df['CUSIP'].nunique() if 'CUSIP' in df.columns else 0
            stats['runs_dealers'] = (
                df['Dealer'].nunique() if 'Dealer' in df.columns else 0
            )
            
            # Date range
            if 'Date' in df.columns:
                date_col = df['Date'].dropna()
                if len(date_col) > 0:
                    min_date = date_col.min()
                    max_date = date_col.max()
                    stats['runs_date_range'] = (min_date, max_date)  # Fixed: was 'date_range'
            
        except Exception as e:
            self.logger.error(f"Error reading parquet stats: {str(e)}")
        
        return stats

