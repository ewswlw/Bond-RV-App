"""
Load module for writing data to parquet files.
Handles append and override modes.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Set
import logging

from config import (
    HISTORICAL_PARQUET, 
    UNIVERSE_PARQUET,
    DATE_COLUMN
)
from utils import setup_logging, format_date_string


class ParquetLoader:
    """Load data to parquet files."""
    
    def __init__(self, log_file: Path):
        """
        Initialize loader with logging.
        
        Args:
            log_file: Path to log file
        """
        self.logger = setup_logging(log_file, 'load')
    
    def get_existing_dates(self) -> Set:
        """
        Get set of dates already in historical parquet file.
        
        Returns:
            Set of datetime objects
        """
        if not HISTORICAL_PARQUET.exists():
            self.logger.info("No existing historical parquet file found")
            return set()
        
        try:
            df = pd.read_parquet(HISTORICAL_PARQUET, columns=[DATE_COLUMN])
            existing_dates = set(df[DATE_COLUMN].unique())
            
            self.logger.info(
                f"Found {len(existing_dates)} existing dates in historical parquet"
            )
            
            return existing_dates
            
        except Exception as e:
            self.logger.error(f"Error reading existing parquet: {str(e)}")
            return set()
    
    def load_historical_append(self, new_data: dict) -> bool:
        """
        Append new data to historical parquet file.
        Only adds dates not already present.
        
        Args:
            new_data: Dictionary mapping date to DataFrame
        
        Returns:
            True if successful
        """
        if not new_data:
            self.logger.warning("No new data to append")
            return True
        
        # Get existing dates
        existing_dates = self.get_existing_dates()
        
        # Filter out dates already in parquet
        dates_to_add = {
            date: df for date, df in new_data.items() 
            if date not in existing_dates
        }
        
        if not dates_to_add:
            self.logger.info("All dates already exist in parquet, nothing to append")
            return True
        
        skipped_dates = set(new_data.keys()) - set(dates_to_add.keys())
        if skipped_dates:
            skipped_str = ', '.join([format_date_string(d) for d in sorted(skipped_dates)])
            self.logger.info(f"Skipping existing dates: {skipped_str}")
        
        # Combine new data
        new_df = pd.concat(dates_to_add.values(), ignore_index=True)
        
        self.logger.info(
            f"Appending {len(dates_to_add)} new dates with {len(new_df)} total rows"
        )
        
        # Append to existing or create new
        try:
            if HISTORICAL_PARQUET.exists():
                # Read existing
                existing_df = pd.read_parquet(HISTORICAL_PARQUET)
                self.logger.info(f"Loaded existing parquet: {len(existing_df)} rows")
                
                # Combine
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # Verify no duplicate Date+CUSIP
                dupes = combined_df.duplicated(subset=[DATE_COLUMN, 'CUSIP']).sum()
                if dupes > 0:
                    self.logger.error(f"Found {dupes} duplicate Date+CUSIP combinations!")
                    return False
                
                self.logger.info(f"Combined total: {len(combined_df)} rows")
                
                # Write combined
                combined_df.to_parquet(HISTORICAL_PARQUET, index=False, engine='pyarrow')
                self.logger.info(f"Successfully appended to {HISTORICAL_PARQUET}")
                
            else:
                # Create new
                new_df.to_parquet(HISTORICAL_PARQUET, index=False, engine='pyarrow')
                self.logger.info(f"Created new parquet: {HISTORICAL_PARQUET}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing parquet: {str(e)}")
            return False
    
    def load_historical_override(self, all_data: dict) -> bool:
        """
        Override historical parquet file with all data.
        
        Args:
            all_data: Dictionary mapping date to DataFrame
        
        Returns:
            True if successful
        """
        if not all_data:
            self.logger.error("No data to write")
            return False
        
        # Combine all data
        combined_df = pd.concat(all_data.values(), ignore_index=True)
        
        # Convert all object columns to string to avoid type inference issues
        for col in combined_df.columns:
            if combined_df[col].dtype == 'object':
                combined_df[col] = combined_df[col].astype(str).replace(['nan', '<NA>', 'None'], pd.NA)
        
        self.logger.info(
            f"Override mode: Writing {len(all_data)} dates with {len(combined_df)} total rows"
        )
        
        # Verify no duplicate Date+CUSIP
        dupes = combined_df.duplicated(subset=[DATE_COLUMN, 'CUSIP']).sum()
        if dupes > 0:
            self.logger.error(f"Found {dupes} duplicate Date+CUSIP combinations!")
            return False
        
        try:
            # Delete existing if present
            if HISTORICAL_PARQUET.exists():
                HISTORICAL_PARQUET.unlink()
                self.logger.info("Deleted existing historical parquet")
            
            # Write new
            combined_df.to_parquet(HISTORICAL_PARQUET, index=False, engine='pyarrow')
            self.logger.info(f"Successfully created {HISTORICAL_PARQUET}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing parquet: {str(e)}")
            return False
    
    def load_universe(self, universe_df: pd.DataFrame) -> bool:
        """
        Write universe parquet file (always override).
        
        Args:
            universe_df: Universe DataFrame
        
        Returns:
            True if successful
        """
        if universe_df is None or len(universe_df) == 0:
            self.logger.error("No universe data to write")
            return False
        
        self.logger.info(
            f"Writing universe table: {len(universe_df)} unique CUSIPs"
        )
        
        try:
            # Delete existing if present
            if UNIVERSE_PARQUET.exists():
                UNIVERSE_PARQUET.unlink()
                self.logger.info("Deleted existing universe parquet")
            
            # Write new
            universe_df.to_parquet(UNIVERSE_PARQUET, index=False, engine='pyarrow')
            self.logger.info(f"Successfully created {UNIVERSE_PARQUET}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing universe parquet: {str(e)}")
            return False
    
    def get_summary_stats(self) -> dict:
        """
        Get summary statistics from parquet files.
        
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        try:
            if HISTORICAL_PARQUET.exists():
                df_hist = pd.read_parquet(HISTORICAL_PARQUET)
                stats['historical_rows'] = len(df_hist)
                stats['historical_dates'] = df_hist[DATE_COLUMN].nunique()
                stats['historical_cusips'] = df_hist['CUSIP'].nunique()
                stats['date_range'] = (
                    df_hist[DATE_COLUMN].min(),
                    df_hist[DATE_COLUMN].max()
                )
            
            if UNIVERSE_PARQUET.exists():
                df_univ = pd.read_parquet(UNIVERSE_PARQUET)
                stats['universe_cusips'] = len(df_univ)
            
        except Exception as e:
            self.logger.error(f"Error reading parquet stats: {str(e)}")
        
        return stats

