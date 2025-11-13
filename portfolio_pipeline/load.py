"""
Load module for writing Portfolio data to parquet files.
Handles append and override modes with Date+CUSIP+ACCOUNT+PORTFOLIO primary key.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Set
from datetime import datetime
import logging

from bond_pipeline.config import (
    PORTFOLIO_PARQUET,
    PORTFOLIO_PRIMARY_KEY,
    DATE_COLUMN
)
from bond_pipeline.utils import (
    setup_logging,
    format_date_string
)


class PortfolioLoader:
    """
    Load Portfolio data to historical_portfolio.parquet.
    
    Primary key: Date + CUSIP + ACCOUNT + PORTFOLIO (enforced after deduplication)
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
            log_file, 'portfolio_load', console_level=logging.CRITICAL
        )
    
    @staticmethod
    def _normalize_dates(date_series: pd.Series) -> Set[datetime]:
        """
        Normalize a pandas Series of date-like values into Python datetime objects.

        Args:
            date_series: Series containing date values.

        Returns:
            Set of normalized datetime objects.
        """
        normalized: Set[datetime] = set()

        if date_series is None:
            return normalized

        for value in pd.Series(date_series).dropna().unique():
            if isinstance(value, datetime):
                normalized.add(value)
            elif hasattr(value, 'to_pydatetime'):
                normalized.add(value.to_pydatetime())
            else:
                try:
                    normalized.add(pd.to_datetime(value).to_pydatetime())
                except Exception:
                    continue

        return normalized
    
    def get_existing_dates(self) -> Set[datetime]:
        """
        Get set of dates already in portfolio parquet file.
        
        Returns:
            Set of datetime objects (unique dates)
        """
        if not PORTFOLIO_PARQUET.exists():
            self.logger.info("No existing portfolio parquet file found")
            return set()
        
        try:
            # Read only Date column for efficiency
            df = pd.read_parquet(PORTFOLIO_PARQUET, columns=[DATE_COLUMN])
            existing_dates = set(df[DATE_COLUMN].dropna().unique())
            
            # Convert any non-datetime dates if needed
            converted_dates = self._normalize_dates(df[DATE_COLUMN])
            
            self.logger.info(
                f"Found {len(converted_dates)} existing dates in portfolio parquet"
            )
            
            return converted_dates
            
        except Exception as e:
            self.logger.error(f"Error reading existing parquet: {str(e)}")
            return set()
    
    def validate_primary_key(self, df: pd.DataFrame) -> bool:
        """
        Validate Date+CUSIP+ACCOUNT+PORTFOLIO uniqueness (primary key).
        
        Args:
            df: DataFrame to validate
        
        Returns:
            True if valid, False if duplicates found
        """
        if not all(col in df.columns for col in PORTFOLIO_PRIMARY_KEY):
            missing = [col for col in PORTFOLIO_PRIMARY_KEY if col not in df.columns]
            self.logger.error(f"Required columns missing: {missing}")
            return False
        
        dupes = df.duplicated(subset=PORTFOLIO_PRIMARY_KEY).sum()
        
        if dupes > 0:
            self.logger.error(
                f"Found {dupes} duplicate Date+CUSIP+ACCOUNT+PORTFOLIO combinations!"
            )
            return False
        
        return True
    
    def load_portfolio_append(self, new_data: dict) -> bool:
        """
        Append new data to portfolio parquet file.
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
        
        # Filter out empty DataFrames before concatenation
        non_empty_dfs = [df for df in dates_to_add.values() if not df.empty and len(df) > 0]
        
        if not non_empty_dfs:
            self.logger.info("No non-empty data to append")
            return True
        
        # Combine new data
        new_df = pd.concat(non_empty_dfs, ignore_index=True)
        
        # Validate primary key
        if not self.validate_primary_key(new_df):
            return False
        
        self.logger.info(
            f"Appending {len(dates_to_add)} new dates with {len(new_df)} total rows"
        )
        
        # Append to existing or create new
        try:
            if PORTFOLIO_PARQUET.exists():
                # Read existing
                existing_df = pd.read_parquet(PORTFOLIO_PARQUET)
                self.logger.info(f"Loaded existing parquet: {len(existing_df)} rows")
                
                # Validate existing data
                if not self.validate_primary_key(existing_df):
                    self.logger.error("Existing parquet has duplicate primary keys!")
                    return False
                
                # Combine
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # Verify no duplicate Date+CUSIP+ACCOUNT+PORTFOLIO
                dupes = combined_df.duplicated(subset=PORTFOLIO_PRIMARY_KEY).sum()
                if dupes > 0:
                    self.logger.error(f"Found {dupes} duplicate primary key combinations after combining!")
                    return False
                
                self.logger.info(f"Combined total: {len(combined_df)} rows")
                
                # Write combined
                combined_df.to_parquet(PORTFOLIO_PARQUET, index=False, engine='pyarrow')
                self.logger.info(f"Successfully appended to {PORTFOLIO_PARQUET}")
                
            else:
                # Create new
                new_df.to_parquet(PORTFOLIO_PARQUET, index=False, engine='pyarrow')
                self.logger.info(f"Created new parquet: {PORTFOLIO_PARQUET}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing parquet: {str(e)}")
            return False
    
    def load_portfolio_override(self, all_data: dict) -> bool:
        """
        Override portfolio parquet file with all data.
        
        Args:
            all_data: Dictionary mapping date to DataFrame
        
        Returns:
            True if successful
        """
        if not all_data:
            self.logger.error("No data to write")
            return False
        
        # Filter out empty DataFrames before concatenation
        non_empty_dfs = [df for df in all_data.values() if not df.empty and len(df) > 0]
        
        if not non_empty_dfs:
            self.logger.error("No non-empty data to write")
            return False
        
        # Combine all data
        combined_df = pd.concat(non_empty_dfs, ignore_index=True)
        
        # Validate primary key
        if not self.validate_primary_key(combined_df):
            return False
        
        self.logger.info(
            f"Override mode: Writing {len(all_data)} dates with {len(combined_df)} total rows"
        )
        
        try:
            # Delete existing if present
            if PORTFOLIO_PARQUET.exists():
                PORTFOLIO_PARQUET.unlink()
                self.logger.info("Deleted existing portfolio parquet")
            
            # Write new
            combined_df.to_parquet(PORTFOLIO_PARQUET, index=False, engine='pyarrow')
            self.logger.info(f"Successfully created {PORTFOLIO_PARQUET}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing parquet: {str(e)}")
            return False

