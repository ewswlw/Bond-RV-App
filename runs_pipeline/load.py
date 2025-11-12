"""
Load module for writing RUNS data to parquet files.
Handles append and override modes with Date+Dealer+CUSIP primary key.
"""

import pandas as pd
from dataclasses import dataclass, field
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

# Allowed dealers for runs_timeseries.parquet
ALLOWED_DEALERS = ['BMO', 'BNS', 'NBF', 'RBC', 'TD']


@dataclass
class LoadResult:
    """Structured response for parquet load operations."""

    success: bool
    new_rows: int = 0
    skipped_rows: int = 0
    existing_rows: int = 0
    total_rows: int = 0
    new_dates: Set[datetime] = field(default_factory=set)
    new_cusips: int = 0
    new_dealers: int = 0


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
    
    def load_append(
        self,
        data: pd.DataFrame,
        existing_dates: Optional[Set[datetime]] = None,
    ) -> LoadResult:
        """
        Append new data to runs_timeseries.parquet.
        Skip dates already captured in the existing parquet file.

        Args:
            data: DataFrame with new data to append.
            existing_dates: Optional pre-fetched set of dates already present.

        Returns:
            LoadResult containing success flag and metrics.
        """
        if data is None or len(data) == 0:
            self.logger.warning("No data to append")
            return LoadResult(success=True)

        original_row_count = len(data)
        dates_to_compare = (
            existing_dates if existing_dates is not None else self.get_existing_dates()
        )

        filtered_data = data
        skipped_rows = 0

        if dates_to_compare and 'Date' in data.columns:
            filtered_data = data[~data['Date'].isin(dates_to_compare)].copy()
            skipped_rows = original_row_count - len(filtered_data)

            if skipped_rows > 0:
                self.logger.info(
                    f"Filtered out {skipped_rows} rows with dates already in parquet"
                )

        if len(filtered_data) == 0:
            self.logger.info(
                "All candidate rows already exist in parquet; nothing to append"
            )
            return LoadResult(
                success=True,
                new_rows=0,
                skipped_rows=skipped_rows,
                new_dates=set(),
            )

        if not self.validate_primary_key(filtered_data):
            return LoadResult(success=False, skipped_rows=skipped_rows)

        new_dates = self._normalize_dates(filtered_data.get('Date'))
        new_cusips = (
            filtered_data['CUSIP'].nunique()
            if 'CUSIP' in filtered_data.columns
            else 0
        )
        new_dealers = (
            filtered_data['Dealer'].nunique()
            if 'Dealer' in filtered_data.columns
            else 0
        )

        try:
            existing_rows = 0
            total_rows = len(filtered_data)

            if bond_config.RUNS_PARQUET.exists():
                existing_df = pd.read_parquet(bond_config.RUNS_PARQUET)
                
                # Replace SCM with BNS in existing data
                if 'Dealer' in existing_df.columns:
                    scm_count = (existing_df['Dealer'].astype(str) == 'SCM').sum()
                    if scm_count > 0:
                        self.logger.info(
                            f"Replacing {scm_count} instances of 'SCM' with 'BNS' in existing data"
                        )
                        existing_df = existing_df.copy()
                        existing_df['Dealer'] = existing_df['Dealer'].astype(str).replace('SCM', 'BNS')
                
                # Filter existing data to only allowed dealers
                if 'Dealer' in existing_df.columns:
                    before_filter = len(existing_df)
                    existing_df = existing_df[existing_df['Dealer'].isin(ALLOWED_DEALERS)].copy()
                    filtered_existing = before_filter - len(existing_df)
                    if filtered_existing > 0:
                        self.logger.info(
                            f"Filtered out {filtered_existing} existing rows from dealers not in {ALLOWED_DEALERS}"
                        )
                
                existing_rows = len(existing_df)
                self.logger.info(
                    f"Loaded existing parquet: {existing_rows} rows (after dealer filter)"
                )

                existing_dupes = existing_df.duplicated(
                    subset=RUNS_PRIMARY_KEY, keep=False
                )
                if existing_dupes.any():
                    self.logger.error(
                        "Existing parquet has duplicate Date+Dealer+CUSIP "
                        "combinations; aborting append."
                    )
                    return LoadResult(success=False, skipped_rows=skipped_rows)

                combined_df = pd.concat(
                    [existing_df, filtered_data],
                    ignore_index=True,
                )

                if not self.validate_primary_key(combined_df):
                    return LoadResult(success=False, skipped_rows=skipped_rows)

                total_rows = len(combined_df)
                self.logger.info(
                    f"Combined total rows after append: {total_rows} "
                    f"({existing_rows} existing + {len(filtered_data)} new)"
                )

                write_df = combined_df
            else:
                write_df = filtered_data
                self.logger.info(
                    f"Creating new runs parquet with {len(filtered_data)} rows"
                )

            write_df = write_df.copy()
            
            # Replace SCM with BNS before filtering (in case any SCM values remain)
            if 'Dealer' in write_df.columns:
                scm_count = (write_df['Dealer'].astype(str) == 'SCM').sum()
                if scm_count > 0:
                    self.logger.info(
                        f"Replacing {scm_count} instances of 'SCM' with 'BNS' before writing"
                    )
                    write_df['Dealer'] = write_df['Dealer'].astype(str).replace('SCM', 'BNS')
            
            # Filter to only allowed dealers
            if 'Dealer' in write_df.columns:
                before_filter = len(write_df)
                write_df = write_df[write_df['Dealer'].isin(ALLOWED_DEALERS)].copy()
                filtered_out = before_filter - len(write_df)
                if filtered_out > 0:
                    self.logger.info(
                        f"Filtered out {filtered_out} rows from dealers not in {ALLOWED_DEALERS}"
                    )
            
            if 'Time' in write_df.columns:
                write_df['Time'] = write_df['Time'].apply(
                    lambda x: x.strftime('%H:%M')
                    if x is not None and hasattr(x, 'strftime')
                    else x
                )

            write_df.to_parquet(
                bond_config.RUNS_PARQUET,
                index=False,
                engine='pyarrow',
            )
            self.logger.info(
                f"Successfully wrote runs parquet: {bond_config.RUNS_PARQUET}"
            )

            return LoadResult(
                success=True,
                new_rows=len(filtered_data),
                skipped_rows=skipped_rows,
                existing_rows=existing_rows,
                total_rows=total_rows,
                new_dates=new_dates,
                new_cusips=new_cusips,
                new_dealers=new_dealers,
            )

        except Exception as exc:
            self.logger.error(
                f"Error writing runs parquet in append mode: {exc}",
                exc_info=True,
            )
            return LoadResult(success=False, skipped_rows=skipped_rows)
    
    def load_override(self, data: pd.DataFrame) -> LoadResult:
        """
        Override runs_timeseries.parquet with all data.
        Delete existing file and write new one.
        
        Args:
            data: DataFrame with all data to write
        
        Returns:
            LoadResult containing success flag and metrics.
        """
        if data is None or len(data) == 0:
            self.logger.error("No data to write in override mode")
            return LoadResult(success=False)

        if not self.validate_primary_key(data):
            return LoadResult(success=False)

        self.logger.info(
            f"Override mode: Writing {len(data)} rows to runs_timeseries.parquet"
        )

        new_dates = self._normalize_dates(data.get('Date'))
        new_cusips = data['CUSIP'].nunique() if 'CUSIP' in data.columns else 0
        new_dealers = data['Dealer'].nunique() if 'Dealer' in data.columns else 0

        try:
            if bond_config.RUNS_PARQUET.exists():
                bond_config.RUNS_PARQUET.unlink()
                self.logger.info("Deleted existing runs parquet before overwrite")

            data_write = data.copy()
            
            # Replace SCM with BNS before filtering
            if 'Dealer' in data_write.columns:
                scm_count = (data_write['Dealer'].astype(str) == 'SCM').sum()
                if scm_count > 0:
                    self.logger.info(
                        f"Replacing {scm_count} instances of 'SCM' with 'BNS' before writing"
                    )
                    data_write['Dealer'] = data_write['Dealer'].astype(str).replace('SCM', 'BNS')
            
            # Filter to only allowed dealers
            if 'Dealer' in data_write.columns:
                before_filter = len(data_write)
                data_write = data_write[data_write['Dealer'].isin(ALLOWED_DEALERS)].copy()
                filtered_out = before_filter - len(data_write)
                if filtered_out > 0:
                    self.logger.info(
                        f"Filtered out {filtered_out} rows from dealers not in {ALLOWED_DEALERS}"
                    )
            
            if 'Time' in data_write.columns:
                data_write['Time'] = data_write['Time'].apply(
                    lambda x: x.strftime('%H:%M')
                    if x is not None and hasattr(x, 'strftime')
                    else x
                )

            data_write.to_parquet(
                bond_config.RUNS_PARQUET,
                index=False,
                engine='pyarrow',
            )
            self.logger.info(
                f"Successfully wrote override parquet: {bond_config.RUNS_PARQUET}"
            )

            return LoadResult(
                success=True,
                new_rows=len(data),
                skipped_rows=0,
                existing_rows=0,
                total_rows=len(data),
                new_dates=new_dates,
                new_cusips=new_cusips,
                new_dealers=new_dealers,
            )

        except Exception as exc:
            self.logger.error(
                f"Error writing runs parquet in override mode: {exc}",
                exc_info=True,
            )
            return LoadResult(success=False)
    
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
                    stats['runs_date_range'] = (min_date, max_date)
                    stats['date_range'] = (min_date, max_date)
                    stats['runs_min_date'] = min_date
                    stats['runs_max_date'] = max_date
            
        except Exception as e:
            self.logger.error(f"Error reading parquet stats: {str(e)}")
        
        return stats

