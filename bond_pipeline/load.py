"""
Load module for writing data to parquet files.
Handles append and override modes.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

from .config import (
    BQL_CUSIP_COLUMN_NAME,
    BQL_DATE_COLUMN_NAME,
    BQL_NAME_COLUMN_NAME,
    BQL_PARQUET,
    BQL_VALUE_COLUMN_NAME,
    HISTORICAL_PARQUET,
    UNIVERSE_PARQUET,
    DATE_COLUMN
)
from .utils import setup_logging, format_date_string, format_section_header


class ParquetLoader:
    """Load data to parquet files."""
    
    def __init__(self, log_file: Path):
        """
        Initialize loader with logging.

        Args:
            log_file: Path to log file
        """
        # Suppress console output - only write to file
        self.logger = setup_logging(log_file, 'load', console_level=logging.CRITICAL)
    
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
            
            # Convert any string dates back to datetime if needed (backward compatibility)
            converted_dates = set()
            for date_val in existing_dates:
                if pd.isna(date_val):
                    continue
                if isinstance(date_val, str):
                    # Try to parse MMDDYYYY format
                    from .utils import parse_mmddyyyy
                    parsed = parse_mmddyyyy(date_val)
                    if parsed:
                        converted_dates.add(parsed)
                    else:
                        # Keep as-is if can't parse
                        converted_dates.add(date_val)
                else:
                    converted_dates.add(date_val)
            
            self.logger.info(
                f"Found {len(converted_dates)} existing dates in historical parquet"
            )
            
            return converted_dates
            
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
        
        # Filter out empty DataFrames before concatenation
        non_empty_dfs = [df for df in dates_to_add.values() if not df.empty and len(df) > 0]
        
        if not non_empty_dfs:
            self.logger.info("No non-empty data to append")
            return True
        
        # Combine new data
        new_df = pd.concat(non_empty_dfs, ignore_index=True)

        # Convert numeric columns to float64 FIRST (before general object conversion)
        years_columns = ['Yrs Since Issue', 'Yrs (Worst)', 'Yrs (Cvn)']
        spread_metric_columns = [
            'G Sprd',
            'vs BI',
            'vs BCE',
            'MTD Equity',
            'YTD Equity',
            'Retracement',
            'Z Score',
            'Retracement2',
        ]
        all_numeric_columns = years_columns + spread_metric_columns
        
        for col in all_numeric_columns:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                self.logger.info(f"Converted new data '{col}' to numeric in append mode")

        self.logger.info(
            f"Appending {len(dates_to_add)} new dates with {len(new_df)} total rows"
        )
        
        # Append to existing or create new
        try:
            if HISTORICAL_PARQUET.exists():
                # Read existing
                existing_df = pd.read_parquet(HISTORICAL_PARQUET)
                self.logger.info(f"Loaded existing parquet: {len(existing_df)} rows")
                
                # Convert Date column to datetime if needed (backward compatibility)
                if DATE_COLUMN in existing_df.columns:
                    if existing_df[DATE_COLUMN].dtype == 'object':
                        # Try to convert string dates to datetime
                        from .utils import parse_mmddyyyy
                        existing_df[DATE_COLUMN] = existing_df[DATE_COLUMN].apply(
                            lambda x: parse_mmddyyyy(x) if pd.notna(x) and isinstance(x, str) else x
                        )
                        # Remove any None values (unparseable strings)
                        existing_df = existing_df[existing_df[DATE_COLUMN].notna()]
                
                # Convert numeric columns if they exist (for compatibility with new numeric format)
                for col in all_numeric_columns:
                    if col in existing_df.columns:
                        if existing_df[col].dtype == 'object':
                            existing_df[col] = pd.to_numeric(existing_df[col], errors='coerce')
                            self.logger.info(f"Converted existing '{col}' to numeric for compatibility")
                
                # Align column types between existing_df and new_df before concatenation
                # This prevents PyArrow conversion errors when object columns have mixed types
                common_columns = set(existing_df.columns) & set(new_df.columns)
                
                # First pass: Detect and convert numeric columns in existing_df, then align new_df
                # This handles columns like "DoD SprdB" that might have mixed types
                numeric_columns = set()
                for col in common_columns:
                    existing_dtype = existing_df[col].dtype
                    
                    # If existing is object, check if it contains numeric values
                    if existing_dtype == 'object':
                        # Check if existing column has numeric values
                        sample_values = existing_df[col].dropna().head(100)
                        if len(sample_values) > 0:
                            # Try to convert a sample to see if it's numeric
                            try:
                                numeric_sample = pd.to_numeric(sample_values, errors='coerce')
                                # If most values convert successfully, treat as numeric
                                if numeric_sample.notna().sum() / len(sample_values) > 0.5:
                                    numeric_columns.add(col)
                                    # Convert existing column to numeric
                                    existing_df[col] = pd.to_numeric(existing_df[col], errors='coerce')
                                    # Convert new_df column to numeric (handles both object and string)
                                    new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                                    self.logger.debug(f"Aligned '{col}': converted both to numeric (detected numeric values)")
                            except (ValueError, TypeError):
                                pass
                    
                    # If existing is already numeric, ensure new_df is also numeric
                    elif pd.api.types.is_numeric_dtype(existing_dtype):
                        if new_df[col].dtype != existing_dtype:
                            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                            self.logger.debug(f"Aligned '{col}': converted new_df to numeric to match existing")
                
                # Second pass: Convert remaining object columns to strings for consistency
                for col in new_df.columns:
                    if col not in all_numeric_columns:
                        if new_df[col].dtype == 'object':
                            new_df[col] = new_df[col].astype(str).replace(['nan', '<NA>', 'None'], pd.NA)
                
                for col in existing_df.columns:
                    if col not in all_numeric_columns:
                        if existing_df[col].dtype == 'object':
                            existing_df[col] = existing_df[col].astype(str).replace(['nan', '<NA>', 'None'], pd.NA)
                
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
        
        # Filter out empty DataFrames before concatenation
        non_empty_dfs = [df for df in all_data.values() if not df.empty and len(df) > 0]
        
        if not non_empty_dfs:
            self.logger.error("No non-empty data to write")
            return False
        
        # Combine all data
        combined_df = pd.concat(non_empty_dfs, ignore_index=True)
        
        # Convert numeric columns to float64 FIRST (before general object conversion)
        years_columns = ['Yrs Since Issue', 'Yrs (Worst)', 'Yrs (Cvn)']
        spread_metric_columns = [
            'G Sprd',
            'vs BI',
            'vs BCE',
            'MTD Equity',
            'YTD Equity',
            'Retracement',
            'Z Score',
            'Retracement2',
        ]
        all_numeric_columns = years_columns + spread_metric_columns
        
        for col in all_numeric_columns:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                self.logger.info(f"Converted '{col}' to numeric in override mode")
        
        # Convert all other object columns to string to avoid type inference issues
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
                # Handle date range (support both datetime64 and string dates)
                date_col = df_hist[DATE_COLUMN]
                if date_col.dtype == 'object':
                    # If string format, parse to datetime for min/max
                    from .utils import parse_mmddyyyy
                    date_series = date_col.apply(
                        lambda x: parse_mmddyyyy(x) if pd.notna(x) and isinstance(x, str) else None
                    )
                    date_min = date_series.min()
                    date_max = date_series.max()
                else:
                    # If datetime64, use directly
                    date_min = date_col.min()
                    date_max = date_col.max()
                stats['date_range'] = (date_min, date_max)
            
            if UNIVERSE_PARQUET.exists():
                df_univ = pd.read_parquet(UNIVERSE_PARQUET)
                stats['universe_cusips'] = len(df_univ)
            
        except Exception as e:
            self.logger.error(f"Error reading parquet stats: {str(e)}")
        
        return stats

    def write_bql_dataset(self, bql_df: pd.DataFrame, cusip_to_name: Dict[str, str]) -> bool:
        """
        Write the BQL dataset to parquet (always overwrite) and log summary metrics.

        Args:
            bql_df: Long-form BQL DataFrame.
            cusip_to_name: Mapping of CUSIP to security name.

        Returns:
            True when the parquet write succeeds, False otherwise.
        """
        if bql_df is None or bql_df.empty:
            self.logger.error("BQL dataset is empty. Nothing to write.")
            return False

        required_columns = {
            BQL_DATE_COLUMN_NAME,
            BQL_NAME_COLUMN_NAME,
            BQL_CUSIP_COLUMN_NAME,
            BQL_VALUE_COLUMN_NAME,
        }
        if missing := required_columns - set(bql_df.columns):
            self.logger.error(
                "BQL dataset missing required columns: %s",
                ", ".join(sorted(missing)),
            )
            return False

        # Ensure value column is numeric
        bql_df = bql_df.copy()
        bql_df[BQL_VALUE_COLUMN_NAME] = pd.to_numeric(
            bql_df[BQL_VALUE_COLUMN_NAME],
            errors="coerce",
        )
        bql_df = bql_df.dropna(subset=[BQL_VALUE_COLUMN_NAME])

        if bql_df.empty:
            self.logger.error("BQL dataset contains no numeric values after cleaning.")
            return False

        total_rows = len(bql_df)
        unique_cusips = bql_df[BQL_CUSIP_COLUMN_NAME].nunique()
        unique_dates = bql_df[BQL_DATE_COLUMN_NAME].nunique()

        self.logger.info(
            "BQL dataset stats: %s rows, %s unique CUSIPs, %s unique dates",
            total_rows,
            unique_cusips,
            unique_dates,
        )

        # Determine orphan CUSIPs compared to universe
        bql_cusips = set(bql_df[BQL_CUSIP_COLUMN_NAME].unique())
        universe_cusips: Set[str] = set()
        if UNIVERSE_PARQUET.exists():
            try:
                universe_df = pd.read_parquet(UNIVERSE_PARQUET, columns=["CUSIP"])
                universe_cusips = set(
                    universe_df["CUSIP"].dropna().astype(str).str.upper().unique().tolist()
                )
            except Exception as exc:
                self.logger.error(
                    "Failed to read universe parquet for BQL orphan comparison: %s",
                    exc,
                )
        else:
            self.logger.warning(
                "Universe parquet missing at %s. Skipping orphan comparison.",
                UNIVERSE_PARQUET,
            )

        orphan_cusips = sorted(bql_cusips - universe_cusips)
        if orphan_cusips:
            self.logger.warning(
                "Found %s BQL CUSIPs missing from universe.parquet.",
                len(orphan_cusips),
            )
            for cusip in orphan_cusips[:20]:
                name = cusip_to_name.get(cusip, "").strip()
                display_name = name or "Unknown name"
                self.logger.warning("  Orphan CUSIP: %s | Name: %s", cusip, display_name)
            if len(orphan_cusips) > 20:
                self.logger.warning(
                    "  ... and %s additional orphan CUSIPs not shown.",
                    len(orphan_cusips) - 20,
                )
        else:
            self.logger.info("All BQL CUSIPs are present in universe.parquet.")

        try:
            if BQL_PARQUET.exists():
                BQL_PARQUET.unlink()
                self.logger.info("Deleted existing BQL parquet: %s", BQL_PARQUET)

            bql_df.to_parquet(BQL_PARQUET, index=False, engine="pyarrow")
            self.logger.info("BQL parquet written successfully: %s", BQL_PARQUET)
        except Exception as exc:
            self.logger.error("Failed to write BQL parquet: %s", exc)
            return False

        return True

