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
from bond_pipeline.config import RUNS_PRIMARY_KEY, UNIVERSE_PARQUET
from bond_pipeline.utils import (
    setup_logging,
    format_date_string
)

# Allowed dealers for runs_timeseries.parquet
ALLOWED_DEALERS = ['BMO', 'BNS', 'NBF', 'RBC', 'TD']

# Custom_Sector values to exclude from spread outlier filtering
EXCLUDED_SECTORS = ['Non Financial Hybrid', 'Non Financial Hybrids', 'Financial Hybrid', 'HY']


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
    
    def filter_outlier_spreads(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out rows with outlier negative spreads based on Date+CUSIP group statistics.
        
        For each Date+CUSIP group:
        - Calculate high and low of Bid Spread (excluding negatives/zeros/NaN)
        - Calculate high and low of Ask Spread (excluding negatives/zeros/NaN)
        - If Bid Spread < 0 and outside [low - 20, high + 20], drop the row
        - If Ask Spread < 0 and outside [low - 20, high + 20], drop the row
        - Skip this check for CUSIPs with excluded Custom_Sector values
        
        Args:
            df: DataFrame with Date, CUSIP, Bid Spread, Ask Spread columns
        
        Returns:
            DataFrame with outlier rows removed
        """
        if df is None or len(df) == 0:
            return df
        
        # Check required columns
        required_cols = ['Date', 'CUSIP']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            self.logger.warning(
                f"Cannot perform spread outlier filtering: missing columns {missing}"
            )
            return df
        
        if 'Bid Spread' not in df.columns and 'Ask Spread' not in df.columns:
            self.logger.info("No Bid Spread or Ask Spread columns found, skipping outlier filtering")
            return df
        
        initial_count = len(df)
        
        # Read universe.parquet to build CUSIP → Custom_Sector mapping
        cusip_sector_map = {}
        excluded_cusips = set()
        
        if UNIVERSE_PARQUET.exists():
            try:
                universe_df = pd.read_parquet(UNIVERSE_PARQUET, columns=['CUSIP', 'Custom_Sector'])
                # Build mapping: CUSIP -> Custom_Sector (normalize to uppercase for comparison)
                for _, row in universe_df.iterrows():
                    cusip = str(row['CUSIP']).strip().upper() if pd.notna(row['CUSIP']) else None
                    sector = str(row['Custom_Sector']).strip() if pd.notna(row['Custom_Sector']) else None
                    if cusip and sector:
                        cusip_sector_map[cusip] = sector
                        if sector in EXCLUDED_SECTORS:
                            excluded_cusips.add(cusip)
                
                self.logger.info(
                    f"Loaded {len(cusip_sector_map)} CUSIPs from universe.parquet, "
                    f"{len(excluded_cusips)} excluded from outlier filtering"
                )
            except Exception as e:
                self.logger.warning(
                    f"Error reading universe.parquet for Custom_Sector mapping: {e}. "
                    "Proceeding without sector-based exclusions."
                )
        else:
            self.logger.warning(
                "universe.parquet not found. Proceeding without sector-based exclusions."
            )
        
        # Create a copy to avoid SettingWithCopyWarning
        df_check = df.copy()
        
        # Track rows to drop
        rows_to_drop = []
        dropped_details = []
        
        # Group by Date+CUSIP
        grouped = df_check.groupby(['Date', 'CUSIP'], group_keys=False)
        
        for (date, cusip), group in grouped:
            # Skip if CUSIP is in excluded sectors (normalize to uppercase for comparison)
            cusip_str = str(cusip).strip().upper() if pd.notna(cusip) else None
            if cusip_str and cusip_str in excluded_cusips:
                continue
            
            # Calculate high/low for Bid Spread (excluding negatives, zeros, NaN)
            bid_spread_col = 'Bid Spread' if 'Bid Spread' in group.columns else None
            ask_spread_col = 'Ask Spread' if 'Ask Spread' in group.columns else None
            
            # Check each row in the group
            for idx, row in group.iterrows():
                drop_reasons = []
                
                # Calculate high/low EXCLUDING current row to avoid outlier skewing the calculation
                if bid_spread_col:
                    # Filter to positive values only, excluding current row
                    bid_valid = group[bid_spread_col].dropna()
                    bid_valid = bid_valid[bid_valid > 0]
                    # Exclude current row's value from calculation
                    if idx in bid_valid.index:
                        bid_valid = bid_valid.drop(index=idx)
                    
                    if len(bid_valid) > 0:
                        bid_high = float(bid_valid.max())
                        bid_low = float(bid_valid.min())
                        
                        # Check Bid Spread (both positive and negative outliers)
                        if pd.notna(row[bid_spread_col]):
                            bid_val = float(row[bid_spread_col])
                            # Check if outside [low - 20, high + 20] (for both positive and negative values)
                            if bid_val < (bid_low - 20) or bid_val > (bid_high + 20):
                                drop_reasons.append(
                                    f"Bid Spread={bid_val:.2f} outside range "
                                    f"[{bid_low - 20:.2f}, {bid_high + 20:.2f}] "
                                    f"(group range excluding this row: [{bid_low:.2f}, {bid_high:.2f}])"
                                )
                    else:
                        # Edge case: no valid positive spreads in group (only negatives/zeros/NaN or only current row)
                        if pd.notna(row[bid_spread_col]):
                            bid_val = float(row[bid_spread_col])
                            if bid_val < 0:
                                self.logger.warning(
                                    f"Date={format_date_string(date)}, CUSIP={cusip_str}: "
                                    f"No valid positive Bid Spread values found in group "
                                    f"(excluding current row). Bid Spread={bid_val:.2f}"
                                )
                                # Drop negative spreads when no valid high/low exists
                                drop_reasons.append(
                                    f"Bid Spread={bid_val:.2f} (no valid high/low in group)"
                                )
                
                # Calculate high/low EXCLUDING current row to avoid outlier skewing the calculation
                if ask_spread_col:
                    # Filter to positive values only, excluding current row
                    ask_valid = group[ask_spread_col].dropna()
                    ask_valid = ask_valid[ask_valid > 0]
                    # Exclude current row's value from calculation
                    if idx in ask_valid.index:
                        ask_valid = ask_valid.drop(index=idx)
                    
                    if len(ask_valid) > 0:
                        ask_high = float(ask_valid.max())
                        ask_low = float(ask_valid.min())
                        
                        # Check Ask Spread (both positive and negative outliers)
                        if pd.notna(row[ask_spread_col]):
                            ask_val = float(row[ask_spread_col])
                            # Check if outside [low - 20, high + 20] (for both positive and negative values)
                            if ask_val < (ask_low - 20) or ask_val > (ask_high + 20):
                                drop_reasons.append(
                                    f"Ask Spread={ask_val:.2f} outside range "
                                    f"[{ask_low - 20:.2f}, {ask_high + 20:.2f}] "
                                    f"(group range excluding this row: [{ask_low:.2f}, {ask_high:.2f}])"
                                )
                    else:
                        # Edge case: no valid positive spreads in group (only negatives/zeros/NaN or only current row)
                        if pd.notna(row[ask_spread_col]):
                            ask_val = float(row[ask_spread_col])
                            if ask_val < 0:
                                self.logger.warning(
                                    f"Date={format_date_string(date)}, CUSIP={cusip_str}: "
                                    f"No valid positive Ask Spread values found in group "
                                    f"(excluding current row). Ask Spread={ask_val:.2f}"
                                )
                                # Drop negative spreads when no valid high/low exists
                                drop_reasons.append(
                                    f"Ask Spread={ask_val:.2f} (no valid high/low in group)"
                                )
                
                # If any reason to drop, mark the row
                if drop_reasons:
                    rows_to_drop.append(idx)
                    dealer = row.get('Dealer', 'N/A')
                    security = row.get('Security', 'N/A')
                    time_val = row.get('Time', 'N/A')
                    
                    dropped_details.append({
                        'Date': date,
                        'CUSIP': cusip_str,
                        'Dealer': dealer,
                        'Security': security,
                        'Time': time_val,
                        'Reasons': '; '.join(drop_reasons)
                    })
        
        # Drop marked rows
        if rows_to_drop:
            df_filtered = df_check.drop(index=rows_to_drop).reset_index(drop=True)
            dropped_count = len(rows_to_drop)
            
            # Log summary
            self.logger.info(
                f"Spread outlier filtering: Dropped {dropped_count} rows "
                f"({initial_count} -> {len(df_filtered)})"
            )
            
            # Log details (limit to first 50 for readability)
            self.logger.info("Dropped rows details:")
            for i, detail in enumerate(dropped_details[:50]):
                self.logger.info(
                    f"  [{i+1}] Date={format_date_string(detail['Date'])}, "
                    f"CUSIP={detail['CUSIP']}, Dealer={detail['Dealer']}, "
                    f"Security={detail['Security']}, Time={detail['Time']}, "
                    f"Reasons: {detail['Reasons']}"
                )
            
            if len(dropped_details) > 50:
                self.logger.info(
                    f"  ... and {len(dropped_details) - 50} more rows "
                    f"(total {len(dropped_details)} dropped)"
                )
            
            return df_filtered
        else:
            self.logger.info(
                f"No outlier spreads found. All {initial_count} rows passed the check."
            )
            return df_check
    
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
                
                # Filter out negative bid/ask spreads in existing data (set to NaN) - data quality issue
                if 'Bid Spread' in existing_df.columns:
                    negative_bid_count = (existing_df['Bid Spread'] < 0).sum()
                    if negative_bid_count > 0:
                        self.logger.info(
                            f"Filtering out {negative_bid_count} negative Bid Spread values in existing data (setting to NaN)"
                        )
                        existing_df = existing_df.copy()
                        existing_df.loc[existing_df['Bid Spread'] < 0, 'Bid Spread'] = pd.NA
                
                if 'Ask Spread' in existing_df.columns:
                    negative_ask_count = (existing_df['Ask Spread'] < 0).sum()
                    if negative_ask_count > 0:
                        self.logger.info(
                            f"Filtering out {negative_ask_count} negative Ask Spread values in existing data (setting to NaN)"
                        )
                        if 'Bid Spread' not in existing_df.columns or negative_bid_count == 0:
                            existing_df = existing_df.copy()
                        existing_df.loc[existing_df['Ask Spread'] < 0, 'Ask Spread'] = pd.NA
                
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
            
            # Filter out negative bid/ask spreads (set to NaN) - data quality issue
            if 'Bid Spread' in write_df.columns:
                negative_bid_count = (write_df['Bid Spread'] < 0).sum()
                if negative_bid_count > 0:
                    self.logger.info(
                        f"Filtering out {negative_bid_count} negative Bid Spread values (setting to NaN)"
                    )
                    write_df.loc[write_df['Bid Spread'] < 0, 'Bid Spread'] = pd.NA
            
            if 'Ask Spread' in write_df.columns:
                negative_ask_count = (write_df['Ask Spread'] < 0).sum()
                if negative_ask_count > 0:
                    self.logger.info(
                        f"Filtering out {negative_ask_count} negative Ask Spread values (setting to NaN)"
                    )
                    write_df.loc[write_df['Ask Spread'] < 0, 'Ask Spread'] = pd.NA
            
            # Filter to only allowed dealers
            if 'Dealer' in write_df.columns:
                before_filter = len(write_df)
                write_df = write_df[write_df['Dealer'].isin(ALLOWED_DEALERS)].copy()
                filtered_out = before_filter - len(write_df)
                if filtered_out > 0:
                    self.logger.info(
                        f"Filtered out {filtered_out} rows from dealers not in {ALLOWED_DEALERS}"
                    )
            
            # Data quality check: Filter outlier negative spreads
            self.logger.info("Performing spread outlier filtering...")
            write_df = self.filter_outlier_spreads(write_df)
            
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
            
            # Filter out negative bid/ask spreads (set to NaN) - data quality issue
            if 'Bid Spread' in data_write.columns:
                negative_bid_count = (data_write['Bid Spread'] < 0).sum()
                if negative_bid_count > 0:
                    self.logger.info(
                        f"Filtering out {negative_bid_count} negative Bid Spread values (setting to NaN)"
                    )
                    data_write.loc[data_write['Bid Spread'] < 0, 'Bid Spread'] = pd.NA
            
            if 'Ask Spread' in data_write.columns:
                negative_ask_count = (data_write['Ask Spread'] < 0).sum()
                if negative_ask_count > 0:
                    self.logger.info(
                        f"Filtering out {negative_ask_count} negative Ask Spread values (setting to NaN)"
                    )
                    data_write.loc[data_write['Ask Spread'] < 0, 'Ask Spread'] = pd.NA
            
            # Filter to only allowed dealers
            if 'Dealer' in data_write.columns:
                before_filter = len(data_write)
                data_write = data_write[data_write['Dealer'].isin(ALLOWED_DEALERS)].copy()
                filtered_out = before_filter - len(data_write)
                if filtered_out > 0:
                    self.logger.info(
                        f"Filtered out {filtered_out} rows from dealers not in {ALLOWED_DEALERS}"
                    )
            
            # Data quality check: Filter outlier negative spreads
            self.logger.info("Performing spread outlier filtering...")
            data_write = self.filter_outlier_spreads(data_write)
            
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

