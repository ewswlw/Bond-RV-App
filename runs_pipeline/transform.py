"""
Transform module for RUNS data cleaning, normalization, and deduplication.
Handles end-of-day snapshot deduplication, CUSIP validation/orphan tracking,
and schema alignment.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

from bond_pipeline.config import (
    RUNS_PRIMARY_KEY,
    RUNS_COLUMNS,
    RUNS_LOG_INVALID_CUSIPS,
    RUNS_TRACK_ORPHAN_CUSIPS,
    UNIVERSE_PARQUET,
    CUSIP_LENGTH
)
from bond_pipeline.utils import (
    setup_logging,
    clean_na_values,
    align_to_master_schema,
    check_cusip_orphans,
    format_date_string
)


class RunsTransformer:
    """
    Transform and clean RUNS data.
    
    Key logic:
    1. End-of-day deduplication: Keep latest Time per Date+Dealer+CUSIP
    2. CUSIP validation: Check length, log invalid, but keep in data
    3. CUSIP orphan tracking: Compare with universe.parquet
    4. Schema alignment: Align to 30-column master schema
    5. Data cleaning: Replace NA values with NaN
    """
    
    def __init__(self, log_file_dupes: Path, log_file_valid: Path):
        """
        Initialize transformer with logging.

        Args:
            log_file_dupes: Path to duplicates log file
            log_file_valid: Path to validation log file
        """
        # Suppress console output - only write to files
        self.logger_dupes = setup_logging(
            log_file_dupes, 'runs_dupes', console_level=logging.CRITICAL
        )
        self.logger_valid = setup_logging(
            log_file_valid, 'runs_valid', console_level=logging.CRITICAL
        )
    
    def deduplicate_end_of_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate to keep only end-of-day snapshots.
        
        For each Date+Dealer+CUSIP group:
        - Keep row with latest Time (end-of-day snapshot)
        - If multiple rows with same latest Time, keep last row by position
        - Log all removed duplicates
        
        Args:
            df: DataFrame with Date, Time, Dealer, CUSIP columns
        
        Returns:
            Deduplicated DataFrame (one row per Date+Dealer+CUSIP)
        """
        if not all(col in df.columns for col in RUNS_PRIMARY_KEY):
            missing = [col for col in RUNS_PRIMARY_KEY if col not in df.columns]
            self.logger_dupes.error(
                f"Required columns missing for deduplication: {missing}"
            )
            return df
        
        initial_count = len(df)
        
        # Check for missing Date, Dealer, or CUSIP
        missing_mask = df['Date'].isna() | df['Dealer'].isna() | df['CUSIP'].isna()
        missing_keys = df[missing_mask]
        
        if len(missing_keys) > 0:
            self.logger_dupes.warning(
                f"Found {len(missing_keys)} rows with missing Date/Dealer/CUSIP. "
                "These will be excluded from deduplication."
            )
        
        # Separate rows with missing keys (keep as-is)
        df_valid = df[~missing_mask].copy()
        df_missing = df[missing_mask].copy()
        
        if len(df_valid) == 0:
            # All rows have missing keys, return as-is
            return df
        
        # Create index column for tie-breaking (preserve original position)
        df_valid['_original_index'] = df_valid.index
        
        # Sort by Date, Dealer, CUSIP (for grouping), then Time descending, then original index descending
        # This ensures latest Time per group, and last position tiebreaker
        df_valid_sorted = df_valid.sort_values(
            RUNS_PRIMARY_KEY + ['Time', '_original_index'],
            ascending=[True, True, True, False, False],
            na_position='last'
        )
        
        # Keep first row per Date+Dealer+CUSIP group (which is the latest Time)
        df_deduped_valid = df_valid_sorted.drop_duplicates(
            subset=RUNS_PRIMARY_KEY,
            keep='first'
        )
        
        # Remove temporary column
        df_deduped_valid = df_deduped_valid.drop(columns=['_original_index'])
        
        # Count duplicates removed
        removed_count = len(df_valid) - len(df_deduped_valid)
        
        if removed_count > 0:
            # Count unique duplicate groups for logging (vectorized)
            duplicates_mask = df_valid_sorted.duplicated(subset=RUNS_PRIMARY_KEY, keep=False)
            duplicates_count = duplicates_mask.sum()
            unique_dupe_groups = len(df_valid_sorted[duplicates_mask][RUNS_PRIMARY_KEY].drop_duplicates())
            
            self.logger_dupes.info(
                f"Found {duplicates_count} duplicate rows "
                f"({unique_dupe_groups} unique Date+Dealer+CUSIP groups)"
            )
        
        # Combine valid deduplicated rows with missing-key rows
        if len(df_missing) > 0:
            df_deduped = pd.concat([df_deduped_valid, df_missing], ignore_index=True)
        else:
            df_deduped = df_deduped_valid.reset_index(drop=True)
        
        removed_total = initial_count - len(df_deduped)
        
        if removed_total > 0:
            self.logger_dupes.info(
                f"Deduplication complete: Removed {removed_total} rows, "
                f"kept {len(df_deduped)} unique Date+Dealer+CUSIP combinations"
            )
        else:
            self.logger_dupes.info(
                f"No duplicates found. All {initial_count} rows are unique."
            )
        
        return df_deduped
    
    def validate_cusips(
        self, df: pd.DataFrame, universe_parquet: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Validate CUSIPs and track orphans vs universe.parquet.
        
        Checks:
        - CUSIP length (should be 9, but don't normalize - keep as-is)
        - Log invalid CUSIPs (wrong length, empty)
        - Track orphan CUSIPs vs universe.parquet
        - Keep invalid CUSIPs in data (just log warnings)
        
        Args:
            df: DataFrame with CUSIP column
            universe_parquet: Path to universe.parquet file (for orphan tracking)
        
        Returns:
            DataFrame with validated CUSIPs (no normalization, all kept)
        """
        if 'CUSIP' not in df.columns:
            self.logger_valid.error("CUSIP column not found in DataFrame")
            return df
        
        if RUNS_LOG_INVALID_CUSIPS:
            # Vectorized validation - much faster than row-by-row
            # Check for empty/NaN CUSIPs
            empty_mask = df['CUSIP'].isna() | (df['CUSIP'].astype(str).str.strip() == '')
            empty_count = empty_mask.sum()
            
            # Check for invalid length (vectorized)
            cusip_str = df['CUSIP'].astype(str).str.strip()
            invalid_length_mask = cusip_str.str.len() != CUSIP_LENGTH
            # Exclude empty from invalid length count (already counted)
            invalid_length_mask = invalid_length_mask & ~empty_mask
            invalid_length_count = invalid_length_mask.sum()
            
            invalid_count = empty_count + invalid_length_count
            
            if invalid_count > 0:
                # Log summary only (avoid row-by-row iteration)
                if empty_count > 0 and invalid_length_count > 0:
                    self.logger_valid.warning(
                        f"Found {invalid_count} invalid CUSIPs "
                        f"({empty_count} empty, {invalid_length_count} wrong length). "
                        "All invalid CUSIPs are kept in data."
                    )
                elif empty_count > 0:
                    self.logger_valid.warning(
                        f"Found {empty_count} empty CUSIPs. "
                        "All invalid CUSIPs are kept in data."
                    )
                else:
                    self.logger_valid.warning(
                        f"Found {invalid_length_count} CUSIPs with wrong length. "
                        "All invalid CUSIPs are kept in data."
                    )
            else:
                self.logger_valid.info(
                    f"All {len(df)} CUSIPs are valid (length {CUSIP_LENGTH})"
                )
        
        # Track orphan CUSIPs vs universe.parquet
        if RUNS_TRACK_ORPHAN_CUSIPS:
            if universe_parquet is None:
                universe_parquet = UNIVERSE_PARQUET
            
            # Get unique CUSIPs from runs data (vectorized)
            runs_cusips = set(df['CUSIP'].dropna().unique())
            
            if len(runs_cusips) > 0:
                # Check orphans (pass DataFrame for detailed logging)
                orphans = check_cusip_orphans(
                    runs_cusips, universe_parquet, self.logger_valid, runs_df=df
                )
                
                if len(orphans) > 0:
                    self.logger_valid.warning(
                        f"Found {len(orphans)} orphan CUSIPs not in universe.parquet"
                    )
        
        return df
    
    def align_to_master_schema(
        self, df: pd.DataFrame, master_schema: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Align DataFrame to 30-column master schema.
        
        Fill missing columns (e.g., old 28-column file) with NaN.
        Reorder columns to match master schema.
        
        Args:
            df: DataFrame to align
            master_schema: Master schema list (default: RUNS_COLUMNS)
        
        Returns:
            Aligned DataFrame with all master schema columns
        """
        if master_schema is None:
            master_schema = RUNS_COLUMNS
        
        # Align to master schema using utility function
        df_aligned = align_to_master_schema(df, master_schema)
        
        # Check for missing columns
        missing_cols = set(master_schema) - set(df.columns)
        if missing_cols:
            self.logger_valid.info(
                f"Added {len(missing_cols)} missing columns from master schema: "
                f"{', '.join(sorted(missing_cols))}"
            )
        
        # Ensure Date and Time are first columns
        cols = df_aligned.columns.tolist()
        if 'Date' in cols:
            cols.remove('Date')
        if 'Time' in cols:
            cols.remove('Time')
        
        df_aligned = df_aligned[['Date', 'Time'] + cols]
        
        return df_aligned
    
    def normalize_dealer_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize dealer names by replacing "SCM" with "BNS".
        
        This replacement happens before deduplication since Dealer is part
        of the primary key (Date+Dealer+CUSIP).
        
        Args:
            df: DataFrame with Dealer column
        
        Returns:
            DataFrame with normalized dealer names
        """
        if 'Dealer' not in df.columns:
            return df
        
        # Count SCM occurrences before replacement
        scm_count = (df['Dealer'].astype(str) == 'SCM').sum()
        
        if scm_count > 0:
            self.logger_valid.info(
                f"Replacing {scm_count} instances of 'SCM' with 'BNS' in Dealer column"
            )
            # Replace SCM with BNS in Dealer column
            df = df.copy()
            df['Dealer'] = df['Dealer'].astype(str).replace('SCM', 'BNS')
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by replacing NA values with NaN.
        
        Args:
            df: DataFrame to clean
        
        Returns:
            Cleaned DataFrame
        """
        return clean_na_values(df)
    
    def transform(self, df: pd.DataFrame, universe_parquet: Optional[Path] = None) -> pd.DataFrame:
        """
        Apply all transformations to DataFrame.
        
        Order:
        1. Normalize dealer names (SCM → BNS)
        2. Deduplicate (end-of-day snapshots)
        3. Clean NA values
        4. Validate CUSIPs and track orphans
        5. Align to master schema
        
        Args:
            df: DataFrame to transform
            universe_parquet: Path to universe.parquet (for orphan tracking)
        
        Returns:
            Transformed DataFrame
        """
        # Step 1: Normalize dealer names (before deduplication since Dealer is part of primary key)
        self.logger_valid.info("Normalizing dealer names...")
        df = self.normalize_dealer_names(df)
        
        # Step 2: Deduplicate (end-of-day snapshots)
        self.logger_dupes.info("Starting end-of-day deduplication...")
        df = self.deduplicate_end_of_day(df)
        
        # Step 3: Clean NA values
        self.logger_valid.info("Cleaning NA values...")
        df = self.clean_data(df)
        
        # Step 4: Validate CUSIPs and track orphans
        self.logger_valid.info("Validating CUSIPs and checking orphans...")
        df = self.validate_cusips(df, universe_parquet)
        
        # Step 5: Align to master schema
        self.logger_valid.info("Aligning to master schema...")
        df = self.align_to_master_schema(df)
        
        return df

