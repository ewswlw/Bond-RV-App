"""
Transform module for data cleaning, normalization, and deduplication.
Handles CUSIP validation, NA cleaning, and duplicate removal.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

from .config import DATE_COLUMN, UNIVERSE_COLUMNS, BOND_NAME_COLUMN
from .utils import (
    setup_logging,
    validate_cusip,
    clean_na_values,
    align_to_master_schema,
    format_date_string,
    format_section_header
)


class DataTransformer:
    """Transform and clean bond data."""
    
    def __init__(self, log_file_duplicates: Path, log_file_validation: Path):
        """
        Initialize transformer with logging.

        Args:
            log_file_duplicates: Path to duplicates log file
            log_file_validation: Path to validation log file
        """
        # Suppress console output - only write to files
        self.logger_dupes = setup_logging(log_file_duplicates, 'duplicates', console_level=logging.CRITICAL)
        self.logger_valid = setup_logging(log_file_validation, 'validation', console_level=logging.CRITICAL)
        self.master_schema = None
    
    def set_master_schema(self, columns: list):
        """
        Set master schema for alignment.
        
        Args:
            columns: List of column names (excluding Date)
        """
        # Master schema should include Date as first column
        if DATE_COLUMN not in columns:
            self.master_schema = [DATE_COLUMN] + columns
        else:
            self.master_schema = columns
        
        self.logger_valid.info(f"Master schema set with {len(self.master_schema)} columns")
    
    def validate_and_normalize_cusips(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and normalize CUSIPs in DataFrame.

        Args:
            df: DataFrame with CUSIP column

        Returns:
            DataFrame with normalized CUSIPs and validation flag
        """
        if 'CUSIP' not in df.columns:
            self.logger_valid.error("CUSIP column not found in DataFrame")
            return df

        # Add validation columns
        df['CUSIP_ORIGINAL'] = df['CUSIP'].copy()
        df['CUSIP_VALID'] = True
        df['CUSIP_ERROR'] = ''

        invalid_count = 0
        has_bond_name = BOND_NAME_COLUMN in df.columns

        for idx, cusip in df['CUSIP'].items():
            normalized, is_valid, error_msg = validate_cusip(cusip)

            df.at[idx, 'CUSIP'] = normalized
            df.at[idx, 'CUSIP_VALID'] = is_valid
            df.at[idx, 'CUSIP_ERROR'] = error_msg

            if not is_valid:
                invalid_count += 1

                # Get bond name if available
                bond_name = ""
                if has_bond_name and idx in df.index:
                    name_value = df.at[idx, BOND_NAME_COLUMN]
                    if pd.notna(name_value) and str(name_value) not in ['nan', '<NA>', 'None', '']:
                        bond_name = f" ({name_value})"

                self.logger_valid.warning(
                    f"Invalid CUSIP: {cusip}{bond_name} -> {normalized} | Error: {error_msg}"
                )

        if invalid_count > 0:
            self.logger_valid.warning(f"Total invalid CUSIPs: {invalid_count} out of {len(df)}")
        else:
            self.logger_valid.info(f"All {len(df)} CUSIPs are valid")

        return df
    
    def remove_duplicates(self, df: pd.DataFrame, keep: str = 'last') -> pd.DataFrame:
        """
        Remove duplicate CUSIPs within same date, keeping last occurrence.
        
        Args:
            df: DataFrame with Date and CUSIP columns
            keep: Which duplicate to keep ('first', 'last')
        
        Returns:
            Deduplicated DataFrame
        """
        if 'CUSIP' not in df.columns or DATE_COLUMN not in df.columns:
            self.logger_dupes.error("Required columns (Date, CUSIP) not found")
            return df
        
        initial_count = len(df)
        
        # Find duplicates
        duplicates = df[df.duplicated(subset=[DATE_COLUMN, 'CUSIP'], keep=False)]

        if len(duplicates) > 0:
            date_str = format_date_string(df[DATE_COLUMN].iloc[0])
            unique_dupes = duplicates['CUSIP'].nunique()

            self.logger_dupes.warning(
                f"Date {date_str}: Found {len(duplicates)} duplicate rows "
                f"({unique_dupes} unique CUSIPs)"
            )

            # Log sample duplicates with bond names if available
            sample_dupes = duplicates.groupby('CUSIP').size().head(10)

            if BOND_NAME_COLUMN in df.columns:
                # Add bond names to duplicate report
                self.logger_dupes.info("Sample duplicate CUSIPs with bond names:")
                for cusip in sample_dupes.index[:10]:
                    count = sample_dupes[cusip]
                    # Get first bond name for this CUSIP
                    bond_rows = df[df['CUSIP'] == cusip]
                    if len(bond_rows) > 0:
                        bond_name = bond_rows[BOND_NAME_COLUMN].iloc[0]
                        if pd.notna(bond_name) and str(bond_name) not in ['nan', '<NA>', 'None', '']:
                            self.logger_dupes.info(f"  {cusip} ({bond_name}): {count} occurrences")
                        else:
                            self.logger_dupes.info(f"  {cusip}: {count} occurrences")
            else:
                self.logger_dupes.info(f"Sample duplicate CUSIPs:\n{sample_dupes}")
        
        # Remove duplicates, keeping last occurrence
        df_deduped = df.drop_duplicates(subset=[DATE_COLUMN, 'CUSIP'], keep=keep)
        
        removed_count = initial_count - len(df_deduped)
        
        if removed_count > 0:
            self.logger_dupes.info(
                f"Removed {removed_count} duplicate rows, kept {len(df_deduped)} unique records"
            )
        
        return df_deduped
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by replacing NA values.
        
        Args:
            df: DataFrame to clean
        
        Returns:
            Cleaned DataFrame
        """
        return clean_na_values(df)
    
    def align_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align DataFrame to master schema.
        
        Args:
            df: DataFrame to align
        
        Returns:
            Aligned DataFrame
        """
        if self.master_schema is None:
            self.logger_valid.warning("Master schema not set, skipping alignment")
            return df
        
        return align_to_master_schema(df, self.master_schema)
    
    def transform_single_file(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformations to a single file's data.
        
        Args:
            df: DataFrame from single file
        
        Returns:
            Transformed DataFrame
        """
        # 1. Validate and normalize CUSIPs
        df = self.validate_and_normalize_cusips(df)
        
        # 2. Remove duplicates (keep last)
        df = self.remove_duplicates(df, keep='last')
        
        # 3. Clean NA values
        df = self.clean_data(df)
        
        # 4. Align to master schema
        df = self.align_schema(df)
        
        return df
    
    def create_universe_table(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create universe table from historical data.
        Keep most recent record for each unique CUSIP.
        
        Args:
            historical_df: Complete historical bond details DataFrame
        
        Returns:
            Universe DataFrame with unique CUSIPs
        """
        self.logger_valid.info("Creating universe table...")
        
        # Sort by date descending to get most recent first
        df_sorted = historical_df.sort_values(DATE_COLUMN, ascending=False)
        
        # Keep first occurrence of each CUSIP (which is the most recent)
        universe_df = df_sorted.drop_duplicates(subset=['CUSIP'], keep='first')
        
        # Select only universe columns
        available_cols = [col for col in UNIVERSE_COLUMNS if col in universe_df.columns]
        missing_cols = [col for col in UNIVERSE_COLUMNS if col not in universe_df.columns]
        
        if missing_cols:
            self.logger_valid.warning(f"Missing columns in universe: {missing_cols}")
        
        universe_df = universe_df[available_cols].copy()
        
        self.logger_valid.info(
            f"Universe table created: {len(universe_df)} unique CUSIPs, "
            f"{len(available_cols)} columns"
        )
        
        return universe_df

