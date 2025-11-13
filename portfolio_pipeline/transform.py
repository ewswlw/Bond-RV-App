"""
Transform module for Portfolio data cleaning, normalization, and deduplication.
Handles CUSIP validation, NA cleaning, and duplicate removal with Date+CUSIP+ACCOUNT+PORTFOLIO primary key.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from bond_pipeline.config import (
    PORTFOLIO_PRIMARY_KEY,
    DATE_COLUMN,
    BOND_NAME_COLUMN,
)
from bond_pipeline.utils import (
    setup_logging,
    validate_cusip,
    clean_na_values,
    align_to_master_schema,
    format_date_string,
    sanitize_log_message,
)


class PortfolioTransformer:
    """
    Transform and clean Portfolio data.
    
    Key logic:
    1. CUSIP normalization: Same as bond pipeline (uppercase, validate 9 chars)
    2. Deduplication: Keep last occurrence per Date+CUSIP+ACCOUNT+PORTFOLIO
    3. Schema alignment: Align to master schema (82 columns, dynamically detected)
    4. Data cleaning: Replace NA values with NaN
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
            log_file_dupes, 'portfolio_dupes', console_level=logging.CRITICAL
        )
        self.logger_valid = setup_logging(
            log_file_valid, 'portfolio_valid', console_level=logging.CRITICAL
        )
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
        has_security = 'SECURITY' in df.columns

        for idx, cusip in df['CUSIP'].items():
            normalized, is_valid, error_msg = validate_cusip(cusip)

            df.at[idx, 'CUSIP'] = normalized
            df.at[idx, 'CUSIP_VALID'] = is_valid
            df.at[idx, 'CUSIP_ERROR'] = error_msg

            if not is_valid:
                invalid_count += 1

                # Get security name if available
                security_name = ""
                if has_security and idx in df.index:
                    name_value = df.at[idx, 'SECURITY']
                    if pd.notna(name_value) and str(name_value) not in ['nan', '<NA>', 'None', '']:
                        security_name = f" ({name_value})"

                self.logger_valid.warning(
                    f"Invalid CUSIP: {cusip}{security_name} -> {normalized} | Error: {error_msg}"
                )

        if invalid_count > 0:
            self.logger_valid.warning(f"Total invalid CUSIPs: {invalid_count} out of {len(df)}")
        else:
            self.logger_valid.info(f"All {len(df)} CUSIPs are valid")

        return df
    
    def remove_duplicates(self, df: pd.DataFrame, keep: str = 'last') -> pd.DataFrame:
        """
        Remove duplicate Date+CUSIP+ACCOUNT+PORTFOLIO combinations, keeping last occurrence.
        
        Args:
            df: DataFrame with Date, CUSIP, ACCOUNT, PORTFOLIO columns
            keep: Which duplicate to keep ('first', 'last')
        
        Returns:
            Deduplicated DataFrame
        """
        if not all(col in df.columns for col in PORTFOLIO_PRIMARY_KEY):
            missing = [col for col in PORTFOLIO_PRIMARY_KEY if col not in df.columns]
            self.logger_dupes.error(f"Required columns missing: {missing}")
            return df
        
        initial_count = len(df)
        
        # Find duplicates
        duplicates = df[df.duplicated(subset=PORTFOLIO_PRIMARY_KEY, keep=False)]

        if len(duplicates) > 0:
            date_str = format_date_string(df[DATE_COLUMN].iloc[0]) if len(df) > 0 else "Unknown"
            unique_dupes = duplicates[PORTFOLIO_PRIMARY_KEY].drop_duplicates()

            self.logger_dupes.warning(
                f"Date {date_str}: Found {len(duplicates)} duplicate rows "
                f"({len(unique_dupes)} unique Date+CUSIP+ACCOUNT+PORTFOLIO combinations)"
            )

            # Log sample duplicates with security names if available
            if 'SECURITY' in df.columns:
                self.logger_dupes.info("Sample duplicate combinations:")
                for _, row in unique_dupes.head(10).iterrows():
                    cusip = row['CUSIP']
                    account = row.get('ACCOUNT', 'N/A')
                    portfolio = row.get('PORTFOLIO', 'N/A')
                    # Get first security name for this combination
                    matching_rows = df[
                        (df['CUSIP'] == cusip) &
                        (df['ACCOUNT'] == account) &
                        (df['PORTFOLIO'] == portfolio)
                    ]
                    if len(matching_rows) > 0:
                        security_name = matching_rows['SECURITY'].iloc[0]
                        if pd.notna(security_name) and str(security_name) not in ['nan', '<NA>', 'None', '']:
                            self.logger_dupes.info(
                                f"  {cusip} | {account} | {portfolio} | {security_name}"
                            )
                        else:
                            self.logger_dupes.info(
                                f"  {cusip} | {account} | {portfolio}"
                            )
        
        # Remove duplicates, keeping last occurrence
        df_deduped = df.drop_duplicates(subset=PORTFOLIO_PRIMARY_KEY, keep=keep)
        
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
    
    def drop_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows where either SECURITY or CUSIP is blank/NaN.
        
        Args:
            df: DataFrame to filter
        
        Returns:
            DataFrame with invalid rows removed
        """
        initial_count = len(df)
        
        # Check for required columns
        has_security = 'SECURITY' in df.columns
        has_cusip = 'CUSIP' in df.columns
        
        if not has_security and not has_cusip:
            self.logger_valid.warning("Neither SECURITY nor CUSIP columns found, skipping row filtering")
            return df
        
        # Create mask for valid rows (both SECURITY and CUSIP must be non-null and non-empty)
        if has_security and has_cusip:
            # Both columns exist - both must be valid
            valid_mask = (
                df['SECURITY'].notna() & 
                (df['SECURITY'].astype(str).str.strip() != '') &
                df['CUSIP'].notna() & 
                (df['CUSIP'].astype(str).str.strip() != '')
            )
        elif has_security:
            # Only SECURITY exists
            valid_mask = (
                df['SECURITY'].notna() & 
                (df['SECURITY'].astype(str).str.strip() != '')
            )
        else:
            # Only CUSIP exists
            valid_mask = (
                df['CUSIP'].notna() & 
                (df['CUSIP'].astype(str).str.strip() != '')
            )
        
        invalid_count = (~valid_mask).sum()
        
        if invalid_count > 0:
            self.logger_valid.warning(
                f"Dropping {invalid_count} rows with blank SECURITY or CUSIP "
                f"(kept {valid_mask.sum()} valid rows)"
            )
            df = df[valid_mask].copy()
        else:
            self.logger_valid.info(f"All {initial_count} rows have valid SECURITY and CUSIP")
        
        return df
    
    def transform_single_file(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformations to a single file's data.
        
        Args:
            df: DataFrame from single file
        
        Returns:
            Transformed DataFrame
        """
        # 0. Drop rows with blank SECURITY or CUSIP (do this first)
        df = self.drop_invalid_rows(df)
        
        # 1. Validate and normalize CUSIPs
        df = self.validate_and_normalize_cusips(df)
        
        # 2. Remove duplicates (keep last)
        df = self.remove_duplicates(df, keep='last')
        
        # 3. Clean NA values
        df = self.clean_data(df)
        
        # 4. Align to master schema
        df = self.align_schema(df)
        
        return df

