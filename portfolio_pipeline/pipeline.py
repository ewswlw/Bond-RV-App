"""
Main pipeline orchestration for Portfolio holdings data.
Extends bond_pipeline pattern with CLI interface for portfolio processing.
"""

import argparse
import logging
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List

import pandas as pd

from bond_pipeline.config import (
    PORTFOLIO_INPUT_DIR,
    PORTFOLIO_PARQUET,
    PORTFOLIO_FILE_PATTERN,
    LOG_FILE_PROCESSING,
    LOG_FILE_DUPLICATES,
    LOG_FILE_VALIDATION,
    LOG_FILE_SUMMARY,
    DATE_COLUMN,
)
from bond_pipeline.utils import (
    setup_logging,
    setup_console_logger,
    format_date_string,
    format_run_header,
    get_run_id,
    save_run_metadata,
    check_and_rotate_logs,
    get_file_list,
)
from portfolio_pipeline.extract import PortfolioExtractor
from portfolio_pipeline.transform import PortfolioTransformer
from portfolio_pipeline.load import PortfolioLoader


class PortfolioDataPipeline:
    """
    Main orchestrator for Portfolio data pipeline.
    
    Processing flow:
    1. Get file list (chronologically sorted by date in filename)
    2. Extract data from all Excel files (date from filename)
    3. Transform: normalize CUSIPs, deduplicate (Date+CUSIP+ACCOUNT+PORTFOLIO),
       align schema, clean data
    4. Load to parquet (append or override mode)
    5. Generate and log summary statistics
    """
    
    def __init__(self, input_dir: Path, mode: str = 'append'):
        """
        Initialize pipeline.

        Args:
            input_dir: Directory containing Portfolio Excel files
            mode: 'append' or 'override'
        """
        self.input_dir = Path(input_dir)
        self.mode = mode
        self.start_time = datetime.now()

        # Get run ID and save metadata
        self.run_id = get_run_id()
        save_run_metadata(self.run_id, self.start_time, mode)

        # Check and rotate logs if needed
        for log_file in [
            LOG_FILE_PROCESSING, LOG_FILE_DUPLICATES,
            LOG_FILE_VALIDATION, LOG_FILE_SUMMARY
        ]:
            check_and_rotate_logs(log_file)

        # Initialize components
        self.extractor = PortfolioExtractor(LOG_FILE_PROCESSING)
        self.transformer = PortfolioTransformer(
            LOG_FILE_DUPLICATES, LOG_FILE_VALIDATION
        )
        self.loader = PortfolioLoader(LOG_FILE_PROCESSING)

        # File logger - captures all details, suppress console
        self.logger = setup_logging(
            LOG_FILE_SUMMARY, 'portfolio_pipeline', console_level=logging.CRITICAL
        )

        # Console logger - only essential messages
        self.console = setup_console_logger('portfolio_pipeline_console')

        # Log enhanced run header to file only
        header = format_run_header(
            self.run_id, self.start_time, mode, self.input_dir
        )
        self.logger.info(header)
    
    def _sort_files_chronologically(self, file_paths: List[Path]) -> List[Path]:
        """
        Sort files chronologically by date extracted from filename.
        
        Args:
            file_paths: List of Path objects to Portfolio Excel files
        
        Returns:
            List of Path objects sorted chronologically (earliest first)
        """
        from bond_pipeline.utils import extract_portfolio_date_from_filename
        
        def extract_date_from_path(file_path: Path) -> datetime:
            """Extract date from Portfolio filename for sorting."""
            date_obj = extract_portfolio_date_from_filename(file_path.name)
            if date_obj:
                return date_obj
            else:
                # If parsing fails, return epoch for sorting (puts at end)
                return datetime(1970, 1, 1)
        
        # Sort by extracted date
        sorted_files = sorted(file_paths, key=extract_date_from_path)
        
        self.logger.info(
            f"Sorted {len(sorted_files)} files chronologically "
            f"(earliest to latest)"
        )
        
        return sorted_files
    
    def run(self) -> bool:
        """
        Run the complete pipeline.

        Returns:
            True if successful
        """
        try:
            self.console.info(">> Processing Portfolio data...")

            total_console_steps = 4

            # Step 1: Get file list
            self.logger.info("\n[STEP 1] Getting file list...")
            file_paths = get_file_list(self.input_dir, '*.xlsx')

            if not file_paths:
                self.logger.error(f"No Excel files found in {self.input_dir}")
                self.console.info(
                    f"[ERROR] No Excel files found in {self.input_dir}"
                )
                return False

            portfolio_files = [
                f for f in file_paths
                if re.search(PORTFOLIO_FILE_PATTERN, f.name, re.IGNORECASE)
            ]

            if not portfolio_files:
                self.logger.error(
                    f"No Portfolio files found matching pattern {PORTFOLIO_FILE_PATTERN}"
                )
                self.console.info(
                    f"[ERROR] No Portfolio files found in {self.input_dir}"
                )
                return False

            self.logger.info(f"Found {len(portfolio_files)} Portfolio Excel files")
            self.console.info(
                f"   [1/{total_console_steps}] Found {len(portfolio_files)} Portfolio files"
            )

            # Sort files chronologically
            portfolio_files = self._sort_files_chronologically(portfolio_files)

            # Step 2: Extract data from Excel files
            self.logger.info("\n[STEP 2] Extracting data from Excel files...")
            self.console.info(f"   [2/{total_console_steps}] Extracting data...")
            raw_data = self.extractor.extract_all_files(portfolio_files)

            if not raw_data:
                self.logger.error("No data extracted from files")
                self.console.info("[ERROR] No data extracted from files")
                return False

            self.logger.info(f"Successfully extracted {len(raw_data)} files")

            # Step 3: Set master schema from latest file
            self.logger.info("\n[STEP 3] Setting master schema...")
            latest_date = max(raw_data.keys())
            latest_df = raw_data[latest_date]
            master_columns = latest_df.columns.tolist()

            self.transformer.set_master_schema(master_columns)
            self.logger.info(f"Master schema set with {len(master_columns)} columns")

            # Step 4: Transform data
            self.logger.info("\n[STEP 4] Transforming data...")
            self.console.info(f"   [3/{total_console_steps}] Transforming & validating...")
            transformed_data = {}

            for date, df in raw_data.items():
                transformed_df = self.transformer.transform_single_file(df)
                transformed_data[date] = transformed_df

            self.logger.info(f"Successfully transformed {len(transformed_data)} files")

            # Step 5: Load to parquet
            self.logger.info("\n[STEP 5] Loading to parquet...")
            self.console.info(f"   [4/{total_console_steps}] Writing parquet...")

            if self.mode == 'override':
                success = self.loader.load_portfolio_override(transformed_data)
            else:
                success = self.loader.load_portfolio_append(transformed_data)

            if not success:
                self.logger.error("Failed to write parquet file")
                self.console.info("[ERROR] Failed to write parquet file")
                return False

            # Step 6: Summary statistics
            self.logger.info("\n[STEP 6] Generating summary statistics...")
            self._log_summary_stats(transformed_data)

            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(f"\nPipeline completed successfully in {elapsed:.2f} seconds")
            self.console.info(f"[OK] Portfolio Pipeline completed ({elapsed:.1f}s)")

            return True

        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}", exc_info=True)
            self.console.info(f"[ERROR] Portfolio Pipeline failed: {str(e)}")
            return False
    
    def _log_summary_stats(self, transformed_data: dict):
        """
        Log summary statistics for the pipeline run.
        
        Args:
            transformed_data: Dictionary mapping date to transformed DataFrame
        """
        total_rows = sum(len(df) for df in transformed_data.values())
        total_dates = len(transformed_data)
        
        # Count unique CUSIPs across all dates
        all_cusips = set()
        for df in transformed_data.values():
            if 'CUSIP' in df.columns:
                all_cusips.update(df['CUSIP'].dropna().unique())
        
        # Count unique accounts and portfolios
        all_accounts = set()
        all_portfolios = set()
        for df in transformed_data.values():
            if 'ACCOUNT' in df.columns:
                all_accounts.update(df['ACCOUNT'].dropna().unique())
            if 'PORTFOLIO' in df.columns:
                all_portfolios.update(df['PORTFOLIO'].dropna().unique())
        
        self.logger.info("=" * 80)
        self.logger.info("PORTFOLIO PIPELINE SUMMARY STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"Mode: {self.mode.upper()}")
        self.logger.info(f"Files processed: {len(transformed_data)}")
        self.logger.info(f"Total rows: {total_rows:,}")
        self.logger.info(f"Unique CUSIPs: {len(all_cusips):,}")
        self.logger.info(f"Unique Accounts: {len(all_accounts):,}")
        self.logger.info(f"Unique Portfolios: {len(all_portfolios):,}")
        
        if transformed_data:
            dates_sorted = sorted(transformed_data.keys())
            self.logger.info(
                f"Date range: {format_date_string(dates_sorted[0])} to "
                f"{format_date_string(dates_sorted[-1])}"
            )
        
        self.logger.info("=" * 80)


def main():
    """CLI entry point for portfolio pipeline."""
    parser = argparse.ArgumentParser(
        description='Process Portfolio holdings Excel files into parquet'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=str(PORTFOLIO_INPUT_DIR),
        help=f'Input directory containing Portfolio Excel files (default: {PORTFOLIO_INPUT_DIR})'
    )
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['append', 'override'],
        default='append',
        help='Processing mode: append (add new dates) or override (rebuild all)'
    )
    
    args = parser.parse_args()
    
    pipeline = PortfolioDataPipeline(Path(args.input), args.mode)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

