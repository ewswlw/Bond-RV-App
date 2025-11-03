"""
Main pipeline orchestration for RUNS data.
Extends bond_pipeline pattern with CLI interface for runs processing.
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
    RUNS_INPUT_DIR,
    RUNS_PARQUET,
    RUNS_FILE_PATTERN,
    LOG_FILE_PROCESSING,
    LOG_FILE_DUPLICATES,
    LOG_FILE_VALIDATION,
    LOG_FILE_SUMMARY,
    UNIVERSE_PARQUET
)
from bond_pipeline.utils import (
    setup_logging,
    setup_console_logger,
    format_date_string,
    format_run_header,
    get_run_id,
    save_run_metadata,
    check_and_rotate_logs,
    extract_date_from_filename,
    get_file_list,
    validate_runs_data
)
from runs_pipeline.extract import RunsExtractor
from runs_pipeline.transform import RunsTransformer
from runs_pipeline.load import RunsLoader


class RunsDataPipeline:
    """
    Main orchestrator for RUNS data pipeline.
    
    Processing flow:
    1. Get file list (chronologically sorted by date in filename)
    2. Extract data from all Excel files (combine into single DataFrame)
    3. Transform: deduplicate (end-of-day snapshots), validate CUSIPs/orphans,
       align schema, clean data, validate data quality
    4. Sort combined DataFrame by Date ascending (earliest to latest)
    5. Load to parquet (append or override mode)
    6. Generate and log summary statistics
    """
    
    def __init__(self, input_dir: Path, mode: str = 'append'):
        """
        Initialize pipeline.

        Args:
            input_dir: Directory containing RUNS Excel files
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
        self.extractor = RunsExtractor(LOG_FILE_PROCESSING)
        self.transformer = RunsTransformer(
            LOG_FILE_DUPLICATES, LOG_FILE_VALIDATION
        )
        self.loader = RunsLoader(LOG_FILE_PROCESSING)

        # File logger - captures all details, suppress console
        self.logger = setup_logging(
            LOG_FILE_SUMMARY, 'runs_pipeline', console_level=logging.CRITICAL
        )

        # Console logger - only essential messages
        self.console = setup_console_logger('runs_pipeline_console')

        # Log enhanced run header to file only
        header = format_run_header(
            self.run_id, self.start_time, mode, self.input_dir
        )
        self.logger.info(header)
    
    def _sort_files_chronologically(
        self, file_paths: List[Path]
    ) -> List[Path]:
        """
        Sort files chronologically by date extracted from filename.
        
        Args:
            file_paths: List of Path objects to RUNS Excel files
        
        Returns:
            List of Path objects sorted chronologically (earliest first)
        """
        def extract_date_from_path(file_path: Path) -> datetime:
            """Extract date from RUNS filename for sorting."""
            filename = file_path.name
            match = re.search(RUNS_FILE_PATTERN, filename, re.IGNORECASE)
            
            if match:
                month, day, year = match.groups()
                # Convert 2-digit year to 4-digit
                year_int = int(year)
                full_year = int(f"20{year}") if year_int < 50 else int(f"19{year}")
                
                try:
                    return datetime(full_year, int(month), int(day))
                except ValueError:
                    # If parsing fails, return epoch for sorting (puts at end)
                    return datetime(1970, 1, 1)
            else:
                # If no match, return epoch (puts at end)
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
            # Console progress indicator
            self.console.info(">> Processing RUNS data...")

            # Step 1: Get file list
            self.logger.info("\n[STEP 1] Getting file list...")
            file_paths = get_file_list(self.input_dir, '*.xlsx')

            if not file_paths:
                self.logger.error(f"No Excel files found in {self.input_dir}")
                self.console.info(
                    f"[ERROR] No Excel files found in {self.input_dir}"
                )
                return False

            # Filter to RUNS files only
            runs_files = [
                f for f in file_paths
                if re.search(RUNS_FILE_PATTERN, f.name, re.IGNORECASE)
            ]

            if not runs_files:
                self.logger.error(
                    f"No RUNS files found matching pattern {RUNS_FILE_PATTERN}"
                )
                self.console.info(
                    f"[ERROR] No RUNS files found in {self.input_dir}"
                )
                return False

            self.logger.info(f"Found {len(runs_files)} RUNS Excel files")
            self.console.info(f"   [1/4] Found {len(runs_files)} RUNS files")

            # Step 2: Sort files chronologically
            self.logger.info("\n[STEP 2] Sorting files chronologically...")
            sorted_files = self._sort_files_chronologically(runs_files)

            # Step 3: Extract data from Excel files
            self.logger.info("\n[STEP 3] Extracting data from Excel files...")
            self.console.info(f"   [2/4] Extracting data...")
            
            extracted_data = self.extractor.extract_all_files(sorted_files)

            if not extracted_data:
                self.logger.error("No data extracted from files")
                self.console.info("[ERROR] No data extracted from files")
                return False

            self.logger.info(
                f"Successfully extracted {len(extracted_data)} files"
            )

            # Combine all DataFrames into single DataFrame
            self.logger.info("Combining extracted DataFrames...")
            combined_df = pd.concat(extracted_data, ignore_index=True)
            self.logger.info(
                f"Combined DataFrame: {len(combined_df)} rows, "
                f"{len(combined_df.columns)} columns"
            )

            # Step 4: Transform data
            self.logger.info("\n[STEP 4] Transforming data...")
            self.console.info(f"   [3/4] Transforming & validating...")
            
            transformed_df = self.transformer.transform(
                combined_df, universe_parquet=UNIVERSE_PARQUET
            )

            self.logger.info(
                f"Transformation complete: {len(transformed_df)} rows "
                f"({len(combined_df) - len(transformed_df)} rows removed)"
            )

            # Step 5: Validate data quality
            self.logger.info("\n[STEP 5] Validating data quality...")
            validate_runs_data(transformed_df, self.transformer.logger_valid)

            # Step 6: Sort by Date ascending (earliest to latest)
            self.logger.info("\n[STEP 6] Sorting by Date ascending...")
            if 'Date' in transformed_df.columns:
                transformed_df = transformed_df.sort_values(
                    'Date', ascending=True, na_position='last'
                )
                self.logger.info(
                    f"Sorted by Date: {format_date_string(transformed_df['Date'].min())} "
                    f"to {format_date_string(transformed_df['Date'].max())}"
                )

            # Step 7: Load to parquet
            self.logger.info("\n[STEP 7] Loading data to parquet...")
            self.console.info(f"   [4/4] Writing to Parquet...")

            if self.mode == 'append':
                success = self.loader.load_append(transformed_df)
            elif self.mode == 'override':
                success = self.loader.load_override(transformed_df)
            else:
                self.logger.error(f"Invalid mode: {self.mode}")
                self.console.info(f"[ERROR] Invalid mode: {self.mode}")
                return False

            if not success:
                self.logger.error("Failed to load runs data")
                self.console.info("[ERROR] Failed to load runs data")
                return False

            # Step 8: Summary statistics
            self.logger.info("\n[STEP 8] Generating summary statistics...")
            stats = self.loader.get_summary_stats()

            # Log detailed stats to file
            self.logger.info("\n" + "=" * 80)
            self.logger.info("RUNS PIPELINE SUMMARY")
            self.logger.info("=" * 80)

            if 'runs_rows' in stats:
                self.logger.info(f"Runs Table:")
                self.logger.info(f"  Total Rows: {stats['runs_rows']:,}")
                self.logger.info(f"  Unique Dates: {stats['runs_dates']}")
                self.logger.info(f"  Unique CUSIPs: {stats['runs_cusips']:,}")
                self.logger.info(f"  Unique Dealers: {stats['runs_dealers']}")

                if 'date_range' in stats:
                    date_min, date_max = stats['date_range']
                    self.logger.info(
                        f"  Date Range: {format_date_string(date_min)} to "
                        f"{format_date_string(date_max)}"
                    )

            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()

            # Detailed completion to file
            self.logger.info("\n" + "=" * 80)
            self.logger.info("  PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"Run ID: #{self.run_id}")
            self.logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Duration: {duration:.2f} seconds")
            self.logger.info("=" * 80 + "\n")

            # Concise summary to console
            self.console.info("\n[SUCCESS] Pipeline completed!")
            if 'runs_rows' in stats:
                self.console.info(f"   Total rows: {stats['runs_rows']:,}")
                self.console.info(f"   Unique CUSIPs: {stats['runs_cusips']:,}")
                if 'date_range' in stats:
                    date_min, date_max = stats['date_range']
                    self.console.info(
                        f"   Date range: {format_date_string(date_min)} to "
                        f"{format_date_string(date_max)}"
                    )
            self.console.info(f"   Duration: {duration:.1f} seconds")
            self.console.info(f"\n   Detailed logs: bond_data/logs/")

            return True

        except Exception as e:
            self.logger.error(
                f"Pipeline failed with error: {str(e)}", exc_info=True
            )
            self.console.info(f"\n[ERROR] Pipeline failed: {str(e)}")
            self.console.info(f"   Check logs for details: bond_data/logs/")
            return False


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='RUNS Pipeline - Process RUNS Excel files to runs_timeseries.parquet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Append new data (default)
  python -m runs_pipeline.pipeline -i "Historical Runs/" -m append
  
  # Override all data
  python -m runs_pipeline.pipeline -i "Historical Runs/" -m override
  
  # Use default input directory
  python -m runs_pipeline.pipeline -m override
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=False,
        default=None,
        help='Input directory containing RUNS Excel files (default: Historical Runs folder)'
    )
    
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['append', 'override'],
        default='append',
        help='Processing mode: append (skip existing dates) or override (rebuild all)'
    )
    
    args = parser.parse_args()

    # Use default input directory if not specified
    if args.input is None:
        input_dir = RUNS_INPUT_DIR
        print(f"Using default input directory: {input_dir}")
    else:
        input_dir = Path(args.input)

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = RunsDataPipeline(input_dir, args.mode)
    success = pipeline.run()
    
    if success:
        print("\n✓ Pipeline completed successfully!")
        print(f"  Runs data: {RUNS_PARQUET}")
        sys.exit(0)
    else:
        print("\n✗ Pipeline failed. Check logs for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()

