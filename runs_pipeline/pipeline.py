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
from typing import List, Set, Tuple

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
from runs_pipeline.load import RunsLoader, LoadResult


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
            self.console.info(">> Processing RUNS data...")

            def _collect_dates(series) -> Set[datetime]:
                normalized: Set[datetime] = set()
                if series is None:
                    return normalized

                for value in pd.Series(series).dropna().unique():
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

            total_console_steps = 5

            # Step 1: Get file list
            self.logger.info("\n[STEP 1] Getting file list...")
            file_paths = get_file_list(self.input_dir, '*.xlsx')

            if not file_paths:
                self.logger.error(f"No Excel files found in {self.input_dir}")
                self.console.info(
                    f"[ERROR] No Excel files found in {self.input_dir}"
                )
                return False

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
            self.console.info(
                f"   [1/{total_console_steps}] Found {len(runs_files)} RUNS files"
            )

            # Step 2: Sort files chronologically
            self.logger.info("\n[STEP 2] Sorting files chronologically...")
            sorted_files = self._sort_files_chronologically(runs_files)

            existing_dates: Set[datetime] = set()
            files_to_process: List[Tuple[Path, Set[datetime]]] = []
            skipped_files: List[Tuple[Path, Set[datetime]]] = []

            run_metrics = {
                'files_total': len(sorted_files),
                'files_processed': 0,
                'files_skipped': 0,
                'rows_extracted': 0,
                'rows_after_filter': 0,
                'new_dates': set(),
            }

            # Step 3: Determine files requiring processing
            if self.mode == 'append':
                self.logger.info("\n[STEP 3] Selecting files to process (append mode)...")
                existing_dates = self.loader.get_existing_dates()

                for file_path in sorted_files:
                    file_dates = self.extractor.peek_file_dates(file_path)
                    if not file_dates:
                        files_to_process.append((file_path, set()))
                        self.logger.info(
                            f"  {file_path.name}: Date peek unavailable; will process file"
                        )
                        continue

                    missing_dates = {d for d in file_dates if d not in existing_dates}
                    if not missing_dates:
                        skipped_files.append((file_path, file_dates))
                        self.logger.info(
                            f"  Skipping {file_path.name}: all {len(file_dates)} date(s) already loaded"
                        )
                    else:
                        files_to_process.append((file_path, missing_dates))
                        self.logger.info(
                            f"  Processing {file_path.name}: {len(missing_dates)} new date(s)"
                        )

                run_metrics['files_skipped'] = len(skipped_files)
                self.console.info(
                    f"   [2/{total_console_steps}] "
                    f"Queued {len(files_to_process)} file(s); skipped {len(skipped_files)} already loaded"
                )
            else:
                self.logger.info("\n[STEP 3] Preparing file queue...")
                files_to_process = [(path, set()) for path in sorted_files]
                self.console.info(
                    f"   [2/{total_console_steps}] Queued {len(files_to_process)} file(s) for processing"
                )

            if not files_to_process:
                self.logger.info(
                    "All RUNS files are already reflected in the parquet dataset; no processing required."
                )
                stats = self.loader.get_summary_stats()
                self.logger.info("Exiting early with no changes.")
                self.console.info(
                    "[INFO] No new RUNS files detected; parquet already up to date."
                )
                if 'runs_rows' in stats:
                    self.console.info(
                        f"   Current rows: {stats['runs_rows']:,}"
                    )
                if 'runs_date_range' in stats:
                    date_min, date_max = stats['runs_date_range']
                    self.console.info(
                        f"   Date range: {format_date_string(date_min)} to "
                        f"{format_date_string(date_max)}"
                    )
                return True

            # Step 4: Extract data from Excel files
            self.logger.info("\n[STEP 4] Extracting data from Excel files...")
            self.console.info(
                f"   [3/{total_console_steps}] Extracting data..."
            )

            extracted_data: List[pd.DataFrame] = []

            for file_path, targeted_dates in files_to_process:
                df = self.extractor.read_excel_file(file_path)
                if df is None:
                    self.logger.warning(
                        f"Skipping file due to extraction error: {file_path.name}"
                    )
                    continue

                original_rows = len(df)
                filtered_df = df

                if self.mode == 'append':
                    if targeted_dates:
                        filtered_df = df[df['Date'].isin(targeted_dates)].copy()
                    else:
                        filtered_df = df[
                            ~df['Date'].isin(existing_dates)
                        ].copy()

                    removed_rows = original_rows - len(filtered_df)
                    if removed_rows > 0:
                        self.logger.info(
                            f"  {file_path.name}: Removed {removed_rows} rows already loaded"
                        )

                if len(filtered_df) == 0:
                    self.logger.info(
                        f"  {file_path.name}: No new rows after filtering; skipping."
                    )
                    continue

                run_metrics['files_processed'] += 1
                run_metrics['rows_extracted'] += original_rows
                run_metrics['rows_after_filter'] += len(filtered_df)
                run_metrics['new_dates'].update(_collect_dates(filtered_df.get('Date')))

                extracted_data.append(filtered_df)

            if not extracted_data:
                self.logger.error("No data extracted from files")
                self.console.info("[ERROR] No data extracted from files")
                return False

            combined_df = pd.concat(extracted_data, ignore_index=True)
            self.logger.info(
                f"Combined DataFrame: {len(combined_df)} rows, "
                f"{len(combined_df.columns)} columns"
            )

            # Step 5: Transform data
            self.logger.info("\n[STEP 5] Transforming data...")
            self.console.info(
                f"   [4/{total_console_steps}] Transforming & validating..."
            )

            transformed_df = self.transformer.transform(
                combined_df, universe_parquet=UNIVERSE_PARQUET
            )

            self.logger.info(
                f"Transformation complete: {len(transformed_df)} rows "
                f"({len(combined_df) - len(transformed_df)} rows removed)"
            )

            # Step 6: Validate data quality
            self.logger.info("\n[STEP 6] Validating data quality...")
            validate_runs_data(transformed_df, self.transformer.logger_valid)

            # Step 7: Sort by Date ascending (earliest to latest)
            self.logger.info("\n[STEP 7] Sorting by Date ascending...")
            if 'Date' in transformed_df.columns:
                non_null_dates = transformed_df['Date'].dropna()
                transformed_df = transformed_df.sort_values(
                    'Date', ascending=True, na_position='last'
                )
                if not non_null_dates.empty:
                    self.logger.info(
                        f"Sorted by Date: {format_date_string(non_null_dates.min())} "
                        f"to {format_date_string(non_null_dates.max())}"
                    )

            # Step 8: Load to parquet
            self.logger.info("\n[STEP 8] Loading data to parquet...")
            self.console.info(
                f"   [5/{total_console_steps}] Writing to Parquet..."
            )

            if self.mode == 'append':
                load_result = self.loader.load_append(
                    transformed_df,
                    existing_dates=existing_dates,
                )
            elif self.mode == 'override':
                load_result = self.loader.load_override(transformed_df)
            else:
                self.logger.error(f"Invalid mode: {self.mode}")
                self.console.info(f"[ERROR] Invalid mode: {self.mode}")
                return False

            if not load_result.success:
                self.logger.error("Failed to load runs data")
                self.console.info("[ERROR] Failed to load runs data")
                return False

            # Step 9: Summary statistics
            self.logger.info("\n[STEP 9] Generating summary statistics...")
            stats = self.loader.get_summary_stats()

            self.logger.info("\n" + "=" * 80)
            self.logger.info("RUNS PIPELINE SUMMARY")
            self.logger.info("=" * 80)

            self.logger.info(
                f"Files processed: {run_metrics['files_processed']} "
                f"(skipped {run_metrics['files_skipped']})"
            )
            self.logger.info(
                f"Rows read from Excel: {run_metrics['rows_extracted']:,}"
            )
            self.logger.info(
                f"Rows retained after filtering: {run_metrics['rows_after_filter']:,}"
            )
            self.logger.info(
                f"New rows written: {load_result.new_rows:,}"
            )
            self.logger.info(
                f"Rows skipped (existing dates): {load_result.skipped_rows:,}"
            )
            self.logger.info(
                f"New dates this run: {len(load_result.new_dates)}"
            )
            if load_result.new_dates:
                sorted_new_dates = sorted(load_result.new_dates)
                self.logger.info(
                    f"  New date range: {format_date_string(sorted_new_dates[0])} "
                    f"to {format_date_string(sorted_new_dates[-1])}"
                )
            self.logger.info(
                f"New CUSIPs this run: {load_result.new_cusips:,}"
            )
            self.logger.info(
                f"New Dealers this run: {load_result.new_dealers:,}"
            )

            if 'runs_rows' in stats:
                self.logger.info("Runs Table:")
                self.logger.info(
                    f"  Total Rows: {stats.get('runs_rows', 0):,}"
                )
                self.logger.info(
                    f"  Unique Dates: {stats.get('runs_dates', 0)}"
                )
                self.logger.info(
                    f"  Unique CUSIPs: {stats.get('runs_cusips', 0):,}"
                )
                self.logger.info(
                    f"  Unique Dealers: {stats.get('runs_dealers', 0)}"
                )

                if 'runs_date_range' in stats:
                    date_min, date_max = stats['runs_date_range']
                    self.logger.info(
                        f"  Date Range: {format_date_string(date_min)} to "
                        f"{format_date_string(date_max)}"
                    )

            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()

            self.logger.info("\n" + "=" * 80)
            self.logger.info("  PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"Run ID: #{self.run_id}")
            self.logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Duration: {duration:.2f} seconds")
            self.logger.info("=" * 80 + "\n")

            self.console.info("\n[SUCCESS] Pipeline completed!")
            self.console.info(
                f"   Files processed: {run_metrics['files_processed']} "
                f"(skipped {run_metrics['files_skipped']})"
            )
            self.console.info(
                f"   New rows: {load_result.new_rows:,}"
            )
            if load_result.new_dates:
                sorted_new_dates = sorted(load_result.new_dates)
                if len(sorted_new_dates) == 1:
                    date_msg = format_date_string(sorted_new_dates[0])
                else:
                    date_msg = (
                        f"{format_date_string(sorted_new_dates[0])} -> "
                        f"{format_date_string(sorted_new_dates[-1])}"
                    )
                self.console.info(
                    f"   New dates: {len(load_result.new_dates)} ({date_msg})"
                )
            else:
                self.console.info("   New dates: 0")
            self.console.info(
                f"   New CUSIPs: {load_result.new_cusips:,} | "
                f"New Dealers: {load_result.new_dealers:,}"
            )
            if 'runs_rows' in stats:
                self.console.info(
                    f"   Total rows: {stats.get('runs_rows', 0):,}"
                )
            if 'runs_date_range' in stats:
                date_min, date_max = stats['runs_date_range']
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

