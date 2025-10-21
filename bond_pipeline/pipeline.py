"""
Main pipeline orchestration script.
Coordinates extraction, transformation, and loading of bond data.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from config import (
    LOG_FILE_PROCESSING,
    LOG_FILE_DUPLICATES,
    LOG_FILE_VALIDATION,
    LOG_FILE_SUMMARY,
    HISTORICAL_PARQUET,
    UNIVERSE_PARQUET
)
from utils import setup_logging, get_file_list, format_date_string
from extract import ExcelExtractor
from transform import DataTransformer
from load import ParquetLoader


class BondDataPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, input_dir: Path, mode: str = 'append'):
        """
        Initialize pipeline.
        
        Args:
            input_dir: Directory containing Excel files
            mode: 'append' or 'override'
        """
        self.input_dir = Path(input_dir)
        self.mode = mode
        
        # Initialize components
        self.extractor = ExcelExtractor(LOG_FILE_PROCESSING)
        self.transformer = DataTransformer(LOG_FILE_DUPLICATES, LOG_FILE_VALIDATION)
        self.loader = ParquetLoader(LOG_FILE_PROCESSING)
        self.logger = setup_logging(LOG_FILE_SUMMARY, 'pipeline')
        
        self.logger.info("=" * 80)
        self.logger.info(f"Bond Data Pipeline - Mode: {mode.upper()}")
        self.logger.info(f"Input Directory: {self.input_dir}")
        self.logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
    
    def run(self) -> bool:
        """
        Run the complete pipeline.
        
        Returns:
            True if successful
        """
        try:
            # Step 1: Get file list
            self.logger.info("\n[STEP 1] Getting file list...")
            file_paths = get_file_list(self.input_dir, '*.xlsx')
            
            if not file_paths:
                self.logger.error(f"No Excel files found in {self.input_dir}")
                return False
            
            self.logger.info(f"Found {len(file_paths)} Excel files")
            
            # Step 2: Extract data from Excel files
            self.logger.info("\n[STEP 2] Extracting data from Excel files...")
            raw_data = self.extractor.extract_all_files(file_paths)
            
            if not raw_data:
                self.logger.error("No data extracted from files")
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
            transformed_data = {}
            
            for date, df in raw_data.items():
                self.logger.info(f"Transforming data for {format_date_string(date)}...")
                transformed_df = self.transformer.transform_single_file(df)
                transformed_data[date] = transformed_df
            
            self.logger.info(f"Transformed {len(transformed_data)} files")
            
            # Step 5: Load historical data
            self.logger.info("\n[STEP 5] Loading historical data to parquet...")
            
            if self.mode == 'append':
                success = self.loader.load_historical_append(transformed_data)
            elif self.mode == 'override':
                success = self.loader.load_historical_override(transformed_data)
            else:
                self.logger.error(f"Invalid mode: {self.mode}")
                return False
            
            if not success:
                self.logger.error("Failed to load historical data")
                return False
            
            # Step 6: Create and load universe table
            self.logger.info("\n[STEP 6] Creating universe table...")
            
            # Read complete historical data
            import pandas as pd
            historical_df = pd.read_parquet(HISTORICAL_PARQUET)
            
            # Create universe
            universe_df = self.transformer.create_universe_table(historical_df)
            
            # Load universe
            success = self.loader.load_universe(universe_df)
            
            if not success:
                self.logger.error("Failed to load universe data")
                return False
            
            # Step 7: Summary statistics
            self.logger.info("\n[STEP 7] Generating summary statistics...")
            stats = self.loader.get_summary_stats()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("PIPELINE SUMMARY")
            self.logger.info("=" * 80)
            
            if 'historical_rows' in stats:
                self.logger.info(f"Historical Table:")
                self.logger.info(f"  Total Rows: {stats['historical_rows']:,}")
                self.logger.info(f"  Unique Dates: {stats['historical_dates']}")
                self.logger.info(f"  Unique CUSIPs: {stats['historical_cusips']:,}")
                
                if 'date_range' in stats:
                    date_min, date_max = stats['date_range']
                    self.logger.info(f"  Date Range: {format_date_string(date_min)} to {format_date_string(date_max)}")
            
            if 'universe_cusips' in stats:
                self.logger.info(f"\nUniverse Table:")
                self.logger.info(f"  Unique CUSIPs: {stats['universe_cusips']:,}")
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            return False


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Bond Data Pipeline - Process Excel files to Parquet tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Append new data (default)
  python pipeline.py --input "Universe Historical/" --mode append
  
  # Override all data
  python pipeline.py --input "Universe Historical/" --mode override
  
  # Short form
  python pipeline.py -i "Universe Historical/" -m append
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input directory containing Excel files'
    )
    
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['append', 'override'],
        default='append',
        help='Processing mode: append (add new dates) or override (rebuild all)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = BondDataPipeline(input_dir, args.mode)
    success = pipeline.run()
    
    if success:
        print("\n✓ Pipeline completed successfully!")
        print(f"  Historical data: {HISTORICAL_PARQUET}")
        print(f"  Universe data: {UNIVERSE_PARQUET}")
        sys.exit(0)
    else:
        print("\n✗ Pipeline failed. Check logs for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()

