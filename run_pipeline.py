"""
Main script to run the bond, runs, and portfolio data pipelines.
Run this from the project root directory.

This script orchestrates:
1. Bond Pipeline - Processes API Historical Excel files
2. Runs Pipeline - Processes Historical Runs Excel files
3. Portfolio Pipeline - Processes Portfolio Holdings Excel files
4. Individual Parquet Files - Regenerate specific parquet outputs
5. All Pipelines - Run Bond, Runs, and Portfolio together
"""

import sys
from pathlib import Path

# Add bond_pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from bond_pipeline.config import (
    DEFAULT_INPUT_DIR,
    RUNS_INPUT_DIR,
    PORTFOLIO_INPUT_DIR,
    HISTORICAL_PARQUET,
    RUNS_PARQUET,
    PORTFOLIO_PARQUET,
    UNIVERSE_PARQUET,
    BQL_PARQUET,
    LOG_FILE_PROCESSING,
    LOG_FILE_DUPLICATES,
    LOG_FILE_VALIDATION,
    LOG_FILE_SUMMARY,
)
from bond_pipeline.pipeline import BondDataPipeline
from bond_pipeline.extract import ExcelExtractor
from bond_pipeline.transform import DataTransformer
from bond_pipeline.load import ParquetLoader
from bond_pipeline.utils import log_parquet_diagnostics
from runs_pipeline.pipeline import RunsDataPipeline
from portfolio_pipeline.pipeline import PortfolioDataPipeline


def run_bond_pipeline(mode: str, process_bql: bool) -> bool:
    """Run the bond data pipeline."""
    print("\n" + "=" * 70)
    print("  BOND DATA PIPELINE")
    print("=" * 70)
    print(f"\n>> Starting Bond Pipeline in {mode.upper()} mode...")
    print(f"   Input: {DEFAULT_INPUT_DIR}")
    
    try:
        pipeline = BondDataPipeline(DEFAULT_INPUT_DIR, mode, process_bql=process_bql)
        success = pipeline.run()
        
        if success:
            print("\n" + "-" * 70)
            print("   [OK] Bond Pipeline completed successfully")
            print(f"   Output: {HISTORICAL_PARQUET}")
            print(f"   Universe: {UNIVERSE_PARQUET}")
            if process_bql:
                print(f"   BQL: {BQL_PARQUET}")
            print("-" * 70)
        else:
            print("\n" + "-" * 70)
            print("   [FAIL] Bond Pipeline failed")
            print(f"   Logs: {HISTORICAL_PARQUET.parent.parent / 'logs'}")
            print("-" * 70)
        
        return success
    except Exception as e:
        print(f"\n[ERROR] Bond Pipeline error: {str(e)}")
        return False


def run_runs_pipeline(mode: str) -> bool:
    """Run the runs data pipeline."""
    print("\n" + "=" * 70)
    print("  RUNS DATA PIPELINE")
    print("=" * 70)
    print(f"\n>> Starting Runs Pipeline in {mode.upper()} mode...")
    print(f"   Input: {RUNS_INPUT_DIR}")
    
    try:
        pipeline = RunsDataPipeline(RUNS_INPUT_DIR, mode)
        success = pipeline.run()
        
        if success:
            print("\n" + "-" * 70)
            print("   [OK] Runs Pipeline completed successfully")
            print(f"   Output: {RUNS_PARQUET}")
            print("-" * 70)
        else:
            print("\n" + "-" * 70)
            print("   [FAIL] Runs Pipeline failed")
            print(f"   Logs: {RUNS_PARQUET.parent.parent / 'logs'}")
            print("-" * 70)
        
        return success
    except Exception as e:
        print(f"\n[ERROR] Runs Pipeline error: {str(e)}")
        return False


def run_portfolio_pipeline(mode: str) -> bool:
    """Run the portfolio data pipeline."""
    print("\n" + "=" * 70)
    print("  PORTFOLIO DATA PIPELINE")
    print("=" * 70)
    print(f"\n>> Starting Portfolio Pipeline in {mode.upper()} mode...")
    print(f"   Input: {PORTFOLIO_INPUT_DIR}")
    
    try:
        pipeline = PortfolioDataPipeline(PORTFOLIO_INPUT_DIR, mode)
        success = pipeline.run()
        
        if success:
            print("\n" + "-" * 70)
            print("   [OK] Portfolio Pipeline completed successfully")
            print(f"   Output: {PORTFOLIO_PARQUET}")
            print("-" * 70)
        else:
            print("\n" + "-" * 70)
            print("   [FAIL] Portfolio Pipeline failed")
            print(f"   Logs: {PORTFOLIO_PARQUET.parent.parent / 'logs'}")
            print("-" * 70)
        
        return success
    except Exception as e:
        print(f"\n[ERROR] Portfolio Pipeline error: {str(e)}")
        return False


def run_bql_only() -> bool:
    """Run only BQL parquet generation."""
    print("\n" + "=" * 70)
    print("  BQL PARQUET GENERATION")
    print("=" * 70)
    print("\n>> Processing BQL workbook...")
    
    try:
        extractor = ExcelExtractor(LOG_FILE_PROCESSING)
        transformer = DataTransformer(LOG_FILE_DUPLICATES, LOG_FILE_VALIDATION)
        loader = ParquetLoader(LOG_FILE_PROCESSING)
        
        # Read BQL workbook
        bql_raw_df = extractor.read_bql_workbook()
        if bql_raw_df is None or bql_raw_df.empty:
            print("   [FAIL] Failed to load BQL workbook or workbook is empty.")
            return False
        
        # Transform BQL data
        bql_artifacts = transformer.transform_bql_data(bql_raw_df)
        if bql_artifacts.dataframe.empty:
            print("   [FAIL] BQL transformation produced no rows.")
            return False
        
        # Write BQL dataset
        success = loader.write_bql_dataset(
            bql_artifacts.dataframe,
            bql_artifacts.cusip_to_name,
        )
        
        if success:
            print("\n" + "-" * 70)
            print("   [OK] BQL parquet generated successfully")
            print(f"   Output: {BQL_PARQUET}")
            print("-" * 70)
        else:
            print("\n" + "-" * 70)
            print("   [FAIL] Failed to write BQL parquet")
            print("-" * 70)
        
        return success
    except Exception as e:
        print(f"\n[ERROR] BQL processing error: {str(e)}")
        return False


def run_historical_only(mode: str) -> bool:
    """Run only historical bond details parquet generation (also regenerates universe)."""
    print("\n" + "=" * 70)
    print("  HISTORICAL BOND DETAILS PARQUET GENERATION")
    print("=" * 70)
    print(f"\n>> Starting Historical Bond Pipeline in {mode.upper()} mode...")
    print(f"   Input: {DEFAULT_INPUT_DIR}")
    print("   Note: This will also regenerate universe.parquet")
    
    try:
        pipeline = BondDataPipeline(DEFAULT_INPUT_DIR, mode, process_bql=False)
        success = pipeline.run()
        
        if success:
            print("\n" + "-" * 70)
            print("   [OK] Historical Bond Details parquet generated successfully")
            print(f"   Output: {HISTORICAL_PARQUET}")
            print(f"   Universe: {UNIVERSE_PARQUET}")
            print("-" * 70)
        else:
            print("\n" + "-" * 70)
            print("   [FAIL] Historical Bond Pipeline failed")
            print(f"   Logs: {HISTORICAL_PARQUET.parent.parent / 'logs'}")
            print("-" * 70)
        
        return success
    except Exception as e:
        print(f"\n[ERROR] Historical Bond Pipeline error: {str(e)}")
        return False


def run_individual_parquet() -> bool:
    """Run individual parquet file regeneration."""
    print("\n" + "=" * 70)
    print("  INDIVIDUAL PARQUET REGENERATION")
    print("=" * 70)
    
    parquet_options = {
        '1': ('historical_bond_details.parquet', 'Historical Bond Details (also regenerates universe)', run_historical_only),
        '2': ('bql.parquet', 'BQL Dataset', run_bql_only),
        '3': ('runs_timeseries.parquet', 'Runs Timeseries', None),  # Handled separately
        '4': ('historical_portfolio.parquet', 'Portfolio Holdings', None),  # Handled separately
    }
    
    print("\nSelect parquet file to regenerate:")
    for key, (filename, description, _) in parquet_options.items():
        print(f"  [{key}] {filename}")
        print(f"       {description}")
    
    choice = input("\nChoice (1, 2, 3, or 4): ").strip()
    
    if choice not in parquet_options:
        print("Invalid choice.")
        return False
    
    filename, description, func = parquet_options[choice]
    
    if choice == '3':
        # Runs pipeline needs mode selection
        print("\nSelect processing mode:")
        print("  [1] Override - Rebuild everything from scratch")
        print("  [2] Append   - Add only new dates (default)")
        mode_choice = input("\nChoice (1 or 2): ").strip() or "2"
        mode = 'override' if mode_choice == '1' else 'append'
        return run_runs_pipeline(mode)
    elif choice == '4':
        # Portfolio pipeline needs mode selection
        print("\nSelect processing mode:")
        print("  [1] Override - Rebuild everything from scratch")
        print("  [2] Append   - Add only new dates (default)")
        mode_choice = input("\nChoice (1 or 2): ").strip() or "2"
        mode = 'override' if mode_choice == '1' else 'append'
        return run_portfolio_pipeline(mode)
    elif choice == '1':
        # Historical needs mode selection
        print("\nSelect processing mode:")
        print("  [1] Override - Rebuild everything from scratch")
        print("  [2] Append   - Add only new dates (default)")
        mode_choice = input("\nChoice (1 or 2): ").strip() or "2"
        mode = 'override' if mode_choice == '1' else 'append'
        return func(mode)
    else:
        # BQL doesn't need mode (always override)
        return func()


def main():
    """Run the pipeline(s) with user selection."""
    
    print("\n" + "=" * 70)
    print("  DATA PIPELINE ORCHESTRATOR")
    print("=" * 70)
    
    # Ask user which pipeline(s) to run
    print("\nSelect pipeline(s) to run:")
    print("  [1] Bond Pipeline only")
    print("  [2] Runs Pipeline only")
    print("  [3] Portfolio Pipeline only")
    print("  [4] Both Bond and Runs Pipelines (default)")
    print("  [5] Individual Parquet Files")
    print("  [6] All Pipelines (Bond, Runs, and Portfolio)")
    
    pipeline_choice = input("\nChoice (1, 2, 3, 4, 5, or 6): ").strip() or "4"
    
    if pipeline_choice not in ['1', '2', '3', '4', '5', '6']:
        print("Invalid choice. Using default: Both Bond and Runs Pipelines")
        pipeline_choice = '4'
    
    # Handle individual parquet selection
    if pipeline_choice == '5':
        result = run_individual_parquet()
        try:
            log_parquet_diagnostics()
        except Exception as exc:
            print(
                "\nWarning: Unable to log parquet diagnostics "
                f"({exc})"
            )
        return 0 if result else 1
    
    # Ask user for mode
    print("\nSelect processing mode:")
    print("  [1] Override - Rebuild everything from scratch")
    print("  [2] Append   - Add only new dates (default)")
    
    mode_choice = input("\nChoice (1 or 2): ").strip() or "2"
    
    if mode_choice == '1':
        mode = 'override'
    else:
        mode = 'append'
    
    # Track results
    results = {}
    process_bql = False

    if pipeline_choice in ['1', '4', '6']:
        bql_choice = input("\nProcess BQL workbook as part of Bond Pipeline? (Y/n): ").strip().lower()
        process_bql = bql_choice != 'n'
    
    # Run selected pipeline(s)
    if pipeline_choice in ['1', '4', '6']:
        # Run Bond Pipeline
        results['bond'] = run_bond_pipeline(mode, process_bql)
    
    if pipeline_choice in ['2', '4', '6']:
        # Run Runs Pipeline
        results['runs'] = run_runs_pipeline(mode)
    
    if pipeline_choice in ['3', '6']:
        # Run Portfolio Pipeline
        results['portfolio'] = run_portfolio_pipeline(mode)
    
    # Summary
    print("\n" + "=" * 70)
    print("  EXECUTION SUMMARY")
    print("=" * 70)
    
    if 'bond' in results:
        status = "SUCCESS" if results['bond'] else "FAILED"
        print(f"\n  Bond Pipeline: {status}")
    
    if 'runs' in results:
        status = "SUCCESS" if results['runs'] else "FAILED"
        print(f"  Runs Pipeline: {status}")
    
    if 'portfolio' in results:
        status = "SUCCESS" if results['portfolio'] else "FAILED"
        print(f"  Portfolio Pipeline: {status}")
    
    print("=" * 70)
    
    # Return exit code
    try:
        log_parquet_diagnostics()
    except Exception as exc:
        print(
            "\nWarning: Unable to log parquet diagnostics "
            f"({exc})"
        )

    if all(results.values()):
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
