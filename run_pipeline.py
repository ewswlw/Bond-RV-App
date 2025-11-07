"""
Main script to run the bond and runs data pipelines.
Run this from the project root directory.

This script orchestrates both:
1. Bond Pipeline - Processes API Historical Excel files
2. Runs Pipeline - Processes Historical Runs Excel files
"""

import sys
from pathlib import Path

# Add bond_pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from bond_pipeline.pipeline import BondDataPipeline
from runs_pipeline.pipeline import RunsDataPipeline
from bond_pipeline.config import (
    DEFAULT_INPUT_DIR,
    RUNS_INPUT_DIR,
    HISTORICAL_PARQUET,
    RUNS_PARQUET,
    UNIVERSE_PARQUET,
    BQL_PARQUET,
)


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
            print(f"   ✓ Bond Pipeline completed successfully")
            print(f"   Output: {HISTORICAL_PARQUET}")
            print(f"   Universe: {UNIVERSE_PARQUET}")
            if process_bql:
                print(f"   BQL: {BQL_PARQUET}")
            print("-" * 70)
        else:
            print("\n" + "-" * 70)
            print(f"   ✗ Bond Pipeline failed")
            print(f"   Logs: {HISTORICAL_PARQUET.parent.parent / 'logs'}")
            print("-" * 70)
        
        return success
    except Exception as e:
        print(f"\n✗ Bond Pipeline error: {str(e)}")
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
            print(f"   ✓ Runs Pipeline completed successfully")
            print(f"   Output: {RUNS_PARQUET}")
            print("-" * 70)
        else:
            print("\n" + "-" * 70)
            print(f"   ✗ Runs Pipeline failed")
            print(f"   Logs: {RUNS_PARQUET.parent.parent / 'logs'}")
            print("-" * 70)
        
        return success
    except Exception as e:
        print(f"\n✗ Runs Pipeline error: {str(e)}")
        return False


def main():
    """Run the pipeline(s) with user selection."""
    
    print("\n" + "=" * 70)
    print("  DATA PIPELINE ORCHESTRATOR")
    print("=" * 70)
    
    # Ask user which pipeline(s) to run
    print("\nSelect pipeline(s) to run:")
    print("  [1] Bond Pipeline only")
    print("  [2] Runs Pipeline only")
    print("  [3] Both Pipelines (default)")
    
    pipeline_choice = input("\nChoice (1, 2, or 3): ").strip() or "3"
    
    if pipeline_choice not in ['1', '2', '3']:
        print("Invalid choice. Using default: Both Pipelines")
        pipeline_choice = '3'
    
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

    if pipeline_choice in ['1', '3']:
        bql_choice = input("\nProcess BQL workbook as part of Bond Pipeline? (Y/n): ").strip().lower()
        process_bql = bql_choice != 'n'
    
    # Run selected pipeline(s)
    if pipeline_choice in ['1', '3']:
        # Run Bond Pipeline
        results['bond'] = run_bond_pipeline(mode, process_bql)
    
    if pipeline_choice in ['2', '3']:
        # Run Runs Pipeline
        results['runs'] = run_runs_pipeline(mode)
    
    # Summary
    print("\n" + "=" * 70)
    print("  EXECUTION SUMMARY")
    print("=" * 70)
    
    if 'bond' in results:
        status = "✓ SUCCESS" if results['bond'] else "✗ FAILED"
        print(f"\n  Bond Pipeline: {status}")
    
    if 'runs' in results:
        status = "✓ SUCCESS" if results['runs'] else "✗ FAILED"
        print(f"  Runs Pipeline: {status}")
    
    print("=" * 70)
    
    # Return exit code
    if all(results.values()):
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
