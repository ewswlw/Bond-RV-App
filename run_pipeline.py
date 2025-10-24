"""
Simple script to run the bond data pipeline.
Run this from the project root directory.
"""

import sys
from pathlib import Path

# Add bond_pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "bond_pipeline"))

from bond_pipeline.pipeline import BondDataPipeline
from bond_pipeline.config import DEFAULT_INPUT_DIR, HISTORICAL_PARQUET, UNIVERSE_PARQUET


def main():
    """Run the pipeline with default settings."""

    print("=" * 80)
    print("BOND DATA PIPELINE - AUTOMATED RUN")
    print("=" * 80)
    print(f"\nInput Directory: {DEFAULT_INPUT_DIR}")
    print(f"Output Directory: {HISTORICAL_PARQUET.parent}")

    # Ask user for mode
    print("\nSelect processing mode:")
    print("  1. OVERRIDE - Rebuild everything from scratch (recommended for first run)")
    print("  2. APPEND   - Add only new dates (for daily updates)")

    choice = input("\nEnter choice (1 or 2) [default: 2]: ").strip()

    if choice == '1':
        mode = 'override'
    else:
        mode = 'append'

    print(f"\nRunning pipeline in {mode.upper()} mode...")
    print("=" * 80)

    # Run pipeline
    pipeline = BondDataPipeline(DEFAULT_INPUT_DIR, mode)
    success = pipeline.run()

    if success:
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nOutput files:")
        print(f"  Historical: {HISTORICAL_PARQUET}")
        print(f"  Universe:   {UNIVERSE_PARQUET}")
        print(f"\nLogs: {HISTORICAL_PARQUET.parent.parent / 'logs'}")
        return 0
    else:
        print("\n" + "=" * 80)
        print("PIPELINE FAILED!")
        print("=" * 80)
        print(f"\nCheck logs for details: {HISTORICAL_PARQUET.parent.parent / 'logs'}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
