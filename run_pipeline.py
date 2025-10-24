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

    print("\n" + "=" * 70)
    print("  BOND DATA PIPELINE")
    print("=" * 70)

    # Ask user for mode
    print("\nSelect processing mode:")
    print("  [1] Override - Rebuild everything from scratch")
    print("  [2] Append   - Add only new dates (default)")

    choice = input("\nChoice (1 or 2): ").strip()

    if choice == '1':
        mode = 'override'
        mode_display = 'OVERRIDE'
    else:
        mode = 'append'
        mode_display = 'APPEND'

    print(f"\n>> Starting pipeline in {mode_display} mode...")

    # Run pipeline
    pipeline = BondDataPipeline(DEFAULT_INPUT_DIR, mode)
    success = pipeline.run()

    if success:
        print("\n" + "-" * 70)
        print(f"   Output: {HISTORICAL_PARQUET.parent}")
        print("-" * 70)
        return 0
    else:
        print("\n" + "-" * 70)
        print(f"   Logs: {HISTORICAL_PARQUET.parent.parent / 'logs'}")
        print("-" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
