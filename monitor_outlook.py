"""
Outlook Email Monitor - CLI Wrapper

Archives emails from Outlook RUNS folder to CSV files organized by date.

Usage:
    python monitor_outlook.py                    # Archive new emails only (default)
    python monitor_outlook.py --all              # Archive ALL emails (rebuilds from scratch)

Created: October 24, 2025 12:30 PM
Updated: October 24, 2025 2:30 PM - Simplified to two modes only
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Add utils directory to path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

from outlook_monitor import OutlookMonitor

# Default output directory for email archiving (always used)
OUTPUT_DIR = r"C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Support Files\Outlook Runs"


def progress_reporter(current, total, subject, skipped=False):
    """Callback function to report sync progress."""
    percent = (current / total) * 100
    status = "SKIP" if skipped else "NEW"
    subject_short = subject[:60] + "..." if len(subject) > 60 else subject
    print(f"\r[{percent:5.1f}%] {current}/{total} - {status}: {subject_short}", end='', flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Archive Outlook RUNS folder emails to CSV files (one CSV per date)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                         Archive new emails only (incremental, default)
  %(prog)s --all                   Archive ALL emails (rebuilds everything)
  %(prog)s --days 2                Archive emails from last 2 days to today
  %(prog)s --days 7                Archive emails from last 7 days to today
        """
    )

    parser.add_argument('--all', action='store_true',
                       help='Archive ALL emails (rebuilds from scratch, deletes existing index)')
    parser.add_argument('--days', type=int, default=None,
                       help='Archive emails from X days ago to today (e.g., --days 2 for last 2 days)')

    args = parser.parse_args()

    # Initialize monitor
    monitor = OutlookMonitor(email_address="eddy.winiarz@ytmcapital.com")

    print("Connecting to Outlook...")
    if not monitor.connect():
        print("Failed to connect to Outlook. Ensure Outlook is installed and configured.")
        return

    print("Accessing RUNS folder...")
    if not monitor.get_runs_folder("RUNS"):
        print("Failed to access RUNS folder. Check that the folder exists in your Inbox.")
        return

    # Handle --all flag by deleting sync index
    if args.all:
        index_path = Path(OUTPUT_DIR) / 'sync_index.csv'
        if index_path.exists():
            print(f"Removing existing sync index (rebuild mode): {index_path}")
            index_path.unlink()

    # Archive emails
    print(f"\nArchiving emails to: {OUTPUT_DIR}")
    if args.days:
        print(f"Date range: Last {args.days} days to today")
    elif args.all:
        print("Mode: Archive ALL emails (rebuild from scratch)")
    else:
        print("Mode: Incremental (new emails only)")
    print("=" * 80)

    start_time = datetime.now()
    stats = monitor.sync_emails_to_csv(
        output_dir=OUTPUT_DIR,
        progress_callback=progress_reporter,
        days_back=args.days
    )

    print("\n")  # New line after progress
    print("=" * 80)
    print("Archive Complete!")
    print(f"  Total emails in folder:   {stats['total']}")
    print(f"  New emails archived:      {stats['new']}")
    print(f"  Already synced (skipped): {stats['skipped']}")
    print(f"  CSV files created:        {stats['files_created']}")
    print(f"  Errors:                   {stats['errors']}")
    print(f"  Duration:                 {(datetime.now() - start_time).total_seconds():.1f}s")
    print(f"\nArchive location: {OUTPUT_DIR}")
    print(f"Index file: {Path(OUTPUT_DIR) / 'sync_index.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
