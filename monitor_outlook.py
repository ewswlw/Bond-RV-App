"""
Outlook Email Monitor - CLI Wrapper

Archives emails from Outlook RUNS folder to CSV files organized by date.

Usage:
    python monitor_outlook.py                    # Archive new emails only (default)
    python monitor_outlook.py --rebuild          # Full rebuild (delete all data, then re-archive everything)
    python monitor_outlook.py --days 2           # Archive last 2 days to today

Created: October 24, 2025 12:30 PM
Updated: October 26, 2025 - Added --rebuild option and comprehensive logging
"""

import argparse
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add utils directory to path
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

from outlook_monitor import OutlookMonitor

# Default output directory for email archiving (always used)
OUTPUT_DIR = r"C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Support Files\Outlook Runs"

# Log directory
LOG_DIR = r"C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Eddy\Python Projects\Bond-RV-App\bond_data\logs"


def setup_logging(log_dir: str) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory to save log files

    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger('outlook_monitor')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler
    log_file = log_path / 'outlook_monitor.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Console handler (only for errors)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)

    # Formatter
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


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
  %(prog)s --rebuild               Full rebuild (delete all data, re-archive everything)
  %(prog)s --days 2                Archive emails from last 2 days to today
  %(prog)s --days 7                Archive emails from last 7 days to today
        """
    )

    parser.add_argument('--rebuild', action='store_true',
                       help='Full rebuild: delete all CSV files, sync index, and logs, then re-archive everything')
    parser.add_argument('--days', type=int, default=None,
                       help='Archive emails from X days ago to today (e.g., --days 2 for last 2 days)')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(LOG_DIR)

    # Initialize monitor with logger
    monitor = OutlookMonitor(email_address="eddy.winiarz@ytmcapital.com", logger=logger)

    print("Connecting to Outlook...")
    if not monitor.connect():
        print("Failed to connect to Outlook. Ensure Outlook is installed and configured.")
        return

    print("Accessing RUNS folder...")
    if not monitor.get_runs_folder("RUNS"):
        print("Failed to access RUNS folder. Check that the folder exists in your Inbox.")
        return

    # Handle --rebuild flag by clearing all data
    if args.rebuild:
        print("\n*** FULL REBUILD MODE ***")
        print("Clearing all CSV files, sync index, and logs...")
        monitor._clear_all_data(OUTPUT_DIR, LOG_DIR)
        print("Data cleared. Re-archiving all emails...\n")

    # Archive emails
    print(f"\nArchiving emails to: {OUTPUT_DIR}")
    if args.days:
        print(f"Date range: Last {args.days} days to today")
    elif args.rebuild:
        print("Mode: Full rebuild (re-archive all emails)")
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
    print(f"  CSV files updated:        {stats['files_updated']}")
    print(f"  Errors:                   {stats['errors']}")
    print(f"  Duration:                 {(datetime.now() - start_time).total_seconds():.1f}s")
    print(f"\nArchive location: {OUTPUT_DIR}")
    print(f"Index file: {Path(OUTPUT_DIR) / 'sync_index.csv'}")
    print(f"Log file: {Path(LOG_DIR) / 'outlook_monitor.log'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
