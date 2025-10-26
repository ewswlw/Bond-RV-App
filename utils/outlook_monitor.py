"""
Outlook Email Monitor Utility

Monitors the "RUNS" folder in Outlook inbox for emails.
Uses pywin32 COM automation to interact with Windows desktop Outlook.

Created: October 24, 2025 12:30 PM
Updated: October 26, 2025 - Added comprehensive logging
"""

import win32com.client
from datetime import datetime
from typing import List, Dict, Optional, Set
import time
import csv
import os
from pathlib import Path
import re
from collections import defaultdict
import logging


class OutlookMonitor:
    """
    Monitor and access emails from Outlook using COM automation.

    Requires:
    - Windows OS
    - Microsoft Outlook installed and configured
    - pywin32 package
    """

    def __init__(self, email_address: str = "eddy.winiarz@ytmcapital.com", logger: Optional[logging.Logger] = None):
        """
        Initialize connection to Outlook.

        Args:
            email_address: Email account to monitor (default: eddy.winiarz@ytmcapital.com)
            logger: Optional logger instance for detailed logging
        """
        self.email_address = email_address
        self.outlook = None
        self.namespace = None
        self.inbox = None
        self.runs_folder = None
        self.logger = logger or logging.getLogger(__name__)

    def connect(self):
        """Establish connection to Outlook application."""
        try:
            self.outlook = win32com.client.Dispatch("Outlook.Application")
            self.namespace = self.outlook.GetNamespace("MAPI")
            print(f"Connected to Outlook for account: {self.email_address}")
            return True
        except Exception as e:
            print(f"Error connecting to Outlook: {e}")
            return False

    def get_runs_folder(self, folder_name: str = "RUNS") -> bool:
        """
        Navigate to the RUNS folder inside Inbox.

        Args:
            folder_name: Name of the subfolder in Inbox (default: "RUNS")

        Returns:
            True if folder found, False otherwise
        """
        try:
            # Get default inbox (olFolderInbox = 6)
            self.inbox = self.namespace.GetDefaultFolder(6)
            print(f"Accessing Inbox: {self.inbox.Name}")

            # Find the RUNS subfolder
            for folder in self.inbox.Folders:
                if folder.Name.upper() == folder_name.upper():
                    self.runs_folder = folder
                    print(f"Found folder: {folder_name} ({self.runs_folder.Items.Count} items)")
                    return True

            print(f"Folder '{folder_name}' not found in Inbox")
            print(f"Available folders: {[f.Name for f in self.inbox.Folders]}")
            return False

        except Exception as e:
            print(f"Error accessing folders: {e}")
            return False

    def list_emails(self, limit: Optional[int] = None) -> List[Dict]:
        """
        List all emails in the RUNS folder.

        Args:
            limit: Maximum number of emails to return (None = all)

        Returns:
            List of dictionaries containing email metadata
        """
        if not self.runs_folder:
            print("RUNS folder not connected. Call get_runs_folder() first.")
            return []

        emails = []
        items = self.runs_folder.Items
        items.Sort("[ReceivedTime]", True)  # Sort by received time, descending

        count = 0
        for item in items:
            try:
                # Check if it's a MailItem (type 43)
                if item.Class == 43:
                    email_data = {
                        'subject': item.Subject,
                        'sender': item.SenderName,
                        'sender_email': item.SenderEmailAddress,
                        'received_time': item.ReceivedTime.strftime('%Y-%m-%d %H:%M:%S'),
                        'size_kb': round(item.Size / 1024, 2),
                        'has_attachments': item.Attachments.Count > 0,
                        'attachment_count': item.Attachments.Count,
                        'unread': item.UnRead,
                        'body_preview': item.Body[:100].replace('\n', ' ') if item.Body else ''
                    }
                    emails.append(email_data)
                    count += 1

                    if limit and count >= limit:
                        break

            except Exception as e:
                print(f"Error processing email: {e}")
                continue

        return emails

    def get_latest_email(self) -> Optional[Dict]:
        """
        Get the most recent email from RUNS folder.

        Returns:
            Dictionary containing email metadata, or None if no emails
        """
        emails = self.list_emails(limit=1)
        return emails[0] if emails else None

    def monitor_new_emails(self, check_interval: int = 60, callback=None):
        """
        Continuously monitor for new emails.

        Args:
            check_interval: Seconds between checks (default: 60)
            callback: Optional function to call when new email detected
                     Function signature: callback(email_data: Dict)
        """
        if not self.runs_folder:
            print("RUNS folder not connected. Call get_runs_folder() first.")
            return

        print(f"\nMonitoring RUNS folder for new emails (checking every {check_interval}s)")
        print("Press Ctrl+C to stop\n")

        # Get initial count
        last_count = self.runs_folder.Items.Count

        try:
            while True:
                # Refresh folder
                self.runs_folder = None
                if not self.get_runs_folder():
                    print("Lost connection to folder, retrying...")
                    time.sleep(check_interval)
                    continue

                current_count = self.runs_folder.Items.Count

                if current_count > last_count:
                    new_emails = current_count - last_count
                    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                          f"Detected {new_emails} new email(s)!")

                    # Get the latest emails
                    latest_emails = self.list_emails(limit=new_emails)

                    for email in latest_emails:
                        print(f"\nNew Email:")
                        print(f"  Subject: {email['subject']}")
                        print(f"  From: {email['sender']} ({email['sender_email']})")
                        print(f"  Received: {email['received_time']}")
                        print(f"  Attachments: {email['attachment_count']}")

                        if callback:
                            callback(email)

                    last_count = current_count
                else:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                          f"No new emails (Total: {current_count})", end='\r')

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")

    def get_email_details(self, subject_filter: str = None) -> List[Dict]:
        """
        Get detailed information about emails matching a subject filter.

        Args:
            subject_filter: String to filter emails by subject (case-insensitive)

        Returns:
            List of detailed email dictionaries
        """
        if not self.runs_folder:
            print("RUNS folder not connected. Call get_runs_folder() first.")
            return []

        emails = []
        items = self.runs_folder.Items
        items.Sort("[ReceivedTime]", True)

        for item in items:
            try:
                if item.Class == 43:  # MailItem
                    if subject_filter and subject_filter.lower() not in item.Subject.lower():
                        continue

                    attachments = []
                    for att in item.Attachments:
                        attachments.append({
                            'filename': att.FileName,
                            'size_kb': round(att.Size / 1024, 2)
                        })

                    email_data = {
                        'subject': item.Subject,
                        'sender': item.SenderName,
                        'sender_email': item.SenderEmailAddress,
                        'received_time': item.ReceivedTime.strftime('%Y-%m-%d %H:%M:%S'),
                        'size_kb': round(item.Size / 1024, 2),
                        'attachments': attachments,
                        'unread': item.UnRead,
                        'body': item.Body
                    }
                    emails.append(email_data)

            except Exception as e:
                print(f"Error processing email: {e}")
                continue

        return emails

    def _sanitize_filename(self, text: str, max_length: int = 100) -> str:
        """
        Sanitize text for use in filename.

        Args:
            text: Text to sanitize
            max_length: Maximum filename length

        Returns:
            Sanitized filename-safe string
        """
        # Remove invalid Windows filename characters
        text = re.sub(r'[<>:"/\\|?*]', '', text)
        # Replace spaces and multiple underscores
        text = re.sub(r'\s+', '_', text)
        text = re.sub(r'_+', '_', text)
        # Truncate to max length
        return text[:max_length].strip('_')

    def _load_synced_emails(self, index_path: Path) -> Set[str]:
        """
        Load set of already-synced email IDs from CSV index.

        Args:
            index_path: Path to the CSV index file

        Returns:
            Set of email EntryIDs that have been synced
        """
        synced_ids = set()

        if not index_path.exists():
            return synced_ids

        try:
            with open(index_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    synced_ids.add(row['EntryID'])
        except Exception as e:
            print(f"Warning: Could not load index file: {e}")

        return synced_ids

    def _update_index(self, index_path: Path, email_id: str, subject: str,
                     received_time: str, filename: str):
        """
        Add a new email entry to the CSV index.

        Args:
            index_path: Path to the CSV index file
            email_id: Unique EntryID of the email
            subject: Email subject
            received_time: When email was received
            filename: Name of the saved .eml file
        """
        file_exists = index_path.exists()

        try:
            with open(index_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['EntryID', 'Subject', 'ReceivedTime', 'Filename', 'SyncedAt'])

                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    'EntryID': email_id,
                    'Subject': subject,
                    'ReceivedTime': received_time,
                    'Filename': filename,
                    'SyncedAt': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        except Exception as e:
            print(f"Warning: Could not update index: {e}")

    def _clear_all_data(self, output_dir: str, log_dir: str = None):
        """
        Clear all CSV files, sync index, and log files (full rebuild mode).

        Args:
            output_dir: Directory containing Outlook Data CSV files and sync index
            log_dir: Optional directory containing log files to clear
        """
        output_path = Path(output_dir)
        deleted_csvs = 0
        deleted_index = False
        deleted_logs = 0

        self.logger.info("=" * 80)
        self.logger.info("CLEARING ALL DATA (FULL REBUILD MODE)")
        self.logger.info("=" * 80)

        # Delete all Outlook Data CSV files
        if output_path.exists():
            for csv_file in output_path.glob("Outlook Data *.csv"):
                try:
                    csv_file.unlink()
                    deleted_csvs += 1
                    self.logger.info(f"Deleted: {csv_file.name}")
                except Exception as e:
                    self.logger.error(f"Failed to delete {csv_file.name}: {e}")

            # Delete sync index
            index_path = output_path / 'sync_index.csv'
            if index_path.exists():
                try:
                    index_path.unlink()
                    deleted_index = True
                    self.logger.info(f"Deleted: sync_index.csv")
                except Exception as e:
                    self.logger.error(f"Failed to delete sync_index.csv: {e}")

        # Delete log files if log_dir specified
        if log_dir:
            log_path = Path(log_dir)
            if log_path.exists():
                log_file = log_path / 'outlook_monitor.log'
                if log_file.exists():
                    try:
                        # Close existing handlers to release file lock
                        for handler in self.logger.handlers[:]:
                            if isinstance(handler, logging.FileHandler):
                                handler.close()
                                self.logger.removeHandler(handler)

                        log_file.unlink()
                        deleted_logs += 1
                        self.logger.info(f"Deleted: outlook_monitor.log")
                    except Exception as e:
                        self.logger.error(f"Failed to delete outlook_monitor.log: {e}")

        self.logger.info("")
        self.logger.info(f"Summary: Deleted {deleted_csvs} CSV files, "
                        f"{'sync index' if deleted_index else 'no sync index'}, "
                        f"{deleted_logs} log files")
        self.logger.info("=" * 80)
        self.logger.info("")

    def sync_emails_to_csv(self, output_dir: str,
                            progress_callback=None,
                            days_back: int = None) -> Dict[str, int]:
        """
        Sync all emails from RUNS folder to CSV files organized by date.
        Creates one CSV file per date (e.g., Outlook Data 10.24.2025.csv).
        Only syncs emails not already in the index (incremental sync).

        Args:
            output_dir: Directory to save CSV files
            progress_callback: Optional callback function(current, total, email_subject, skipped)
            days_back: If specified, only archive emails from X days ago to today (e.g., 2 = last 2 days)

        Returns:
            Dictionary with sync statistics: {'total', 'new', 'skipped', 'errors', 'files_created'}
        """
        start_time = datetime.now()

        if not self.runs_folder:
            print("RUNS folder not connected. Call get_runs_folder() first.")
            return {'total': 0, 'new': 0, 'skipped': 0, 'errors': 0, 'files_created': 0}

        # Setup paths
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        index_path = output_path / 'sync_index.csv'

        # Log sync start
        self.logger.info("=" * 80)
        self.logger.info(f"OUTLOOK EMAIL ARCHIVING - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
        self.logger.info(f"Mode: {'Last ' + str(days_back) + ' days' if days_back else 'Incremental sync'}")
        self.logger.info(f"Input: Outlook RUNS folder ({self.email_address})")
        self.logger.info(f"Output: {output_dir}")
        self.logger.info("")

        # Load already-synced email IDs
        synced_ids = self._load_synced_emails(index_path)
        self.logger.info(f"--- SCAN PHASE ---")
        print(f"Found {len(synced_ids)} already-synced emails in index")

        # Calculate date range if days_back is specified
        cutoff_date = None
        if days_back is not None:
            from datetime import timedelta
            import pywintypes
            # Create timezone-aware datetime to match Outlook's ReceivedTime
            cutoff_date_naive = datetime.now() - timedelta(days=days_back)
            cutoff_date_naive = cutoff_date_naive.replace(hour=0, minute=0, second=0, microsecond=0)
            # Convert to pywintypes.datetime (timezone-aware)
            cutoff_date = pywintypes.Time(cutoff_date_naive)
            print(f"Filtering emails from {cutoff_date_naive.strftime('%Y-%m-%d')} to today")

        # Get emails (filtered by date if days_back specified)
        items = self.runs_folder.Items

        # Apply Outlook filter if days_back is specified
        if days_back is not None:
            # Format date for Outlook filter (MM/DD/YYYY format)
            filter_date = cutoff_date_naive.strftime('%m/%d/%Y')
            # Outlook filter syntax for ReceivedTime >= cutoff_date
            outlook_filter = f"[ReceivedTime] >= '{filter_date}'"
            items = items.Restrict(outlook_filter)
            print(f"Applied Outlook filter: {outlook_filter}")

        total_items = items.Count

        stats = {'total': total_items, 'new': 0, 'skipped': 0, 'errors': 0, 'files_created': 0, 'files_updated': 0}

        # Log scan results
        self.logger.info(f"Total emails in folder: {total_items:,}")
        self.logger.info(f"Already synced: {len(synced_ids):,}")
        self.logger.info("")

        # Group emails by date
        emails_by_date = defaultdict(list)

        self.logger.info(f"--- PROCESSING PHASE ---")
        print(f"Processing {total_items} emails from RUNS folder...")

        # Collect all new emails grouped by date
        for i in range(1, total_items + 1):
            try:
                item = items[i]

                # Only process MailItems
                if item.Class != 43:
                    stats['skipped'] += 1
                    continue

                # Check if already synced
                entry_id = item.EntryID
                if entry_id in synced_ids:
                    stats['skipped'] += 1
                    if progress_callback:
                        progress_callback(i, total_items, item.Subject, skipped=True)
                    continue

                # Extract email data
                received_time = item.ReceivedTime
                date_key = received_time.strftime('%Y-%m-%d')

                # Get attachments info
                attachments = []
                for att in item.Attachments:
                    attachments.append(f"{att.FileName}({round(att.Size/1024, 2)}KB)")

                email_data = {
                    'EntryID': entry_id,
                    'ReceivedDate': received_time.strftime('%Y-%m-%d'),
                    'ReceivedTime': received_time.strftime('%H:%M:%S'),
                    'ReceivedDateTime': received_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Subject': item.Subject,
                    'SenderName': item.SenderName,
                    'SenderEmail': item.SenderEmailAddress,
                    'SizeKB': round(item.Size / 1024, 2),
                    'AttachmentCount': item.Attachments.Count,
                    'Attachments': '; '.join(attachments) if attachments else '',
                    'Unread': item.UnRead,
                    'Body': item.Body[:5000] if item.Body else ''  # First 5000 chars
                }

                emails_by_date[date_key].append(email_data)
                stats['new'] += 1

                if progress_callback:
                    progress_callback(i, total_items, item.Subject, skipped=False)

            except Exception as e:
                stats['errors'] += 1
                if stats['errors'] <= 10:
                    error_msg = f"Error processing email at index {i}: {e}"
                    print(f"\n{error_msg}")
                    self.logger.error(error_msg)
                elif stats['errors'] == 11:
                    print(f"\n... suppressing further error messages")
                continue

        # Log per-date summary
        if emails_by_date:
            self.logger.info("")
            for date_key in sorted(emails_by_date.keys()):
                count = len(emails_by_date[date_key])
                self.logger.info(f"Date: {date_key} | {count} emails")

        # Write CSV files per date
        self.logger.info("")
        self.logger.info(f"--- WRITING FILES ---")
        print(f"\n\nWriting {len(emails_by_date)} CSV files...")

        csv_fieldnames = ['EntryID', 'ReceivedDate', 'ReceivedTime', 'ReceivedDateTime',
                         'Subject', 'SenderName', 'SenderEmail', 'SizeKB',
                         'AttachmentCount', 'Attachments', 'Unread', 'Body']

        for date_key, emails in sorted(emails_by_date.items()):
            # Convert YYYY-MM-DD to MM.DD.YYYY format
            date_obj = datetime.strptime(date_key, '%Y-%m-%d')
            formatted_date = date_obj.strftime('%m.%d.%Y')
            csv_filename = output_path / f"Outlook Data {formatted_date}.csv"

            # Check if file exists to determine if we need header
            file_exists = csv_filename.exists()

            try:
                with open(csv_filename, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)

                    if not file_exists:
                        writer.writeheader()
                        stats['files_created'] += 1
                    else:
                        stats['files_updated'] += 1

                    writer.writerows(emails)

                # Update sync index for all emails in this batch
                for email in emails:
                    self._update_index(
                        index_path,
                        email['EntryID'],
                        email['Subject'],
                        email['ReceivedDateTime'],
                        f"Outlook Data {formatted_date}.csv"
                    )

                msg = f"  Wrote {len(emails)} emails to {csv_filename.name}"
                print(msg)
                self.logger.info(f"{csv_filename.name}: {len(emails)} emails {'(new file)' if not file_exists else '(updated)'}")

            except Exception as e:
                error_msg = f"Error writing CSV for {date_key}: {e}"
                print(f"\n{error_msg}")
                self.logger.error(error_msg)
                stats['errors'] += len(emails)

        # Final summary log
        duration = (datetime.now() - start_time).total_seconds()
        emails_per_sec = stats['new'] / duration if duration > 0 else 0

        self.logger.info("")
        self.logger.info(f"--- RESULTS ---")
        self.logger.info(f"Files created: {stats['files_created']}")
        self.logger.info(f"Files updated: {stats['files_updated']}")
        self.logger.info(f"Total emails archived: {stats['new']:,}")
        self.logger.info(f"Errors: {stats['errors']}")
        self.logger.info("")
        self.logger.info(f"Performance: {stats['new']} emails in {duration:.1f}s ({emails_per_sec:.1f} emails/sec)")
        self.logger.info("=" * 80)
        self.logger.info("")

        return stats


def print_email_summary(emails: List[Dict]):
    """Helper function to print email summary in a formatted table."""
    if not emails:
        print("No emails found.")
        return

    print(f"\nFound {len(emails)} email(s) in RUNS folder:\n")
    print("-" * 120)
    print(f"{'Received':<20} {'From':<30} {'Subject':<50} {'Atts':<5} {'Size':<8}")
    print("-" * 120)

    for email in emails:
        subject = email['subject'][:47] + "..." if len(email['subject']) > 50 else email['subject']
        sender = email['sender'][:27] + "..." if len(email['sender']) > 30 else email['sender']

        print(f"{email['received_time']:<20} "
              f"{sender:<30} "
              f"{subject:<50} "
              f"{email['attachment_count']:<5} "
              f"{email['size_kb']}KB")

    print("-" * 120)
