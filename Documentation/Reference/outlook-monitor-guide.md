# Outlook Email Monitor - Quick Reference Guide

**Last Updated**: October 26, 2025 6:30 PM

This is a quick reference guide for the Outlook Email Monitor utility. For comprehensive technical documentation including architecture, implementation details, and optimization strategies, see **[Outlook Email Archiving System](outlook-email-archiving.md)**.

## Overview

The Outlook Email Monitor is a standalone utility that uses Windows COM automation (pywin32) to archive emails from the "RUNS" folder (eddy.winiarz@ytmcapital.com) to CSV files organized by date. It supports incremental sync, date-range filtering, and full rebuild modes with comprehensive logging to `bond_data/logs/outlook_monitor.log`.

## Prerequisites

1. **Windows OS** - COM automation only works on Windows
2. **Microsoft Outlook** - Desktop client must be installed and configured
3. **Virtual Environment** - Bond-RV-App virtual environment activated
4. **Dependencies Installed** - Run `pip install -r requirements.txt` to install pywin32

## Installation

```bash
# Activate virtual environment
Bond-RV-App\Scripts\activate

# Install/update dependencies (includes pywin32)
pip install -r requirements.txt
```

## Quick Start

All commands assume you're in the project root directory with the virtual environment activated.

### Archive New Emails (Default)

```bash
python monitor_outlook.py
```

Archives only emails not already synced (incremental mode).

### Archive Last X Days

```bash
# Last 2 days
python monitor_outlook.py --days 2

# Last 7 days
python monitor_outlook.py --days 7
```

Only processes emails from X days ago to today (much faster than full archive).

### Rebuild Entire Archive

```bash
python monitor_outlook.py --rebuild
```

Deletes all CSV files, sync index, and log files, then rebuilds the entire archive from scratch.

### Show Latest Email

```bash
python monitor_outlook.py latest
```

Displays detailed information about the most recent email:
- Subject
- Sender (name and email address)
- Received timestamp
- Size in KB
- Attachment count
- Read/unread status
- Body preview (first 100 characters)

### Monitor for New Emails (Continuous)

```bash
# Check for new emails every 60 seconds (default)
python monitor_outlook.py monitor

# Check every 30 seconds
python monitor_outlook.py monitor --interval 30
```

The monitor runs continuously and:
- Checks the RUNS folder at the specified interval
- Displays a timestamp with current status
- Alerts when new emails arrive with full details
- Press Ctrl+C to stop monitoring

### Search Emails by Subject

```bash
# Find all emails with "API" in the subject
python monitor_outlook.py search --subject "API"

# Case-insensitive partial matching
python monitor_outlook.py search --subject "bond data"
```

Shows detailed information for matching emails including:
- Full metadata (subject, sender, date, size)
- Complete attachment list with filenames and sizes
- Body preview (first 500 characters)

## Technical Details

### How It Works

1. **COM Automation** - Uses `win32com.client` to connect to the local Outlook application
2. **MAPI Namespace** - Accesses Outlook folders through MAPI (Messaging API)
3. **Folder Navigation** - Finds the "RUNS" subfolder within the default Inbox
4. **Email Retrieval** - Reads email metadata and content without moving/modifying emails

### File Structure

```
Bond-RV-App/
├── utils/
│   └── outlook_monitor.py      # Core OutlookMonitor class
├── monitor_outlook.py           # CLI wrapper (project root)
└── requirements.txt             # Includes pywin32>=305
```

### OutlookMonitor Class

The `OutlookMonitor` class in `utils/outlook_monitor.py` provides programmatic access:

```python
from utils.outlook_monitor import OutlookMonitor

# Initialize and connect
monitor = OutlookMonitor(email_address="eddy.winiarz@ytmcapital.com")
monitor.connect()
monitor.get_runs_folder("RUNS")

# Get emails
emails = monitor.list_emails(limit=10)
latest = monitor.get_latest_email()
filtered = monitor.get_email_details(subject_filter="API")

# Monitor with custom callback
def handle_new_email(email_data):
    print(f"New email: {email_data['subject']}")

monitor.monitor_new_emails(check_interval=60, callback=handle_new_email)
```

### Logging

All operations are logged to `bond_data/logs/outlook_monitor.log` with comprehensive details:

**Log Contents**:
- Scan phase: Total emails found, date range filtering
- Processing phase: Emails processed, new vs skipped, progress updates
- Summary statistics: Total/new/skipped counts, files created/updated, errors
- Performance metrics: Emails per second, total duration
- Error details: Failed email processing with subject and EntryID

**Log Format**:
- Formatted sections with separators (80-char lines)
- UTF-8 encoding with proper Unicode support
- Timestamped entries for each run
- Console shows only errors; file has full details

**Example Log Output**:
```
================================================================================
OUTLOOK EMAIL ARCHIVING - 2025-10-26 14:30:22
================================================================================
Mode: Incremental (new emails only)

SCAN PHASE
--------------------------------------------------------------------------------
Total emails in RUNS folder: 803
Date range: 2025-10-22 to 2025-10-26

PROCESSING PHASE
--------------------------------------------------------------------------------
[85.2%] 684/803 - NEW: Bond market update for Oct 26...

RESULTS
--------------------------------------------------------------------------------
Total emails:             803
New emails archived:      127
Already synced (skipped): 676
CSV files created:        1
CSV files updated:        2
Errors:                   0

PERFORMANCE
--------------------------------------------------------------------------------
Duration:      2.3 seconds
Emails/second: 349.1
```

## Troubleshooting

### "Failed to connect to Outlook"

**Cause**: Outlook application is not running or not configured.

**Solution**:
1. Open Microsoft Outlook desktop application
2. Ensure your email account is configured and syncing
3. Try running the monitor again

### "Folder 'RUNS' not found in Inbox"

**Cause**: The RUNS folder doesn't exist or has a different name.

**Solution**:
1. Check Outlook to confirm the folder exists
2. Verify the folder is a direct subfolder of Inbox (not nested deeper)
3. The error message will show available folders - check the spelling
4. Folder name matching is case-insensitive

### "ImportError: No module named win32com"

**Cause**: pywin32 package not installed.

**Solution**:
```bash
# Activate virtual environment first
Bond-RV-App\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Monitor Shows "No new emails" Continuously

**Cause**: This is normal behavior - it's checking but nothing new has arrived.

**Solution**: The monitor is working correctly. It will alert when new emails are detected. The counter on the right shows the current total count.

## Integration with Bond Pipeline

The Outlook Monitor is designed as a **standalone utility** and is not integrated into the bond data pipeline. However, you can extend it for automation:

### Example: Trigger Pipeline on New Email

```python
from utils.outlook_monitor import OutlookMonitor
import subprocess

def run_pipeline_on_new_email(email_data):
    """Run pipeline when email with 'API' in subject arrives."""
    if 'API' in email_data['subject']:
        print(f"Triggering pipeline for: {email_data['subject']}")
        subprocess.run(['python', 'run_pipeline.py'], input='2\n', text=True)

monitor = OutlookMonitor()
monitor.connect()
monitor.get_runs_folder()
monitor.monitor_new_emails(check_interval=300, callback=run_pipeline_on_new_email)
```

### Example: Download Attachments

```python
# This functionality can be added to OutlookMonitor class
# by accessing item.Attachments and calling SaveAsFile()
```

## Limitations

1. **Windows Only** - COM automation requires Windows OS
2. **Local Outlook Required** - Won't work with web-based Outlook or other email clients
3. **No Cloud Access** - Only accesses emails synced to local Outlook
4. **Read-Only** - Current implementation doesn't modify/delete emails (by design)
5. **Polling-Based** - Monitor uses polling, not real-time push notifications

## Future Enhancements

Potential features to add:

- Attachment download functionality
- Email filtering by date range
- Mark emails as read/unread
- Move emails between folders
- Extract and parse Excel attachments
- Auto-trigger pipeline when specific emails arrive

## Related Documentation

- **Setup Guide**: `Documentation/Setup/virtual-environment-guide.md`
- **Testing Guide**: `Documentation/Reference/testing-guide.md`
- **Project Architecture**: `Documentation/Architecture/pipeline-architecture.md`

## Support

For issues or questions:
1. Check Troubleshooting section above
2. Verify Outlook is running and configured
3. Ensure virtual environment is activated
4. Check that pywin32 is installed: `pip show pywin32`
