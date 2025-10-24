# Outlook Email Archiving System

**Created**: October 24, 2025 2:45 PM
**Last Updated**: October 24, 2025 8:05 PM

## Overview

Automated email archiving system that syncs emails from the Outlook "RUNS" folder (eddy.winiarz@ytmcapital.com) to CSV files organized by date. Uses Windows COM automation (pywin32) for direct Outlook integration without cloud APIs.

## Key Features

- **Incremental Sync**: Only archives new emails not already processed
- **Date-Based Organization**: One CSV file per date (e.g., "Outlook Data 10.24.2025.csv")
- **Efficient Filtering**: Uses Outlook's native filtering to process only emails in specified date range
- **Index Tracking**: Maintains sync_index.csv to track processed emails by EntryID
- **Three Operating Modes**:
  - Default: Archive new emails only (incremental)
  - `--all`: Rebuild entire archive from scratch
  - `--days X`: Archive emails from X days ago to today

## Architecture

### File Structure

```
Bond-RV-App/
├── utils/
│   └── outlook_monitor.py           # Core OutlookMonitor class
├── monitor_outlook.py                # CLI wrapper (project root)
└── Output Directory/
    ├── Outlook Data MM.DD.YYYY.csv  # One CSV per date
    └── sync_index.csv               # Tracking file for incremental sync
```

### Output Directory (Fixed Path)

```
C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Support Files\Outlook Runs
```

This path is hardcoded in `monitor_outlook.py` as `OUTPUT_DIR` constant.

## Technical Implementation

### 1. COM Automation Connection

```python
# Connect to Outlook via Windows COM
outlook = win32com.client.Dispatch("Outlook.Application")
namespace = outlook.GetNamespace("MAPI")
inbox = namespace.GetDefaultFolder(6)  # olFolderInbox = 6

# Navigate to RUNS subfolder
for folder in inbox.Folders:
    if folder.Name.upper() == "RUNS":
        runs_folder = folder
```

**Requirements**:
- Windows OS only
- Microsoft Outlook desktop client installed and configured
- pywin32 package (`pip install pywin32>=305`)

### 2. Efficient Date Filtering

**Key Optimization**: Instead of checking all emails individually, the system uses Outlook's native `Restrict()` filter to pre-filter emails by date.

```python
# Before optimization: Process all 37,879 emails
items = runs_folder.Items

# After optimization: Process only emails in date range (e.g., 803 emails for 2 days)
if days_back is not None:
    cutoff_date_naive = datetime.now() - timedelta(days=days_back)
    cutoff_date_naive = cutoff_date_naive.replace(hour=0, minute=0, second=0, microsecond=0)

    # Outlook filter syntax
    filter_date = cutoff_date_naive.strftime('%m/%d/%Y')
    outlook_filter = f"[ReceivedTime] >= '{filter_date}'"
    items = items.Restrict(outlook_filter)
```

**Performance Impact**:
- Without filter: ~37,879 emails checked
- With 2-day filter: ~803 emails checked (96% reduction)

### 3. Incremental Sync Logic

**Tracking File**: `sync_index.csv`

```csv
EntryID,Subject,ReceivedTime,Filename,SyncedAt
00000000...,Sample Email,2025-10-24 14:30:00,Outlook Data 10.24.2025.csv,2025-10-24 14:35:22
```

**Process**:

1. **Load Existing Index**:
   ```python
   synced_ids = set()
   with open(index_path, 'r') as f:
       reader = csv.DictReader(f)
       for row in reader:
           synced_ids.add(row['EntryID'])
   ```

2. **Check Each Email**:
   ```python
   entry_id = item.EntryID
   if entry_id in synced_ids:
       stats['skipped'] += 1
       continue  # Skip already-synced emails
   ```

3. **Update Index After Sync**:
   ```python
   with open(index_path, 'a') as f:
       writer = csv.DictWriter(f, fieldnames=['EntryID', 'Subject', ...])
       writer.writerow({
           'EntryID': entry_id,
           'Subject': subject,
           'ReceivedTime': received_time,
           'Filename': f"Outlook Data {formatted_date}.csv",
           'SyncedAt': datetime.now()
       })
   ```

### 4. CSV Organization by Date

**Grouping Strategy**: Emails are grouped by `ReceivedDate` before writing to disk.

```python
emails_by_date = defaultdict(list)

for item in items:
    received_time = item.ReceivedTime
    date_key = received_time.strftime('%Y-%m-%d')  # e.g., "2025-10-24"

    email_data = {...}
    emails_by_date[date_key].append(email_data)

# Write one CSV per date
for date_key, emails in sorted(emails_by_date.items()):
    date_obj = datetime.strptime(date_key, '%Y-%m-%d')
    formatted_date = date_obj.strftime('%m.%d.%Y')  # "10.24.2025"
    csv_filename = f"Outlook Data {formatted_date}.csv"

    with open(csv_filename, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(emails)
```

### 5. CSV Schema

Each CSV file contains the following columns:

| Column           | Type    | Description                                    |
|------------------|---------|------------------------------------------------|
| EntryID          | String  | Unique Outlook identifier for email           |
| ReceivedDate     | Date    | Date email was received (YYYY-MM-DD)          |
| ReceivedTime     | Time    | Time email was received (HH:MM:SS)            |
| ReceivedDateTime | DateTime| Full timestamp (YYYY-MM-DD HH:MM:SS)          |
| Subject          | String  | Email subject line                             |
| SenderName       | String  | Display name of sender                         |
| SenderEmail      | String  | Email address of sender                        |
| SizeKB           | Float   | Email size in kilobytes                        |
| AttachmentCount  | Integer | Number of attachments                          |
| Attachments      | String  | Semicolon-separated list: "file1.xlsx(25KB); file2.pdf(100KB)" |
| Unread           | Boolean | Whether email is unread (True/False)          |
| Body             | String  | First 5000 characters of email body           |

### 6. Timezone Handling

**Challenge**: Outlook's `ReceivedTime` is timezone-aware (pywintypes.datetime), but Python's `datetime.now()` is timezone-naive.

**Solution**: Convert Python datetime to pywintypes.Time for comparison.

```python
import pywintypes

# Create timezone-aware datetime for filtering
cutoff_date_naive = datetime.now() - timedelta(days=days_back)
cutoff_date_naive = cutoff_date_naive.replace(hour=0, minute=0, second=0, microsecond=0)

# Convert to timezone-aware for comparison with Outlook datetimes
cutoff_date = pywintypes.Time(cutoff_date_naive)
```

**Why This Matters**: Without timezone-aware conversion, comparisons fail with:
```
TypeError: can't compare offset-naive and offset-aware datetimes
```

## Usage

### Prerequisites

```bash
# Activate virtual environment
Bond-RV-App\Scripts\activate

# Ensure dependencies are installed
pip install -r requirements.txt
```

### Command-Line Interface

**1. Incremental Archive (Default)**

Archives only new emails not already synced:

```bash
python monitor_outlook.py
```

Output:
```
Archiving emails to: C:\Users\Eddy\...\Support Files\Outlook Runs
Mode: Incremental (new emails only)
================================================================================
Found 1250 already-synced emails in index
Processing 37879 emails from RUNS folder...
  Total emails in folder:   37879
  New emails archived:      45
  Already synced (skipped): 37834
  CSV files created:        2
```

**2. Archive Last X Days**

Archives only emails from X days ago to today:

```bash
# Last 2 days
python monitor_outlook.py --days 2

# Last 7 days
python monitor_outlook.py --days 7
```

Output:
```
Date range: Last 2 days to today
================================================================================
Filtering emails from 2025-10-22 to today
Applied Outlook filter: [ReceivedTime] >= '10/22/2025'
Processing 803 emails from RUNS folder...
```

**Key Benefit**: Only processes emails in date range (803 instead of 37,879 for 2 days).

**3. Rebuild Entire Archive**

Deletes sync index and rebuilds all CSVs from scratch:

```bash
python monitor_outlook.py --all
```

Output:
```
Removing existing sync index (rebuild mode): ...\sync_index.csv
Mode: Archive ALL emails (rebuild from scratch)
================================================================================
Found 0 already-synced emails in index
Processing 37879 emails from RUNS folder...
```

**When to Use**:
- First-time setup
- CSV files were manually deleted
- Need to verify data integrity
- Index file is corrupted

### Help Documentation

```bash
python monitor_outlook.py --help
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Outlook Desktop (eddy.winiarz@ytmcapital.com)              │
│  └─ Inbox                                                   │
│      └─ RUNS Folder (37,879 emails)                        │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  │ COM Automation (pywin32)
                                  │
┌─────────────────────────────────▼───────────────────────────┐
│ OutlookMonitor.sync_emails_to_csv()                        │
│                                                              │
│  1. Load sync_index.csv → Set[EntryID]                     │
│  2. Apply Outlook filter (if --days specified)              │
│  3. For each email in filtered items:                       │
│      - Check if EntryID in synced_ids                       │
│      - Skip if already synced                               │
│      - Extract metadata + body (first 5000 chars)           │
│      - Group by ReceivedDate                                │
│  4. Write CSV files (one per date)                          │
│  5. Update sync_index.csv with new EntryIDs                 │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  │
┌─────────────────────────────────▼───────────────────────────┐
│ Output Directory (Dropbox)                                  │
│                                                              │
│  Outlook Data 10.22.2025.csv  (245 emails)                 │
│  Outlook Data 10.23.2025.csv  (387 emails)                 │
│  Outlook Data 10.24.2025.csv  (171 emails)                 │
│  sync_index.csv               (803 entries)                 │
└──────────────────────────────────────────────────────────────┘
```

## Error Handling

### Common Issues and Solutions

**1. "Failed to connect to Outlook"**

**Cause**: Outlook desktop application not running.

**Solution**: Open Microsoft Outlook and ensure it's connected to your email account.

---

**2. "Folder 'RUNS' not found in Inbox"**

**Cause**: RUNS folder doesn't exist or has different name.

**Solution**: Check Outlook to confirm folder name and location (must be direct subfolder of Inbox).

---

**3. "can't compare offset-naive and offset-aware datetimes"**

**Cause**: Timezone mismatch between Python datetime and Outlook datetime.

**Solution**: Already fixed in code using `pywintypes.Time()` conversion.

---

**4. "ImportError: No module named win32com"**

**Cause**: pywin32 not installed.

**Solution**:
```bash
Bond-RV-App\Scripts\activate
pip install -r requirements.txt
```

---

**5. Large Number of Errors During Processing**

**Cause**: COM automation can fail on individual emails due to permissions, corruption, etc.

**Behavior**: Script logs first 10 errors, then suppresses further messages to avoid clutter.

**Impact**: Failed emails are counted in `stats['errors']` but don't stop the process.

## Performance Characteristics

### Timing Benchmarks

| Operation                    | Email Count | Duration | Notes                      |
|------------------------------|-------------|----------|----------------------------|
| Full archive (--all)         | 37,879      | ~45 min  | First-time setup           |
| Incremental (default)        | 45 new      | ~2 min   | Daily maintenance          |
| 2-day filter (--days 2)      | 803         | ~3 min   | Reduced dataset            |
| 7-day filter (--days 7)      | ~2,500      | ~8 min   | Weekly catch-up            |

**Note**: Times approximate, depend on email complexity, attachment count, and Outlook responsiveness.

### Optimization Strategies

1. **Outlook Native Filtering**: Using `items.Restrict()` reduces email processing by 95%+ for date-range queries
2. **EntryID Index**: In-memory set lookup (O(1)) for skip checks instead of scanning CSV
3. **Batch CSV Writing**: Group emails by date, write once per date instead of per-email
4. **Limited Body Text**: Only extract first 5000 characters to reduce memory/disk usage

## Programmatic Usage

For advanced automation, use the `OutlookMonitor` class directly:

```python
from utils.outlook_monitor import OutlookMonitor

# Initialize and connect
monitor = OutlookMonitor(email_address="eddy.winiarz@ytmcapital.com")
monitor.connect()
monitor.get_runs_folder("RUNS")

# Sync emails
stats = monitor.sync_emails_to_csv(
    output_dir=r"C:\path\to\output",
    progress_callback=my_progress_function,  # Optional
    days_back=2  # Optional: filter to last 2 days
)

print(f"Archived {stats['new']} new emails")
print(f"Skipped {stats['skipped']} already-synced emails")
print(f"Created {stats['files_created']} CSV files")
```

### Progress Callback

```python
def my_progress_function(current, total, subject, skipped=False):
    """Called for each email processed."""
    percent = (current / total) * 100
    status = "SKIP" if skipped else "NEW"
    print(f"[{percent:.1f}%] {current}/{total} - {status}: {subject[:60]}")

monitor.sync_emails_to_csv(
    output_dir=output_dir,
    progress_callback=my_progress_function
)
```

## Integration with Bond Pipeline

The Outlook archiving system is **standalone** and not integrated into the bond data pipeline. However, potential integrations could include:

### Automatic Pipeline Trigger

Monitor for specific email subjects and auto-run pipeline:

```python
from utils.outlook_monitor import OutlookMonitor
import subprocess

def trigger_on_api_email(email_data):
    """Run pipeline when 'API' email arrives."""
    if 'API' in email_data['subject']:
        print(f"Triggering pipeline for: {email_data['subject']}")
        subprocess.run(['python', 'run_pipeline.py'], input='2\n', text=True)

monitor = OutlookMonitor()
monitor.connect()
monitor.get_runs_folder()
monitor.monitor_new_emails(check_interval=300, callback=trigger_on_api_email)
```

### Attachment Extraction

Future enhancement to save email attachments:

```python
for attachment in item.Attachments:
    attachment.SaveAsFile(os.path.join(output_dir, attachment.FileName))
```

## Maintenance

### Regular Cleanup

**Recommended**: Keep sync_index.csv growing indefinitely (it's just tracking metadata).

**Optional Cleanup** (if index gets very large):

```bash
# Backup existing index
copy sync_index.csv sync_index_backup.csv

# Rebuild from scratch
python monitor_outlook.py --all
```

### Monitoring Sync Health

Check sync statistics after each run:

```
Archive Complete!
  Total emails in folder:   37879
  New emails archived:      45      ← Should match expected daily volume
  Already synced (skipped): 37834
  CSV files created:        2       ← One per date with new emails
  Errors:                   0       ← Should be 0 or very low
```

**Alert if**:
- Errors > 10: Investigate COM connection or email corruption
- New emails = 0: Verify new emails are arriving in RUNS folder
- Skipped = 0: Index may have been deleted (rebuilding)

## Limitations

1. **Windows Only**: COM automation requires Windows OS
2. **Local Outlook Required**: Must have Outlook desktop client installed and synced
3. **Single Account**: Configured for eddy.winiarz@ytmcapital.com only
4. **Read-Only Metadata**: Doesn't save full .eml files or attachments (by design)
5. **Polling-Based**: Not real-time event-driven (use `monitor_new_emails()` for polling)

## Future Enhancements

Potential improvements:

- [ ] Attachment download and storage
- [ ] Multi-account support
- [ ] Parse and extract Excel data from attachments
- [ ] Real-time monitoring with Windows event handlers
- [ ] Email content search and filtering
- [ ] Integration with bond pipeline for auto-processing
- [ ] Export to Parquet format for analysis
- [ ] Web dashboard for archive browsing

## Related Documentation

- **Setup Guide**: `Documentation/Setup/virtual-environment-guide.md`
- **Outlook Monitor CLI**: `Documentation/Reference/outlook-monitor-guide.md`
- **Testing Guide**: `Documentation/Reference/testing-guide.md`
- **Project Architecture**: `Documentation/Architecture/pipeline-architecture.md`

## Change Log

### October 24, 2025 8:05 PM
- Added Outlook native filtering using `items.Restrict()` for efficient date-range queries
- Fixed timezone-aware datetime comparison using `pywintypes.Time()`
- Removed `--output` argument (now uses fixed OUTPUT_DIR constant)
- Optimized date filtering to only process emails in specified range

### October 24, 2025 2:45 PM
- Initial implementation with incremental sync
- CSV organization by date
- Three operating modes (default, --all, --days)
- EntryID tracking via sync_index.csv
