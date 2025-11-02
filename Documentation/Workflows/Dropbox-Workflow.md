# Dropbox Workflow - Complete Setup & Usage Guide

**Document Version**: 1.1  
**Last Updated**: November 2, 2025 18:27:11  
**Purpose**: Step-by-step instructions for using Dropbox to store raw data and GitHub for code

---

## 📋 Overview

This workflow keeps:
- **Code** in GitHub (version controlled)
- **Raw Excel files** in Dropbox (shared across computers)
- **Parquet files** generated locally (not committed to Git)

Every time you pull the repo on a new computer, you'll regenerate the parquet files from the raw data in Dropbox.

---

## 🎯 One-Time Setup (Do This Once)

### Step 1: Organize Raw Data in Dropbox

1. **Create folder structure in Dropbox:**
   ```
   ~/Dropbox/Bond-RV-App-Data/
   └── Universe Historical/
       ├── API 08.04.23.xlsx
       ├── API 09.22.25.xlsx
       ├── API 09.23.25.xlsx
       ├── API 09.24.25.xlsx
       ├── API 09.25.25.xlsx
       ├── API 10.09.25.xlsx
       ├── API 10.14.25.xlsx
       ├── API 10.20.25.xlsx
       ├── API 10.31.23.xlsx
       ├── API 12.20.24.xlsx
       └── API 12.29.23.xlsx
   ```

2. **Copy your Excel files to this folder:**
   - On Mac/Linux:
     ```bash
     mkdir -p ~/Dropbox/Bond-RV-App-Data/
     cp -r "/path/to/Universe Historical" ~/Dropbox/Bond-RV-App-Data/
     ```
   
   - On Windows:
     ```powershell
     mkdir "$env:USERPROFILE\Dropbox\Bond-RV-App-Data"
     xcopy "C:\path\to\Universe Historical" "$env:USERPROFILE\Dropbox\Bond-RV-App-Data\Universe Historical" /E /I
     ```

3. **Wait for Dropbox to sync** (check Dropbox icon in system tray)

### Step 2: Verify Dropbox Sync on All Computers

On each computer where you'll use the pipeline:

1. **Install Dropbox** (if not already installed):
   - Download from: https://www.dropbox.com/install
   - Sign in with your account

2. **Verify folder exists:**
   - Mac/Linux: `ls ~/Dropbox/Bond-RV-App-Data/Universe\ Historical/`
   - Windows: `dir %USERPROFILE%\Dropbox\Bond-RV-App-Data\Universe Historical`

3. **Confirm all files are synced** (should see 11 .xlsx files)

---

## 💻 Computer 1: Initial Setup & First Run

### Step 1: Clone Repository

```bash
# Clone the repo
git clone https://github.com/ewswlw/Bond-RV-App.git
cd Bond-RV-App
```

### Step 2: Install Dependencies

```bash
# Install required Python packages
pip install pandas pyarrow openpyxl
```

### Step 3: Run Pipeline (Override Mode)

**Mac/Linux:**
```bash
cd bond_pipeline
python pipeline.py -i ~/Dropbox/Bond-RV-App-Data/Universe\ Historical/ -m override
```

**Windows:**
```powershell
cd bond_pipeline
python pipeline.py -i "%USERPROFILE%\Dropbox\Bond-RV-App-Data\Universe Historical" -m override
```

### Step 4: Verify Output

```bash
# Check that parquet files were created
ls ../bond_data/parquet/
# Should see:
# - historical_bond_details.parquet
# - universe.parquet

# Check logs
ls ../bond_data/logs/
# Should see:
# - processing.log
# - duplicates.log
# - validation.log
# - summary.log
```

### Step 5: Test the Data

```bash
# Quick test in Python
python << EOF
import pandas as pd

# Load historical data
df_hist = pd.read_parquet('../bond_data/parquet/historical_bond_details.parquet')
print(f"Historical: {len(df_hist)} rows, {df_hist['Date'].nunique()} dates")

# Load universe
df_univ = pd.read_parquet('../bond_data/parquet/universe.parquet')
print(f"Universe: {len(df_univ)} unique CUSIPs")
EOF
```

**Expected output:**
```
Historical: 25741 rows, 11 dates
Universe: 3231 unique CUSIPs
```

---

## 💻 Computer 2 (or any new computer): Setup & Run

### Step 1: Ensure Dropbox is Synced

```bash
# Verify Dropbox folder exists and has data
ls ~/Dropbox/Bond-RV-App-Data/Universe\ Historical/
# Should see 11 .xlsx files
```

### Step 2: Clone Repository

```bash
git clone https://github.com/ewswlw/Bond-RV-App.git
cd Bond-RV-App
```

### Step 3: Install Dependencies

```bash
pip install pandas pyarrow openpyxl
```

### Step 4: Run Pipeline

**Mac/Linux:**
```bash
cd bond_pipeline
python pipeline.py -i ~/Dropbox/Bond-RV-App-Data/Universe\ Historical/ -m override
```

**Windows:**
```powershell
cd bond_pipeline
python pipeline.py -i "%USERPROFILE%\Dropbox\Bond-RV-App-Data\Universe Historical" -m override
```

### Step 5: Verify Output

Same as Computer 1 Step 4 above.

---

## 🔄 Daily Workflow: Adding New Data

### When New Excel File is Added to Dropbox

1. **Add new file to Dropbox folder:**
   ```bash
   # Example: New file for October 21, 2025
   cp "API 10.21.25.xlsx" ~/Dropbox/Bond-RV-App-Data/Universe\ Historical/
   ```

2. **Wait for Dropbox to sync across all computers**

3. **On any computer, run pipeline in APPEND mode:**
   ```bash
   cd Bond-RV-App/bond_pipeline
   python pipeline.py -i ~/Dropbox/Bond-RV-App-Data/Universe\ Historical/ -m append
   ```

4. **Pipeline will:**
   - Detect existing dates in parquet
   - Process only the new file (10.21.25)
   - Append new data to historical_bond_details.parquet
   - Rebuild universe.parquet with updated data

---

## 🔄 Workflow: Pulling Latest Code Changes

### When Code is Updated in GitHub

1. **Pull latest changes:**
   ```bash
   cd Bond-RV-App
   git pull origin main
   ```

2. **Regenerate parquet files (recommended):**
   ```bash
   cd bond_pipeline
   python pipeline.py -i ~/Dropbox/Bond-RV-App-Data/Universe\ Historical/ -m override
   ```

   **Why override?** Ensures parquet files match the latest code logic.

3. **Verify output:**
   ```bash
   python << EOF
   import pandas as pd
   df = pd.read_parquet('../bond_data/parquet/historical_bond_details.parquet')
   print(f"Total rows: {len(df)}, Dates: {df['Date'].nunique()}")
   EOF
   ```

---

## 🛠️ Troubleshooting

### Issue: "No such file or directory: Dropbox/Bond-RV-App-Data"

**Solution:**
1. Check Dropbox is installed and running
2. Verify folder path:
   ```bash
   # Mac/Linux
   ls ~/Dropbox/
   
   # Windows
   dir %USERPROFILE%\Dropbox
   ```
3. Ensure Dropbox has finished syncing (check Dropbox icon)

### Issue: "No Excel files found in directory"

**Solution:**
1. Check file pattern matches `API MM.DD.YY.xlsx`
2. Verify files are in correct folder:
   ```bash
   ls ~/Dropbox/Bond-RV-App-Data/Universe\ Historical/*.xlsx
   ```

### Issue: Pipeline fails with "Error reading file"

**Solution:**
1. Check Dropbox files are fully synced (not just placeholders)
2. Right-click file in Dropbox → "Make available offline"
3. Wait for download to complete

### Issue: Different results on different computers

**Solution:**
1. Ensure Dropbox is fully synced on both computers
2. Run pipeline in **override mode** to rebuild from scratch:
   ```bash
   python pipeline.py -i ~/Dropbox/Bond-RV-App-Data/Universe\ Historical/ -m override
   ```

---

## 📊 Quick Reference Commands

### Mac/Linux

```bash
# Clone repo
git clone https://github.com/ewswlw/Bond-RV-App.git

# Install dependencies
pip install pandas pyarrow openpyxl

# Run pipeline (override)
cd Bond-RV-App/bond_pipeline
python pipeline.py -i ~/Dropbox/Bond-RV-App-Data/Universe\ Historical/ -m override

# Run pipeline (append new data)
python pipeline.py -i ~/Dropbox/Bond-RV-App-Data/Universe\ Historical/ -m append

# Check output
ls ../bond_data/parquet/

# View logs
cat ../bond_data/logs/summary.log
```

### Windows (PowerShell)

```powershell
# Clone repo
git clone https://github.com/ewswlw/Bond-RV-App.git

# Install dependencies
pip install pandas pyarrow openpyxl

# Run pipeline (override)
cd Bond-RV-App\bond_pipeline
python pipeline.py -i "$env:USERPROFILE\Dropbox\Bond-RV-App-Data\Universe Historical" -m override

# Run pipeline (append new data)
python pipeline.py -i "$env:USERPROFILE\Dropbox\Bond-RV-App-Data\Universe Historical" -m append

# Check output
dir ..\bond_data\parquet\

# View logs
type ..\bond_data\logs\summary.log
```

---

## 📝 Best Practices

### 1. Always Use Override Mode After Code Updates
```bash
git pull origin main
cd bond_pipeline
python pipeline.py -i ~/Dropbox/Bond-RV-App-Data/Universe\ Historical/ -m override
```

### 2. Use Append Mode for New Data Only
```bash
# When new Excel file is added to Dropbox
python pipeline.py -i ~/Dropbox/Bond-RV-App-Data/Universe\ Historical/ -m append
```

### 3. Check Logs After Each Run
```bash
# Quick check
tail -20 ../bond_data/logs/summary.log

# Check for issues
grep "ERROR\|WARNING" ../bond_data/logs/validation.log
```

### 4. Backup Dropbox Folder Regularly
- Dropbox has version history (30 days free, unlimited with Plus/Professional)
- Can restore deleted/modified files from Dropbox web interface

### 5. Keep Excel Files Organized
```
Universe Historical/
├── API 08.04.23.xlsx
├── API 09.22.25.xlsx
...
└── API 10.20.25.xlsx

# Do NOT mix with other files
# Keep consistent naming: API MM.DD.YY.xlsx
```

---

## 🔐 Security Notes

1. **Dropbox folder is private** - only accessible to accounts with access
2. **No sensitive data in Git** - only code is version controlled
3. **Parquet files are local** - not shared via Git or Dropbox
4. **Logs are local** - contain data quality info, not committed

---

## 📞 Support

If you encounter issues:
1. Check logs in `bond_data/logs/`
2. Review troubleshooting section above
3. Verify Dropbox sync status
4. Try override mode to rebuild from scratch

---

## ✅ Checklist for New Computer Setup

- [ ] Dropbox installed and signed in
- [ ] Verify `~/Dropbox/Bond-RV-App-Data/Universe Historical/` exists
- [ ] Verify 11 .xlsx files are present and synced
- [ ] Git installed
- [ ] Python 3.11+ installed
- [ ] Clone repository: `git clone https://github.com/ewswlw/Bond-RV-App.git`
- [ ] Install dependencies: `pip install pandas pyarrow openpyxl`
- [ ] Run pipeline: `python pipeline.py -i ~/Dropbox/Bond-RV-App-Data/Universe\ Historical/ -m override`
- [ ] Verify output: Check `bond_data/parquet/` for 2 parquet files
- [ ] Test data: Load parquet files in Python and verify row counts

---

**Document maintained by**: Bond RV App Team  
**Last updated**: November 2, 2025 18:27:11

