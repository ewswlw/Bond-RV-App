# Local Workflow - Drag & Drop Excel Files

**Document Version**: 2.0  
**Date**: October 21, 2025  
**Purpose**: Simple drag-and-drop workflow using local Raw Data folder

---

## 📋 Overview

This workflow keeps everything local and simple:
- **Code** in GitHub (version controlled)
- **Raw Excel files** in `Raw Data/` folder (drag & drop)
- **Parquet files** generated in `bond_data/parquet/` (local)

**No Dropbox required!** Just drag files into the `Raw Data/` folder and run the pipeline.

---

## 🎯 One-Time Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/ewswlw/Bond-RV-App.git
cd Bond-RV-App
```

### Step 2: Install Dependencies

```bash
pip install pandas pyarrow openpyxl
```

### Step 3: Add Your Excel Files

**Drag and drop** your Excel files into the `Raw Data/` folder:

```
Bond-RV-App/
├── Raw Data/
│   ├── API 08.04.23.xlsx    ← Drag files here
│   ├── API 09.22.25.xlsx
│   ├── API 09.23.25.xlsx
│   └── ... (more files)
```

**File naming must follow pattern**: `API MM.DD.YY.xlsx`

### Step 4: Run Pipeline (First Time)

```bash
cd bond_pipeline
python pipeline.py -i "../Raw Data/" -m override
```

### Step 5: Verify Output

```bash
# Check parquet files were created
ls ../bond_data/parquet/
# Should see:
# - historical_bond_details.parquet
# - universe.parquet

# Quick test
python << EOF
import pandas as pd
df = pd.read_parquet('../bond_data/parquet/historical_bond_details.parquet')
print(f"Total rows: {len(df)}, Dates: {df['Date'].nunique()}")
EOF
```

---

## 💻 Daily Workflow: Adding New Data

### When You Receive a New Excel File

1. **Drag the new file into `Raw Data/` folder**
   ```
   Raw Data/
   ├── API 08.04.23.xlsx
   ├── ... (existing files)
   └── API 10.21.25.xlsx    ← New file
   ```

2. **Run pipeline in APPEND mode**
   ```bash
   cd bond_pipeline
   python pipeline.py -i "../Raw Data/" -m append
   ```

3. **Pipeline will:**
   - Detect existing dates in parquet
   - Process only the new file
   - Append new data to historical_bond_details.parquet
   - Rebuild universe.parquet with updated data

4. **Verify the update**
   ```bash
   python << EOF
   import pandas as pd
   df = pd.read_parquet('../bond_data/parquet/historical_bond_details.parquet')
   print(f"Total rows: {len(df)}, Dates: {df['Date'].nunique()}")
   print(f"Latest date: {df['Date'].max()}")
   EOF
   ```

---

## 🔄 Workflow: Pulling Code Updates

### When Code is Updated in GitHub

1. **Pull latest changes**
   ```bash
   cd Bond-RV-App
   git pull origin main
   ```

2. **Regenerate parquet files (recommended)**
   ```bash
   cd bond_pipeline
   python pipeline.py -i "../Raw Data/" -m override
   ```

   **Why override?** Ensures parquet files match the latest code logic.

3. **Verify output**
   ```bash
   python << EOF
   import pandas as pd
   df = pd.read_parquet('../bond_data/parquet/historical_bond_details.parquet')
   print(f"Total rows: {len(df)}, Dates: {df['Date'].nunique()}")
   EOF
   ```

---

## 🖥️ Setting Up on Multiple Computers

### Computer 1 (Initial Setup)

```bash
# 1. Clone repo
git clone https://github.com/ewswlw/Bond-RV-App.git
cd Bond-RV-App

# 2. Install dependencies
pip install pandas pyarrow openpyxl

# 3. Add Excel files to Raw Data/ folder (drag & drop)

# 4. Run pipeline
cd bond_pipeline
python pipeline.py -i "../Raw Data/" -m override
```

### Computer 2 (or any additional computer)

```bash
# 1. Clone repo
git clone https://github.com/ewswlw/Bond-RV-App.git
cd Bond-RV-App

# 2. Install dependencies
pip install pandas pyarrow openpyxl

# 3. Copy Excel files to Raw Data/ folder
#    Options:
#    - Copy from Computer 1 via USB drive
#    - Download from shared location (email, Dropbox, etc.)
#    - Drag & drop into Raw Data/ folder

# 4. Run pipeline
cd bond_pipeline
python pipeline.py -i "../Raw Data/" -m override
```

### Keeping Excel Files in Sync

**Option A: Manual Sync**
- Copy new Excel files via USB drive, email, or file sharing
- Drag into `Raw Data/` folder on each computer
- Run pipeline in append mode

**Option B: Shared Folder (Optional)**
- Keep `Raw Data/` folder in Dropbox/Google Drive/OneDrive
- All computers access the same shared folder
- See [Dropbox-Workflow.md](Dropbox-Workflow.md) for details

---

## 🛠️ Troubleshooting

### Issue: "No files found in directory"

**Solution:**
1. Check files are in `Raw Data/` folder (not a subfolder)
2. Verify file names match pattern: `API MM.DD.YY.xlsx`
3. Check path in command:
   ```bash
   # Correct
   python pipeline.py -i "../Raw Data/" -m override
   
   # Wrong (missing ../)
   python pipeline.py -i "Raw Data/" -m override
   ```

### Issue: "Error reading file"

**Solution:**
1. Verify Excel file is not corrupted (try opening in Excel)
2. Check file is not open in another program
3. Ensure file extension is `.xlsx` (not `.xls` or `.csv`)

### Issue: Pipeline processes same date twice

**Solution:**
- Use **append mode** for new files (automatically skips existing dates)
- Use **override mode** only when rebuilding from scratch

### Issue: Different results on different computers

**Solution:**
1. Ensure both computers have the same Excel files in `Raw Data/`
2. Run pipeline in **override mode** to rebuild from scratch:
   ```bash
   python pipeline.py -i "../Raw Data/" -m override
   ```

### Issue: Git shows Raw Data files as changes

**Solution:**
- Excel files should be ignored by Git (check `.gitignore`)
- Only `Raw Data/.gitkeep` should be tracked
- If files show up, they're being tracked by mistake

---

## 📊 Quick Reference Commands

### First Time Setup
```bash
git clone https://github.com/ewswlw/Bond-RV-App.git
cd Bond-RV-App
pip install pandas pyarrow openpyxl
# Drag Excel files into Raw Data/ folder
cd bond_pipeline
python pipeline.py -i "../Raw Data/" -m override
```

### Add New Data
```bash
# 1. Drag new Excel file into Raw Data/ folder
# 2. Run append mode
cd bond_pipeline
python pipeline.py -i "../Raw Data/" -m append
```

### Pull Code Updates
```bash
git pull origin main
cd bond_pipeline
python pipeline.py -i "../Raw Data/" -m override
```

### Check Output
```bash
# List parquet files
ls bond_data/parquet/

# View logs
cat bond_data/logs/summary.log

# Quick data check
python -c "import pandas as pd; df = pd.read_parquet('bond_data/parquet/historical_bond_details.parquet'); print(f'Rows: {len(df)}, Dates: {df[\"Date\"].nunique()}')"
```

---

## 📝 Best Practices

### 1. Use Append Mode for New Data
```bash
# When adding new Excel file
python pipeline.py -i "../Raw Data/" -m append
```

### 2. Use Override Mode After Code Updates
```bash
# After git pull
python pipeline.py -i "../Raw Data/" -m override
```

### 3. Keep Excel Files Organized
```
Raw Data/
├── API 08.04.23.xlsx
├── API 10.31.23.xlsx
├── API 12.29.23.xlsx
├── API 12.20.24.xlsx
├── API 09.22.25.xlsx
├── API 09.23.25.xlsx
├── API 09.24.25.xlsx
├── API 09.25.25.xlsx
├── API 10.09.25.xlsx
├── API 10.14.25.xlsx
└── API 10.20.25.xlsx

# Do NOT mix with other files
# Keep consistent naming: API MM.DD.YY.xlsx
```

### 4. Check Logs After Each Run
```bash
# Quick check
tail -20 bond_data/logs/summary.log

# Check for issues
grep "ERROR\|WARNING" bond_data/logs/validation.log
```

### 5. Backup Raw Data
- Keep a backup copy of Excel files in a safe location
- Excel files are not in Git, so you need to back them up separately

---

## 🔐 Security Notes

1. **Excel files are NOT in Git** - too large, excluded by .gitignore
2. **Parquet files are local** - not committed to Git
3. **Logs are local** - contain data quality info, not committed
4. **Only code is version controlled** - safe to share repo

---

## 📞 Support

If you encounter issues:
1. Check logs in `bond_data/logs/`
2. Review troubleshooting section above
3. Verify Excel files are in `Raw Data/` folder
4. Try override mode to rebuild from scratch

---

## ✅ Checklist for New Computer Setup

- [ ] Git installed
- [ ] Python 3.11+ installed
- [ ] Clone repository: `git clone https://github.com/ewswlw/Bond-RV-App.git`
- [ ] Install dependencies: `pip install pandas pyarrow openpyxl`
- [ ] Copy Excel files to `Raw Data/` folder
- [ ] Verify files match pattern: `API MM.DD.YY.xlsx`
- [ ] Run pipeline: `python pipeline.py -i "../Raw Data/" -m override`
- [ ] Verify output: Check `bond_data/parquet/` for 2 parquet files
- [ ] Test data: Load parquet files in Python and verify row counts

---

## 🎯 Comparison: Local vs Dropbox Workflow

| Feature | Local Workflow | Dropbox Workflow |
|---------|----------------|------------------|
| **Setup Complexity** | ⭐⭐⭐⭐⭐ Simple | ⭐⭐⭐ Moderate |
| **File Sync** | Manual | Automatic |
| **Storage Location** | Local only | Cloud + Local |
| **Team Sharing** | Manual copy | Automatic |
| **Best For** | Single user or small team | Team with frequent updates |

**This document describes the Local Workflow.** For automatic file syncing across computers, see [Dropbox-Workflow.md](Dropbox-Workflow.md).

---

**Document maintained by**: Bond RV App Team  
**Last updated**: October 21, 2025

