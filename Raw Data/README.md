# Raw Data Folder

## Purpose

This folder contains the raw Excel files that the pipeline processes.

## Usage

### Adding Files

Simply **drag and drop** your Excel files into this folder:

```
Raw Data/
├── API 08.04.23.xlsx
├── API 09.22.25.xlsx
├── API 09.23.25.xlsx
└── ... (more files)
```

### File Naming Convention

Files must follow the pattern: **`API MM.DD.YY.xlsx`**

Examples:
- ✅ `API 10.20.25.xlsx` (October 20, 2025)
- ✅ `API 12.29.23.xlsx` (December 29, 2023)
- ❌ `API 2025-10-20.xlsx` (wrong format)
- ❌ `bonds_10_20_25.xlsx` (wrong format)

### Running the Pipeline

After adding files to this folder:

```bash
# From the bond_pipeline directory
cd bond_pipeline

# First time or rebuild everything
python pipeline.py -i "../Raw Data/" -m override

# Add only new dates
python pipeline.py -i "../Raw Data/" -m append
```

## Notes

- **Excel files are NOT committed to Git** (too large)
- Only the `.gitkeep` file is tracked to preserve folder structure
- Share Excel files via Dropbox, email, or other means
- Each computer needs its own copy of the raw data files

## Workflow

1. **Receive new Excel file** (via email, Dropbox, etc.)
2. **Drag file into this folder**
3. **Run pipeline** in append mode
4. **Commit code changes** (if any) to Git
5. **Share Excel file** with team members

## Troubleshooting

### "No files found in directory"

- Check that files are in the `Raw Data/` folder
- Verify file names match pattern: `API MM.DD.YY.xlsx`
- Ensure files are not in a subfolder

### "Error reading file"

- Check that files are valid Excel files (.xlsx)
- Try opening the file in Excel to verify it's not corrupted
- Check that the file is not open in another program

---

**For complete documentation, see [../Documentation/README.md](../Documentation/README.md)**

