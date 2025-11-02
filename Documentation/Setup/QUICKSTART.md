# Bond Data Pipeline - Quick Start Guide

**Last Updated**: November 2, 2025 18:27:11

## 🚀 Getting Started in 5 Minutes

### Step 1: Set Up Virtual Environment ⭐ REQUIRED

**This project requires a virtual environment named `Bond-RV-App`.**

```bash
# Create virtual environment
python -m venv Bond-RV-App

# Activate it
Bond-RV-App\Scripts\activate       # Windows
source Bond-RV-App/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

For detailed virtual environment instructions, see [VENV_SETUP.md](VENV_SETUP.md).

### Step 2: Verify Installation
```bash
python -c "import pandas; import pyarrow; import openpyxl; print('All dependencies installed!')"
```

### Directory Structure
**Current Structure** (as of November 2, 2025):
```
bond_pipeline/          # Pipeline code (7 modules)
├── __init__.py         # Package initialization
├── config.py           # Configuration
├── utils.py            # Helper functions (date parsing, CUSIP validation, logging)
├── extract.py          # Excel reading (ExcelExtractor class)
├── transform.py       # Data cleaning (DataTransformer class)
├── load.py             # Parquet writing (ParquetLoader class)
└── pipeline.py         # Main orchestration (BondDataPipeline class)
```

bond_data/
├── parquet/            # Output parquet files
│   ├── historical_bond_details.parquet
│   └── universe.parquet
└── logs/               # Processing logs
```

## 📂 Input Data

**Current Setup** (as of November 2, 2025):
- **Raw Data/** folder (recommended - simple drag & drop)
- Default Dropbox folder (optional - configured in `config.py`)

**Default Input Directory** (as configured in `bond_pipeline/config.py`):
- Windows: `C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Support Files\API Historical`

See [Local-Workflow.md](../Workflows/Local-Workflow.md) for the recommended workflow.

## 📝 Usage

**IMPORTANT**: Always activate the virtual environment before running commands!

```bash
Bond-RV-App\Scripts\activate       # Windows
source Bond-RV-App/bin/activate    # Mac/Linux
```

### Automated Pipeline Runner (Recommended)
```bash
python run_pipeline.py
# Select: 1=override (first run), 2=append (daily updates)
```

### First Time Setup (Override Mode)
Processes all files and creates new parquet tables:

```bash
# From project root
python run_pipeline.py
# Select option 1 (OVERRIDE)

# OR from bond_pipeline directory
cd bond_pipeline
python pipeline.py -i "../Raw Data/" -m override
```

### Daily Updates (Append Mode)
Adds only new dates to existing parquet tables:

```bash
# From project root
python run_pipeline.py
# Select option 2 (APPEND)

# OR from bond_pipeline directory
cd bond_pipeline
python pipeline.py -i "../Raw Data/" -m append
```

## 📊 Output Tables

### 1. Historical Bond Details
- **File**: `bond_data/parquet/historical_bond_details.parquet`
- **Primary Key**: `Date + CUSIP`
- **Columns**: 76 (all source columns + Date)
- **Purpose**: Time series of all bonds

### 2. Universe
- **File**: `bond_data/parquet/universe.parquet`
- **Primary Key**: `CUSIP`
- **Columns**: 13 key columns
- **Purpose**: Current universe of unique bonds

## 💻 Using the Data

### Python Example
```python
import pandas as pd

# Load historical data
df_hist = pd.read_parquet('bond_data/parquet/historical_bond_details.parquet')

# Load universe
df_universe = pd.read_parquet('bond_data/parquet/universe.parquet')

# Example queries
recent = df_hist[df_hist['Date'] >= '2025-09-01']
cusip_ts = df_hist[df_hist['CUSIP'] == '037833DX5'].sort_values('Date')
```

## 📈 Dataset Stats

Stats depend on the Excel files you have in `Raw Data/` folder:
- Processing time is typically 15-20 seconds for 12 files
- Use `cat bond_data/logs/summary.log` to see your dataset stats

## 🔍 Checking Logs

```bash
# View processing summary
cat bond_data/logs/summary.log

# Check for duplicates
cat bond_data/logs/duplicates.log

# Review validation issues
cat bond_data/logs/validation.log
```

## ⚠️ Important Notes

1. **File Naming**: Excel files must follow pattern `API MM.DD.YY.xlsx`
2. **Append Mode**: Automatically skips dates already in parquet
3. **Override Mode**: Deletes existing parquet and rebuilds from scratch
4. **Invalid CUSIPs**: Logged but included (check validation.log)
5. **Duplicates**: Automatically removed (last occurrence kept)

## 🆘 Troubleshooting

### No files found
- Check input directory path
- Ensure files match `*.xlsx` pattern

### Pipeline fails
- Check logs in `bond_data/logs/`
- Verify Excel files are not corrupted
- Ensure sufficient disk space

### Unexpected results
- Review `validation.log` for data quality issues
- Check `duplicates.log` for deduplication details

## 📚 Full Documentation

See `README.md` and `bond_pipeline_documentation.md` for complete details.

## 🎯 Next Steps

1. Run override mode to build initial tables
2. Test queries on parquet files
3. Integrate into trading application
4. Set up scheduled runs for daily updates

---

**Questions?** Check the full documentation or logs for detailed information.

