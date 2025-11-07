# Bond RV App - Data Pipeline

**A modular data engineering pipeline for processing bond data for relative value trading applications.**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/ewswlw/Bond-RV-App.git
cd Bond-RV-App

# 2. Install dependencies
pip install pandas pyarrow openpyxl

# 3. Drag Excel files into Raw Data/ folder

# 4. Run pipeline (first time)
cd bond_pipeline
python pipeline.py -i "../Raw Data/" -m override

# 5. Daily updates (when new files added)
python pipeline.py -i "../Raw Data/" -m append
```

**📖 For detailed setup instructions, see [Documentation/Workflows/Local-Workflow.md](Documentation/Workflows/Local-Workflow.md)** ⭐ PRIMARY

**🔄 For Dropbox sync workflow, see [Documentation/Workflows/Dropbox-Workflow.md](Documentation/Workflows/Dropbox-Workflow.md)**

---

## 📋 Overview

This project contains a data pipeline for bond trading analysis:

### Excel Pipeline (Bond Pipeline)
Processes Excel files containing bond data and creates two optimized Parquet tables:

1. **`historical_bond_details.parquet`** - Time series data with unique `Date + CUSIP` combinations
2. **`universe.parquet`** - Current universe of all unique CUSIPs (13 key columns)

### Runs Pipeline (Dealer Quotes)
Processes Excel files containing dealer quote data (runs) and creates one optimized Parquet table:

1. **`runs_timeseries.parquet`** - Time series of dealer quotes with unique `Date + Dealer + CUSIP` combinations (end-of-day snapshots)

### BQL Workbook Ingestion
Processes the Bloomberg query workbook (`bql.xlsx`) into a long-form spreads dataset:

1. **`bql.parquet`** - Normalized table with columns `Date`, `Name`, `CUSIP`, `Value`, including CUSIP orphan logging versus `universe.parquet`

### Key Features

✅ **Modular Architecture** - Separate modules with clean separation of concerns  
✅ **Dual Pipelines** - Bond pipeline and Runs pipeline with shared utilities  
✅ **BQL Workbook Ingestion** - Converts Bloomberg query output to long-form spreads dataset with orphan tracking  
✅ **Incremental Loading** - Append mode skips existing dates  
✅ **Data Validation** - CUSIP normalization & validation, orphan tracking  
✅ **Optimized Performance** - Vectorized operations (~100x faster for large datasets)  
✅ **Enhanced Logging** - Orphan CUSIPs logged with full context (Security, Date, Dealer, Time, Ticker)  
✅ **Deduplication** - Removes duplicates (keeps last occurrence / latest time for runs)  
✅ **Schema Evolution** - Handles files with different column counts (59-75 for bonds, 28-30 for runs)  
✅ **Comprehensive Logging** - 4 separate log files track all operations  
✅ **Primary Key Enforcement** - No duplicate Date+CUSIP (bonds) or Date+Dealer+CUSIP (runs)  
✅ **CLI Interface** - Easy command-line usage with unified orchestrator  

---

## 📁 Project Structure

```
bond-rv-app/
├── Raw Data/                   # ⭐ Drag & drop Excel files here
│   ├── README.md               # Instructions for this folder
│   ├── API 08.04.23.xlsx       # Example files (not in Git)
│   ├── API 09.22.25.xlsx
│   └── ... (more files)
│
├── bond_pipeline/              # Bond pipeline code
│   ├── config.py               # Configuration and constants (shared)
│   ├── utils.py                # Helper functions (shared)
│   ├── extract.py              # Excel file reading
│   ├── transform.py            # Data cleaning & transformation
│   ├── load.py                 # Parquet writing
│   ├── pipeline.py             # Bond pipeline orchestration
│   └── README.md               # Module documentation
│
├── runs_pipeline/             # Runs pipeline code
│   ├── __init__.py             # Package exports
│   ├── extract.py              # RUNS Excel file reading
│   ├── transform.py            # End-of-day deduplication & validation
│   ├── load.py                 # Runs parquet writing
│   └── pipeline.py             # Runs pipeline orchestration
│
├── run_pipeline.py             # ⭐ Unified pipeline orchestrator
│
├── bond_data/                  # Data directory (local only)
│   ├── parquet/                # Output parquet files
│   │   ├── historical_bond_details.parquet  # Bond pipeline output
│   │   ├── universe.parquet                 # Bond pipeline output
│   │   ├── bql.parquet                      # BQL spreads dataset
│   │   └── runs_timeseries.parquet          # Runs pipeline output
│   └── logs/                   # Processing logs
│       ├── processing.log      # Excel pipeline logs
│       ├── duplicates.log
│       ├── validation.log
│       └── summary.log
│
├── Documentation/              # Complete documentation
│   ├── README.md               # Documentation index
│   ├── Setup/                  # Getting started guides
│   ├── Workflows/              # Step-by-step procedures
│   │   ├── Local-Workflow.md   # ⭐ PRIMARY workflow
│   │   └── Dropbox-Workflow.md # Optional sync workflow
│   ├── Architecture/           # Technical design docs
│   └── Reference/              # Reference materials
│
└── README.md                   # This file
```

---

## 🔧 Installation

### Requirements
- Python 3.11+
- pandas
- pyarrow
- openpyxl
- Dropbox account (for data storage)

### Install Dependencies
```bash
pip install pandas pyarrow openpyxl
```

---

## 📖 Usage

### Unified Pipeline Orchestrator (Recommended)

```bash
# Run unified orchestrator - select pipeline(s) and mode interactively
python run_pipeline.py

# Prompts:
# 1. Select pipeline: [1] Bond, [2] Runs, [3] Both
# 2. Select mode: [1] Override, [2] Append
# 3. (Bond) Include BQL workbook ingestion? [Y/n]
```

### Direct Pipeline CLI

```bash
# Bond Pipeline
python -m bond_pipeline.pipeline -i "Raw Data/" -m append --process-bql
python -m bond_pipeline.pipeline -i "Raw Data/" -m override --process-bql

# Runs Pipeline
python -m runs_pipeline.pipeline -i "Historical Runs/" -m append
python -m runs_pipeline.pipeline -i "Historical Runs/" -m override
```

### Python API

```python
import pandas as pd

# Load bond pipeline outputs
df_hist = pd.read_parquet('bond_data/parquet/historical_bond_details.parquet')
df_universe = pd.read_parquet('bond_data/parquet/universe.parquet')

# Example: Get time series for specific CUSIP
cusip_ts = df_hist[df_hist['CUSIP'] == '037833DX5'].sort_values('Date')

# Example: Filter by date range
recent = df_hist[df_hist['Date'] >= '2025-09-01']

# Load runs pipeline output
df_runs = pd.read_parquet('bond_data/parquet/runs_timeseries.parquet')

# Example: Get quotes for specific CUSIP and dealer
cusip_dealer = df_runs[
    (df_runs['CUSIP'] == '037833DX5') & 
    (df_runs['Dealer'] == 'RBC')
].sort_values(['Date', 'Time'])
```

---

## 📚 Documentation

### Quick Links

| Document | Purpose |
|----------|---------|
| **[Documentation/README.md](Documentation/README.md)** | Documentation index and navigation |
| **[Documentation/Setup/QUICKSTART.md](Documentation/Setup/QUICKSTART.md)** | 5-minute setup guide |
| **[Documentation/Workflows/Dropbox-Workflow.md](Documentation/Workflows/Dropbox-Workflow.md)** | Complete Dropbox workflow (⭐ PRIMARY) |
| **[Documentation/Architecture/bond_pipeline_documentation.md](Documentation/Architecture/bond_pipeline_documentation.md)** | Technical documentation |
| **[Documentation/Reference/DELIVERABLES.txt](Documentation/Reference/DELIVERABLES.txt)** | Project deliverables |
| **[Documentation/Reference/Data-Distribution-Options.md](Documentation/Reference/Data-Distribution-Options.md)** | Data strategy analysis |

### Documentation Structure

```
Documentation/
├── README.md                           # Documentation index
├── Setup/                              # Getting started
│   └── QUICKSTART.md
├── Workflows/                          # Procedures
│   └── Dropbox-Workflow.md             # ⭐ PRIMARY WORKFLOW
├── Architecture/                       # Technical docs
│   └── bond_pipeline_documentation.md
└── Reference/                          # Reference materials
    ├── DELIVERABLES.txt
    └── Data-Distribution-Options.md
```

---

## 🔄 Workflow: Local Drag & Drop

This project uses a **simple drag-and-drop workflow**:

1. **Raw Excel files** → Drag into `Raw Data/` folder
2. **Code** → Version controlled in GitHub
3. **Parquet files** → Generated locally in `bond_data/parquet/`

### Basic Workflow:

```bash
# 1. Clone repo
git clone https://github.com/ewswlw/Bond-RV-App.git
cd Bond-RV-App

# 2. Drag Excel files into Raw Data/ folder

# 3. Run pipeline
cd bond_pipeline
python pipeline.py -i "../Raw Data/" -m override

# 4. When adding new files, use append mode
python pipeline.py -i "../Raw Data/" -m append
```

**📖 For complete workflow instructions, see [Documentation/Workflows/Local-Workflow.md](Documentation/Workflows/Local-Workflow.md)** ⭐ PRIMARY

**🔄 For automatic sync across computers, see [Documentation/Workflows/Dropbox-Workflow.md](Documentation/Workflows/Dropbox-Workflow.md)**

---

## 📊 Data Schema

### Historical Bond Details Table (Bond Pipeline)
- **Primary Key**: `Date + CUSIP`
- **Columns**: 76 (Date + all source columns)
- **Purpose**: Complete time series of all bonds

### Universe Table (Bond Pipeline)
- **Primary Key**: `CUSIP`
- **Columns**: 13 key columns
  1. CUSIP
  2. Benchmark Cusip
  3. Custom_Sector
  4. Bloomberg Cusip
  5. Security
  6. Benchmark
  7. Pricing Date
  8. Pricing Date (Bench)
  9. Worst Date
  10. Yrs (Worst)
  11. Ticker
  12. Currency
  13. Equity Ticker
- **Purpose**: Current universe of unique bonds with most recent data

### BQL Spreads Table (Bond Pipeline)
- **Primary Key**: `Date + CUSIP`
- **Columns**: `Date`, `Name`, `CUSIP`, `Value`
- **Purpose**: Long-form Bloomberg query spreads with orphan logging against `universe.parquet`

### Runs Timeseries Table (Runs Pipeline)
- **Primary Key**: `Date + Dealer + CUSIP`
- **Columns**: 30 (all source columns with Date and Time as first columns)
- **Purpose**: End-of-day snapshots of dealer quotes over time
- **Deduplication**: Keeps latest Time per Date+Dealer+CUSIP (end-of-day snapshot)

---

## 📈 Performance

### Bond Pipeline
- **Processing Time**: ~15-20 seconds for 11 files
- **Memory Usage**: ~100-200 MB
- **Output Size**: 
  - Historical: ~2-3 MB (compressed)
  - Universe: ~200-300 KB (compressed)

### Runs Pipeline
- **Processing Time**: ~5-30 seconds for ~130,000 rows (after optimization)
- **Speed Improvement**: ~100x faster with vectorized operations
- **Memory Usage**: ~200-400 MB
- **Output Size**: 
  - Runs Timeseries: ~5-10 MB (compressed)

---

## 🔍 Data Quality

The pipeline includes comprehensive data validation:

- **CUSIP Validation**: 
  - Bond pipeline: Normalizes to uppercase, validates 9-character format
  - Runs pipeline: Validates length but keeps as-is (no normalization)
- **Orphan Tracking**: Runs pipeline tracks CUSIPs not in universe.parquet with detailed context (Security, Date, Dealer, Time, Ticker)
- **Deduplication**: 
  - Bond pipeline: Removes duplicate Date+CUSIP combinations (keeps last)
  - Runs pipeline: End-of-day snapshots (latest Time per Date+Dealer+CUSIP)
- **NA Handling**: Standardizes various NA representations
- **Schema Alignment**: Handles files with different column counts (59-75 for bonds, 28-30 for runs)
- **Logging**: Tracks all data quality issues with enhanced context

---

## 🧪 Testing

The pipeline has been tested with:
- 11 Excel files (2023-2025)
- 25,741 total rows
- 3,231 unique CUSIPs
- Schema evolution (59-75 columns)
- Duplicate handling (~500 duplicates)
- Invalid CUSIP detection (~100 invalid)

All tests passed successfully. See [Documentation/Architecture/bond_pipeline_documentation.md](Documentation/Architecture/bond_pipeline_documentation.md) for detailed test results.

---

## 🛠️ Development

### Module Overview

1. **config.py** - Configuration, paths, and constants
2. **utils.py** - Helper functions (date parsing, CUSIP validation, logging)
3. **extract.py** - Excel file reading with date extraction
4. **transform.py** - Data cleaning, normalization, deduplication
5. **load.py** - Parquet writing with append/override logic
6. **pipeline.py** - Main orchestration script with CLI

### Adding New Features

The modular architecture makes it easy to extend:
- Add new validation rules in `transform.py`
- Extend CLI options in `pipeline.py`
- Add new output formats in `load.py`
- Customize logging in `utils.py`

---

## 🚧 Roadmap

Future enhancements:
- [ ] Automated testing suite
- [ ] Data quality dashboard
- [ ] API integration for real-time updates
- [ ] Partitioning for large datasets (>100M rows)
- [ ] Incremental universe updates (optimization)
- [ ] Web UI for monitoring

---

## 🛟 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| No files found | Check Dropbox path and sync status |
| Pipeline fails | Check logs in `bond_data/logs/` |
| Different results on computers | Run in override mode to rebuild |
| Dropbox not synced | Wait for sync, check Dropbox icon |

**📖 For detailed troubleshooting, see [Documentation/Workflows/Dropbox-Workflow.md](Documentation/Workflows/Dropbox-Workflow.md#-troubleshooting)**

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

This is a private project for professional bond trading. For questions or issues, please contact the development team.

---

## 📞 Support

For technical support or questions:
- Review the [Documentation](Documentation/README.md)
- Check the [Dropbox Workflow Guide](Documentation/Workflows/Dropbox-Workflow.md)
- Check the [logs](bond_data/logs/) for detailed error messages
- Refer to the [quick start guide](Documentation/Setup/QUICKSTART.md)

---

**Built with ❤️ for professional bond traders**

**Repository**: https://github.com/ewswlw/Bond-RV-App

