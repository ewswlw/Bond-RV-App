# Bond RV App - Data Pipeline

**A modular data engineering pipeline for processing bond data for relative value trading applications.**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install pandas pyarrow openpyxl

# Run pipeline (first time)
cd bond_pipeline
python pipeline.py -i "/path/to/Universe Historical/" -m override

# Daily updates
python pipeline.py -i "/path/to/Universe Historical/" -m append
```

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

---

## 📋 Overview

This pipeline processes Excel files containing bond data and creates two optimized Parquet tables:

1. **`historical_bond_details.parquet`** - Time series data with unique `Date + CUSIP` combinations
2. **`universe.parquet`** - Current universe of all unique CUSIPs (13 key columns)

### Key Features

✅ **Modular Architecture** - 7 separate modules with clean separation of concerns  
✅ **Incremental Loading** - Append mode skips existing dates  
✅ **Data Validation** - CUSIP normalization & validation  
✅ **Deduplication** - Removes duplicates (keeps last occurrence)  
✅ **Schema Evolution** - Handles files with different column counts (59-75)  
✅ **Comprehensive Logging** - 4 separate log files track all operations  
✅ **Primary Key Enforcement** - No duplicate Date+CUSIP combinations  
✅ **CLI Interface** - Easy command-line usage  

---

## 📁 Project Structure

```
bond-rv-app/
├── bond_pipeline/              # Pipeline code
│   ├── config.py               # Configuration and constants
│   ├── utils.py                # Helper functions
│   ├── extract.py              # Excel file reading
│   ├── transform.py            # Data cleaning & transformation
│   ├── load.py                 # Parquet writing
│   ├── pipeline.py             # Main orchestration script
│   └── README.md               # Module documentation
│
├── bond_data/                  # Data directory
│   ├── parquet/                # Output parquet files
│   │   ├── historical_bond_details.parquet
│   │   └── universe.parquet
│   └── logs/                   # Processing logs
│       ├── processing.log
│       ├── duplicates.log
│       ├── validation.log
│       └── summary.log
│
├── bond_pipeline_documentation.md  # Complete technical documentation
├── QUICKSTART.md                   # Quick start guide
├── DELIVERABLES.txt                # Project deliverables summary
└── README.md                       # This file
```

---

## 🔧 Installation

### Requirements
- Python 3.11+
- pandas
- pyarrow
- openpyxl

### Install Dependencies
```bash
pip install pandas pyarrow openpyxl
```

---

## 📖 Usage

### Command Line Interface

```bash
# Override mode - rebuild everything from scratch
python pipeline.py -i "/path/to/Universe Historical/" -m override

# Append mode - add only new dates
python pipeline.py -i "/path/to/Universe Historical/" -m append
```

### Python API

```python
import pandas as pd

# Load historical time series
df_hist = pd.read_parquet('bond_data/parquet/historical_bond_details.parquet')

# Load current universe
df_universe = pd.read_parquet('bond_data/parquet/universe.parquet')

# Example: Get time series for specific CUSIP
cusip_ts = df_hist[df_hist['CUSIP'] == '037833DX5'].sort_values('Date')

# Example: Filter by date range
recent = df_hist[df_hist['Date'] >= '2025-09-01']
```

---

## 📊 Data Schema

### Historical Bond Details Table
- **Primary Key**: `Date + CUSIP`
- **Columns**: 76 (Date + all source columns)
- **Purpose**: Complete time series of all bonds

### Universe Table
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

---

## 📈 Performance

- **Processing Time**: ~15-20 seconds for 11 files
- **Memory Usage**: ~100-200 MB
- **Output Size**: 
  - Historical: ~2-3 MB (compressed)
  - Universe: ~200-300 KB (compressed)

---

## 🔍 Data Quality

The pipeline includes comprehensive data validation:

- **CUSIP Validation**: Normalizes to uppercase, validates 9-character format
- **Deduplication**: Removes duplicate Date+CUSIP combinations (keeps last)
- **NA Handling**: Standardizes various NA representations
- **Schema Alignment**: Handles files with different column counts
- **Logging**: Tracks all data quality issues

---

## 📝 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[bond_pipeline_documentation.md](bond_pipeline_documentation.md)** - Complete technical documentation
  - Data discovery & analysis
  - Q&A with requirements
  - Implementation details
  - Test results
  - Production recommendations
- **[DELIVERABLES.txt](DELIVERABLES.txt)** - Project deliverables summary
- **[bond_pipeline/README.md](bond_pipeline/README.md)** - Module-level documentation

---

## 🧪 Testing

The pipeline has been tested with:
- 11 Excel files (2023-2025)
- 25,741 total rows
- 3,231 unique CUSIPs
- Schema evolution (59-75 columns)
- Duplicate handling (~500 duplicates)
- Invalid CUSIP detection (~100 invalid)

All tests passed successfully. See [bond_pipeline_documentation.md](bond_pipeline_documentation.md) for detailed test results.

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

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

This is a private project for professional bond trading. For questions or issues, please contact the development team.

---

## 📞 Support

For technical support or questions:
- Review the [documentation](bond_pipeline_documentation.md)
- Check the [logs](bond_data/logs/) for detailed error messages
- Refer to the [quick start guide](QUICKSTART.md)

---

**Built with ❤️ for professional bond traders**

