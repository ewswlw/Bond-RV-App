# CLAUDE.md

**Last Updated**: October 24, 2025 12:00 PM

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bond RV App is a modular ETL pipeline that processes Excel files containing bond market data and transforms them into optimized Parquet tables for relative value trading analysis. The pipeline handles schema evolution (59-75 columns across different file vintages), CUSIP validation, deduplication, and produces two core outputs: a historical time series table and a universe table of unique bonds.

## Architecture

### Pipeline Flow (ETL Pattern)
1. **Extract** (`extract.py`) - Reads Excel files with regex pattern matching on filenames to extract dates
2. **Transform** (`transform.py`) - Validates CUSIPs, removes duplicates, normalizes data, aligns schemas
3. **Load** (`load.py`) - Writes to Parquet with append/override modes

### Key Components
- **config.py** - Central configuration with hardcoded Dropbox paths, schema definitions, NA value mappings
- **utils.py** - Date parsing, CUSIP validation, logging setup
- **pipeline.py** - Main orchestrator that coordinates E-T-L flow
- **run_pipeline.py** - Simple CLI wrapper at project root

### Critical Design Patterns
- **Schema Evolution**: Master schema is dynamically set from the latest file, older files are padded with missing columns
- **Deduplication**: Keeps last occurrence of Date+CUSIP duplicates (not first)
- **Append Mode**: Checks existing Parquet for dates, skips already-processed dates
- **Override Mode**: Deletes existing Parquet and rebuilds from scratch

### Data Flow
```
Excel files (Raw Data/ or Dropbox) → extract.py → raw_data dict[datetime, DataFrame]
→ transform.py → transformed_data dict[datetime, DataFrame]
→ load.py → historical_bond_details.parquet + universe.parquet
```

**Input Flexibility**: The pipeline supports two workflows:
- **Local Workflow**: Files in `Raw Data/` folder (recommended - simple drag & drop)
- **Dropbox Workflow**: Files in Dropbox folder (optional - for automatic syncing across computers)

## Virtual Environment Setup

**CRITICAL**: This project uses a virtual environment named `Bond-RV-App` (matching the project name).

### First-Time Setup
```bash
# Create virtual environment
python -m venv Bond-RV-App

# Activate virtual environment
Bond-RV-App\Scripts\activate       # Windows
source Bond-RV-App/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Daily Use
```bash
# Always activate before running any scripts
Bond-RV-App\Scripts\activate       # Windows
source Bond-RV-App/bin/activate    # Mac/Linux
```

## Running the Pipeline

**Note**: All commands below assume the virtual environment is activated.

### Quick Run (Interactive)
```bash
# From project root
python run_pipeline.py
# Select mode: 1=override (first run), 2=append (daily updates)
```

### Using Virtual Environment Without Activation
```bash
# Windows - use full path to venv Python
Bond-RV-App\Scripts\python.exe run_pipeline.py
```

### Direct Pipeline Execution
```bash
# From bond_pipeline directory
python pipeline.py -i "../Raw Data/" -m override    # First run
python pipeline.py -i "../Raw Data/" -m append      # Daily updates

# With custom input directory
python pipeline.py -i "path/to/excel/files" -m append
```

### Non-Interactive (for automation)
```bash
echo 2 | python run_pipeline.py  # Runs in append mode
```

## Testing

### Run All Tests
```bash
pytest                           # All tests
pytest -v                        # Verbose
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests only
```

### Run Specific Test
```bash
pytest tests/unit/test_utils.py                    # Single file
pytest tests/unit/test_utils.py::test_function_name # Single test
pytest -k "cusip"                                   # Pattern match
```

### Test Coverage
```bash
pytest --cov=bond_pipeline --cov-report=html
```

## Configuration & Paths

### Critical Path Configuration
The DEFAULT_INPUT_DIR in `config.py` is hardcoded to a Dropbox path:
```python
DEFAULT_INPUT_DIR = Path(r"C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Support Files\API Historical")
```

This path is machine-specific and assumes Dropbox sync. When working on different machines, this path must be updated.

### File Naming Convention
Excel files must match pattern: `API MM.DD.YY.xlsx`
- Example: `API 10.20.25.xlsx` (October 20, 2025)
- Regex: `r'API\s+(\d{2})\.(\d{2})\.(\d{2})\.xlsx$'`
- Years 00-49 → 2000-2049, years 50-99 → 1950-1999

### Output Structure
```
bond_data/
├── parquet/
│   ├── historical_bond_details.parquet  # Primary key: Date + CUSIP
│   └── universe.parquet                  # Primary key: CUSIP (13 columns)
└── logs/
    ├── processing.log   # Main pipeline operations
    ├── duplicates.log   # Duplicate detection details
    ├── validation.log   # CUSIP validation warnings
    └── summary.log      # High-level pipeline summary
```

## Data Quality Rules

### CUSIP Validation
- Must be exactly 9 alphanumeric characters
- Automatically uppercased
- Invalid CUSIPs are logged but NOT removed (kept in dataset with validation warnings)
- Common issues: wrong length (8, 10, 12 chars), special characters, Bloomberg IDs

### Duplicate Handling
- Duplicates defined as: same Date + CUSIP combination within a single file
- Strategy: Keep LAST occurrence (not first)
- Logged to `duplicates.log` with sample CUSIPs

### NA Value Normalization
These strings are converted to pandas NA:
- `#N/A Field Not Applicable`
- `#N/A Invalid Security`
- `#N/A`, `N/A`, empty strings

## Coding Standards

### Documentation Requirements
- All new documentation must include **date and timestamp**
- When creating documentation, review and update related existing docs for consistency

### Configuration
- All config must be in `.py` files (NOT `.json`, `.yaml`, `.toml`)
- Use `config.py` for all constants and paths

### Dependencies
- Project uses pip with `requirements.txt` (not Poetry)
- All dependencies (production + testing) are in a single `requirements.txt` file
- Core dependencies: pandas>=2.0.0, pyarrow>=12.0.0, openpyxl>=3.1.0
- Test dependencies: pytest>=7.0.0, pytest-cov>=4.0.0, pytest-mock>=3.10.0

## Common Modifications

### Adding New Validation Rules
Edit `transform.py` → `DataTransformer` class → Add validation logic in `transform_single_file()`

### Changing Output Schema
Edit `config.py` → Modify `UNIVERSE_COLUMNS` for universe table

### Adjusting Duplicate Logic
Edit `transform.py` → `remove_duplicates()` method → Change `keep='last'` parameter

### Adding New CLI Options
Edit `pipeline.py` → `main()` function → Add argparse arguments

## Troubleshooting

### Pipeline Fails with "No files found"
- Check Dropbox sync status (must be fully synced)
- Verify DEFAULT_INPUT_DIR path in `config.py` matches your machine
- Confirm Excel files match naming pattern `API MM.DD.YY.xlsx`

### "Could not convert with type str" Error
- Object columns must be converted to strings before Parquet write
- Fix is in `load.py` lines 92-95 (already implemented for append mode)

### Different Results on Different Computers
- Run in override mode to rebuild from scratch
- Ensure same Excel files are present
- Check Dropbox sync completion

### Missing Dependencies
```bash
# All dependencies are in requirements.txt
pip install -r requirements.txt
```

## Documentation Structure

The project has comprehensive documentation in the `Documentation/` folder:
- **Setup/** - Installation and virtual environment guides
- **Workflows/** - Local and Dropbox workflows
- **Architecture/** - Technical documentation
- **Reference/** - Testing, deliverables, and decision documents

See `Documentation/README.md` for complete documentation index.
