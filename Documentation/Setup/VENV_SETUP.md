# Virtual Environment Setup Guide

**Last Updated:** 2025-10-24

## Overview

This project uses a virtual environment named `Bond-RV-App` to isolate dependencies.

## Quick Reference

### Activate Virtual Environment
```bash
# Windows
Bond-RV-App\Scripts\activate

# Mac/Linux
source Bond-RV-App/bin/activate
```

### Deactivate
```bash
deactivate
```

### Run Without Activation (Windows)
```bash
Bond-RV-App\Scripts\python.exe run_pipeline.py
Bond-RV-App\Scripts\python.exe -m pytest
```

## First-Time Setup

If the virtual environment doesn't exist yet:

```bash
# 1. Create virtual environment
python -m venv Bond-RV-App

# 2. Activate it
Bond-RV-App\Scripts\activate       # Windows
source Bond-RV-App/bin/activate    # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import pandas; import pyarrow; import openpyxl; print('All dependencies installed!')"
```

## Installed Packages

### Production Dependencies
- `pandas>=2.0.0` - Data manipulation
- `pyarrow>=12.0.0` - Parquet file support
- `openpyxl>=3.1.0` - Excel file reading

### Development/Testing Dependencies
- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Test coverage
- `pytest-mock>=3.10.0` - Mocking for tests

## Common Tasks

### Running the Pipeline
```bash
# With activation
Bond-RV-App\Scripts\activate
python run_pipeline.py

# Without activation (Windows)
Bond-RV-App\Scripts\python.exe run_pipeline.py
```

### Running Tests
```bash
# With activation
Bond-RV-App\Scripts\activate
pytest

# Without activation (Windows)
Bond-RV-App\Scripts\python.exe -m pytest
```

### Installing New Packages
```bash
# Activate first
Bond-RV-App\Scripts\activate

# Install package
pip install package-name

# Update requirements.txt
pip freeze > requirements.txt
```

## Troubleshooting

### Virtual Environment Not Found
If you get an error about the virtual environment not existing:
```bash
python -m venv Bond-RV-App
```

### Module Not Found Errors
If you get import errors:
```bash
Bond-RV-App\Scripts\activate
pip install -r requirements.txt
```

### Wrong Python Version
Check your Python version:
```bash
Bond-RV-App\Scripts\python.exe --version
```
Should be Python 3.11+

## Git Ignore

The virtual environment folder `Bond-RV-App/` is automatically excluded from git commits via `.gitignore`.
