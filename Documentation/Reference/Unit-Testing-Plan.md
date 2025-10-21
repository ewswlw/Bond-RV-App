# Unit Testing Plan - Bond Data Pipeline

**Document Version**: 1.0  
**Date**: October 21, 2025  
**Purpose**: Comprehensive plan for unit testing all pipeline modules

---

## 📋 Testing Strategy

### Goals
1. **Code Coverage**: Aim for 80%+ coverage
2. **Edge Cases**: Test all known edge cases and data quality issues
3. **Regression Prevention**: Catch bugs before they reach production
4. **Documentation**: Tests serve as living documentation
5. **Fast Execution**: All tests should run in < 30 seconds

### Testing Framework
- **Framework**: `pytest` (industry standard, great fixtures)
- **Coverage**: `pytest-cov` (coverage reporting)
- **Mocking**: `unittest.mock` (mock file I/O, external dependencies)
- **Fixtures**: Sample data files for testing

---

## 🧪 Module-by-Module Test Plan

### 1. `config.py` - Configuration Tests

**Purpose**: Verify all paths and constants are correctly defined

#### Test Cases:

```python
test_paths_exist()
    # Verify PARQUET_DIR, LOG_DIR exist or can be created
    
test_constants_valid()
    # Verify DATE_COLUMN, MASTER_SCHEMA, UNIVERSE_COLUMNS are valid
    
test_na_values_list()
    # Verify NA_VALUES list contains expected patterns
```

**Priority**: Low (simple constants, but good baseline)

---

### 2. `utils.py` - Utility Functions Tests

**Purpose**: Test helper functions for date parsing, CUSIP validation, logging

#### Test Cases:

##### **Date Parsing (`parse_date_from_filename`)**

```python
test_parse_date_valid_formats()
    # Test: "API 10.20.25.xlsx" -> datetime(2025, 10, 20)
    # Test: "API 08.04.23.xlsx" -> datetime(2023, 8, 4)
    # Test: "API 12.29.23.xlsx" -> datetime(2023, 12, 29)
    
test_parse_date_invalid_formats()
    # Test: "API 2025-10-20.xlsx" -> raises ValueError
    # Test: "bonds_10_20_25.xlsx" -> raises ValueError
    # Test: "API 13.32.25.xlsx" -> raises ValueError (invalid date)
    
test_parse_date_edge_cases()
    # Test: "API 02.29.24.xlsx" -> leap year (valid)
    # Test: "API 02.29.23.xlsx" -> non-leap year (invalid)
    # Test: "API 01.01.00.xlsx" -> year 2000
    # Test: "API 12.31.99.xlsx" -> year 1999 or 2099?
```

##### **CUSIP Validation (`validate_cusip`)**

```python
test_validate_cusip_valid()
    # Test: "037833CY4" -> valid (9 chars, alphanumeric)
    # Test: "89678ZAB2" -> valid (9 chars, alphanumeric)
    # Test: "06051GJG5" -> valid
    
test_validate_cusip_invalid_length()
    # Test: "123" -> invalid (too short)
    # Test: "0636B108" -> invalid (8 chars)
    # Test: "6698Z3Z452" -> invalid (10 chars)
    # Test: "BBG01G27TPY1" -> invalid (12 chars, Bloomberg ID)
    
test_validate_cusip_invalid_characters()
    # Test: "880789A#9" -> invalid (contains #)
    # Test: "880789A 9" -> invalid (contains space)
    # Test: "880789A-9" -> invalid (contains hyphen)
    
test_validate_cusip_normalization()
    # Test: "89678zab2" -> normalized to "89678ZAB2"
    # Test: " 06418GAD9 Corp" -> extract and normalize to "06418GAD9"
    # Test: "38141GYD0 CORP" -> extract "38141GYD0"
    
test_validate_cusip_edge_cases()
    # Test: None -> invalid
    # Test: "" -> invalid
    # Test: "000000000" -> valid (all zeros)
    # Test: "AAAAAAAAA" -> valid (all letters)
```

##### **NA Value Cleaning (`clean_na_values`)**

```python
test_clean_na_values_standard()
    # Test: "#N/A Field Not Applicable" -> pd.NA
    # Test: "#N/A Invalid Security" -> pd.NA
    # Test: "N/A" -> pd.NA
    # Test: "nan" -> pd.NA
    
test_clean_na_values_dataframe()
    # Test: DataFrame with mixed NA values -> all converted to pd.NA
    # Test: Numeric columns preserved
    # Test: String columns preserved
    
test_clean_na_values_edge_cases()
    # Test: Empty DataFrame -> returns empty DataFrame
    # Test: All NA values -> all converted
    # Test: No NA values -> DataFrame unchanged
```

##### **Logging Setup (`setup_logger`)**

```python
test_logger_creation()
    # Test: Logger created with correct name
    # Test: Log file created in correct directory
    # Test: Log level set correctly
    
test_logger_multiple_calls()
    # Test: Multiple calls don't create duplicate handlers
    
test_logger_writes_to_file()
    # Test: Log messages written to file
    # Test: Log format correct (timestamp, level, message)
```

**Priority**: **HIGH** (core utilities used throughout pipeline)

---

### 3. `extract.py` - Excel Reading Tests

**Purpose**: Test Excel file reading and date extraction

#### Test Cases:

##### **File Discovery (`get_excel_files`)**

```python
test_get_excel_files_valid_directory()
    # Test: Directory with .xlsx files -> returns all files
    # Test: Files returned in sorted order
    
test_get_excel_files_empty_directory()
    # Test: Empty directory -> returns empty list
    
test_get_excel_files_no_excel_files()
    # Test: Directory with only .txt, .csv -> returns empty list
    
test_get_excel_files_invalid_directory()
    # Test: Non-existent directory -> raises FileNotFoundError
    
test_get_excel_files_mixed_extensions()
    # Test: Directory with .xlsx, .xls, .csv -> returns only .xlsx
```

##### **Excel Reading (`read_excel_file`)**

```python
test_read_excel_valid_file()
    # Test: Valid Excel file -> returns DataFrame
    # Test: Correct header row (row 2, index 1)
    # Test: All columns present
    
test_read_excel_date_extraction()
    # Test: "API 10.20.25.xlsx" -> Date column added with 2025-10-20
    # Test: All rows have same date
    
test_read_excel_invalid_file()
    # Test: Corrupted Excel file -> raises Exception
    # Test: Non-Excel file (.txt) -> raises Exception
    
test_read_excel_missing_columns()
    # Test: File missing expected columns -> handles gracefully
    
test_read_excel_empty_file()
    # Test: Excel file with no data rows -> returns empty DataFrame with Date column
    
test_read_excel_schema_evolution()
    # Test: Old file (59 cols) -> missing columns filled with NA
    # Test: New file (75 cols) -> all columns present
```

##### **Batch Processing (`extract_all_files`)**

```python
test_extract_all_files_success()
    # Test: Multiple valid files -> returns dict of DataFrames
    # Test: Keys are datetime objects
    # Test: Each DataFrame has correct date
    
test_extract_all_files_partial_failure()
    # Test: Mix of valid and invalid files -> returns only valid ones
    # Test: Logs errors for invalid files
    
test_extract_all_files_no_files()
    # Test: Empty directory -> returns empty dict
```

**Priority**: **HIGH** (critical for data ingestion)

---

### 4. `transform.py` - Data Transformation Tests

**Purpose**: Test data cleaning, normalization, deduplication

#### Test Cases:

##### **CUSIP Normalization (`normalize_cusips`)**

```python
test_normalize_cusips_uppercase()
    # Test: "89678zab2" -> "89678ZAB2"
    # Test: Mixed case -> all uppercase
    
test_normalize_cusips_validation()
    # Test: Valid CUSIPs -> no warnings
    # Test: Invalid CUSIPs -> logged warnings
    # Test: Invalid CUSIPs still included in output
    
test_normalize_cusips_edge_cases()
    # Test: None values -> handled gracefully
    # Test: Empty strings -> handled gracefully
    # Test: Whitespace trimmed
```

##### **Schema Alignment (`align_schema`)**

```python
test_align_schema_add_missing_columns()
    # Test: 59-col DataFrame -> 75 columns with NAs
    # Test: Missing columns added in correct order
    
test_align_schema_preserve_existing()
    # Test: 75-col DataFrame -> unchanged
    # Test: Data values preserved
    
test_align_schema_extra_columns()
    # Test: DataFrame with extra columns -> extra columns kept
    
test_align_schema_empty_dataframe()
    # Test: Empty DataFrame -> correct schema with 0 rows
```

##### **Deduplication (`remove_duplicates`)**

```python
test_remove_duplicates_basic()
    # Test: Duplicate Date+CUSIP -> keeps last occurrence
    # Test: Correct number of rows removed
    
test_remove_duplicates_logging()
    # Test: Duplicates found -> logged with sample CUSIPs
    # Test: No duplicates -> no warning logged
    
test_remove_duplicates_edge_cases()
    # Test: All rows duplicate -> keeps 1 of each Date+CUSIP
    # Test: No duplicates -> DataFrame unchanged
    # Test: Empty DataFrame -> returns empty DataFrame
    
test_remove_duplicates_data_integrity()
    # Test: Last occurrence has most recent data
    # Test: Other columns preserved correctly
```

##### **Full Transformation (`transform_data`)**

```python
test_transform_data_complete_pipeline()
    # Test: Raw DataFrame -> cleaned, normalized, deduplicated
    # Test: All steps applied in correct order
    
test_transform_data_with_issues()
    # Test: Data with invalid CUSIPs, duplicates, NAs -> all handled
    # Test: Appropriate warnings logged
```

**Priority**: **HIGH** (core business logic)

---

### 5. `load.py` - Parquet Writing Tests

**Purpose**: Test parquet file writing with append/override logic

#### Test Cases:

##### **Append Mode (`append_to_historical`)**

```python
test_append_new_dates()
    # Test: New dates added to existing parquet
    # Test: Existing dates skipped
    # Test: Combined DataFrame correct
    
test_append_no_new_dates()
    # Test: All dates already exist -> no changes
    # Test: Returns success with message
    
test_append_first_time()
    # Test: No existing parquet -> creates new file
    
test_append_duplicate_check()
    # Test: Combined data has no Date+CUSIP duplicates
    # Test: Raises error if duplicates found
    
test_append_schema_compatibility()
    # Test: New data with different schema -> merged correctly
```

##### **Override Mode (`override_historical`)**

```python
test_override_creates_new()
    # Test: No existing parquet -> creates new file
    # Test: Correct number of rows and dates
    
test_override_replaces_existing()
    # Test: Existing parquet deleted
    # Test: New parquet created with all data
    
test_override_duplicate_check()
    # Test: Final data has no Date+CUSIP duplicates
    # Test: Raises error if duplicates found
```

##### **Universe Creation (`create_universe`)**

```python
test_create_universe_unique_cusips()
    # Test: One row per CUSIP
    # Test: Most recent date for each CUSIP
    
test_create_universe_columns()
    # Test: Only 13 specified columns included
    # Test: Column order correct
    
test_create_universe_empty_input()
    # Test: Empty historical data -> empty universe
    
test_create_universe_data_integrity()
    # Test: CUSIP from latest date has correct Pricing Date
    # Test: All fields match source data
```

##### **Parquet I/O**

```python
test_parquet_write_read_roundtrip()
    # Test: Write DataFrame -> Read back -> identical
    # Test: Data types preserved
    
test_parquet_compression()
    # Test: File size reasonable (compressed)
    
test_parquet_invalid_path()
    # Test: Invalid path -> raises appropriate error
```

**Priority**: **HIGH** (critical for data persistence)

---

### 6. `pipeline.py` - Integration Tests

**Purpose**: Test end-to-end pipeline orchestration

#### Test Cases:

##### **CLI Argument Parsing**

```python
test_cli_valid_arguments()
    # Test: -i path -m override -> parsed correctly
    # Test: -i path -m append -> parsed correctly
    
test_cli_invalid_arguments()
    # Test: Missing -i -> error
    # Test: Invalid mode -> error
    # Test: --help -> shows help
```

##### **Pipeline Execution**

```python
test_pipeline_override_mode_success()
    # Test: Complete pipeline run in override mode
    # Test: Both parquet files created
    # Test: Logs generated
    
test_pipeline_append_mode_success()
    # Test: Complete pipeline run in append mode
    # Test: Only new dates processed
    # Test: Universe rebuilt
    
test_pipeline_no_files()
    # Test: Empty input directory -> graceful failure
    # Test: Appropriate error message
    
test_pipeline_partial_failure()
    # Test: Some files fail -> continues with valid files
    # Test: Errors logged
```

##### **Summary Statistics**

```python
test_summary_statistics_correct()
    # Test: Row counts correct
    # Test: Date range correct
    # Test: Unique CUSIP count correct
```

**Priority**: **MEDIUM** (integration tests, covered by module tests)

---

## 🎯 Test Data Strategy

### Sample Data Files

Create minimal test fixtures in `tests/fixtures/`:

```
tests/fixtures/
├── excel_files/
│   ├── valid/
│   │   ├── API 01.01.23.xlsx          # Valid, old schema (59 cols)
│   │   ├── API 02.01.23.xlsx          # Valid, new schema (75 cols)
│   │   └── API 03.01.23.xlsx          # Valid, with duplicates
│   ├── invalid/
│   │   ├── corrupted.xlsx             # Corrupted file
│   │   ├── wrong_format.txt           # Not an Excel file
│   │   └── API 13.32.23.xlsx          # Invalid date
│   └── edge_cases/
│       ├── API 02.29.24.xlsx          # Leap year
│       ├── empty.xlsx                 # No data rows
│       └── missing_columns.xlsx       # Missing expected columns
│
├── parquet_files/
│   ├── historical_sample.parquet      # Sample historical data
│   └── universe_sample.parquet        # Sample universe data
│
└── csv_data/
    ├── sample_bonds_59cols.csv        # Old schema
    ├── sample_bonds_75cols.csv        # New schema
    ├── sample_with_duplicates.csv     # Has duplicate Date+CUSIP
    └── sample_with_invalid_cusips.csv # Has invalid CUSIPs
```

### Test Data Characteristics

**Minimal but Representative:**
- 10-20 rows per file (fast tests)
- Include all known edge cases
- Real CUSIP patterns (valid and invalid)
- Mix of data types (strings, numbers, dates)

**Fixtures:**
```python
@pytest.fixture
def sample_dataframe():
    """Sample DataFrame with 10 bonds, valid data"""
    return pd.DataFrame({...})

@pytest.fixture
def sample_with_duplicates():
    """Sample DataFrame with duplicate Date+CUSIP"""
    return pd.DataFrame({...})

@pytest.fixture
def sample_invalid_cusips():
    """Sample DataFrame with invalid CUSIPs"""
    return pd.DataFrame({...})

@pytest.fixture
def temp_excel_file(tmp_path):
    """Create temporary Excel file for testing"""
    # Create Excel file in tmp_path
    # Return path
```

---

## 📊 Test Organization

### Directory Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── fixtures/                      # Test data files
│   ├── excel_files/
│   ├── parquet_files/
│   └── csv_data/
├── unit/                          # Unit tests
│   ├── test_config.py
│   ├── test_utils.py
│   ├── test_extract.py
│   ├── test_transform.py
│   └── test_load.py
├── integration/                   # Integration tests
│   └── test_pipeline.py
└── test_data_quality.py           # Data quality tests
```

---

## 🚀 Implementation Priority

### Phase 1: Core Utilities (Week 1)
- ✅ `test_utils.py` - Date parsing, CUSIP validation
- ✅ `test_extract.py` - Excel reading
- **Impact**: Catches most data ingestion issues

### Phase 2: Business Logic (Week 2)
- ✅ `test_transform.py` - Data cleaning, deduplication
- ✅ `test_load.py` - Parquet writing
- **Impact**: Ensures data integrity

### Phase 3: Integration (Week 3)
- ✅ `test_pipeline.py` - End-to-end tests
- ✅ `test_data_quality.py` - Data quality checks
- **Impact**: Catches regression issues

### Phase 4: Edge Cases (Week 4)
- ✅ Add more edge case tests based on production issues
- ✅ Increase coverage to 90%+
- **Impact**: Production-ready

---

## 📈 Coverage Goals

| Module | Target Coverage | Priority |
|--------|----------------|----------|
| `utils.py` | 95% | HIGH |
| `extract.py` | 90% | HIGH |
| `transform.py` | 95% | HIGH |
| `load.py` | 90% | HIGH |
| `config.py` | 80% | LOW |
| `pipeline.py` | 80% | MEDIUM |
| **Overall** | **85%+** | - |

---

## 🛠️ Testing Tools & Setup

### Installation

```bash
pip install pytest pytest-cov pytest-mock pandas pyarrow openpyxl
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=bond_pipeline --cov-report=html

# Run specific test file
pytest tests/unit/test_utils.py

# Run specific test
pytest tests/unit/test_utils.py::test_validate_cusip_valid

# Run with verbose output
pytest -v

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

### Configuration (`pytest.ini`)

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

---

## 🎯 Success Criteria

### Definition of Done
- [ ] All modules have unit tests
- [ ] Coverage ≥ 85%
- [ ] All tests pass
- [ ] Tests run in < 30 seconds
- [ ] CI/CD integration (GitHub Actions)
- [ ] Test documentation complete

### Quality Metrics
- **Test Coverage**: 85%+ overall, 90%+ for critical modules
- **Test Speed**: < 30 seconds for full suite
- **Test Reliability**: 0 flaky tests
- **Maintainability**: Clear test names, good documentation

---

## 🔄 Continuous Integration

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest --cov=bond_pipeline --cov-report=xml
      - uses: codecov/codecov-action@v2
```

---

## 📝 Next Steps

1. **Create test fixtures** - Build sample data files
2. **Implement Phase 1 tests** - Start with `test_utils.py`
3. **Set up pytest** - Configure pytest.ini and conftest.py
4. **Run and iterate** - Fix failing tests, improve coverage
5. **Add CI/CD** - Integrate with GitHub Actions
6. **Document** - Add testing guide to documentation

---

**Ready to implement? Let me know which phase you'd like to start with!**

