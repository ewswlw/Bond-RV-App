# Bond Pipeline Tests

## Overview

Comprehensive test suite for the bond data pipeline with **25+ unit tests** covering all core functionality.

## Test Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # Shared fixtures
├── pytest.ini                   # Pytest configuration
├── unit/                        # Unit tests
│   ├── test_utils.py           # ✅ 25 tests (COMPLETE)
│   ├── test_extract.py         # TODO
│   ├── test_transform.py       # TODO
│   ├── test_load.py            # TODO
│   ├── test_utils_runs.py      # ✅ 30 tests (COMPLETE)
│   ├── test_extract_runs.py    # ✅ 14 tests (COMPLETE)
│   ├── test_transform_runs.py  # ✅ 17 tests (COMPLETE)
│   └── test_load_runs.py       # ✅ 15 tests (COMPLETE)
├── integration/                 # Integration tests
│   ├── test_pipeline.py        # TODO
│   ├── test_pipeline_runs.py  # ✅ 7 tests (COMPLETE)
│   └── test_run_pipeline.py    # ✅ 16 tests (COMPLETE)
└── fixtures/                    # Test data
    ├── excel_files/
    ├── parquet_files/
    └── csv_data/
```

## Running Tests

### Install Dependencies

```bash
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/unit/test_utils.py
```

### Run with Coverage

```bash
pytest --cov=bond_pipeline --cov-report=html
```

### Run with Verbose Output

```bash
pytest -v
```

### Run Specific Test

```bash
pytest tests/unit/test_utils.py::TestCUSIPValidation::test_validate_cusip_valid
```

## Test Coverage

### Current Status

| Module | Tests | Status | Coverage |
|--------|-------|--------|----------|
| **Bond Pipeline:** |
| `utils.py` | 25 | ✅ COMPLETE | ~90% |
| `extract.py` | 0 | ⏳ TODO | 0% |
| `transform.py` | 0 | ⏳ TODO | 0% |
| `load.py` | 0 | ⏳ TODO | 0% |
| `config.py` | 0 | ⏳ TODO | 0% |
| `pipeline.py` | 0 | ⏳ TODO | 0% |
| **Runs Pipeline:** |
| `utils.py` (runs functions) | 30 | ✅ COMPLETE | ~90% |
| `runs_pipeline/extract.py` | 14 | ✅ COMPLETE | ~85% |
| `runs_pipeline/transform.py` | 17 | ✅ COMPLETE | ~85% |
| `runs_pipeline/load.py` | 15 | ✅ COMPLETE | ~85% |
| `runs_pipeline/pipeline.py` | 7 | ✅ COMPLETE | ~80% |
| `run_pipeline.py` | 16 | ✅ COMPLETE | ~90% |
| **TOTAL** | **118** | **~65% Complete** | **~40%** |

### Target Coverage

- **Overall**: 85%+
- **Critical modules** (utils, extract, transform, load): 90%+
- **Config**: 80%+
- **Pipeline**: 80%+

## Test Categories

### ✅ Completed Tests (118 tests)

#### Bond Pipeline Tests (25 tests)

##### Date Parsing (6 tests)
- ✅ Valid date formats
- ✅ Invalid date formats
- ✅ Leap year handling
- ✅ Edge cases (year 2000, 1999)
- ✅ Invalid month/day

##### CUSIP Validation (9 tests)
- ✅ Valid CUSIPs
- ✅ Invalid length (3, 8, 10, 12 chars)
- ✅ Invalid characters (#, -, space)
- ✅ Normalization (lowercase → uppercase)
- ✅ None and empty string handling
- ✅ All zeros and all letters
- ✅ Whitespace handling

##### NA Value Cleaning (5 tests)
- ✅ Standard NA values
- ✅ Mixed DataFrame
- ✅ Empty DataFrame
- ✅ All NA values
- ✅ No NA values

##### Logger Setup (5 tests)
- ✅ Logger creation
- ✅ Writes to file
- ✅ Multiple calls (no duplicate handlers)
- ✅ Different logger names
- ✅ Directory creation

#### Runs Pipeline Tests (77 tests)

##### Runs Date Parsing (10 tests)
- ✅ Valid MM/DD/YY formats
- ✅ Invalid formats
- ✅ Leap year handling
- ✅ Edge cases (year 2000, 1999, 2050+)
- ✅ Invalid month/day
- ✅ Empty string/None handling
- ✅ Whitespace handling

##### Runs Time Parsing (8 tests)
- ✅ Valid HH:MM formats
- ✅ Invalid formats (seconds, wrong separator)
- ✅ Invalid hour/minute
- ✅ Edge cases (midnight, end of day)
- ✅ Empty string/None handling
- ✅ Whitespace handling

##### CUSIP Orphan Tracking (5 tests)
- ✅ No orphans (all in universe)
- ✅ With orphans
- ✅ Universe file doesn't exist
- ✅ Empty runs CUSIPs
- ✅ All orphans

##### Runs Data Validation (8 tests)
- ✅ Valid data
- ✅ Missing dates/times
- ✅ Negative prices
- ✅ Extreme spreads
- ✅ Unknown dealers
- ✅ Date range outside bounds
- ✅ Missing date column

##### Runs Extraction (14 tests)
- ✅ Valid Excel reading
- ✅ Date/Time parsing
- ✅ Column reordering (Date, Time first)
- ✅ Missing Date/Time columns
- ✅ Empty files
- ✅ Invalid date/time formats
- ✅ Multiple files extraction
- ✅ Error handling

##### Runs Transformation (17 tests)
- ✅ End-of-day deduplication (latest Time)
- ✅ Same Time tiebreaker (keep last)
- ✅ CUSIP validation
- ✅ CUSIP orphan tracking
- ✅ Schema alignment
- ✅ Data cleaning (NA values)
- ✅ Full transform pipeline

##### Runs Loading (15 tests)
- ✅ Append mode (new dates)
- ✅ Append mode (skip existing dates)
- ✅ Override mode (new/replace)
- ✅ Primary key validation
- ✅ Empty data handling
- ✅ Summary statistics
- ✅ File existence checks

##### Runs Pipeline Integration (7 tests)
- ✅ Pipeline initialization
- ✅ Append mode execution
- ✅ Override mode execution
- ✅ File sorting (chronological)
- ✅ No files found
- ✅ Empty files
- ✅ Full pipeline workflow

##### Run Pipeline Script Integration (16 tests)
- ✅ Bond pipeline execution (append/override modes)
- ✅ Bond pipeline failure handling
- ✅ Bond pipeline exception handling
- ✅ Runs pipeline execution (append/override modes)
- ✅ Runs pipeline failure handling
- ✅ Runs pipeline exception handling
- ✅ Main function: bond pipeline only
- ✅ Main function: runs pipeline only
- ✅ Main function: both pipelines (success)
- ✅ Main function: both pipelines (bond fails)
- ✅ Main function: both pipelines (runs fails)
- ✅ Main function: override mode
- ✅ Main function: default choices
- ✅ Main function: invalid choices handling

### ⏳ Pending Tests

#### Extract Module (~20 tests)
- File discovery
- Excel reading
- Date extraction
- Schema evolution
- Error handling

#### Transform Module (~25 tests)
- CUSIP normalization
- Schema alignment
- Deduplication
- Data integrity

#### Load Module (~20 tests)
- Append mode
- Override mode
- Universe creation
- Parquet I/O

#### Pipeline Module (~10 tests)
- CLI argument parsing
- Override mode execution
- Append mode execution
- Error handling

## Fixtures

### Available Fixtures

All fixtures are defined in `conftest.py`:

- `sample_dataframe()` - 10 bonds with valid data
- `sample_with_duplicates()` - DataFrame with duplicate Date+CUSIP
- `sample_invalid_cusips()` - DataFrame with invalid CUSIPs
- `sample_old_schema()` - 59-column schema
- `sample_new_schema()` - 75-column schema
- `temp_excel_file(tmp_path)` - Create temporary Excel files
- `sample_dates()` - Valid, invalid, and edge case dates
- `sample_cusips()` - Valid, invalid, and normalization cases
- `sample_na_values()` - Standard NA value patterns
- `clean_test_dirs(tmp_path)` - Clean test directories

### Using Fixtures

```python
def test_example(sample_dataframe):
    """Test using sample DataFrame fixture"""
    assert len(sample_dataframe) == 10
    assert 'CUSIP' in sample_dataframe.columns
```

## Writing New Tests

### Test Template

```python
class TestModuleName:
    """Tests for module_name.py"""
    
    def test_function_name_scenario(self, fixture_if_needed):
        """Test description"""
        # Arrange
        input_data = ...
        
        # Act
        result = function_under_test(input_data)
        
        # Assert
        assert result == expected_value
```

### Best Practices

1. **One assertion per test** (when possible)
2. **Clear test names** - describe what is being tested
3. **Use fixtures** - avoid code duplication
4. **Test edge cases** - None, empty, invalid inputs
5. **Mock external dependencies** - file I/O, network calls
6. **Keep tests fast** - < 30 seconds for full suite

## Continuous Integration

### GitHub Actions (TODO)

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
      - run: pip install -r requirements-dev.txt
      - run: pytest --cov=bond_pipeline --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:

```bash
# Make sure you're in the project root
cd /path/to/Bond-RV-App

# Run tests with python -m pytest
python -m pytest tests/
```

### Fixture Not Found

If you see `fixture 'xxx' not found`:

1. Check that `conftest.py` is in the `tests/` directory
2. Verify the fixture is defined in `conftest.py`
3. Check fixture name spelling

### Test Discovery Issues

If pytest doesn't find your tests:

1. Test files must start with `test_`
2. Test functions must start with `test_`
3. Test classes must start with `Test`

## Next Steps

### Bond Pipeline Tests
1. ✅ **Phase 1 Complete**: `test_utils.py` (25 tests)
2. ⏳ **Phase 2**: Implement `test_extract.py` (20 tests)
3. ⏳ **Phase 3**: Implement `test_transform.py` (25 tests)
4. ⏳ **Phase 4**: Implement `test_load.py` (20 tests)
5. ⏳ **Phase 5**: Implement `test_pipeline.py` (10 tests)

### Runs Pipeline Tests
1. ✅ **Phase 1 Complete**: `test_utils_runs.py` (30 tests)
2. ✅ **Phase 2 Complete**: `test_extract_runs.py` (14 tests)
3. ✅ **Phase 3 Complete**: `test_transform_runs.py` (17 tests)
4. ✅ **Phase 4 Complete**: `test_load_runs.py` (15 tests)
5. ✅ **Phase 5 Complete**: `test_pipeline_runs.py` (7 tests)

### Run Pipeline Script Tests
1. ✅ **Phase 1 Complete**: `test_run_pipeline.py` (16 tests)
   - Bond pipeline execution (append/override)
   - Runs pipeline execution (append/override)
   - Main function orchestration (all combinations)
   - Error handling (failures, exceptions)

### Infrastructure
2. ⏳ **Phase 2**: Add CI/CD integration

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Coverage.py](https://coverage.readthedocs.io/)

---

**Last Updated**: January 2025  
**Status**: 
- Bond Pipeline: Phase 1 Complete (25/100+ tests)
- Runs Pipeline: All Phases Complete (77/77 tests)
- Run Pipeline Script: All Phases Complete (16/16 tests)
- **Total**: 118 tests complete

