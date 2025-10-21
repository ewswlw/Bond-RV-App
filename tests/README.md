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
│   └── test_load.py            # TODO
├── integration/                 # Integration tests
│   └── test_pipeline.py        # TODO
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
| `utils.py` | 25 | ✅ COMPLETE | ~90% |
| `extract.py` | 0 | ⏳ TODO | 0% |
| `transform.py` | 0 | ⏳ TODO | 0% |
| `load.py` | 0 | ⏳ TODO | 0% |
| `config.py` | 0 | ⏳ TODO | 0% |
| `pipeline.py` | 0 | ⏳ TODO | 0% |
| **TOTAL** | **25** | **25% Complete** | **~15%** |

### Target Coverage

- **Overall**: 85%+
- **Critical modules** (utils, extract, transform, load): 90%+
- **Config**: 80%+
- **Pipeline**: 80%+

## Test Categories

### ✅ Completed Tests (25 tests)

#### Date Parsing (6 tests)
- ✅ Valid date formats
- ✅ Invalid date formats
- ✅ Leap year handling
- ✅ Edge cases (year 2000, 1999)
- ✅ Invalid month/day

#### CUSIP Validation (9 tests)
- ✅ Valid CUSIPs
- ✅ Invalid length (3, 8, 10, 12 chars)
- ✅ Invalid characters (#, -, space)
- ✅ Normalization (lowercase → uppercase)
- ✅ None and empty string handling
- ✅ All zeros and all letters
- ✅ Whitespace handling

#### NA Value Cleaning (5 tests)
- ✅ Standard NA values
- ✅ Mixed DataFrame
- ✅ Empty DataFrame
- ✅ All NA values
- ✅ No NA values

#### Logger Setup (5 tests)
- ✅ Logger creation
- ✅ Writes to file
- ✅ Multiple calls (no duplicate handlers)
- ✅ Different logger names
- ✅ Directory creation

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

1. ✅ **Phase 1 Complete**: `test_utils.py` (25 tests)
2. ⏳ **Phase 2**: Implement `test_extract.py` (20 tests)
3. ⏳ **Phase 3**: Implement `test_transform.py` (25 tests)
4. ⏳ **Phase 4**: Implement `test_load.py` (20 tests)
5. ⏳ **Phase 5**: Implement `test_pipeline.py` (10 tests)
6. ⏳ **Phase 6**: Add CI/CD integration

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Coverage.py](https://coverage.readthedocs.io/)

---

**Last Updated**: October 21, 2025  
**Status**: Phase 1 Complete (25/100+ tests)

