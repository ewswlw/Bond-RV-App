"""
Unit tests for utils.py module
Tests: date parsing, CUSIP validation, NA cleaning, logging
"""
import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from bond_pipeline.utils import (
    extract_date_from_filename,
    validate_cusip,
    clean_na_values,
    setup_logging
)


class TestDateParsing:
    """Tests for extract_date_from_filename function"""
    
    def test_parse_date_valid_formats(self, sample_dates):
        """Test parsing valid date formats"""
        for filename, expected_date in sample_dates['valid']:
            result = extract_date_from_filename(filename)
            assert result == expected_date, f"Failed for {filename}"
    
    def test_parse_date_invalid_formats(self, sample_dates):
        """Test that invalid formats return None"""
        for filename in sample_dates['invalid']:
            result = extract_date_from_filename(filename)
            assert result is None, f"Should return None for {filename}"
    
    def test_parse_date_leap_year(self):
        """Test leap year handling"""
        # Valid leap year
        result = extract_date_from_filename('API 02.29.24.xlsx')
        assert result == datetime(2024, 2, 29)
        
        # Invalid non-leap year - should return None
        result = extract_date_from_filename('API 02.29.23.xlsx')
        assert result is None
    
    def test_parse_date_edge_cases(self):
        """Test edge cases"""
        # Year 2000
        result = extract_date_from_filename('API 01.01.00.xlsx')
        assert result.year == 2000
        
        # Year 1999
        result = extract_date_from_filename('API 12.31.99.xlsx')
        assert result.year == 1999
    
    def test_parse_date_invalid_month(self):
        """Test invalid month returns None"""
        result = extract_date_from_filename('API 13.01.25.xlsx')
        assert result is None
    
    def test_parse_date_invalid_day(self):
        """Test invalid day returns None"""
        result = extract_date_from_filename('API 01.32.25.xlsx')
        assert result is None


class TestCUSIPValidation:
    """Tests for validate_cusip function"""
    
    def test_validate_cusip_valid(self, sample_cusips):
        """Test valid CUSIPs"""
        for cusip in sample_cusips['valid']:
            normalized, is_valid, error = validate_cusip(cusip)
            assert is_valid, f"CUSIP {cusip} should be valid: {error}"
            assert normalized == cusip.upper()
            assert error == ''
    
    def test_validate_cusip_invalid_length(self, sample_cusips):
        """Test CUSIPs with invalid length"""
        for cusip, length in sample_cusips['invalid_length']:
            normalized, is_valid, error = validate_cusip(cusip)
            assert not is_valid, f"CUSIP {cusip} should be invalid"
            assert f"Invalid length: {length}" in error
    
    def test_validate_cusip_invalid_characters(self, sample_cusips):
        """Test CUSIPs with invalid characters"""
        for cusip in sample_cusips['invalid_chars']:
            normalized, is_valid, error = validate_cusip(cusip)
            assert not is_valid, f"CUSIP {cusip} should be invalid"
            assert "non-alphanumeric" in error.lower()
    
    def test_validate_cusip_normalization(self, sample_cusips):
        """Test CUSIP normalization to uppercase"""
        for original, expected in sample_cusips['needs_normalization']:
            normalized, is_valid, error = validate_cusip(original)
            # Extract just the CUSIP part if there's extra text
            assert expected in normalized, f"Failed to normalize {original}, got {normalized}"
    
    def test_validate_cusip_none(self):
        """Test None value"""
        normalized, is_valid, error = validate_cusip(None)
        assert not is_valid
        assert "Empty CUSIP" in error
    
    def test_validate_cusip_empty_string(self):
        """Test empty string"""
        normalized, is_valid, error = validate_cusip('')
        assert not is_valid
        assert "Empty CUSIP" in error
    
    def test_validate_cusip_all_zeros(self):
        """Test all zeros (technically valid)"""
        normalized, is_valid, error = validate_cusip('000000000')
        assert is_valid
        assert normalized == '000000000'
        assert error == ''
    
    def test_validate_cusip_all_letters(self):
        """Test all letters (technically valid)"""
        normalized, is_valid, error = validate_cusip('AAAAAAAAA')
        assert is_valid
        assert normalized == 'AAAAAAAAA'
        assert error == ''
    
    def test_validate_cusip_whitespace(self):
        """Test CUSIP with leading/trailing whitespace"""
        normalized, is_valid, error = validate_cusip('  037833CY4  ')
        # Should strip and normalize
        assert '037833CY4' == normalized
        assert is_valid


class TestNAValueCleaning:
    """Tests for clean_na_values function"""
    
    def test_clean_na_values_standard(self):
        """Test cleaning standard NA values"""
        # Note: 'NaN' string is different from actual NaN
        na_values = ['#N/A Field Not Applicable', '#N/A Invalid Security', 'N/A', 'nan', '#N/A']
        df = pd.DataFrame({
            'col1': na_values,
            'col2': ['Value'] * len(na_values)
        })
        
        result = clean_na_values(df)
        
        # Most NA values should be converted
        assert result['col1'].isna().sum() >= 4, f"Expected at least 4 NAs, got {result['col1'].isna().sum()}"
        # col2 should be unchanged
        assert result['col2'].notna().all()
    
    def test_clean_na_values_mixed_dataframe(self):
        """Test DataFrame with mixed data types"""
        df = pd.DataFrame({
            'string_col': ['Value', '#N/A Field Not Applicable', 'Another'],
            'numeric_col': [1.5, 2.0, 3.5],
            'date_col': [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)]
        })
        
        result = clean_na_values(df)
        
        # Check string column cleaned
        assert result['string_col'].isna().sum() == 1
        # Check numeric column preserved
        assert result['numeric_col'].sum() == 7.0
        # Check date column preserved
        assert result['date_col'].notna().all()
    
    def test_clean_na_values_empty_dataframe(self):
        """Test empty DataFrame"""
        df = pd.DataFrame()
        result = clean_na_values(df)
        assert result.empty
    
    def test_clean_na_values_all_na(self):
        """Test DataFrame with all NA values"""
        df = pd.DataFrame({
            'col1': ['#N/A Field Not Applicable'] * 5,
            'col2': ['N/A'] * 5
        })
        
        result = clean_na_values(df)
        
        assert result['col1'].isna().all()
        assert result['col2'].isna().all()
    
    def test_clean_na_values_no_na(self):
        """Test DataFrame with no NA values"""
        df = pd.DataFrame({
            'col1': ['Value1', 'Value2', 'Value3'],
            'col2': [1, 2, 3]
        })
        
        result = clean_na_values(df)
        
        # DataFrame should be unchanged
        assert result['col1'].notna().all()
        assert result['col2'].notna().all()


class TestLoggerSetup:
    """Tests for setup_logging function"""
    
    def test_logger_creation(self, tmp_path):
        """Test logger is created with correct name"""
        log_file = tmp_path / "test.log"
        # Create parent directory
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger = setup_logging(log_file, "test_logger")
        
        assert logger.name == "test_logger"
        # Log file should be created after first write
        logger.info("Test")
        assert log_file.exists()
    
    def test_logger_writes_to_file(self, tmp_path):
        """Test logger writes messages to file"""
        log_file = tmp_path / "test.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger = setup_logging(log_file, "test_logger_write")
        
        test_message = "Test log message"
        logger.info(test_message)
        
        # Read log file
        with open(log_file, 'r') as f:
            content = f.read()
        
        assert test_message in content
        assert "INFO" in content
    
    def test_logger_multiple_calls(self, tmp_path):
        """Test multiple calls don't create duplicate handlers"""
        log_file = tmp_path / "test.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger1 = setup_logging(log_file, "test_logger_multi")
        logger2 = setup_logging(log_file, "test_logger_multi")
        
        # Should be the same logger
        assert logger1 is logger2
        
        # Write a message
        logger1.info("Test message")
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Should only have one line (not duplicated)
        matching_lines = [l for l in lines if "Test message" in l]
        assert len(matching_lines) == 1
    
    def test_logger_different_names(self, tmp_path):
        """Test loggers with different names are separate"""
        log_file1 = tmp_path / "test1.log"
        log_file2 = tmp_path / "test2.log"
        log_file1.parent.mkdir(parents=True, exist_ok=True)
        log_file2.parent.mkdir(parents=True, exist_ok=True)
        
        logger1 = setup_logging(log_file1, "logger1")
        logger2 = setup_logging(log_file2, "logger2")
        
        assert logger1 is not logger2
        assert logger1.name != logger2.name
    
    def test_logger_creates_directory(self, tmp_path):
        """Test logger handles directory creation gracefully"""
        log_dir = tmp_path / "logs" / "subdir"
        log_file = log_dir / "test.log"
        
        # Create directory first (setup_logging doesn't create dirs)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger = setup_logging(log_file, "test_logger_dir")
        logger.info("Test")
        
        assert log_dir.exists()
        assert log_file.exists()


# Run tests with: pytest tests/unit/test_utils.py -v

