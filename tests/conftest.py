"""
Shared pytest fixtures for bond pipeline tests
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import openpyxl
from openpyxl import Workbook


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame with 10 bonds, valid data"""
    return pd.DataFrame({
        'CUSIP': ['037833CY4', '03524BAH9', '008911BL2', '06051GJG5', '89678ZAB2',
                  '06048WW63', '06051GJK6', '06051GJZ3', '06051GKE8', '06051GKJ7'],
        'Security': ['Bond A', 'Bond B', 'Bond C', 'Bond D', 'Bond E',
                     'Bond F', 'Bond G', 'Bond H', 'Bond I', 'Bond J'],
        'Benchmark Cusip': ['135087B45', '135087M68', '135087J39', '135087E67', '135087M27',
                            '135087B45', '135087M68', '135087J39', '135087E67', '135087M27'],
        'Custom_Sector': ['Non Financial Maple', 'Non Financial Maple', 'HY', 'IG', 'IG',
                          'HY', 'IG', 'Non Financial Maple', 'HY', 'IG'],
        'Pricing Date': [datetime(2020, 8, 13), datetime(2017, 5, 8), datetime(2021, 7, 27),
                        datetime(2019, 3, 15), datetime(2020, 4, 3),
                        datetime(2021, 1, 10), datetime(2019, 6, 20), datetime(2020, 9, 5),
                        datetime(2018, 11, 30), datetime(2022, 2, 14)],
        'Benchmark': ['Benchmark 1', 'Benchmark 2', 'Benchmark 3', 'Benchmark 4', 'Benchmark 5',
                      'Benchmark 1', 'Benchmark 2', 'Benchmark 3', 'Benchmark 4', 'Benchmark 5'],
        'Ticker': ['TICK1', 'TICK2', 'TICK3', 'TICK4', 'TICK5',
                   'TICK6', 'TICK7', 'TICK8', 'TICK9', 'TICK10'],
        'Currency': ['USD', 'USD', 'CAD', 'USD', 'CAD',
                     'USD', 'CAD', 'USD', 'CAD', 'USD'],
    })


@pytest.fixture
def sample_with_duplicates():
    """Sample DataFrame with duplicate Date+CUSIP"""
    df = pd.DataFrame({
        'Date': [datetime(2025, 10, 20)] * 6,
        'CUSIP': ['037833CY4', '037833CY4', '03524BAH9', '03524BAH9', '008911BL2', '008911BL2'],
        'Security': ['Bond A v1', 'Bond A v2', 'Bond B v1', 'Bond B v2', 'Bond C v1', 'Bond C v2'],
        'Benchmark Cusip': ['135087B45'] * 6,
        'Custom_Sector': ['IG'] * 6,
    })
    return df


@pytest.fixture
def sample_invalid_cusips():
    """Sample DataFrame with invalid CUSIPs"""
    return pd.DataFrame({
        'CUSIP': [
            '037833CY4',      # Valid
            '123',            # Too short
            '0636B108',       # 8 chars
            '6698Z3Z452',     # 10 chars
            '880789A#9',      # Invalid character
            'BBG01G27TPY1',   # Bloomberg ID
            '89678zab2',      # Lowercase (needs normalization)
            ' 06418GAD9 Corp', # Extra text
        ],
        'Security': [f'Bond {i}' for i in range(8)],
    })


@pytest.fixture
def sample_old_schema():
    """Sample DataFrame with old schema (59 columns)"""
    data = {
        'CUSIP': ['037833CY4', '03524BAH9'],
        'Security': ['Bond A', 'Bond B'],
        'Benchmark Cusip': ['135087B45', '135087M68'],
    }
    # Add 56 more columns to make 59 total
    for i in range(56):
        data[f'Column_{i}'] = ['Value'] * 2
    return pd.DataFrame(data)


@pytest.fixture
def sample_new_schema():
    """Sample DataFrame with new schema (75 columns)"""
    data = {
        'CUSIP': ['037833CY4', '03524BAH9'],
        'Security': ['Bond A', 'Bond B'],
        'Benchmark Cusip': ['135087B45', '135087M68'],
    }
    # Add 72 more columns to make 75 total
    for i in range(72):
        data[f'Column_{i}'] = ['Value'] * 2
    return pd.DataFrame(data)


@pytest.fixture
def temp_excel_file(tmp_path):
    """Create a temporary Excel file with sample data"""
    def _create_excel(filename, data_df, date_str='10.20.25'):
        """
        Create an Excel file with proper structure
        
        Args:
            filename: Name of the file (e.g., 'API 10.20.25.xlsx')
            data_df: DataFrame to write
            date_str: Date string for filename (MM.DD.YY format)
        """
        filepath = tmp_path / filename
        
        # Create workbook
        wb = Workbook()
        ws = wb.active
        
        # Row 1: Empty (matches real files)
        # Row 2: Headers
        headers = list(data_df.columns)
        for col_idx, header in enumerate(headers, start=1):
            ws.cell(row=2, column=col_idx, value=header)
        
        # Row 3+: Data
        for row_idx, row_data in enumerate(data_df.itertuples(index=False), start=3):
            for col_idx, value in enumerate(row_data, start=1):
                ws.cell(row=row_idx, column=col_idx, value=value)
        
        wb.save(filepath)
        return filepath
    
    return _create_excel


@pytest.fixture
def sample_dates():
    """Sample dates for testing"""
    return {
        'valid': [
            ('API 10.20.25.xlsx', datetime(2025, 10, 20)),
            ('API 08.04.23.xlsx', datetime(2023, 8, 4)),
            ('API 12.29.23.xlsx', datetime(2023, 12, 29)),
            ('API 01.01.00.xlsx', datetime(2000, 1, 1)),
        ],
        'invalid': [
            'API 2025-10-20.xlsx',  # Wrong format
            'bonds_10_20_25.xlsx',  # Wrong format
            'API 13.32.25.xlsx',    # Invalid date
            'API 02.29.23.xlsx',    # Non-leap year
        ],
        'edge_cases': [
            ('API 02.29.24.xlsx', datetime(2024, 2, 29)),  # Leap year
            ('API 12.31.99.xlsx', datetime(1999, 12, 31)),  # Year 1999
        ]
    }


@pytest.fixture
def sample_cusips():
    """Sample CUSIPs for testing"""
    return {
        'valid': [
            '037833CY4',
            '89678ZAB2',
            '06051GJG5',
            '000000000',  # All zeros
            'AAAAAAAAA',  # All letters
        ],
        'invalid_length': [
            ('123', 3),
            ('0636B108', 8),
            ('6698Z3Z452', 10),
            ('BBG01G27TPY1', 12),
        ],
        'invalid_chars': [
            '880789A#9',   # Contains #
            '880789A 9',   # Contains space
            '880789A-9',   # Contains hyphen
        ],
        'needs_normalization': [
            ('89678zab2', '89678ZAB2'),
            (' 06418GAD9 Corp', '06418GAD9'),
            ('38141GYD0 CORP', '38141GYD0'),
        ]
    }


@pytest.fixture
def sample_na_values():
    """Sample NA values for testing"""
    return [
        '#N/A Field Not Applicable',
        '#N/A Invalid Security',
        'N/A',
        'nan',
        'NaN',
        '#N/A',
    ]


@pytest.fixture
def mock_logger(mocker):
    """Mock logger for testing"""
    return mocker.Mock()


@pytest.fixture
def clean_test_dirs(tmp_path):
    """Create clean test directories"""
    parquet_dir = tmp_path / "parquet"
    logs_dir = tmp_path / "logs"
    raw_data_dir = tmp_path / "raw_data"
    
    parquet_dir.mkdir()
    logs_dir.mkdir()
    raw_data_dir.mkdir()
    
    return {
        'parquet': parquet_dir,
        'logs': logs_dir,
        'raw_data': raw_data_dir,
    }

