"""
Portfolio pipeline package for processing portfolio holdings Excel files.

This module processes portfolio holdings data from AD History directory,
extracting dates from filenames (Aggies MM.DD.YY.xlsx pattern) and
creating historical_portfolio.parquet with Date+CUSIP+ACCOUNT+PORTFOLIO primary key.
"""

from .pipeline import PortfolioDataPipeline

__all__ = ['PortfolioDataPipeline']

