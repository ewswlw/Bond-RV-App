"""
Runs views script.

This module creates custom formatted tables from runs_today.csv data.
Outputs nicely formatted tables to portfolio_runs_view.txt for portfolio monitoring.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Get script directory and build paths relative to it
SCRIPT_DIR = Path(__file__).parent.resolve()
RUNS_TODAY_CSV_PATH = SCRIPT_DIR.parent / "processed_data" / "runs_today.csv"
RUNS_PARQUET_PATH = SCRIPT_DIR.parent.parent / "bond_data" / "parquet" / "runs_timeseries.parquet"
OUTPUT_DIR = SCRIPT_DIR.parent / "processed_data"
OUTPUT_FILE = OUTPUT_DIR / "portfolio_runs_view.txt"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Column display names (shorter versions for table display)
COLUMN_DISPLAY_NAMES = {
    "Security": "Security",
    "QUANTITY": "Qty",
    "POSITION CR01": "Pos CR01",
    "Yrs (Cvn)": "Yrs Cvn",
    "Tight Bid >3mm": "TB >3mm",
    "Wide Offer >3mm": "WO >3mm",
    "Tight Bid": "TB",
    "Wide Offer": "WO",
    "Bid/Offer>3mm": "B/O>3mm",
    "Bid/Offer": "B/O",
    "Dealer @ Tight Bid >3mm": "Dealer TB>3mm",
    "Dealer @ Wide Offer >3mm": "Dealer WO>3mm",
    "Size @ Tight Bid >3mm": "Size TB>3mm",
    "Size @ Wide Offer >3mm": "Size WO>3mm",
    "CR01 @ Tight Bid": "CR01 TB",
    "CR01 @ Wide Offer": "CR01 WO",
    "# of Bids >3mm": "# Bids>3mm",
    "# of Offers >3mm": "# Offers>3mm",
    "DoD Chg Tight Bid >3mm": "DoD TB>3mm",
    "DoD Chg Wide Offer >3mm": "DoD WO>3mm",
    "DoD Chg Tight Bid": "DoD TB",
    "DoD Chg Wide Offer": "DoD WO",
    "DoD Chg Size @ Tight Bid >3mm": "DoD Size TB>3mm",
    "DoD Chg Size @ Wide Offer >3mm": "DoD Size WO>3mm",
    "MTD Chg Tight Bid": "MTD TB",
    "YTD Chg Tight Bid": "YTD TB",
    "1yr Chg Tight Bid": "1yr TB",
    "MTD Equity": "MTD Equity",
    "YTD Equity": "YTD Equity",
    "Retracement": "Retracement",
    "Custom_Sector": "Custom Sector",
}

# Column order for Portfolio Sorted By CR01 Risk table
PORTFOLIO_CR01_RISK_COLUMNS = [
    "Security",
    "QUANTITY",
    "POSITION CR01",
    "Yrs (Cvn)",
    "Tight Bid >3mm",
    "Wide Offer >3mm",
    "Tight Bid",
    "Wide Offer",
    "Bid/Offer>3mm",
    "Bid/Offer",
    "Dealer @ Tight Bid >3mm",
    "Dealer @ Wide Offer >3mm",
    "Size @ Tight Bid >3mm",
    "Size @ Wide Offer >3mm",
    "CR01 @ Tight Bid",
    "CR01 @ Wide Offer",
    "# of Bids >3mm",
    "# of Offers >3mm",
    "DoD Chg Tight Bid >3mm",
    "DoD Chg Wide Offer >3mm",
    "DoD Chg Tight Bid",
    "DoD Chg Wide Offer",
    "DoD Chg Size @ Tight Bid >3mm",
    "DoD Chg Size @ Wide Offer >3mm",
    "MTD Chg Tight Bid",
    "YTD Chg Tight Bid",
    "1yr Chg Tight Bid",
    "MTD Equity",
    "YTD Equity",
    "Retracement",
    "Custom_Sector",
]

# Column order for MTD table (excludes DoD columns)
PORTFOLIO_MTD_COLUMNS = [
    "Security",
    "QUANTITY",
    "POSITION CR01",
    "Yrs (Cvn)",
    "Tight Bid >3mm",
    "Wide Offer >3mm",
    "Tight Bid",
    "Wide Offer",
    "Bid/Offer>3mm",
    "Bid/Offer",
    "Dealer @ Tight Bid >3mm",
    "Dealer @ Wide Offer >3mm",
    "Size @ Tight Bid >3mm",
    "Size @ Wide Offer >3mm",
    "CR01 @ Tight Bid",
    "CR01 @ Wide Offer",
    "# of Bids >3mm",
    "# of Offers >3mm",
    "MTD Chg Tight Bid",
    "YTD Chg Tight Bid",
    "1yr Chg Tight Bid",
    "MTD Equity",
    "YTD Equity",
    "Retracement",
    "Custom_Sector",
]

# Column order for YTD table (excludes DoD and MTD columns)
PORTFOLIO_YTD_COLUMNS = [
    "Security",
    "QUANTITY",
    "POSITION CR01",
    "Yrs (Cvn)",
    "Tight Bid >3mm",
    "Wide Offer >3mm",
    "Tight Bid",
    "Wide Offer",
    "Bid/Offer>3mm",
    "Bid/Offer",
    "Dealer @ Tight Bid >3mm",
    "Dealer @ Wide Offer >3mm",
    "Size @ Tight Bid >3mm",
    "Size @ Wide Offer >3mm",
    "CR01 @ Tight Bid",
    "CR01 @ Wide Offer",
    "# of Bids >3mm",
    "# of Offers >3mm",
    "YTD Chg Tight Bid",
    "1yr Chg Tight Bid",
    "MTD Equity",
    "YTD Equity",
    "Retracement",
    "Custom_Sector",
]

# Column order for 1yr table (excludes DoD, MTD, and YTD columns)
PORTFOLIO_1YR_COLUMNS = [
    "Security",
    "QUANTITY",
    "POSITION CR01",
    "Yrs (Cvn)",
    "Tight Bid >3mm",
    "Wide Offer >3mm",
    "Tight Bid",
    "Wide Offer",
    "Bid/Offer>3mm",
    "Bid/Offer",
    "Dealer @ Tight Bid >3mm",
    "Dealer @ Wide Offer >3mm",
    "Size @ Tight Bid >3mm",
    "Size @ Wide Offer >3mm",
    "CR01 @ Tight Bid",
    "CR01 @ Wide Offer",
    "# of Bids >3mm",
    "# of Offers >3mm",
    "1yr Chg Tight Bid",
    "MTD Equity",
    "YTD Equity",
    "Retracement",
    "Custom_Sector",
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_ascii(value: Optional[str]) -> str:
    """
    Convert text to ASCII by removing or replacing unsupported characters.
    
    Args:
        value: Input string that may contain non-ASCII characters.
    
    Returns:
        ASCII-safe string representation.
    """
    if value is None or pd.isna(value):
        return ""
    sanitized = str(value).encode("ascii", errors="ignore").decode("ascii")
    if sanitized:
        return sanitized
    return str(value).encode("ascii", errors="replace").decode("ascii")


def format_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format numeric columns with thousand separators (no decimals, rounded).
    Retracement column formatted as percentage with 2 decimals and "%" suffix.
    Yrs (Cvn) column formatted to 1 decimal place.
    
    Args:
        df: DataFrame to format.
    
    Returns:
        Formatted DataFrame with numeric columns as formatted strings.
    """
    df_formatted = df.copy()
    for col in df_formatted.columns:
        if pd.api.types.is_numeric_dtype(df_formatted[col]):
            if col == "Retracement":
                # Format as percentage with 2 decimals (multiply by 100: 1.0 = 100%)
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: f"{x * 100:.2f}%" if pd.notna(x) else ""
                )
            elif col == "Yrs (Cvn)":
                # Format to 1 decimal place
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else ""
                )
            else:
                # Format with thousand separators, no decimals (rounded)
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: f"{int(round(x)):,}" if pd.notna(x) else ""
                )
    return df_formatted


def get_reference_dates(runs_path: Path) -> dict:
    """
    Extract MTD and YTD reference dates from runs_timeseries.parquet.
    
    Args:
        runs_path: Path to runs_timeseries.parquet file.
    
    Returns:
        Dictionary with 'last_date', 'mtd_ref_date', 'ytd_ref_date', 'one_yr_ref_date'.
    """
    df = pd.read_parquet(runs_path)
    unique_dates = sorted(df["Date"].unique())
    
    if len(unique_dates) < 1:
        return {
            "last_date": None,
            "mtd_ref_date": None,
            "ytd_ref_date": None,
            "one_yr_ref_date": None,
        }
    
    last_date = unique_dates[-1]
    
    # MTD: First date of current month
    mtd_ref_date = pd.Timestamp(last_date.year, last_date.month, 1)
    mtd_dates = [d for d in unique_dates if d >= mtd_ref_date and d < last_date]
    mtd_ref_date = mtd_dates[0] if mtd_dates else None
    
    # YTD: First date of current year
    ytd_ref_date = pd.Timestamp(last_date.year, 1, 1)
    ytd_dates = [d for d in unique_dates if d >= ytd_ref_date and d < last_date]
    ytd_ref_date = ytd_dates[0] if ytd_dates else None
    
    # 1yr: Approximately 1 year ago (closest available date)
    one_year_ago = last_date - pd.DateOffset(years=1)
    one_yr_dates = [d for d in unique_dates if d <= one_year_ago]
    one_yr_ref_date = one_yr_dates[-1] if one_yr_dates else None
    
    return {
        "last_date": last_date,
        "mtd_ref_date": mtd_ref_date,
        "ytd_ref_date": ytd_ref_date,
        "one_yr_ref_date": one_yr_ref_date,
    }


def format_table(df: pd.DataFrame, title: str, column_display_names: dict) -> str:
    """
    Format DataFrame as a nicely formatted text table with center alignment.
    
    Args:
        df: DataFrame to format.
        title: Table title.
        column_display_names: Dictionary mapping original column names to display names.
    
    Returns:
        Formatted table string with center-aligned columns.
    """
    if df.empty:
        return f"\n{'='*100}\n{title}\n{'='*100}\nNo data available.\n"
    
    # Create copy for formatting
    df_formatted = df.copy()
    
    # Format numeric columns (thousand separators, Retracement as percentage)
    df_formatted = format_numeric_columns(df_formatted)
    
    # Replace NaN with empty string (blank)
    df_formatted = df_formatted.fillna("")
    
    # Ensure ASCII-safe strings
    for col in df_formatted.columns:
        if df_formatted[col].dtype == 'object':
            df_formatted[col] = df_formatted[col].apply(
                lambda x: ensure_ascii(str(x)) if x != "" else ""
            )
    
    # Rename columns to display names
    display_columns = {}
    for col in df_formatted.columns:
        display_columns[col] = column_display_names.get(col, col)
    df_formatted = df_formatted.rename(columns=display_columns)
    
    # Convert all columns to string for consistent formatting
    for col in df_formatted.columns:
        df_formatted[col] = df_formatted[col].astype(str)
    
    # Calculate column widths (max of header length and max content length)
    col_widths = {}
    for col in df_formatted.columns:
        header_len = len(str(col))
        content_len = df_formatted[col].str.len().max() if len(df_formatted) > 0 else 0
        col_widths[col] = max(header_len, content_len, 3)  # Minimum width of 3
    
    # Build table with center alignment
    output_lines = []
    
    # Header row (center-aligned)
    header_parts = []
    for col in df_formatted.columns:
        header_parts.append(str(col).center(col_widths[col]))
    output_lines.append("  ".join(header_parts))
    
    # Separator row
    separator_parts = []
    for col in df_formatted.columns:
        separator_parts.append("-" * col_widths[col])
    output_lines.append("  ".join(separator_parts))
    
    # Data rows (center-aligned)
    for idx, row in df_formatted.iterrows():
        row_parts = []
        for col in df_formatted.columns:
            value = str(row[col])
            row_parts.append(value.center(col_widths[col]))
        output_lines.append("  ".join(row_parts))
    
    table_str = "\n".join(output_lines)
    
    # Build formatted output
    output = f"\n{'='*100}\n{title}\n{'='*100}\n"
    output += f"Total rows: {len(df):,}\n"
    output += f"{'-'*100}\n"
    output += table_str
    output += f"\n{'='*100}\n"
    
    return output


def create_portfolio_cr01_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Portfolio Sorted By CR01 Risk table.
    
    Args:
        df: Input DataFrame from runs_today.csv.
    
    Returns:
        Filtered and sorted DataFrame with selected columns.
    """
    # Filter to QUANTITY > 0
    df_filtered = df[df["QUANTITY"] > 0].copy()
    
    # Sort by POSITION CR01 descending (largest to smallest)
    df_filtered = df_filtered.sort_values("POSITION CR01", ascending=False, na_position='last')
    
    # Select only required columns (in order)
    available_columns = [col for col in PORTFOLIO_CR01_RISK_COLUMNS if col in df_filtered.columns]
    df_filtered = df_filtered[available_columns].copy()
    
    return df_filtered


def create_portfolio_dod_bid_chg_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Portfolio Sorted By DoD Bid Chg With >3MM on Bid table.
    
    Filters to rows where Tight Bid >3mm has a value (non-blank) and DoD Chg Tight Bid >3mm is non-zero (positive or negative).
    
    Args:
        df: Input DataFrame from runs_today.csv.
    
    Returns:
        Filtered and sorted DataFrame with selected columns.
    """
    # Filter to QUANTITY > 0
    df_filtered = df[df["QUANTITY"] > 0].copy()
    
    # Filter to rows where Tight Bid >3mm has a value (non-blank)
    tb_col = "Tight Bid >3mm"
    if tb_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[tb_col].notna()].copy()
    
    # Filter to rows where DoD Chg Tight Bid >3mm is non-zero (positive or negative) and non-blank
    dod_col = "DoD Chg Tight Bid >3mm"
    if dod_col in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered[dod_col].notna() & (df_filtered[dod_col] != 0)
        ].copy()
    
    # Sort by DoD Chg Tight Bid >3mm descending (largest changes first)
    if dod_col in df_filtered.columns:
        df_filtered = df_filtered.sort_values(dod_col, ascending=False, na_position='last')
    
    # Select only required columns (in order)
    available_columns = [col for col in PORTFOLIO_CR01_RISK_COLUMNS if col in df_filtered.columns]
    df_filtered = df_filtered[available_columns].copy()
    
    return df_filtered


def create_portfolio_mtd_bid_chg_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Portfolio Sorted By MTD Bid Chg With >3MM on Bid table.
    
    Filters to rows where Tight Bid >3mm has a value (non-blank) and MTD Chg Tight Bid is non-zero (positive or negative).
    
    Args:
        df: Input DataFrame from runs_today.csv.
    
    Returns:
        Filtered and sorted DataFrame with selected columns.
    """
    # Filter to QUANTITY > 0
    df_filtered = df[df["QUANTITY"] > 0].copy()
    
    # Filter to rows where Tight Bid >3mm has a value (non-blank)
    tb_col = "Tight Bid >3mm"
    if tb_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[tb_col].notna()].copy()
    
    # Filter to rows where MTD Chg Tight Bid is non-zero (positive or negative) and non-blank
    mtd_col = "MTD Chg Tight Bid"
    if mtd_col in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered[mtd_col].notna() & (df_filtered[mtd_col] != 0)
        ].copy()
    
    # Sort by MTD Chg Tight Bid descending (largest changes first)
    if mtd_col in df_filtered.columns:
        df_filtered = df_filtered.sort_values(mtd_col, ascending=False, na_position='last')
    
    # Select only required columns (in order) - excludes DoD columns
    available_columns = [col for col in PORTFOLIO_MTD_COLUMNS if col in df_filtered.columns]
    df_filtered = df_filtered[available_columns].copy()
    
    return df_filtered


def create_portfolio_ytd_bid_chg_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Portfolio Sorted By YTD Bid Chg With >3MM on Bid table.
    
    Filters to rows where Tight Bid >3mm has a value (non-blank) and YTD Chg Tight Bid is non-zero (positive or negative).
    
    Args:
        df: Input DataFrame from runs_today.csv.
    
    Returns:
        Filtered and sorted DataFrame with selected columns.
    """
    # Filter to QUANTITY > 0
    df_filtered = df[df["QUANTITY"] > 0].copy()
    
    # Filter to rows where Tight Bid >3mm has a value (non-blank)
    tb_col = "Tight Bid >3mm"
    if tb_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[tb_col].notna()].copy()
    
    # Filter to rows where YTD Chg Tight Bid is non-zero (positive or negative) and non-blank
    ytd_col = "YTD Chg Tight Bid"
    if ytd_col in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered[ytd_col].notna() & (df_filtered[ytd_col] != 0)
        ].copy()
    
    # Sort by YTD Chg Tight Bid descending (largest changes first)
    if ytd_col in df_filtered.columns:
        df_filtered = df_filtered.sort_values(ytd_col, ascending=False, na_position='last')
    
    # Select only required columns (in order) - excludes DoD and MTD columns
    available_columns = [col for col in PORTFOLIO_YTD_COLUMNS if col in df_filtered.columns]
    df_filtered = df_filtered[available_columns].copy()
    
    return df_filtered


def create_portfolio_1yr_bid_chg_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Portfolio Sorted By 1yr Bid Chg With >3MM on Bid table.
    
    Filters to rows where Tight Bid >3mm has a value (non-blank) and 1yr Chg Tight Bid is non-zero (positive or negative).
    
    Args:
        df: Input DataFrame from runs_today.csv.
    
    Returns:
        Filtered and sorted DataFrame with selected columns.
    """
    # Filter to QUANTITY > 0
    df_filtered = df[df["QUANTITY"] > 0].copy()
    
    # Filter to rows where Tight Bid >3mm has a value (non-blank)
    tb_col = "Tight Bid >3mm"
    if tb_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[tb_col].notna()].copy()
    
    # Filter to rows where 1yr Chg Tight Bid is non-zero (positive or negative) and non-blank
    one_yr_col = "1yr Chg Tight Bid"
    if one_yr_col in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered[one_yr_col].notna() & (df_filtered[one_yr_col] != 0)
        ].copy()
    
    # Sort by 1yr Chg Tight Bid descending (largest changes first)
    if one_yr_col in df_filtered.columns:
        df_filtered = df_filtered.sort_values(one_yr_col, ascending=False, na_position='last')
    
    # Select only required columns (in order) - excludes DoD, MTD, and YTD columns
    available_columns = [col for col in PORTFOLIO_1YR_COLUMNS if col in df_filtered.columns]
    df_filtered = df_filtered[available_columns].copy()
    
    return df_filtered


def main() -> None:
    """Main execution function."""
    print("="*100)
    print("RUNS VIEWS - Portfolio Runs View Generator")
    print("="*100)
    
    # Step 1: Load runs_today.csv
    print("\n[STEP 1] Loading runs_today.csv...")
    if not RUNS_TODAY_CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {RUNS_TODAY_CSV_PATH}")
    
    df = pd.read_csv(RUNS_TODAY_CSV_PATH)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Step 2: Get reference dates from runs_timeseries.parquet
    print("\n[STEP 2] Extracting reference dates from runs_timeseries.parquet...")
    if not RUNS_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found: {RUNS_PARQUET_PATH}")
    
    ref_dates = get_reference_dates(RUNS_PARQUET_PATH)
    last_date = ref_dates["last_date"]
    mtd_ref_date = ref_dates["mtd_ref_date"]
    ytd_ref_date = ref_dates["ytd_ref_date"]
    
    print(f"Last date: {last_date}")
    print(f"MTD reference date: {mtd_ref_date}")
    print(f"YTD reference date: {ytd_ref_date}")
    
    # Step 3: Create Portfolio Sorted By CR01 Risk table
    print("\n[STEP 3] Creating Portfolio Sorted By CR01 Risk table...")
    portfolio_cr01_df = create_portfolio_cr01_risk_table(df)
    print(f"Filtered to {len(portfolio_cr01_df):,} rows with QUANTITY > 0")
    
    # Step 4: Create Portfolio Sorted By DoD Bid Chg table
    print("\n[STEP 4] Creating Portfolio Sorted By DoD Bid Chg With >3MM on Bid table...")
    portfolio_dod_bid_df = create_portfolio_dod_bid_chg_table(df)
    print(f"Filtered to {len(portfolio_dod_bid_df):,} rows with QUANTITY > 0, TB >3mm has value, and DoD TB>3mm non-zero")
    
    # Step 5: Create Portfolio Sorted By MTD Bid Chg table
    print("\n[STEP 5] Creating Portfolio Sorted By MTD Bid Chg With >3MM on Bid table...")
    portfolio_mtd_bid_df = create_portfolio_mtd_bid_chg_table(df)
    print(f"Filtered to {len(portfolio_mtd_bid_df):,} rows with QUANTITY > 0, TB >3mm has value, and MTD TB non-zero")
    
    # Step 6: Create Portfolio Sorted By YTD Bid Chg table
    print("\n[STEP 6] Creating Portfolio Sorted By YTD Bid Chg With >3MM on Bid table...")
    portfolio_ytd_bid_df = create_portfolio_ytd_bid_chg_table(df)
    print(f"Filtered to {len(portfolio_ytd_bid_df):,} rows with QUANTITY > 0, TB >3mm has value, and YTD TB non-zero")
    
    # Step 7: Create Portfolio Sorted By 1yr Bid Chg table
    print("\n[STEP 7] Creating Portfolio Sorted By 1yr Bid Chg With >3MM on Bid table...")
    portfolio_1yr_bid_df = create_portfolio_1yr_bid_chg_table(df)
    print(f"Filtered to {len(portfolio_1yr_bid_df):,} rows with QUANTITY > 0, TB >3mm has value, and 1yr TB non-zero")
    
    # Step 8: Format and write output
    print("\n[STEP 8] Formatting and writing output...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Write header
        f.write("PORTFOLIO RUNS VIEW\n")
        f.write("="*100 + "\n")
        f.write(f"Last Refreshed: {timestamp}\n")
        f.write(f"\nReference Dates:\n")
        f.write(f"  Last Date: {last_date}\n")
        if mtd_ref_date:
            f.write(f"  MTD Reference Date: {mtd_ref_date.strftime('%Y-%m-%d')}\n")
        else:
            f.write(f"  MTD Reference Date: N/A\n")
        if ytd_ref_date:
            f.write(f"  YTD Reference Date: {ytd_ref_date.strftime('%Y-%m-%d')}\n")
        else:
            f.write(f"  YTD Reference Date: N/A\n")
        f.write("="*100 + "\n")
        
        # Write Portfolio Sorted By CR01 Risk table
        table_str = format_table(
            portfolio_cr01_df,
            "Portfolio Sorted By CR01 Risk",
            COLUMN_DISPLAY_NAMES
        )
        f.write(table_str)
        
        # Write Portfolio Sorted By DoD Bid Chg table
        table_str = format_table(
            portfolio_dod_bid_df,
            "Portfolio Sorted By DoD Bid Chg With >3MM on Bid",
            COLUMN_DISPLAY_NAMES
        )
        f.write(table_str)
        
        # Write Portfolio Sorted By MTD Bid Chg table
        table_str = format_table(
            portfolio_mtd_bid_df,
            "Portfolio Sorted By MTD Bid Chg With >3MM on Bid",
            COLUMN_DISPLAY_NAMES
        )
        f.write(table_str)
        
        # Write Portfolio Sorted By YTD Bid Chg table
        table_str = format_table(
            portfolio_ytd_bid_df,
            "Portfolio Sorted By YTD Bid Chg With >3MM on Bid",
            COLUMN_DISPLAY_NAMES
        )
        f.write(table_str)
        
        # Write Portfolio Sorted By 1yr Bid Chg table
        table_str = format_table(
            portfolio_1yr_bid_df,
            "Portfolio Sorted By 1yr Bid Chg With >3MM on Bid",
            COLUMN_DISPLAY_NAMES
        )
        f.write(table_str)
        
        f.write("\n" + "="*100 + "\n")
        f.write("END OF REPORT\n")
    
    print(f"\nOutput written to: {OUTPUT_FILE}")
    print(f"Total rows in Portfolio CR01 Risk table: {len(portfolio_cr01_df):,}")
    print(f"Total rows in Portfolio DoD Bid Chg table: {len(portfolio_dod_bid_df):,}")
    print(f"Total rows in Portfolio MTD Bid Chg table: {len(portfolio_mtd_bid_df):,}")
    print(f"Total rows in Portfolio YTD Bid Chg table: {len(portfolio_ytd_bid_df):,}")
    print(f"Total rows in Portfolio 1yr Bid Chg table: {len(portfolio_1yr_bid_df):,}")
    print("\nDone!")


if __name__ == "__main__":
    main()

