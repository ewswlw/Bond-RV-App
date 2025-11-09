"""
Runs today analytics script.

This module reads `analytics/processed_data/runs_adjusted_ts.csv`, filters to the most
recent date, computes Day-over-Day (DoD) changes by comparing the last date with the
second-to-last date, and exports the results to CSV for daily monitoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# Get script directory and build paths relative to it
SCRIPT_DIR = Path(__file__).parent.resolve()
RUNS_ADJUSTED_TS_CSV = SCRIPT_DIR / "processed_data" / "runs_adjusted_ts.csv"
OUTPUT_DIR = SCRIPT_DIR / "processed_data"

# Columns to calculate DoD changes for
DOD_COLUMNS = [
    "Tight Bid >3mm",
    "Wide Offer >3mm",
    "Tight Bid",
    "Wide Offer",
    "Size @ Tight Bid >3mm",
    "Size @ Wide Offer >3mm",
    "CR01 @ Tight Bid",
    "CR01 @ Wide Offer",
    "Cumm. Bid Size",
    "Cumm. Offer Size",
    "# of Bids >3mm",
    "# of Offers >3mm",
    "Bid RBC",
    "Ask RBC",
    "Bid Size RBC",
    "Offer Size RBC",
]


def ensure_ascii(value: Optional[str]) -> str:
    """
    Convert text to ASCII by removing or replacing unsupported characters.

    Args:
        value: Input string that may contain non-ASCII characters.

    Returns:
        ASCII-safe string representation.
    """
    if value is None:
        return ""
    sanitized = value.encode("ascii", errors="ignore").decode("ascii")
    if sanitized:
        return sanitized
    return value.encode("ascii", errors="replace").decode("ascii")


def run_analysis(
    csv_path: Path = RUNS_ADJUSTED_TS_CSV,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Execute the runs today analytics workflow.

    Args:
        csv_path: Path to runs_adjusted_ts.csv file.
        output_dir: Directory for CSV export.

    Returns:
        DataFrame containing today's data with DoD changes.
    """
    print("Loading runs_adjusted_ts.csv...")
    df = pd.read_csv(csv_path)
    
    # Convert Date column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
    
    print(f"Loaded {len(df):,} rows")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Get unique dates sorted
    unique_dates = sorted(df["Date"].unique())
    
    if len(unique_dates) < 2:
        raise ValueError(
            f"Need at least 2 dates to calculate DoD changes. Found {len(unique_dates)} date(s)."
        )
    
    last_date = unique_dates[-1]
    second_last_date = unique_dates[-2]
    
    print(f"\nLast date: {last_date}")
    print(f"Second-to-last date: {second_last_date}")
    
    # Filter to last date and second-to-last date
    last_date_df = df[df["Date"] == last_date].copy()
    second_last_date_df = df[df["Date"] == second_last_date].copy()
    
    print(f"\nRows on last date: {len(last_date_df):,}")
    print(f"Rows on second-to-last date: {len(second_last_date_df):,}")
    
    # Handle duplicate CUSIPs (shouldn't happen after grouping in runs-adjusted_ts.py, but handle just in case)
    # Group by CUSIP and keep first row if duplicates exist
    last_date_grouped = last_date_df.groupby("CUSIP", as_index=False)
    if last_date_df["CUSIP"].duplicated().any():
        print("\nWarning: Found duplicate CUSIPs on last date, keeping first occurrence...")
        last_date_df = last_date_df.drop_duplicates(subset=["CUSIP"], keep="first")
        print(f"Rows after deduplication: {len(last_date_df):,}")
    
    # Get CUSIPs that exist on both dates
    last_date_cusips = set(last_date_df["CUSIP"].unique())
    second_last_date_cusips = set(second_last_date_df["CUSIP"].unique())
    common_cusips = last_date_cusips & second_last_date_cusips
    
    print(f"\nCUSIPs on last date: {len(last_date_cusips):,}")
    print(f"CUSIPs on second-to-last date: {len(second_last_date_cusips):,}")
    print(f"CUSIPs on both dates: {len(common_cusips):,}")
    
    # Filter to only CUSIPs that exist on both dates
    result_df = last_date_df[last_date_df["CUSIP"].isin(common_cusips)].copy()
    
    # Create lookup dictionary for second-to-last date values
    # Handle duplicates by keeping first occurrence
    if second_last_date_df["CUSIP"].duplicated().any():
        second_last_date_df = second_last_date_df.drop_duplicates(subset=["CUSIP"], keep="first")
    
    second_last_lookup = {}
    for idx, row in second_last_date_df.iterrows():
        second_last_lookup[row["CUSIP"]] = row
    
    # Calculate DoD changes
    print("\nCalculating Day-over-Day changes...")
    for col in DOD_COLUMNS:
        if col not in result_df.columns:
            print(f"  Warning: Column '{col}' not found, skipping DoD calculation")
            continue
        
        dod_col_name = f"DoD Chg {col}"
        dod_values = []
        
        for idx, row in result_df.iterrows():
            cusip = row["CUSIP"]
            last_value = row[col]
            
            if cusip in second_last_lookup:
                second_last_value = second_last_lookup[cusip][col]
                
                # Calculate DoD: Last Date - Second Last Date
                if pd.notna(last_value) and pd.notna(second_last_value):
                    dod_values.append(float(last_value) - float(second_last_value))
                else:
                    dod_values.append(pd.NA)
            else:
                dod_values.append(pd.NA)
        
        result_df[dod_col_name] = dod_values
    
    # Calculate Bid/Offer spread columns
    print("\nCalculating Bid/Offer spreads...")
    # Bid/Offer>3mm = Tight Bid >3mm - Wide Offer >3mm (only if both have values)
    result_df["Bid/Offer>3mm"] = result_df.apply(
        lambda row: (
            row["Tight Bid >3mm"] - row["Wide Offer >3mm"]
            if pd.notna(row["Tight Bid >3mm"]) and pd.notna(row["Wide Offer >3mm"])
            else pd.NA
        ),
        axis=1
    )
    
    # Bid/Offer = Tight Bid - Wide Offer (only if both have values)
    result_df["Bid/Offer"] = result_df.apply(
        lambda row: (
            row["Tight Bid"] - row["Wide Offer"]
            if pd.notna(row["Tight Bid"]) and pd.notna(row["Wide Offer"])
            else pd.NA
        ),
        axis=1
    )
    
    # Add T-1 dealer columns (from second last date)
    print("\nAdding T-1 dealer columns...")
    dealer_tight_t1_values = []
    dealer_wide_t1_values = []
    
    for idx, row in result_df.iterrows():
        cusip = row["CUSIP"]
        if cusip in second_last_lookup:
            second_last_row = second_last_lookup[cusip]
            dealer_tight_t1 = second_last_row.get("Dealer @ Tight Bid >3mm", pd.NA)
            dealer_wide_t1 = second_last_row.get("Dealer @ Wide Offer >3mm", pd.NA)
            dealer_tight_t1_values.append(dealer_tight_t1 if pd.notna(dealer_tight_t1) else pd.NA)
            dealer_wide_t1_values.append(dealer_wide_t1 if pd.notna(dealer_wide_t1) else pd.NA)
        else:
            dealer_tight_t1_values.append(pd.NA)
            dealer_wide_t1_values.append(pd.NA)
    
    result_df["Dealer @ Tight T-1"] = dealer_tight_t1_values
    result_df["Dealer @ Wide T-1"] = dealer_wide_t1_values
    
    # Ensure column order
    column_order = [
        "CUSIP",
        "Security",
        "Bid Workout Risk",
        "Tight Bid >3mm",
        "Wide Offer >3mm",
        "Bid/Offer>3mm",
        "Tight Bid",
        "Wide Offer",
        "Bid/Offer",
        "Dealer @ Tight Bid >3mm",
        "Dealer @ Wide Offer >3mm",
        "Dealer @ Tight T-1",
        "Dealer @ Wide T-1",
        "Size @ Tight Bid >3mm",
        "Size @ Wide Offer >3mm",
        "CR01 @ Tight Bid",
        "CR01 @ Wide Offer",
        "Cumm. Bid Size",
        "Cumm. Offer Size",
        "# of Bids >3mm",
        "# of Offers >3mm",
        "Bid RBC",
        "Ask RBC",
        "Bid Size RBC",
        "Offer Size RBC",
    ]
    
    # Add DoD columns at the end
    for col in DOD_COLUMNS:
        dod_col_name = f"DoD Chg {col}"
        if dod_col_name in result_df.columns:
            column_order.append(dod_col_name)
    
    # Ensure all columns exist (fill missing with NaN)
    for col in column_order:
        if col not in result_df.columns:
            result_df[col] = pd.NA
    
    result_df = result_df[column_order]
    
    # Convert Bid/Offer columns to float64
    for col in ["Bid/Offer>3mm", "Bid/Offer"]:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")
    
    # Convert DoD columns to float64
    for col in DOD_COLUMNS:
        dod_col_name = f"DoD Chg {col}"
        if dod_col_name in result_df.columns:
            result_df[dod_col_name] = pd.to_numeric(result_df[dod_col_name], errors="coerce")
    
    # Ensure ASCII-safe Security and Dealer columns
    if "Security" in result_df.columns:
        result_df["Security"] = result_df["Security"].apply(
            lambda x: ensure_ascii(x) if pd.notna(x) else x
        )
    if "Dealer @ Tight Bid >3mm" in result_df.columns:
        result_df["Dealer @ Tight Bid >3mm"] = result_df["Dealer @ Tight Bid >3mm"].apply(
            lambda x: ensure_ascii(x) if pd.notna(x) else x
        )
    if "Dealer @ Wide Offer >3mm" in result_df.columns:
        result_df["Dealer @ Wide Offer >3mm"] = result_df["Dealer @ Wide Offer >3mm"].apply(
            lambda x: ensure_ascii(x) if pd.notna(x) else x
        )
    if "Dealer @ Tight T-1" in result_df.columns:
        result_df["Dealer @ Tight T-1"] = result_df["Dealer @ Tight T-1"].apply(
            lambda x: ensure_ascii(x) if pd.notna(x) else x
        )
    if "Dealer @ Wide T-1" in result_df.columns:
        result_df["Dealer @ Wide T-1"] = result_df["Dealer @ Wide T-1"].apply(
            lambda x: ensure_ascii(x) if pd.notna(x) else x
        )
    
    # Sort by CUSIP
    result_df = result_df.sort_values("CUSIP", ascending=True).reset_index(drop=True)
    
    # Write to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "runs_today.csv"
    result_df.to_csv(output_path, index=False)
    
    # Display results
    print("\n" + "=" * 80)
    print("Runs Today Analytics")
    print("=" * 80)
    print(f"\nDate: {last_date}")
    print(f"Total rows: {len(result_df):,}")
    print(f"Unique CUSIPs: {result_df['CUSIP'].nunique():,}")
    
    print("\n" + "-" * 80)
    print("DataFrame Info:")
    print("-" * 80)
    result_df.info()
    
    print("\n" + "-" * 80)
    print("First 20 rows:")
    print("-" * 80)
    print(result_df.head(20).to_string(index=False))
    
    print("\n" + "-" * 80)
    print("Last 10 rows:")
    print("-" * 80)
    print(result_df.tail(10).to_string(index=False))
    
    print(f"\nCSV written to: {output_path}")
    
    return result_df


def main() -> None:
    """Entry point for running the runs today analytics script."""
    run_analysis()


if __name__ == "__main__":
    main()

