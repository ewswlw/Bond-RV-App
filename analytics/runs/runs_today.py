"""
Runs today analytics script.

This module reads `bond_data/parquet/runs_timeseries.parquet`, aggregates metrics by Date+CUSIP+Benchmark,
filters to the most recent date, computes Day-over-Day (DoD), Month-to-Date (MTD), Year-to-Date (YTD), and
1-year (1yr) changes by comparing the last date with reference dates, and exports the results to CSV for
daily monitoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# Get script directory and build paths relative to it
SCRIPT_DIR = Path(__file__).parent.resolve()
RUNS_PARQUET_PATH = SCRIPT_DIR.parent.parent / "bond_data" / "parquet" / "runs_timeseries.parquet"
OUTPUT_DIR = SCRIPT_DIR.parent / "processed_data"

# Size threshold for >3mm filters
SIZE_THRESHOLD = 3000000

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


def compute_group_metrics(group: pd.DataFrame) -> dict:
    """
    Compute aggregated metrics for a Date+CUSIP+Benchmark group.

    Args:
        group: DataFrame subset containing all rows for one Date+CUSIP+Benchmark.

    Returns:
        Dictionary with computed metrics for the group.
    """
    # Basic columns
    result = {
        "Date": group["Date"].iloc[0],
        "CUSIP": group["CUSIP"].iloc[0],
        "Time": group["Time"].max(),  # Latest Time in group
        "Bid Workout Risk": group["Bid Workout Risk"].mean(),  # Average
        "Security": group["Security"].iloc[0],  # Any row (should all be same)
    }

    # Filter to rows with Bid Size > 3mm
    bid_gt_3mm = group[group["Bid Size"] > SIZE_THRESHOLD].copy()
    
    # Filter to rows with Ask Size > 3mm
    ask_gt_3mm = group[group["Ask Size"] > SIZE_THRESHOLD].copy()

    # Tight Bid >3mm (smallest Bid Spread with Bid Size > 3000000)
    if len(bid_gt_3mm) > 0 and bid_gt_3mm["Bid Spread"].notna().any():
        tight_bid_3mm_idx = bid_gt_3mm["Bid Spread"].idxmin()
        if pd.notna(tight_bid_3mm_idx):
            result["Tight Bid >3mm"] = bid_gt_3mm.loc[tight_bid_3mm_idx, "Bid Spread"]
            result["Dealer @ Tight Bid >3mm"] = bid_gt_3mm.loc[tight_bid_3mm_idx, "Dealer"]
            result["Size @ Tight Bid >3mm"] = bid_gt_3mm.loc[tight_bid_3mm_idx, "Bid Size"]
        else:
            result["Tight Bid >3mm"] = pd.NA
            result["Dealer @ Tight Bid >3mm"] = pd.NA
            result["Size @ Tight Bid >3mm"] = pd.NA
    else:
        result["Tight Bid >3mm"] = pd.NA
        result["Dealer @ Tight Bid >3mm"] = pd.NA
        result["Size @ Tight Bid >3mm"] = pd.NA

    # Wide Offer >3mm (largest Ask Spread with Ask Size > 3000000)
    if len(ask_gt_3mm) > 0 and ask_gt_3mm["Ask Spread"].notna().any():
        wide_offer_3mm_idx = ask_gt_3mm["Ask Spread"].idxmax()
        if pd.notna(wide_offer_3mm_idx):
            result["Wide Offer >3mm"] = ask_gt_3mm.loc[wide_offer_3mm_idx, "Ask Spread"]
            result["Dealer @ Wide Offer >3mm"] = ask_gt_3mm.loc[wide_offer_3mm_idx, "Dealer"]
            result["Size @ Wide Offer >3mm"] = ask_gt_3mm.loc[wide_offer_3mm_idx, "Ask Size"]
        else:
            result["Wide Offer >3mm"] = pd.NA
            result["Dealer @ Wide Offer >3mm"] = pd.NA
            result["Size @ Wide Offer >3mm"] = pd.NA
    else:
        result["Wide Offer >3mm"] = pd.NA
        result["Dealer @ Wide Offer >3mm"] = pd.NA
        result["Size @ Wide Offer >3mm"] = pd.NA

    # Tight Bid (smallest Bid Spread overall)
    if len(group) > 0 and group["Bid Spread"].notna().any():
        result["Tight Bid"] = group["Bid Spread"].min()
    else:
        result["Tight Bid"] = pd.NA

    # Wide Offer (largest Ask Spread overall)
    if len(group) > 0 and group["Ask Spread"].notna().any():
        result["Wide Offer"] = group["Ask Spread"].max()
    else:
        result["Wide Offer"] = pd.NA

    # CR01 calculations (using average Bid Workout Risk)
    avg_workout_risk = result["Bid Workout Risk"]
    if pd.notna(result["Size @ Tight Bid >3mm"]) and pd.notna(avg_workout_risk):
        result["CR01 @ Tight Bid"] = avg_workout_risk * (result["Size @ Tight Bid >3mm"] / 10000)
    else:
        result["CR01 @ Tight Bid"] = pd.NA

    if pd.notna(result["Size @ Wide Offer >3mm"]) and pd.notna(avg_workout_risk):
        result["CR01 @ Wide Offer"] = avg_workout_risk * (result["Size @ Wide Offer >3mm"] / 10000)
    else:
        result["CR01 @ Wide Offer"] = pd.NA

    # Cumulative sizes (sum ignoring NaN)
    result["Cumm. Bid Size"] = group["Bid Size"].sum(skipna=True) if group["Bid Size"].notna().any() else 0
    result["Cumm. Offer Size"] = group["Ask Size"].sum(skipna=True) if group["Ask Size"].notna().any() else 0

    # Count unique dealers with Bid Size > 3mm
    if len(bid_gt_3mm) > 0:
        result["# of Bids >3mm"] = bid_gt_3mm["Dealer"].nunique()
    else:
        result["# of Bids >3mm"] = 0

    # Count unique dealers with Ask Size > 3mm
    if len(ask_gt_3mm) > 0:
        result["# of Offers >3mm"] = ask_gt_3mm["Dealer"].nunique()
    else:
        result["# of Offers >3mm"] = 0

    # RBC dealer columns
    rbc_rows = group[group["Dealer"] == "RBC"].copy()
    if len(rbc_rows) > 0:
        # If multiple RBC rows, take the first one (or could use latest time)
        rbc_row = rbc_rows.iloc[0]
        result["Bid RBC"] = rbc_row["Bid Spread"] if pd.notna(rbc_row["Bid Spread"]) else pd.NA
        result["Ask RBC"] = rbc_row["Ask Spread"] if pd.notna(rbc_row["Ask Spread"]) else pd.NA
        result["Bid Size RBC"] = rbc_row["Bid Size"] if pd.notna(rbc_row["Bid Size"]) else pd.NA
        result["Offer Size RBC"] = rbc_row["Ask Size"] if pd.notna(rbc_row["Ask Size"]) else pd.NA
    else:
        result["Bid RBC"] = pd.NA
        result["Ask RBC"] = pd.NA
        result["Bid Size RBC"] = pd.NA
        result["Offer Size RBC"] = pd.NA

    return result


def run_analysis(
    runs_path: Path = RUNS_PARQUET_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Execute the runs today analytics workflow.

    Reads parquet, aggregates metrics for required dates only, filters to today,
    computes DoD/MTD/YTD/1yr changes, and exports to CSV.

    Args:
        runs_path: Path to runs_timeseries.parquet file.
        output_dir: Directory for CSV export.

    Returns:
        DataFrame containing today's data with DoD, MTD, YTD, and 1yr changes.
    """
    # Step 1: Load parquet and find required dates
    print("Loading runs_timeseries.parquet...")
    df = pd.read_parquet(runs_path)
    
    print(f"Loaded {len(df):,} rows")
    
    # Ensure required columns exist
    required_cols = [
        "Date", "CUSIP", "Benchmark", "Time", "Bid Workout Risk", "Security",
        "Bid Spread", "Ask Spread", "Bid Size", "Ask Size", "Dealer"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Get unique dates sorted
    unique_dates = sorted(df["Date"].unique())
    
    if len(unique_dates) < 2:
        raise ValueError(
            f"Need at least 2 dates to calculate DoD changes. Found {len(unique_dates)} date(s)."
        )
    
    last_date = unique_dates[-1]
    second_last_date = unique_dates[-2]
    
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Last date: {last_date}")
    print(f"Second-to-last date: {second_last_date}")
    
    # Find reference dates for MTD, YTD, and 1yr calculations
    print("\nFinding reference dates for MTD, YTD, and 1yr calculations...")
    
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
    
    print(f"  MTD reference date: {mtd_ref_date}")
    print(f"  YTD reference date: {ytd_ref_date}")
    print(f"  1yr reference date: {one_yr_ref_date}")
    
    # Collect all dates we need to process
    required_dates = {last_date, second_last_date}
    if mtd_ref_date:
        required_dates.add(mtd_ref_date)
    if ytd_ref_date:
        required_dates.add(ytd_ref_date)
    if one_yr_ref_date:
        required_dates.add(one_yr_ref_date)
    
    print(f"\nProcessing {len(required_dates)} required dates (optimized from {len(unique_dates)} total dates)")
    
    # Step 2: Filter to only required dates before aggregation
    df_filtered = df[df["Date"].isin(required_dates)].copy()
    print(f"Filtered to {len(df_filtered):,} rows for required dates")
    
    # Step 3: Group by Date, CUSIP, Benchmark and compute aggregated metrics
    print("\nGrouping by Date, CUSIP, Benchmark and computing metrics...")
    results = []
    
    grouped = df_filtered.groupby(["Date", "CUSIP", "Benchmark"], as_index=False)
    num_groups = grouped.ngroups
    
    for idx, (name, group) in enumerate(grouped):
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1:,} / {num_groups:,} groups...")
        
        metrics = compute_group_metrics(group)
        results.append(metrics)
    
    print(f"Computed metrics for {len(results):,} groups")
    
    # Step 4: Convert to DataFrame and format
    print("Building aggregated DataFrame...")
    aggregated_df = pd.DataFrame(results)
    
    # Ensure column order for aggregated data
    aggregated_column_order = [
        "Date",
        "CUSIP",
        "Time",
        "Bid Workout Risk",
        "Security",
        "Tight Bid >3mm",
        "Wide Offer >3mm",
        "Tight Bid",
        "Wide Offer",
        "Dealer @ Tight Bid >3mm",
        "Dealer @ Wide Offer >3mm",
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
    
    # Ensure all columns exist (fill missing with NaN)
    for col in aggregated_column_order:
        if col not in aggregated_df.columns:
            aggregated_df[col] = pd.NA
    
    aggregated_df = aggregated_df[aggregated_column_order]
    
    # Convert numeric columns to float64
    float_columns = [
        "Tight Bid >3mm",
        "Wide Offer >3mm",
        "Tight Bid",
        "Wide Offer",
        "Size @ Tight Bid >3mm",
        "Size @ Wide Offer >3mm",
        "CR01 @ Tight Bid",
        "CR01 @ Wide Offer",
        "Bid RBC",
        "Ask RBC",
        "Bid Size RBC",
        "Offer Size RBC",
    ]
    for col in float_columns:
        if col in aggregated_df.columns:
            aggregated_df[col] = pd.to_numeric(aggregated_df[col], errors="coerce")
    
    # Ensure ASCII-safe Security and Dealer columns
    if "Security" in aggregated_df.columns:
        aggregated_df["Security"] = aggregated_df["Security"].apply(
            lambda x: ensure_ascii(x) if pd.notna(x) else x
        )
    if "Dealer @ Tight Bid >3mm" in aggregated_df.columns:
        aggregated_df["Dealer @ Tight Bid >3mm"] = aggregated_df["Dealer @ Tight Bid >3mm"].apply(
            lambda x: ensure_ascii(x) if pd.notna(x) else x
        )
    if "Dealer @ Wide Offer >3mm" in aggregated_df.columns:
        aggregated_df["Dealer @ Wide Offer >3mm"] = aggregated_df["Dealer @ Wide Offer >3mm"].apply(
            lambda x: ensure_ascii(x) if pd.notna(x) else x
        )
    
    # Step 5: Filter to last date and prepare for today analysis
    print("\nFiltering to last date for today analysis...")
    last_date_df = aggregated_df[aggregated_df["Date"] == last_date].copy()
    
    # Handle duplicate CUSIPs (keep first occurrence)
    if last_date_df["CUSIP"].duplicated().any():
        print("Warning: Found duplicate CUSIPs on last date, keeping first occurrence...")
        last_date_df = last_date_df.drop_duplicates(subset=["CUSIP"], keep="first")
    
    print(f"Rows on last date: {len(last_date_df):,}")
    
    # Step 6: Get CUSIPs that exist on both last date and second-to-last date
    second_last_date_df = aggregated_df[aggregated_df["Date"] == second_last_date].copy()
    if second_last_date_df["CUSIP"].duplicated().any():
        second_last_date_df = second_last_date_df.drop_duplicates(subset=["CUSIP"], keep="first")
    
    last_date_cusips = set(last_date_df["CUSIP"].unique())
    second_last_date_cusips = set(second_last_date_df["CUSIP"].unique())
    common_cusips = last_date_cusips & second_last_date_cusips
    
    print(f"CUSIPs on last date: {len(last_date_cusips):,}")
    print(f"CUSIPs on second-to-last date: {len(second_last_date_cusips):,}")
    print(f"CUSIPs on both dates: {len(common_cusips):,}")
    
    # Filter to only CUSIPs that exist on both dates
    result_df = last_date_df[last_date_df["CUSIP"].isin(common_cusips)].copy()
    
    # Step 7: Create lookup dictionaries for reference dates
    second_last_lookup = {}
    for idx, row in second_last_date_df.iterrows():
        second_last_lookup[row["CUSIP"]] = row
    
    mtd_lookup = {}
    if mtd_ref_date:
        mtd_df = aggregated_df[aggregated_df["Date"] == mtd_ref_date].copy()
        if mtd_df["CUSIP"].duplicated().any():
            mtd_df = mtd_df.drop_duplicates(subset=["CUSIP"], keep="first")
        for idx, row in mtd_df.iterrows():
            mtd_lookup[row["CUSIP"]] = row
    
    ytd_lookup = {}
    if ytd_ref_date:
        ytd_df = aggregated_df[aggregated_df["Date"] == ytd_ref_date].copy()
        if ytd_df["CUSIP"].duplicated().any():
            ytd_df = ytd_df.drop_duplicates(subset=["CUSIP"], keep="first")
        for idx, row in ytd_df.iterrows():
            ytd_lookup[row["CUSIP"]] = row
    
    one_yr_lookup = {}
    if one_yr_ref_date:
        one_yr_df = aggregated_df[aggregated_df["Date"] == one_yr_ref_date].copy()
        if one_yr_df["CUSIP"].duplicated().any():
            one_yr_df = one_yr_df.drop_duplicates(subset=["CUSIP"], keep="first")
        for idx, row in one_yr_df.iterrows():
            one_yr_lookup[row["CUSIP"]] = row
    
    # Step 8: Calculate DoD changes
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
    
    # Step 9: Calculate MTD, YTD, and 1yr changes
    print("\nCalculating MTD, YTD, and 1yr changes...")
    for col in DOD_COLUMNS:
        if col not in result_df.columns:
            continue
        
        # MTD changes
        mtd_col_name = f"MTD Chg {col}"
        mtd_values = []
        for idx, row in result_df.iterrows():
            cusip = row["CUSIP"]
            last_value = row[col]
            if cusip in mtd_lookup:
                mtd_value = mtd_lookup[cusip][col]
                if pd.notna(last_value) and pd.notna(mtd_value):
                    mtd_values.append(float(last_value) - float(mtd_value))
                else:
                    mtd_values.append(pd.NA)
            else:
                mtd_values.append(pd.NA)
        result_df[mtd_col_name] = mtd_values
        
        # YTD changes
        ytd_col_name = f"YTD Chg {col}"
        ytd_values = []
        for idx, row in result_df.iterrows():
            cusip = row["CUSIP"]
            last_value = row[col]
            if cusip in ytd_lookup:
                ytd_value = ytd_lookup[cusip][col]
                if pd.notna(last_value) and pd.notna(ytd_value):
                    ytd_values.append(float(last_value) - float(ytd_value))
                else:
                    ytd_values.append(pd.NA)
            else:
                ytd_values.append(pd.NA)
        result_df[ytd_col_name] = ytd_values
        
        # 1yr changes
        one_yr_col_name = f"1yr Chg {col}"
        one_yr_values = []
        for idx, row in result_df.iterrows():
            cusip = row["CUSIP"]
            last_value = row[col]
            if cusip in one_yr_lookup:
                one_yr_value = one_yr_lookup[cusip][col]
                if pd.notna(last_value) and pd.notna(one_yr_value):
                    one_yr_values.append(float(last_value) - float(one_yr_value))
                else:
                    one_yr_values.append(pd.NA)
            else:
                one_yr_values.append(pd.NA)
        result_df[one_yr_col_name] = one_yr_values
    
    # Step 10: Calculate Bid/Offer spread columns
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
    
    # Step 11: Add T-1 dealer columns (from second last date)
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
    
    # Step 12: Ensure final column order
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
    
    # Add MTD columns after DoD columns
    for col in DOD_COLUMNS:
        mtd_col_name = f"MTD Chg {col}"
        if mtd_col_name in result_df.columns:
            column_order.append(mtd_col_name)
    
    # Add YTD columns after MTD columns
    for col in DOD_COLUMNS:
        ytd_col_name = f"YTD Chg {col}"
        if ytd_col_name in result_df.columns:
            column_order.append(ytd_col_name)
    
    # Add 1yr columns after YTD columns
    for col in DOD_COLUMNS:
        one_yr_col_name = f"1yr Chg {col}"
        if one_yr_col_name in result_df.columns:
            column_order.append(one_yr_col_name)
    
    # Ensure all columns exist (fill missing with NaN)
    for col in column_order:
        if col not in result_df.columns:
            result_df[col] = pd.NA
    
    result_df = result_df[column_order]
    
    # Step 13: Convert all change columns to float64
    for col in ["Bid/Offer>3mm", "Bid/Offer"]:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")
    
    for col in DOD_COLUMNS:
        for prefix in ["DoD Chg ", "MTD Chg ", "YTD Chg ", "1yr Chg "]:
            col_name = f"{prefix}{col}"
            if col_name in result_df.columns:
                result_df[col_name] = pd.to_numeric(result_df[col_name], errors="coerce")
    
    # Step 14: Ensure ASCII-safe Dealer columns
    if "Dealer @ Tight T-1" in result_df.columns:
        result_df["Dealer @ Tight T-1"] = result_df["Dealer @ Tight T-1"].apply(
            lambda x: ensure_ascii(x) if pd.notna(x) else x
        )
    if "Dealer @ Wide T-1" in result_df.columns:
        result_df["Dealer @ Wide T-1"] = result_df["Dealer @ Wide T-1"].apply(
            lambda x: ensure_ascii(x) if pd.notna(x) else x
        )
    
    # Step 15: Sort by CUSIP
    result_df = result_df.sort_values("CUSIP", ascending=True).reset_index(drop=True)
    
    # Step 16: Write to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "runs_today.csv"
    result_df.to_csv(output_path, index=False)
    
    # Step 17: Display results
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
