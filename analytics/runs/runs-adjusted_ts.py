"""
Runs adjusted timeseries analytics script.

This module reads `bond_data/parquet/runs_timeseries.parquet`, groups by Date, CUSIP,
and Benchmark, computes aggregated metrics (tightest/widest spreads, sizes, CR01
calculations), and exports the results to CSV for analysis.
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
    Execute the runs adjusted timeseries analytics workflow.

    Args:
        runs_path: Path to runs_timeseries.parquet file.
        output_dir: Directory for CSV export.

    Returns:
        DataFrame containing aggregated metrics sorted by Date.
    """
    print("Loading runs_timeseries.parquet...")
    df = pd.read_parquet(runs_path)
    
    print(f"Loaded {len(df):,} rows")
    print(f"Unique Date+CUSIP+Benchmark combinations: {df.groupby(['Date', 'CUSIP', 'Benchmark']).ngroups:,}")
    
    # Ensure required columns exist
    required_cols = [
        "Date", "CUSIP", "Benchmark", "Time", "Bid Workout Risk", "Security",
        "Bid Spread", "Ask Spread", "Bid Size", "Ask Size", "Dealer"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Group by Date, CUSIP, Benchmark
    print("Grouping by Date, CUSIP, Benchmark and computing metrics...")
    results = []
    
    grouped = df.groupby(["Date", "CUSIP", "Benchmark"], as_index=False)
    num_groups = grouped.ngroups
    
    for idx, (name, group) in enumerate(grouped):
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1:,} / {num_groups:,} groups...")
        
        metrics = compute_group_metrics(group)
        results.append(metrics)
    
    print(f"Computed metrics for {len(results):,} groups")
    
    # Convert to DataFrame
    print("Building result DataFrame...")
    result_df = pd.DataFrame(results)
    
    # Ensure column order
    column_order = [
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
    for col in column_order:
        if col not in result_df.columns:
            result_df[col] = pd.NA
    
    result_df = result_df[column_order]
    
    # Sort by Date (earliest to latest)
    result_df = result_df.sort_values("Date", ascending=True).reset_index(drop=True)
    
    # Convert numeric columns to float64 (handles pd.NA by converting to NaN)
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
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")
    
    # Ensure ASCII-safe Security and Dealer columns (only apply to non-null values)
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
    
    # Write to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "runs_adjusted_ts.csv"
    result_df.to_csv(output_path, index=False)
    
    # Display results
    print("\n" + "=" * 80)
    print("Runs Adjusted Timeseries Analytics")
    print("=" * 80)
    print(f"\nTotal rows: {len(result_df):,}")
    print(f"Date range: {result_df['Date'].min()} to {result_df['Date'].max()}")
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
    """Entry point for running the runs adjusted timeseries analytics script."""
    run_analysis()


if __name__ == "__main__":
    main()

