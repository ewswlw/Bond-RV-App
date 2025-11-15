"""
Runs today analytics script.

This module reads `bond_data/parquet/runs_timeseries.parquet`, aggregates metrics by Date+CUSIP+Benchmark,
filters to the most recent date, computes Day-over-Day (DoD), Month-to-Date (MTD), Year-to-Date (YTD), and
1-year (1yr) changes by comparing the last date with reference dates, and exports the results to CSV for
daily monitoring.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Import logging setup from bond_pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from bond_pipeline.utils import setup_logging

# Get script directory and build paths relative to it
SCRIPT_DIR = Path(__file__).parent.resolve()
RUNS_PARQUET_PATH = SCRIPT_DIR.parent.parent / "bond_data" / "parquet" / "runs_timeseries.parquet"
HISTORICAL_PARQUET_PATH = SCRIPT_DIR.parent.parent / "bond_data" / "parquet" / "historical_bond_details.parquet"
PORTFOLIO_PARQUET_PATH = SCRIPT_DIR.parent.parent / "bond_data" / "parquet" / "historical_portfolio.parquet"
OUTPUT_DIR = SCRIPT_DIR.parent / "processed_data"
LOGS_DIR = SCRIPT_DIR.parent.parent / "bond_data" / "logs"
RUNS_TODAY_LOG = LOGS_DIR / "runs_today.log"

# Size threshold for >3mm filters
SIZE_THRESHOLD = 3000000

# ============================================================================
# MERGE CONFIGURATION
# ============================================================================

# Columns to merge from historical_portfolio.parquet (last date only)
PORTFOLIO_MERGE_COLUMNS = [
    "QUANTITY",
    "POSITION CR01",
]

# Aggregation strategy for portfolio columns when multiple rows per CUSIP exist
# Options: "sum" (for numeric columns like QUANTITY, POSITION CR01) or "first"
PORTFOLIO_AGGREGATION = {
    "QUANTITY": "sum",
    "POSITION CR01": "sum",
}

# Columns to merge from historical_bond_details.parquet (last date only)
BOND_DETAILS_MERGE_COLUMNS = [
    "G Sprd",
    "Yrs (Cvn)",
    "vs BI",
    "vs BCE",
    "MTD Equity",
    "YTD Equity",
    "Retracement",
    "Yrs Since Issue",
    "Z Score",
    "Retracement2",
    "Rating",
    "Custom_Sector",
]

# Columns to merge from bond details that should be converted to numeric
BOND_DETAILS_NUMERIC_COLUMNS = [
    "G Sprd",
    "Yrs (Cvn)",
    "vs BI",
    "vs BCE",
    "MTD Equity",
    "YTD Equity",
    "Retracement",
    "Yrs Since Issue",
    "Z Score",
    "Retracement2",
]

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


def normalize_cusip(cusip: str) -> str:
    """
    Normalize a CUSIP to 9-character uppercase format for matching.
    
    This matches the normalization used in bond_pipeline (uppercase, remove extra text).
    Runs pipeline keeps CUSIPs as-is, but we need normalized CUSIPs to match with
    historical_bond_details.parquet which has normalized CUSIPs.
    
    Args:
        cusip: CUSIP string (may be lowercase, have extra text, etc.)
    
    Returns:
        Normalized 9-character uppercase CUSIP.
    """
    if pd.isna(cusip) or cusip == '':
        return ''
    
    # Convert to string and normalize
    cusip_str = str(cusip).strip().upper()
    
    # Remove trailing " CORP" or "Corp" if present
    cusip_str = re.sub(r'\s+CORP$', '', cusip_str, flags=re.IGNORECASE)
    cusip_str = cusip_str.replace(' CORP', '')
    
    # Remove any whitespace
    cusip_str = re.sub(r'\s+', '', cusip_str)
    
    # Take first 9 characters (in case there's extra text)
    if len(cusip_str) >= 9:
        return cusip_str[:9]
    
    return cusip_str


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
        "Benchmark": group["Benchmark"].iloc[0],  # Any row (should all be same)
        "Time": group["Time"].max(),  # Latest Time in group
        "Bid Workout Risk": group["Bid Workout Risk"].mean(),  # Average
        "Security": group["Security"].iloc[0],  # Any row (should all be same)
    }

    # Filter out negative bid/ask spreads (set to NaN) - these are data quality issues
    group_cleaned = group.copy()
    if "Bid Spread" in group_cleaned.columns:
        group_cleaned.loc[group_cleaned["Bid Spread"] < 0, "Bid Spread"] = pd.NA
    if "Ask Spread" in group_cleaned.columns:
        group_cleaned.loc[group_cleaned["Ask Spread"] < 0, "Ask Spread"] = pd.NA
    
    # Filter to rows with Bid Size > 3mm
    bid_gt_3mm = group_cleaned[group_cleaned["Bid Size"] > SIZE_THRESHOLD].copy()
    
    # Filter to rows with Ask Size > 3mm
    ask_gt_3mm = group_cleaned[group_cleaned["Ask Size"] > SIZE_THRESHOLD].copy()
    
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
    
    # Tight Bid (smallest Bid Spread overall, excluding negatives)
    if len(group_cleaned) > 0 and group_cleaned["Bid Spread"].notna().any():
        result["Tight Bid"] = group_cleaned["Bid Spread"].min()
    else:
        result["Tight Bid"] = pd.NA
    
    # Wide Offer (largest Ask Spread overall, excluding negatives)
    if len(group_cleaned) > 0 and group_cleaned["Ask Spread"].notna().any():
        result["Wide Offer"] = group_cleaned["Ask Spread"].max()
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

    # RBC dealer columns (use cleaned data to exclude negative spreads)
    rbc_rows = group_cleaned[group_cleaned["Dealer"] == "RBC"].copy()
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


def merge_external_columns(
    result_df: pd.DataFrame,
    last_date: pd.Timestamp,
    portfolio_path: Path = PORTFOLIO_PARQUET_PATH,
    historical_path: Path = HISTORICAL_PARQUET_PATH,
    portfolio_columns: list[str] = None,
    bond_columns: list[str] = None,
    portfolio_agg: dict[str, str] = None,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Merge columns from portfolio and bond details parquet files into result DataFrame.
    
    Args:
        result_df: DataFrame to merge columns into (must have CUSIP column).
        last_date: Date to filter parquet files to (last date only).
        portfolio_path: Path to historical_portfolio.parquet file.
        historical_path: Path to historical_bond_details.parquet file.
        portfolio_columns: List of column names to merge from portfolio.
        bond_columns: List of column names to merge from bond details.
        portfolio_agg: Dictionary mapping column names to aggregation strategy ("sum" or "first").
        logger: Logger instance for detailed logging.
    
    Returns:
        DataFrame with merged columns added.
    """
    if portfolio_columns is None:
        portfolio_columns = PORTFOLIO_MERGE_COLUMNS
    if bond_columns is None:
        bond_columns = BOND_DETAILS_MERGE_COLUMNS
    if portfolio_agg is None:
        portfolio_agg = PORTFOLIO_AGGREGATION
    
    log_func = logger.info if logger else print
    log_warn = logger.warning if logger else print
    
    log_func("\n" + "=" * 80)
    log_func("MERGING EXTERNAL COLUMNS")
    log_func("=" * 80)
    log_func(f"Date: {last_date}")
    log_func(f"Input DataFrame shape: {result_df.shape[0]:,} rows x {result_df.shape[1]} columns")
    log_func(f"Unique CUSIPs in input: {result_df['CUSIP'].nunique():,}")
    
    # Normalize CUSIPs in result_df for matching (runs CUSIPs may not be normalized)
    log_func("\nNormalizing CUSIPs for matching...")
    result_df["CUSIP_NORMALIZED"] = result_df["CUSIP"].apply(normalize_cusip)
    
    # Count normalization changes
    normalization_changes = (result_df["CUSIP"] != result_df["CUSIP_NORMALIZED"]).sum()
    log_func(f"  CUSIPs that changed during normalization: {normalization_changes:,} / {len(result_df):,}")
    
    # Debug: Show sample of CUSIP normalization
    sample_size = min(10, len(result_df))
    if sample_size > 0:
        sample_df = result_df[["CUSIP", "CUSIP_NORMALIZED"]].head(sample_size)
        changed_samples = sample_df[sample_df["CUSIP"] != sample_df["CUSIP_NORMALIZED"]]
        if len(changed_samples) > 0:
            log_func(f"  Sample CUSIP normalization changes (first {min(5, len(changed_samples))}):")
            for idx, row in changed_samples.head(5).iterrows():
                log_func(f"    '{row['CUSIP']}' -> '{row['CUSIP_NORMALIZED']}'")
    
    # Merge from historical_portfolio.parquet (last date only)
    try:
        portfolio_df = pd.read_parquet(portfolio_path)
        if "Date" in portfolio_df.columns and "CUSIP" in portfolio_df.columns:
            portfolio_last_date = portfolio_df[portfolio_df["Date"] == last_date].copy()
            if len(portfolio_last_date) > 0:
                # Portfolio CUSIPs are already normalized (portfolio pipeline normalizes them)
                # Normalize to be safe
                portfolio_last_date["CUSIP_NORMALIZED"] = portfolio_last_date["CUSIP"].apply(normalize_cusip)
                
                # Filter to columns that exist
                available_cols = [col for col in portfolio_columns if col in portfolio_last_date.columns]
                
                if available_cols:
                    # Build aggregation dictionary
                    agg_dict = {}
                    for col in available_cols:
                        agg_strategy = portfolio_agg.get(col, "first")
                        agg_dict[col] = agg_strategy
                    
                    # Aggregate by normalized CUSIP (handles multiple ACCOUNT/PORTFOLIO rows)
                    portfolio_agg_df = portfolio_last_date.groupby("CUSIP_NORMALIZED", as_index=False).agg(agg_dict)
                    
                    # Merge into result_df using normalized CUSIP
                    result_df = result_df.merge(
                        portfolio_agg_df[["CUSIP_NORMALIZED"] + available_cols],
                        on="CUSIP_NORMALIZED",
                        how="left",
                        suffixes=("", "_portfolio")
                    )
                    
                    log_func(f"  Merged {len(available_cols)} portfolio columns: {available_cols}")
                    
                    # Log match statistics
                    matched_count = result_df[available_cols[0]].notna().sum() if available_cols else 0
                    total_count = len(result_df)
                    match_pct = 100 * matched_count / total_count if total_count > 0 else 0
                    log_func(f"  Portfolio match statistics: {matched_count:,} / {total_count:,} CUSIPs matched ({match_pct:.1f}%)")
                    
                    # Log unmatched CUSIPs sample
                    if matched_count < total_count and available_cols:
                        unmatched = result_df[result_df[available_cols[0]].isna()][["CUSIP", "CUSIP_NORMALIZED"]].head(10)
                        if len(unmatched) > 0:
                            log_func(f"  Sample unmatched CUSIPs (first {len(unmatched)}):")
                            for idx, row in unmatched.iterrows():
                                log_func(f"    '{row['CUSIP']}' (normalized: '{row['CUSIP_NORMALIZED']}')")
                
                # Add missing columns as NA
                for col in portfolio_columns:
                    if col not in available_cols:
                        log_warn(f"  Warning: Portfolio column '{col}' not found in parquet file")
                        result_df[col] = pd.NA
            else:
                log_warn(f"  Warning: No portfolio data found for last date {last_date}")
                for col in portfolio_columns:
                    result_df[col] = pd.NA
        else:
            log_warn("  Warning: Portfolio parquet missing Date or CUSIP columns")
            for col in portfolio_columns:
                result_df[col] = pd.NA
    except Exception as e:
        log_warn(f"  Warning: Failed to load portfolio parquet: {e}")
        if logger:
            logger.exception("Portfolio merge error")
        for col in portfolio_columns:
            result_df[col] = pd.NA
    
    # Merge from historical_bond_details.parquet (last date only)
    try:
        bond_df = pd.read_parquet(historical_path)
        if "Date" in bond_df.columns and "CUSIP" in bond_df.columns:
            # Find the most recent date available (may not be exactly last_date)
            bond_dates = sorted(bond_df["Date"].unique())
            bond_most_recent_date = bond_dates[-1] if bond_dates else None
            
            log_func(f"  Bond details parquet date range: {bond_dates[0] if bond_dates else 'N/A'} to {bond_most_recent_date}")
            log_func(f"  Looking for data on date: {last_date}")
            
            if bond_most_recent_date and bond_most_recent_date != last_date:
                log_warn(f"  Warning: Bond details last date ({bond_most_recent_date}) differs from runs last date ({last_date})")
                log_func(f"  Using bond details date: {bond_most_recent_date}")
                bond_last_date = bond_df[bond_df["Date"] == bond_most_recent_date].copy()
            else:
                bond_last_date = bond_df[bond_df["Date"] == last_date].copy()
            
            if len(bond_last_date) > 0:
                # Bond details CUSIPs are already normalized (bond pipeline normalizes them)
                # Normalize to be safe and create normalized column for matching
                bond_last_date["CUSIP_NORMALIZED"] = bond_last_date["CUSIP"].apply(normalize_cusip)
                
                # Filter to columns that exist
                available_cols = [col for col in bond_columns if col in bond_last_date.columns]
                
                if available_cols:
                    # Handle duplicate normalized CUSIPs (keep first occurrence)
                    if bond_last_date["CUSIP_NORMALIZED"].duplicated().any():
                        bond_last_date = bond_last_date.drop_duplicates(subset=["CUSIP_NORMALIZED"], keep="first")
                    
                    bond_merge_df = bond_last_date[["CUSIP_NORMALIZED"] + available_cols].copy()
                    
                    # Merge into result_df using normalized CUSIP
                    result_df = result_df.merge(
                        bond_merge_df,
                        on="CUSIP_NORMALIZED",
                        how="left",
                        suffixes=("", "_bond")
                    )
                    
                    log_func(f"  Merged {len(available_cols)} bond details columns: {available_cols}")
                    
                    # Report match statistics
                    matched_count = result_df[available_cols[0]].notna().sum() if available_cols else 0
                    total_count = len(result_df)
                    match_pct = 100 * matched_count / total_count if total_count > 0 else 0
                    log_func(f"  Bond details match statistics: {matched_count:,} / {total_count:,} CUSIPs matched ({match_pct:.1f}%)")
                    
                    # Log detailed match statistics per column
                    log_func("  Column-level match statistics:")
                    for col in available_cols:
                        col_matched = result_df[col].notna().sum()
                        col_pct = 100 * col_matched / total_count if total_count > 0 else 0
                        log_func(f"    {col}: {col_matched:,} / {total_count:,} ({col_pct:.1f}%)")
                    
                    # Debug: Show sample of unmatched CUSIPs
                    if matched_count < total_count and available_cols:
                        unmatched = result_df[result_df[available_cols[0]].isna()][["CUSIP", "CUSIP_NORMALIZED"]].head(10)
                        if len(unmatched) > 0:
                            log_func(f"  Sample unmatched CUSIPs (first {len(unmatched)}):")
                            for idx, row in unmatched.iterrows():
                                log_func(f"    '{row['CUSIP']}' (normalized: '{row['CUSIP_NORMALIZED']}')")
                    
                    # Log sample of matched CUSIPs with values
                    if matched_count > 0 and available_cols:
                        matched_sample = result_df[result_df[available_cols[0]].notna()][["CUSIP"] + available_cols[:3]].head(5)
                        if len(matched_sample) > 0:
                            log_func(f"  Sample matched CUSIPs with values (first {len(matched_sample)}):")
                            for idx, row in matched_sample.iterrows():
                                values_str = ", ".join([f"{col}={row[col]}" for col in available_cols[:3] if pd.notna(row[col])])
                                log_func(f"    {row['CUSIP']}: {values_str}")
                    
                    # Convert numeric columns
                    for col in available_cols:
                        if col in BOND_DETAILS_NUMERIC_COLUMNS:
                            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")
                
                # Add missing columns as NA
                for col in bond_columns:
                    if col not in available_cols:
                        log_warn(f"  Warning: Bond details column '{col}' not found in parquet file")
                        result_df[col] = pd.NA
            else:
                log_warn(f"  Warning: No bond details data found for date {bond_most_recent_date if bond_most_recent_date else 'N/A'}")
                log_warn(f"  All bond details columns will be set to NaN")
                for col in bond_columns:
                    result_df[col] = pd.NA
        else:
            log_warn("  Warning: Bond details parquet missing Date or CUSIP columns")
            for col in bond_columns:
                result_df[col] = pd.NA
    except Exception as e:
        log_warn(f"  Warning: Failed to load bond details parquet: {e}")
        if logger:
            logger.exception("Bond details merge error")
        for col in bond_columns:
            result_df[col] = pd.NA
    
    # Remove temporary normalized CUSIP column
    if "CUSIP_NORMALIZED" in result_df.columns:
        result_df = result_df.drop(columns=["CUSIP_NORMALIZED"])
    
    log_func(f"\nFinal DataFrame shape after merge: {result_df.shape[0]:,} rows x {result_df.shape[1]} columns")
    log_func("=" * 80)
    
    return result_df


def run_analysis(
    runs_path: Path = RUNS_PARQUET_PATH,
    output_dir: Path = OUTPUT_DIR,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Execute the runs today analytics workflow.

    Reads parquet, aggregates metrics for required dates only, filters to today,
    computes DoD/MTD/YTD/1yr changes, and exports to CSV.

    Args:
        runs_path: Path to runs_timeseries.parquet file.
        output_dir: Directory for CSV export.
        logger: Logger instance for detailed logging.

    Returns:
        DataFrame containing today's data with DoD, MTD, YTD, and 1yr changes.
    """
    log_func = logger.info if logger else print
    
    # Log run header
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_func("\n" + "=" * 80)
    log_func(f"RUNS TODAY ANALYTICS - {timestamp}")
    log_func("=" * 80)
    log_func(f"Input parquet: {runs_path}")
    log_func(f"Output directory: {output_dir}")
    
    # Step 1: Load parquet and find required dates
    log_func("\n[STEP 1] Loading runs_timeseries.parquet...")
    df = pd.read_parquet(runs_path)
    
    log_func(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Log df.info() to log file
    if logger:
        log_func("\nDataFrame Info (runs_timeseries.parquet):")
        log_func("-" * 80)
        buffer = io.StringIO()
        df.info(buf=buffer)
        log_func(buffer.getvalue())
        log_func("-" * 80)
    
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
    
    log_func(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    log_func(f"Last date: {last_date}")
    log_func(f"Second-to-last date: {second_last_date}")
    
    # Find reference dates for MTD, YTD, and 1yr calculations
    log_func("\n[STEP 2] Finding reference dates for MTD, YTD, and 1yr calculations...")
    
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
    
    log_func(f"  MTD reference date: {mtd_ref_date}")
    log_func(f"  YTD reference date: {ytd_ref_date}")
    log_func(f"  1yr reference date: {one_yr_ref_date}")
    
    # Collect all dates we need to process
    required_dates = {last_date, second_last_date}
    if mtd_ref_date:
        required_dates.add(mtd_ref_date)
    if ytd_ref_date:
        required_dates.add(ytd_ref_date)
    if one_yr_ref_date:
        required_dates.add(one_yr_ref_date)
    
    log_func(f"\n[STEP 3] Processing {len(required_dates)} required dates (optimized from {len(unique_dates)} total dates)")
    
    # Step 3: Filter to only required dates before aggregation
    df_filtered = df[df["Date"].isin(required_dates)].copy()
    log_func(f"Filtered to {len(df_filtered):,} rows for required dates")
    
    # Step 4: Group by Date, CUSIP, Benchmark and compute aggregated metrics
    log_func("\n[STEP 4] Grouping by Date, CUSIP, Benchmark and computing metrics...")
    results = []
    
    grouped = df_filtered.groupby(["Date", "CUSIP", "Benchmark"], as_index=False)
    num_groups = grouped.ngroups
    
    for idx, (name, group) in enumerate(grouped):
        if (idx + 1) % 1000 == 0:
            log_func(f"  Processed {idx + 1:,} / {num_groups:,} groups...")
        
        metrics = compute_group_metrics(group)
        results.append(metrics)
    
    log_func(f"Computed metrics for {len(results):,} groups")
    
    # Step 5: Convert to DataFrame and format
    log_func("\n[STEP 5] Building aggregated DataFrame...")
    aggregated_df = pd.DataFrame(results)
    
    # Ensure column order for aggregated data
    # Note: Benchmark must be included to match by (CUSIP, Benchmark) in lookups
    aggregated_column_order = [
        "Date",
        "CUSIP",
        "Benchmark",
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
    
    # Preserve Benchmark column if it exists but isn't in the order list
    if "Benchmark" in aggregated_df.columns and "Benchmark" not in aggregated_column_order:
        aggregated_column_order.insert(2, "Benchmark")  # Insert after CUSIP
    
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
    
    # Note: We keep all rows (including duplicates by CUSIP with different Benchmarks)
    # This is important because we match by (CUSIP, Benchmark) tuple
    print(f"Rows on last date: {len(last_date_df):,}")
    
    # Step 6: Get rows that exist on both last date and second-to-last date
    # Match by (CUSIP, Benchmark) to ensure correct pairing
    second_last_date_df = aggregated_df[aggregated_df["Date"] == second_last_date].copy()
    
    # Create sets of (CUSIP, Benchmark) tuples for matching
    last_date_keys = set(zip(last_date_df["CUSIP"], last_date_df["Benchmark"]))
    second_last_date_keys = set(zip(second_last_date_df["CUSIP"], second_last_date_df["Benchmark"]))
    common_keys = last_date_keys & second_last_date_keys
    
    print(f"Unique (CUSIP, Benchmark) pairs on last date: {len(last_date_keys):,}")
    print(f"Unique (CUSIP, Benchmark) pairs on second-to-last date: {len(second_last_date_keys):,}")
    print(f"Common (CUSIP, Benchmark) pairs: {len(common_keys):,}")
    
    # Filter to only (CUSIP, Benchmark) pairs that exist on both dates
    result_df = last_date_df[
        last_date_df.apply(lambda row: (row["CUSIP"], row["Benchmark"]) in common_keys, axis=1)
    ].copy()
    
    # Step 7: Create lookup dictionaries for reference dates
    # Use (CUSIP, Benchmark) tuple as key to ensure correct matching
    second_last_lookup = {}
    for idx, row in second_last_date_df.iterrows():
        key = (row["CUSIP"], row["Benchmark"])
        second_last_lookup[key] = row
    
    mtd_lookup = {}
    if mtd_ref_date:
        mtd_df = aggregated_df[aggregated_df["Date"] == mtd_ref_date].copy()
        for idx, row in mtd_df.iterrows():
            key = (row["CUSIP"], row["Benchmark"])
            # If duplicate key exists, keep first occurrence
            if key not in mtd_lookup:
                mtd_lookup[key] = row
    
    ytd_lookup = {}
    if ytd_ref_date:
        ytd_df = aggregated_df[aggregated_df["Date"] == ytd_ref_date].copy()
        for idx, row in ytd_df.iterrows():
            key = (row["CUSIP"], row["Benchmark"])
            # If duplicate key exists, keep first occurrence
            if key not in ytd_lookup:
                ytd_lookup[key] = row
    
    one_yr_lookup = {}
    if one_yr_ref_date:
        one_yr_df = aggregated_df[aggregated_df["Date"] == one_yr_ref_date].copy()
        for idx, row in one_yr_df.iterrows():
            key = (row["CUSIP"], row["Benchmark"])
            # If duplicate key exists, keep first occurrence
            if key not in one_yr_lookup:
                one_yr_lookup[key] = row
    
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
            benchmark = row["Benchmark"]
            last_value = row[col]
            lookup_key = (cusip, benchmark)
            
            if lookup_key in second_last_lookup:
                second_last_value = second_last_lookup[lookup_key][col]
                
                # Calculate DoD: Last Date - Second Last Date
                # Only calculate if BOTH values exist and are not blank/empty
                # If either value is blank/NaN/empty, set DoD to blank (pd.NA)
                if (pd.notna(last_value) and pd.notna(second_last_value) and 
                    last_value != '' and second_last_value != ''):
                    try:
                        dod_values.append(float(last_value) - float(second_last_value))
                    except (ValueError, TypeError):
                        # If conversion fails, set to blank
                        dod_values.append(pd.NA)
                else:
                    # Either value is blank/NaN/empty - set DoD to blank
                    dod_values.append(pd.NA)
            else:
                # CUSIP not found in second-to-last date - set DoD to blank
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
            benchmark = row["Benchmark"]
            last_value = row[col]
            lookup_key = (cusip, benchmark)
            if lookup_key in mtd_lookup:
                mtd_value = mtd_lookup[lookup_key][col]
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
            benchmark = row["Benchmark"]
            last_value = row[col]
            lookup_key = (cusip, benchmark)
            if lookup_key in ytd_lookup:
                ytd_value = ytd_lookup[lookup_key][col]
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
            benchmark = row["Benchmark"]
            last_value = row[col]
            lookup_key = (cusip, benchmark)
            if lookup_key in one_yr_lookup:
                one_yr_value = one_yr_lookup[lookup_key][col]
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
        benchmark = row["Benchmark"]
        lookup_key = (cusip, benchmark)
        if lookup_key in second_last_lookup:
            second_last_row = second_last_lookup[lookup_key]
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
    
    # Add portfolio columns if they exist
    for col in PORTFOLIO_MERGE_COLUMNS:
        if col in result_df.columns:
            column_order.append(col)
    
    # Add bond details columns if they exist
    for col in BOND_DETAILS_MERGE_COLUMNS:
        if col in result_df.columns:
            column_order.append(col)
    
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
    
    # Convert portfolio numeric columns
    for col in PORTFOLIO_MERGE_COLUMNS:
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
    
    # Step 15.5: Merge columns from portfolio and bond details parquet files
    log_func("\n[STEP 15.5] Merging external columns...")
    result_df = merge_external_columns(
        result_df,
        last_date,
        portfolio_path=PORTFOLIO_PARQUET_PATH,
        historical_path=HISTORICAL_PARQUET_PATH,
        portfolio_columns=PORTFOLIO_MERGE_COLUMNS,
        bond_columns=BOND_DETAILS_MERGE_COLUMNS,
        portfolio_agg=PORTFOLIO_AGGREGATION,
        logger=logger,
    )
    
    # Re-apply column order after merge (merged columns may have been added)
    existing_cols = [col for col in column_order if col in result_df.columns]
    remaining_cols = [col for col in result_df.columns if col not in existing_cols]
    result_df = result_df[existing_cols + remaining_cols]
    
    # Step 16: Write to CSV
    log_func("\n[STEP 16] Writing CSV output...")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "runs_today.csv"
    result_df.to_csv(output_path, index=False)
    log_func(f"CSV written to: {output_path}")
    
    # Step 17: Read CSV back and log df.info() for verification
    log_func("\n[STEP 17] Reading CSV back for verification...")
    csv_df = pd.read_csv(output_path)
    log_func(f"CSV read back: {len(csv_df):,} rows, {len(csv_df.columns)} columns")
    
    # Log df.info() for the CSV file (verbose to show all columns)
    if logger:
        log_func("\n" + "-" * 80)
        log_func("DataFrame Info (runs_today.csv) - Full Column Details:")
        log_func("-" * 80)
        buffer = io.StringIO()
        csv_df.info(buf=buffer, verbose=True, show_counts=True)
        log_func(buffer.getvalue())
        log_func("-" * 80)
    
    # Step 18: Log final results summary
    log_func("\n" + "=" * 80)
    log_func("FINAL RESULTS SUMMARY")
    log_func("=" * 80)
    log_func(f"Date: {last_date}")
    log_func(f"Total rows: {len(csv_df):,}")
    log_func(f"Total columns: {len(csv_df.columns)}")
    log_func(f"Unique CUSIPs: {csv_df['CUSIP'].nunique():,}")
    
    # Log column summary
    log_func("\nColumn Summary:")
    log_func(f"  Total columns: {len(result_df.columns)}")
    log_func(f"  Portfolio columns: {len([c for c in PORTFOLIO_MERGE_COLUMNS if c in result_df.columns])}")
    log_func(f"  Bond details columns: {len([c for c in BOND_DETAILS_MERGE_COLUMNS if c in result_df.columns])}")
    log_func(f"  DoD change columns: {len([c for c in result_df.columns if c.startswith('DoD Chg ')])}")
    log_func(f"  MTD change columns: {len([c for c in result_df.columns if c.startswith('MTD Chg ')])}")
    log_func(f"  YTD change columns: {len([c for c in result_df.columns if c.startswith('YTD Chg ')])}")
    log_func(f"  1yr change columns: {len([c for c in result_df.columns if c.startswith('1yr Chg ')])}")
    
    # Log sample data from CSV
    log_func("\n" + "-" * 80)
    log_func("First 10 rows (sample from CSV):")
    log_func("-" * 80)
    log_func(csv_df.head(10).to_string(index=False))
    
    log_func("\n" + "-" * 80)
    log_func("Last 10 rows (sample from CSV):")
    log_func("-" * 80)
    log_func(csv_df.tail(10).to_string(index=False))
    
    log_func("\n" + "=" * 80)
    log_func(f"RUN COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_func("=" * 80 + "\n")
    
    # Console output (minimal)
    print(f"\nRuns Today Analytics completed")
    print(f"Date: {last_date}")
    print(f"Total rows: {len(csv_df):,}")
    print(f"Unique CUSIPs: {csv_df['CUSIP'].nunique():,}")
    print(f"CSV written to: {output_path}")
    if logger:
        print(f"Detailed log written to: {RUNS_TODAY_LOG}")
    
    return result_df


def main() -> None:
    """Entry point for running the runs today analytics script."""
    # Set up logging
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        RUNS_TODAY_LOG,
        name='runs_today',
        console_level=logging.CRITICAL,  # Suppress console output, only log to file
    )
    
    # Run analysis with logger
    run_analysis(logger=logger)


if __name__ == "__main__":
    main()
