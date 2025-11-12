"""
Portfolio executable combinations pair analytics script.

This module reads `bond_data/parquet/bql.parquet`, filters CUSIPs present in the most
recent dates, computes all pairwise spreads, filters to only pairs where cusip_1 matches
CUSIPs from runs_today.csv with CR01 @ Wide Offer >= 2000, and cusip_2 is in the
portfolio CUSIP list, and exports all pairs sorted by Z Score to CSV.
The top 80 pairs are displayed to console for monitoring relative value opportunities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Get script directory and build paths relative to it
SCRIPT_DIR = Path(__file__).parent.resolve()
BQL_PARQUET_PATH = SCRIPT_DIR.parent.parent / "bond_data" / "parquet" / "bql.parquet"
HISTORICAL_PARQUET_PATH = SCRIPT_DIR.parent.parent / "bond_data" / "parquet" / "historical_bond_details.parquet"
RUNS_TODAY_CSV_PATH = SCRIPT_DIR.parent / "processed_data" / "runs_today.csv"
OUTPUT_DIR = SCRIPT_DIR.parent / "processed_data"

# Filter to CUSIPs present in most recent 75% of dates
RECENT_DATE_PERCENT = 0.75

# CR01 @ Wide Offer threshold for filtering cusip_1
CR01_WIDE_OFFER_THRESHOLD = 2000.0

# Portfolio CUSIPs (normalized to uppercase, 9 characters)
PORTFOLIO_CUSIPS = {
    "13607PXH2",
    "44810ZCS7",
    "29251ZCJ4",
    "06418YXB9",
    "83179XAL2",
    "13607HR79",
    "06418MM43",
    "766910BT9",
    "779926FY5",
    "63306AHT6",
    "83179XAH1",
    "25675TAP2",
    "07813ZCL6",
    "89116CST5",
    "780086XL3",
    "7800867G3",
    "89117FPG8",
    "87971MCC5",
    "13607PBA1",
    "89156VAC0",
    "387427AM9",
    "34527ACW8",
    "064164QM1",
    "26153WAJ8",
    "92938WAD5",
    "31430W3J1",
    "375916AA1",
    "759480AN6",
    "375916AC7",
    "16141AAG8",
    "759480AM8",
    "019456AK8",
    "667495AN5",
    "15135UAT6",
    "949746TJO",
    "780086WG5",
    "019456AM4",
    "06369ZCL6",
    "55279QAE0",
    "891102AE5",
    "136765BX1",
    "12658MAD3",
    "02138ZAQ6",
    "63306AHF6",
    "16141AAF0",
    "775109BT7",
    "06415GDJ6",
    "89117GX51",
    "375916AE3",
    "89353ZCF3",
    "31430WU44",
    "11291ZAM9",
    "190330AQ3",
    "172967MJ7",
    "31943BBY5",
    "918423BJ2",
    "56501RAQ9",
    "86682ZAT3",
    "12658MAC5",
}


@dataclass
class PairSummary:
    """Container with pair analytics metrics."""

    Bond_1: str
    Bond_2: str
    last_value: float
    average_value: float
    vs_average: float
    z_score: Optional[float]
    percentile: float
    cusip_1: str
    cusip_2: str


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


def get_cad_cusips(historical_path: Path) -> Set[str]:
    """
    Get CUSIPs with Currency="CAD" from the last date in historical_bond_details.parquet.

    Args:
        historical_path: Path to historical_bond_details.parquet file.

    Returns:
        Set of CUSIPs with Currency="CAD" on the last date.
    """
    print("Loading historical bond details to filter CAD CUSIPs...")
    historical_df = pd.read_parquet(historical_path)
    
    # Get the last date
    last_date = historical_df["Date"].max()
    print(f"Last date in historical data: {last_date}")
    
    # Filter to last date and Currency="CAD"
    last_date_df = historical_df[historical_df["Date"] == last_date]
    cad_cusips = set(last_date_df[last_date_df["Currency"] == "CAD"]["CUSIP"].unique())
    
    print(f"Found {len(cad_cusips)} CAD CUSIPs on last date")
    return cad_cusips


def get_executable_cusips(runs_today_path: Path, threshold: float = CR01_WIDE_OFFER_THRESHOLD) -> Set[str]:
    """
    Get CUSIPs from runs_today.csv where CR01 @ Wide Offer >= threshold.

    Args:
        runs_today_path: Path to runs_today.csv file.
        threshold: Minimum CR01 @ Wide Offer value (default 2000.0).

    Returns:
        Set of CUSIPs with CR01 @ Wide Offer >= threshold.

    Raises:
        FileNotFoundError: If runs_today.csv does not exist.
        ValueError: If required column is missing or no matching CUSIPs found.
    """
    if not runs_today_path.exists():
        raise FileNotFoundError(f"runs_today.csv not found at: {runs_today_path}")
    
    print(f"Loading runs_today.csv from {runs_today_path}...")
    runs_df = pd.read_csv(runs_today_path)
    
    # Check if required column exists
    required_column = "CR01 @ Wide Offer"
    if required_column not in runs_df.columns:
        raise ValueError(
            f"Required column '{required_column}' not found in runs_today.csv. "
            f"Available columns: {list(runs_df.columns)}"
        )
    
    # Filter to rows with valid (non-null) CR01 @ Wide Offer values >= threshold
    valid_mask = runs_df[required_column].notna()
    filtered_df = runs_df[valid_mask].copy()
    
    if filtered_df.empty:
        raise ValueError(f"No rows with valid CR01 @ Wide Offer values in runs_today.csv")
    
    # Filter to rows where CR01 @ Wide Offer >= threshold
    threshold_mask = filtered_df[required_column] >= threshold
    executable_df = filtered_df[threshold_mask].copy()
    
    if executable_df.empty:
        raise ValueError(
            f"No CUSIPs found with CR01 @ Wide Offer >= {threshold} in runs_today.csv"
        )
    
    # Extract CUSIPs
    executable_cusips = set(executable_df["CUSIP"].unique())
    
    print(
        f"Found {len(executable_cusips)} CUSIPs with CR01 @ Wide Offer >= {threshold} "
        f"(from {len(executable_df)} rows)"
    )
    
    return executable_cusips


def filter_recent_cusips(data: pd.DataFrame, percent: float = RECENT_DATE_PERCENT) -> pd.DataFrame:
    """
    Filter to CUSIPs present in the most recent percent of dates.

    Args:
        data: BQL DataFrame with Date and CUSIP columns.
        percent: Percentage of most recent dates to consider (default 0.75).

    Returns:
        Filtered DataFrame containing only CUSIPs present in recent dates.
    """
    unique_dates = sorted(data["Date"].unique())
    num_recent_dates = max(1, int(len(unique_dates) * percent))
    recent_dates = set(unique_dates[-num_recent_dates:])
    
    # Filter to recent dates first
    recent_data = data[data["Date"].isin(recent_dates)].copy()
    
    # Find CUSIPs that appear in all recent dates (or at least most of them)
    # For efficiency, we'll use CUSIPs that have data in at least 90% of recent dates
    cusip_date_counts = recent_data.groupby("CUSIP")["Date"].nunique()
    min_dates_required = max(1, int(num_recent_dates * 0.9))
    valid_cusips = cusip_date_counts[cusip_date_counts >= min_dates_required].index.tolist()
    
    # Return all data for valid CUSIPs (not just recent dates)
    return data[data["CUSIP"].isin(valid_cusips)].copy()


def build_name_lookup(data: pd.DataFrame) -> Dict[str, str]:
    """
    Generate a mapping from CUSIP to security name.

    Args:
        data: BQL DataFrame containing `CUSIP` and `Name`.

    Returns:
        Dictionary mapping each CUSIP to its most recent non-null name.
    """
    name_series = (
        data.dropna(subset=["Name"])
        .sort_values("Date")
        .groupby("CUSIP")["Name"]
        .last()
    )
    return {
        cusip: ensure_ascii(name)
        for cusip, name in name_series.to_dict().items()
    }


def compute_pair_stats_vectorized(
    cusip_1_values: np.ndarray,
    cusip_2_values: np.ndarray,
) -> Optional[Tuple[float, float, float, Optional[float], float]]:
    """
    Compute pair statistics using vectorized operations.

    Args:
        cusip_1_values: Array of values for first CUSIP (aligned by date).
        cusip_2_values: Array of values for second CUSIP (aligned by date).

    Returns:
        Tuple of (last_value, average_value, vs_average, z_score, percentile)
        or None if insufficient data.
    """
    # Compute spreads
    spreads = cusip_1_values - cusip_2_values
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(spreads))
    if valid_mask.sum() < 2:  # Need at least 2 data points
        return None
    
    valid_spreads = spreads[valid_mask]
    
    # Compute statistics
    last_value = float(valid_spreads[-1])
    average_value = float(np.mean(valid_spreads))
    vs_average = last_value - average_value
    
    # Compute Z score
    if len(valid_spreads) > 1:
        spread_std = float(np.std(valid_spreads, ddof=1))
        if spread_std > 0:
            z_score = vs_average / spread_std
        else:
            z_score = None
    else:
        z_score = None
    
    # Compute percentile (rank of last value)
    percentile = float((valid_spreads <= last_value).sum() / len(valid_spreads) * 100.0)
    
    return (last_value, average_value, vs_average, z_score, percentile)


def check_portfolio_cusips_in_data(
    portfolio_cusips: Set[str],
    data_cusips: Set[str],
    historical_path: Path,
) -> None:
    """
    Check for portfolio CUSIPs missing from BQL data and log warnings with bond names.

    Args:
        portfolio_cusips: Set of portfolio CUSIPs to check.
        data_cusips: Set of CUSIPs present in BQL data.
        historical_path: Path to historical_bond_details.parquet to look up bond names.
    """
    missing_cusips = portfolio_cusips - data_cusips
    if missing_cusips:
        # Load historical data to get bond names
        historical_df = pd.read_parquet(historical_path)
        last_date = historical_df["Date"].max()
        last_date_df = historical_df[historical_df["Date"] == last_date]
        
        # Create CUSIP to Security name mapping
        cusip_to_name = dict(zip(last_date_df["CUSIP"], last_date_df["Security"]))
        
        print(f"\nWARNING: {len(missing_cusips)} portfolio CUSIPs not found in BQL data:")
        for cusip in sorted(missing_cusips):
            bond_name = cusip_to_name.get(cusip, "Unknown")
            # Ensure ASCII-safe for console
            bond_name_safe = ensure_ascii(bond_name)
            print(f"  - {bond_name_safe} ({cusip})")


def run_analysis(
    bql_path: Path = BQL_PARQUET_PATH,
    historical_path: Path = HISTORICAL_PARQUET_PATH,
    runs_today_path: Path = RUNS_TODAY_CSV_PATH,
    output_dir: Path = OUTPUT_DIR,
    top_n: int = 80,
) -> pd.DataFrame:
    """
    Execute the portfolio executable combinations pair analytics workflow.

    Args:
        bql_path: Path to the BQL parquet file.
        historical_path: Path to historical_bond_details.parquet file.
        runs_today_path: Path to runs_today.csv file.
        output_dir: Directory for CSV export.
        top_n: Number of top pairs to return (default 80).

    Returns:
        DataFrame containing top N pair analytics filtered to executable CUSIPs
        (cusip_1) and portfolio CUSIPs (cusip_2), sorted by Z Score.
    """
    # First, get CAD CUSIPs from historical data
    cad_cusips = get_cad_cusips(historical_path)
    
    if not cad_cusips:
        raise ValueError("No CAD CUSIPs found in historical data")
    
    # Get executable CUSIPs from runs_today.csv
    executable_cusips = get_executable_cusips(runs_today_path, CR01_WIDE_OFFER_THRESHOLD)
    
    print("Loading BQL data...")
    data = pd.read_parquet(bql_path)
    
    # Filter to only CAD CUSIPs
    print(f"Filtering BQL data to {len(cad_cusips)} CAD CUSIPs...")
    data = data[data["CUSIP"].isin(cad_cusips)].copy()
    
    if data.empty:
        raise ValueError("No BQL data found for CAD CUSIPs")
    
    # Check for missing portfolio CUSIPs in BQL data
    data_cusips = set(data["CUSIP"].unique())
    check_portfolio_cusips_in_data(PORTFOLIO_CUSIPS, data_cusips, historical_path)
    
    # Ensure all columns have complete data - drop any rows with missing values
    print("Filtering for complete data...")
    data = data.dropna()
    
    if data.empty:
        raise ValueError("No data remaining after filtering for complete rows")
    
    print(f"Total rows: {len(data)}, Unique CUSIPs: {data['CUSIP'].nunique()}, Unique Dates: {data['Date'].nunique()}")
    
    # Filter to CUSIPs present in most recent dates
    print(f"Filtering to CUSIPs present in most recent {RECENT_DATE_PERCENT*100:.0f}% of dates...")
    filtered_data = filter_recent_cusips(data, RECENT_DATE_PERCENT)
    
    if filtered_data.empty:
        raise ValueError("No data remaining after filtering for recent CUSIPs")
    
    print(f"After filtering: {len(filtered_data)} rows, {filtered_data['CUSIP'].nunique()} CUSIPs")
    
    # Build name lookup
    print("Building name lookup...")
    name_lookup = build_name_lookup(filtered_data)
    
    # Create pivoted table (Date × CUSIP)
    print("Creating pivoted table...")
    wide_values = filtered_data.pivot_table(
        index="Date",
        columns="CUSIP",
        values="Value",
        aggfunc="last",
    ).sort_index()
    
    # Get list of CUSIPs
    cusips = wide_values.columns.tolist()
    
    # Pre-filter: Only include pairs where cusip_1 is executable and cusip_2 is in portfolio
    executable_cusips_in_data = executable_cusips & set(cusips)
    portfolio_cusips_in_data = PORTFOLIO_CUSIPS & set(cusips)
    
    print(f"Pre-filtering: {len(executable_cusips_in_data)} executable CUSIPs, {len(portfolio_cusips_in_data)} portfolio CUSIPs found in data")
    
    if not executable_cusips_in_data:
        raise ValueError(f"No executable CUSIPs (CR01 @ Wide Offer >= {CR01_WIDE_OFFER_THRESHOLD}) found in filtered BQL data")
    
    if not portfolio_cusips_in_data:
        raise ValueError("No portfolio CUSIPs found in filtered BQL data")
    
    # Create mapping from CUSIP to its index in the pivoted table
    cusip_to_idx = {cusip: i for i, cusip in enumerate(cusips)}
    
    # Get indices of executable CUSIPs for cusip_1 and portfolio CUSIPs for cusip_2
    executable_indices_1 = [cusip_to_idx[cusip] for cusip in cusips if cusip in executable_cusips_in_data]
    portfolio_indices_2 = [cusip_to_idx[cusip] for cusip in cusips if cusip in portfolio_cusips_in_data]
    
    num_cusips_1 = len(executable_indices_1)
    num_cusips_2 = len(portfolio_indices_2)
    num_pairs = num_cusips_1 * num_cusips_2
    
    print(f"Pre-filtered to {num_pairs:,} pairs ({num_cusips_1} executable CUSIPs × {num_cusips_2} portfolio CUSIPs)")
    
    # Convert to numpy array for faster operations
    values_array = wide_values.values  # Shape: (num_dates, num_cusips)
    
    # Pre-allocate results list
    summaries: List[PairSummary] = []
    
    # Process pairs in batches for progress tracking
    batch_size = max(1000, num_pairs // 100)  # Show progress every ~1%
    processed = 0
    
    # Iterate over executable CUSIPs (cusip_1) paired with portfolio CUSIPs (cusip_2)
    for i_idx in executable_indices_1:
        cusip_1 = cusips[i_idx]
        cusip_1_vals = values_array[:, i_idx]
        
        for j_idx in portfolio_indices_2:
            cusip_2 = cusips[j_idx]
            cusip_2_vals = values_array[:, j_idx]
            
            # Find dates where both have values
            both_valid = ~(np.isnan(cusip_1_vals) | np.isnan(cusip_2_vals))
            
            if both_valid.sum() < 2:  # Need at least 2 overlapping dates
                continue
            
            # Extract valid values
            valid_1 = cusip_1_vals[both_valid]
            valid_2 = cusip_2_vals[both_valid]
            
            # Compute statistics
            stats = compute_pair_stats_vectorized(valid_1, valid_2)
            if stats is None:
                continue
            
            last_value, average_value, vs_average, z_score, percentile = stats
            
            # Only include pairs with valid Z scores (for sorting)
            if z_score is None:
                continue
            
            summaries.append(
                PairSummary(
                    Bond_1=name_lookup.get(cusip_1, cusip_1),
                    Bond_2=name_lookup.get(cusip_2, cusip_2),
                    last_value=last_value,
                    average_value=average_value,
                    vs_average=vs_average,
                    z_score=z_score,
                    percentile=percentile,
                    cusip_1=cusip_1,
                    cusip_2=cusip_2,
                )
            )
            
            processed += 1
            if processed % batch_size == 0:
                print(f"  Processed {processed:,} / {num_pairs:,} pairs ({processed*100/num_pairs:.1f}%)...")
    
    print(f"Computed {len(summaries):,} valid pairs")
    
    if len(summaries) == 0:
        raise ValueError("No pairs remaining after filtering!")
    
    # Convert to DataFrame
    print("Converting to DataFrame...")
    results_df = pd.DataFrame(
        [
            {
                "Bond_1": pair.Bond_1,
                "Bond_2": pair.Bond_2,
                "Last": pair.last_value,
                "Avg": pair.average_value,
                "vs Avg": pair.vs_average,
                "Z Score": pair.z_score,
                "Percentile": pair.percentile,
                "cusip_1": pair.cusip_1,
                "cusip_2": pair.cusip_2,
            }
            for pair in summaries
        ]
    )
    
    # Sort by Z Score descending
    print("Sorting results...")
    results_df = results_df.sort_values("Z Score", ascending=False, na_position="last")
    
    # Ensure ASCII-safe names for all rows
    for column in ["Bond_1", "Bond_2"]:
        results_df[column] = results_df[column].map(ensure_ascii)
    
    # Write all rows to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "port_executable.csv"
    results_df.to_csv(output_path, index=False)
    
    # Display top N to console
    top_results = results_df.head(top_n).copy()
    print(f"\nTop {len(top_results)} Portfolio Executable Pair Analytics (by Z Score):")
    print(top_results.to_string(index=False))
    print(f"\nCSV written to: {output_path} (all {len(results_df):,} rows)")
    
    return top_results


def main() -> None:
    """Entry point for running the portfolio executable combinations pair analytics script."""
    run_analysis()


if __name__ == "__main__":
    main()

