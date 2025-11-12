"""
Term combinations pair analytics script.

This module reads `bond_data/parquet/bql.parquet`, filters CUSIPs present in the most
recent dates, filters pairs where "Yrs (Cvn)" values are within 0.8 years of each other,
computes pairwise spreads, and exports all pairs sorted by Z Score to CSV. The top 80
pairs are displayed to console for monitoring relative value opportunities.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Get script directory and build paths relative to it
SCRIPT_DIR = Path(__file__).parent.resolve()
BQL_PARQUET_PATH = SCRIPT_DIR.parent.parent / "bond_data" / "parquet" / "bql.parquet"
HISTORICAL_PARQUET_PATH = SCRIPT_DIR.parent.parent / "bond_data" / "parquet" / "historical_bond_details.parquet"
OUTPUT_DIR = SCRIPT_DIR.parent / "processed_data"

# Filter to CUSIPs present in most recent 75% of dates
RECENT_DATE_PERCENT = 0.75

# Maximum absolute difference in "Yrs (Cvn)" for pair filtering
MAX_YRS_CVN_DIFF = 0.8


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


def get_cad_cusips_with_yrs_cvn(historical_path: Path) -> Tuple[Set[str], Dict[str, float]]:
    """
    Get CAD CUSIPs and "Yrs (Cvn)" mapping from the last date in historical_bond_details.parquet.

    Args:
        historical_path: Path to historical_bond_details.parquet file.

    Returns:
        Tuple of (cad_cusips_set, yrs_cvn_mapping) where:
        - cad_cusips_set: Set of CUSIPs with Currency="CAD" on the last date
        - yrs_cvn_mapping: Dictionary mapping CUSIP to "Yrs (Cvn)" value (only includes CUSIPs with valid "Yrs (Cvn)")
    """
    print("Loading historical bond details to filter CAD CUSIPs and get Yrs (Cvn) mappings...")
    historical_df = pd.read_parquet(historical_path)
    
    # Get the last date
    last_date = historical_df["Date"].max()
    print(f"Last date in historical data: {last_date}")
    
    # Filter to last date
    last_date_df = historical_df[historical_df["Date"] == last_date].copy()
    
    # Filter to CAD CUSIPs
    cad_df = last_date_df[last_date_df["Currency"] == "CAD"].copy()
    cad_cusips = set(cad_df["CUSIP"].unique())
    
    print(f"Found {len(cad_cusips)} CAD CUSIPs on last date")
    
    # Filter to CAD CUSIPs with valid "Yrs (Cvn)" data
    valid_df = cad_df[cad_df["Yrs (Cvn)"].notna()].copy()
    
    # Convert "Yrs (Cvn)" to float
    valid_df["Yrs (Cvn)"] = pd.to_numeric(valid_df["Yrs (Cvn)"], errors="coerce")
    valid_df = valid_df[valid_df["Yrs (Cvn)"].notna()].copy()
    
    # Create mapping
    yrs_cvn_mapping = dict(zip(valid_df["CUSIP"], valid_df["Yrs (Cvn)"].astype(float)))
    
    excluded_count = len(cad_cusips) - len(yrs_cvn_mapping)
    if excluded_count > 0:
        print(f"Excluded {excluded_count} CAD CUSIPs with missing/invalid Yrs (Cvn) data")
    
    print(f"Found {len(yrs_cvn_mapping)} CAD CUSIPs with valid Yrs (Cvn) data")
    
    return cad_cusips, yrs_cvn_mapping


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


def run_analysis(
    bql_path: Path = BQL_PARQUET_PATH,
    historical_path: Path = HISTORICAL_PARQUET_PATH,
    output_dir: Path = OUTPUT_DIR,
    top_n: int = 80,
    max_yrs_cvn_diff: float = MAX_YRS_CVN_DIFF,
) -> pd.DataFrame:
    """
    Execute the term combinations pair analytics workflow.

    Args:
        bql_path: Path to the BQL parquet file.
        historical_path: Path to historical_bond_details.parquet file.
        output_dir: Directory for CSV export.
        top_n: Number of top pairs to return (default 80).
        max_yrs_cvn_diff: Maximum absolute difference in "Yrs (Cvn)" for pair filtering (default 0.8).

    Returns:
        DataFrame containing top N pair analytics sorted by Z Score.
    """
    # Get CAD CUSIPs and "Yrs (Cvn)" mapping from historical data
    cad_cusips, yrs_cvn_mapping = get_cad_cusips_with_yrs_cvn(historical_path)
    
    if not cad_cusips:
        raise ValueError("No CAD CUSIPs found in historical data")
    
    if not yrs_cvn_mapping:
        raise ValueError("No CAD CUSIPs with valid Yrs (Cvn) data found")
    
    print("Loading BQL data...")
    data = pd.read_parquet(bql_path)
    
    # Filter to only CAD CUSIPs that have "Yrs (Cvn)" data
    valid_cusips = set(yrs_cvn_mapping.keys())
    print(f"Filtering BQL data to {len(valid_cusips)} CAD CUSIPs with valid Yrs (Cvn) data...")
    data = data[data["CUSIP"].isin(valid_cusips)].copy()
    
    if data.empty:
        raise ValueError("No BQL data found for CAD CUSIPs with valid Yrs (Cvn) data")
    
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
    
    # Filter CUSIPs to only those with "Yrs (Cvn)" data (should already be filtered, but double-check)
    cusips = [c for c in cusips if c in yrs_cvn_mapping]
    
    if not cusips:
        raise ValueError("No CUSIPs remaining after Yrs (Cvn) filtering")
    
    num_cusips = len(cusips)
    
    # Pre-filter pairs based on "Yrs (Cvn)" difference
    print(f"Pre-filtering pairs where abs(Yrs (Cvn) difference) <= {max_yrs_cvn_diff}...")
    valid_pairs: List[Tuple[int, int]] = []
    for i, j in combinations(range(num_cusips), 2):
        cusip_1 = cusips[i]
        cusip_2 = cusips[j]
        
        yrs_cvn_1 = yrs_cvn_mapping.get(cusip_1)
        yrs_cvn_2 = yrs_cvn_mapping.get(cusip_2)
        
        # Both should be in mapping (already filtered), but check anyway
        if yrs_cvn_1 is None or yrs_cvn_2 is None:
            continue
        
        # Check if difference is within threshold
        if abs(yrs_cvn_1 - yrs_cvn_2) <= max_yrs_cvn_diff:
            valid_pairs.append((i, j))
    
    num_pairs = len(valid_pairs)
    print(f"Found {num_pairs:,} valid pairs (out of {num_cusips * (num_cusips - 1) // 2:,} possible)")
    
    if num_pairs == 0:
        raise ValueError(f"No pairs found with Yrs (Cvn) difference <= {max_yrs_cvn_diff}")
    
    # Convert to numpy array for faster operations
    values_array = wide_values[cusips].values  # Shape: (num_dates, num_cusips)
    dates_index = wide_values.index
    
    # Pre-allocate results list
    summaries: List[PairSummary] = []
    
    # Process pairs in batches for progress tracking
    batch_size = max(1000, num_pairs // 100)  # Show progress every ~1%
    processed = 0
    
    for idx, (i, j) in enumerate(valid_pairs):
        cusip_1 = cusips[i]
        cusip_2 = cusips[j]
        
        # Get aligned values (handling NaN alignment)
        cusip_1_vals = values_array[:, i]
        cusip_2_vals = values_array[:, j]
        
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
    
    # Convert to DataFrame and sort by Z Score descending
    print("Sorting results...")
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
    results_df = results_df.sort_values("Z Score", ascending=False, na_position="last")
    
    # Ensure ASCII-safe names for all rows
    for column in ["Bond_1", "Bond_2"]:
        results_df[column] = results_df[column].map(ensure_ascii)
    
    # Write all rows to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "term_comb.csv"
    results_df.to_csv(output_path, index=False)
    
    # Display top N to console
    top_results = results_df.head(top_n).copy()
    print(f"\nTop {top_n} Pair Analytics (by Z Score):")
    print(top_results.to_string(index=False))
    print(f"\nCSV written to: {output_path} (all {len(results_df):,} rows)")
    
    return top_results


def main() -> None:
    """Entry point for running the term combinations pair analytics script."""
    run_analysis()


if __name__ == "__main__":
    main()

