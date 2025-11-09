"""
All combinations pair analytics script.

This module reads `bond_data/parquet/bql.parquet`, filters CUSIPs present in the most
recent dates, computes all pairwise spreads, and exports top 80 pairs sorted by
Z Score for monitoring relative value opportunities.
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
BQL_PARQUET_PATH = SCRIPT_DIR.parent / "bond_data" / "parquet" / "bql.parquet"
HISTORICAL_PARQUET_PATH = SCRIPT_DIR.parent / "bond_data" / "parquet" / "historical_bond_details.parquet"
OUTPUT_DIR = SCRIPT_DIR / "processed_data"

# Filter to CUSIPs present in most recent 75% of dates
RECENT_DATE_PERCENT = 0.75


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
) -> pd.DataFrame:
    """
    Execute the all combinations pair analytics workflow.

    Args:
        bql_path: Path to the BQL parquet file.
        historical_path: Path to historical_bond_details.parquet file.
        output_dir: Directory for CSV export.
        top_n: Number of top pairs to return (default 80).

    Returns:
        DataFrame containing top N pair analytics sorted by Z Score.
    """
    # First, get CAD CUSIPs from historical data
    cad_cusips = get_cad_cusips(historical_path)
    
    if not cad_cusips:
        raise ValueError("No CAD CUSIPs found in historical data")
    
    print("Loading BQL data...")
    data = pd.read_parquet(bql_path)
    
    # Filter to only CAD CUSIPs
    print(f"Filtering BQL data to {len(cad_cusips)} CAD CUSIPs...")
    data = data[data["CUSIP"].isin(cad_cusips)].copy()
    
    if data.empty:
        raise ValueError("No BQL data found for CAD CUSIPs")
    
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
    num_cusips = len(cusips)
    num_pairs = num_cusips * (num_cusips - 1) // 2
    
    print(f"Computing {num_pairs:,} pairwise combinations...")
    
    # Convert to numpy array for faster operations
    values_array = wide_values.values  # Shape: (num_dates, num_cusips)
    dates_index = wide_values.index
    
    # Pre-allocate results list
    summaries: List[PairSummary] = []
    
    # Process pairs in batches for progress tracking
    batch_size = max(1000, num_pairs // 100)  # Show progress every ~1%
    processed = 0
    
    for idx, (i, j) in enumerate(combinations(range(num_cusips), 2)):
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
    
    # Sort by Z Score descending, take top N
    results_df = results_df.sort_values("Z Score", ascending=False, na_position="last")
    top_results = results_df.head(top_n).copy()
    
    # Ensure ASCII-safe names
    for column in ["Bond_1", "Bond_2"]:
        top_results[column] = top_results[column].map(ensure_ascii)
    
    # Write to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_comb.csv"
    top_results.to_csv(output_path, index=False)
    
    print(f"\nTop {top_n} Pair Analytics (by Z Score):")
    print(top_results.to_string(index=False))
    print(f"\nCSV written to: {output_path}")
    
    return top_results


def main() -> None:
    """Entry point for running the all combinations pair analytics script."""
    run_analysis()


if __name__ == "__main__":
    main()

