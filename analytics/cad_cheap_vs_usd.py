"""
CAD cheap vs USD pair analytics script.

This module reads `bond_data/parquet/bql.parquet`, filters CUSIPs present in the most
recent dates, computes all pairwise spreads, filters to pairs where cusip_1 has
Currency="CAD" and cusip_2 has Currency="USD" with matching Ticker and Custom_Sector
values and absolute difference in Yrs (Cvn) <= 2 (from the last date in
historical_bond_details.parquet), and exports top 80 pairs sorted by Z Score for
monitoring relative value opportunities.
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


def get_currency_ticker_sector_mappings(historical_path: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, float]]:
    """
    Get Currency, Ticker, Custom_Sector, and Yrs (Cvn) mappings from the last date in historical_bond_details.parquet.

    Args:
        historical_path: Path to historical_bond_details.parquet file.

    Returns:
        Tuple of (currency_mapping, ticker_mapping, sector_mapping, yrs_cvn_mapping) dictionaries mapping CUSIP to Currency/Ticker/Custom_Sector/Yrs (Cvn).
        CUSIPs with missing/null Currency, Ticker, Custom_Sector, or Yrs (Cvn) are excluded.
    """
    print("Loading historical bond details to get Currency, Ticker, Custom_Sector, and Yrs (Cvn) mappings...")
    historical_df = pd.read_parquet(historical_path)
    
    # Get the last date
    last_date = historical_df["Date"].max()
    print(f"Last date in historical data: {last_date}")
    
    # Filter to last date
    last_date_df = historical_df[historical_df["Date"] == last_date].copy()
    
    # Filter out rows with missing Currency, Ticker, Custom_Sector, or Yrs (Cvn)
    valid_df = last_date_df[
        last_date_df["Currency"].notna() &
        last_date_df["Ticker"].notna() &
        last_date_df["Custom_Sector"].notna() &
        last_date_df["Yrs (Cvn)"].notna()
    ].copy()
    
    # Create mappings
    currency_mapping = dict(zip(valid_df["CUSIP"], valid_df["Currency"]))
    ticker_mapping = dict(zip(valid_df["CUSIP"], valid_df["Ticker"]))
    sector_mapping = dict(zip(valid_df["CUSIP"], valid_df["Custom_Sector"]))
    yrs_cvn_mapping = dict(zip(valid_df["CUSIP"], valid_df["Yrs (Cvn)"].astype(float)))
    
    print(
        f"Found {len(currency_mapping)} CUSIPs with valid Currency, Ticker, Custom_Sector, and Yrs (Cvn) "
        f"(excluded {len(last_date_df) - len(valid_df)} with missing values)"
    )
    
    return currency_mapping, ticker_mapping, sector_mapping, yrs_cvn_mapping


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
    Execute the CAD cheap vs USD pair analytics workflow.

    Args:
        bql_path: Path to the BQL parquet file.
        historical_path: Path to historical_bond_details.parquet file.
        output_dir: Directory for CSV export.
        top_n: Number of top pairs to return (default 80).

    Returns:
        DataFrame containing top N pair analytics filtered to CAD/USD pairs
        with matching Tickers, Custom_Sectors, and Yrs (Cvn) difference <= 2,
        sorted by Z Score.
    """
    print("Loading BQL data...")
    data = pd.read_parquet(bql_path)
    
    if data.empty:
        raise ValueError("No BQL data found")
    
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
    
    # Get Currency, Ticker, Custom_Sector, and Yrs (Cvn) mappings from historical data
    currency_mapping, ticker_mapping, sector_mapping, yrs_cvn_mapping = get_currency_ticker_sector_mappings(historical_path)
    
    # Filter to pairs where:
    # 1. cusip_1 has Currency="CAD"
    # 2. cusip_2 has Currency="USD"
    # 3. Both have the same Ticker value
    # 4. Both have the same Custom_Sector value
    # 5. Absolute difference in Yrs (Cvn) <= 2
    print("\nFiltering to CAD/USD pairs with matching Tickers, Custom_Sectors, and Yrs (Cvn) difference <= 2...")
    before_filter_count = len(results_df)
    
    # Filter to pairs where both CUSIPs exist in mappings
    results_df = results_df[
        results_df["cusip_1"].isin(currency_mapping.keys()) &
        results_df["cusip_2"].isin(currency_mapping.keys())
    ].copy()
    
    # Add Currency, Ticker, Custom_Sector, and Yrs (Cvn) columns for filtering
    results_df["currency_1"] = results_df["cusip_1"].map(currency_mapping)
    results_df["currency_2"] = results_df["cusip_2"].map(currency_mapping)
    results_df["ticker_1"] = results_df["cusip_1"].map(ticker_mapping)
    results_df["ticker_2"] = results_df["cusip_2"].map(ticker_mapping)
    results_df["sector_1"] = results_df["cusip_1"].map(sector_mapping)
    results_df["sector_2"] = results_df["cusip_2"].map(sector_mapping)
    results_df["yrs_cvn_1"] = results_df["cusip_1"].map(yrs_cvn_mapping)
    results_df["yrs_cvn_2"] = results_df["cusip_2"].map(yrs_cvn_mapping)
    
    # Apply filters
    results_df = results_df[
        (results_df["currency_1"] == "CAD") &
        (results_df["currency_2"] == "USD") &
        (results_df["ticker_1"] == results_df["ticker_2"]) &
        (results_df["sector_1"] == results_df["sector_2"]) &
        ((results_df["yrs_cvn_1"] - results_df["yrs_cvn_2"]).abs() <= 2.0)
    ].copy()
    
    # Drop temporary columns
    results_df = results_df.drop(columns=["currency_1", "currency_2", "ticker_1", "ticker_2", "sector_1", "sector_2", "yrs_cvn_1", "yrs_cvn_2"])
    
    after_filter_count = len(results_df)
    print(f"Filtered from {before_filter_count:,} to {after_filter_count:,} pairs")
    
    if results_df.empty:
        raise ValueError("No pairs remaining after CAD/USD/Ticker/Custom_Sector/Yrs (Cvn) filtering!")
    
    # Sort by Z Score descending, take top N
    print("Sorting results...")
    results_df = results_df.sort_values("Z Score", ascending=False, na_position="last")
    top_results = results_df.head(top_n).copy()
    
    # Ensure ASCII-safe names
    for column in ["Bond_1", "Bond_2"]:
        top_results[column] = top_results[column].map(ensure_ascii)
    
    # Write to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cad_cheap_vs_usd.csv"
    top_results.to_csv(output_path, index=False)
    
    print(f"\nTop {len(top_results)} CAD vs USD Pair Analytics (by Z Score):")
    print(top_results.to_string(index=False))
    print(f"\nCSV written to: {output_path}")
    
    return top_results


def main() -> None:
    """Entry point for running the CAD cheap vs USD pair analytics script."""
    run_analysis()


if __name__ == "__main__":
    main()

