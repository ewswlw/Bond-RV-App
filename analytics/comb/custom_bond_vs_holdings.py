"""
Custom bond vs holdings pair analytics script.

This module reads `bond_data/parquet/bql.parquet`, filters cusip_1 to CAD CUSIPs and finds
target bond(s) via TARGET_BOND_TICKER/SECURITY, pairs each with holdings (cusip_2) that
have CR01 @ Tight Bid > 2000 matched with portfolio, computes pairwise spreads (cusip_1 - cusip_2),
and exports all pairs sorted by Z Score to CSV. The top N pairs are displayed to console
for monitoring relative value opportunities.

Configuration:
    - TARGET_BOND_TICKER: Ticker of target bond(s) - takes precedence over Security (default: None)
    - TARGET_BOND_SECURITY: Security name of target bond (default: None)
    - RECENT_DATE_PERCENT: Percentage of recent dates to filter (default: 0.75)
    - TOP_N_PAIRS: Number of top pairs to display (default: 80)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

# Target bond Ticker to find in universe.parquet (exact match, case-sensitive)
# If provided, takes precedence over TARGET_BOND_SECURITY
# Set to None to use Security name instead
# Example: "TCN" will find all bonds with Ticker="TCN"
TARGET_BOND_TICKER = "TCN"  # Set to "TCN" to use Ticker-based search

# Target bond Security name to find in universe.parquet
# Used only if TARGET_BOND_TICKER is None
# Change this to analyze pairs against a different bond
TARGET_BOND_SECURITY = None

# Filter to CUSIPs present in most recent X% of dates (default: 0.75 = 75%)
RECENT_DATE_PERCENT = 0.75

# Number of top pairs to display in console (default: 80)
TOP_N_PAIRS = 80

# ============================================================================
# PATHS (auto-configured based on script location)
# ============================================================================

# Get script directory and build paths relative to it
SCRIPT_DIR = Path(__file__).parent.resolve()
BQL_PARQUET_PATH = SCRIPT_DIR.parent.parent / "bond_data" / "parquet" / "bql.parquet"
HISTORICAL_PARQUET_PATH = SCRIPT_DIR.parent.parent / "bond_data" / "parquet" / "historical_bond_details.parquet"
UNIVERSE_PARQUET_PATH = SCRIPT_DIR.parent.parent / "bond_data" / "parquet" / "universe.parquet"
RUNS_TODAY_CSV_PATH = SCRIPT_DIR.parent / "processed_data" / "runs_today.csv"
PORTFOLIO_PARQUET_PATH = SCRIPT_DIR.parent.parent / "bond_data" / "parquet" / "historical_portfolio.parquet"
OUTPUT_DIR = SCRIPT_DIR.parent / "processed_data"


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


def normalize_cusip(cusip: str) -> str:
    """
    Normalize a CUSIP to 9-character uppercase format.
    
    Handles CUSIPs that may already be normalized or may have trailing descriptors.
    
    Args:
        cusip: CUSIP string to normalize.
    
    Returns:
        Normalized 9-character uppercase CUSIP.
    """
    if pd.isna(cusip) or cusip == '':
        return ''
    cusip_str = str(cusip).strip().upper()
    # Remove trailing " CORP" if present
    cusip_str = cusip_str.replace(" CORP", "")
    # Take first 9 characters
    return cusip_str[:9]


def find_target_bond_cusips(
    universe_path: Path,
    cad_cusips: Set[str],
    ticker: Optional[str] = None,
    security_name: Optional[str] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Find CUSIP(s) for target bond(s) by searching Ticker or Security name in universe.parquet,
    filtered to CAD CUSIPs only.
    
    If ticker is provided, searches by Ticker (exact match, case-sensitive).
    If ticker not found, falls back to security_name if provided.
    Returns all matching CUSIPs if multiple bonds share the same Ticker.
    Only includes CUSIPs that are in the CAD CUSIPs set.

    Args:
        universe_path: Path to universe.parquet file.
        cad_cusips: Set of CAD CUSIPs to filter results.
        ticker: Ticker to search for (exact match, case-sensitive). Takes precedence.
        security_name: Security name to search for (fallback if ticker not found).

    Returns:
        Tuple of (list of CUSIP strings, mapping of CUSIP to Security name).

    Raises:
        ValueError: If no bond found and no fallback available.
    """
    print("Loading universe.parquet to find target bond(s)...")
    universe_df = pd.read_parquet(universe_path)
    
    # Filter to CAD CUSIPs first
    universe_df = universe_df[universe_df["CUSIP"].isin(cad_cusips)].copy()
    
    matches = None
    search_type = None
    
    # Try Ticker first if provided
    if ticker is not None and ticker.strip():
        print(f"Searching by Ticker: '{ticker}' (exact match, case-sensitive) in CAD CUSIPs...")
        matches = universe_df[universe_df["Ticker"] == ticker].copy()
        search_type = "Ticker"
        
        if len(matches) == 0:
            print(f"WARNING: No CAD bonds found with Ticker '{ticker}'")
            if security_name is not None and security_name.strip():
                print(f"Falling back to Security name: '{security_name}'")
                matches = universe_df[universe_df["Security"] == security_name].copy()
                search_type = "Security"
        else:
            # Log each match
            print(f"Found {len(matches)} CAD bond(s) with Ticker '{ticker}':")
            for idx, row in matches.iterrows():
                print(f"  CUSIP: {row['CUSIP']}, Security: {row['Security']}, Ticker: {row['Ticker']}")
    
    # Try Security if Ticker not used or not found
    elif security_name is not None and security_name.strip():
        print(f"Searching by Security name: '{security_name}' in CAD CUSIPs...")
        matches = universe_df[universe_df["Security"] == security_name].copy()
        search_type = "Security"
    
    # No search criteria provided
    if matches is None or len(matches) == 0:
        error_msg = "No target bond specified. Provide either TARGET_BOND_TICKER or TARGET_BOND_SECURITY."
        if ticker is not None and ticker.strip():
            error_msg = f"No CAD bonds found with Ticker '{ticker}'"
            if security_name is not None and security_name.strip():
                error_msg += f" or Security '{security_name}'"
        raise ValueError(error_msg)
    
    # Extract CUSIPs and Security names
    target_cusips = [str(cusip) for cusip in matches["CUSIP"].tolist()]
    cusip_to_security = dict(zip(matches["CUSIP"], matches["Security"]))
    
    print(f"Found {len(target_cusips)} target CAD bond(s): {', '.join(target_cusips)}")
    
    return target_cusips, cusip_to_security


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


def load_cr01_holdings(
    runs_today_path: Path = RUNS_TODAY_CSV_PATH,
    portfolio_path: Path = PORTFOLIO_PARQUET_PATH,
) -> List[str]:
    """
    Load CR01 holdings CUSIPs dynamically from data sources.
    
    Filters runs_today.csv for CUSIPs where:
    - CR01 @ Tight Bid > 2000
    
    Then matches these CUSIPs with CUSIPs in historical_portfolio.parquet
    (last date only, unique CUSIPs from any account/portfolio).
    
    Args:
        runs_today_path: Path to runs_today.csv file.
        portfolio_path: Path to historical_portfolio.parquet file.
    
    Returns:
        Sorted list of normalized CUSIPs matching the criteria.
    
    Raises:
        FileNotFoundError: If required data files don't exist.
        ValueError: If data files are empty or no matching CUSIPs found.
    """
    # Read runs_today.csv
    if not runs_today_path.exists():
        raise FileNotFoundError(f"runs_today.csv not found at: {runs_today_path}")
    
    runs_df = pd.read_csv(runs_today_path)
    
    if runs_df.empty:
        raise ValueError(f"runs_today.csv is empty at: {runs_today_path}")
    
    # Check required columns exist
    required_cols = ["CUSIP", "CR01 @ Tight Bid"]
    missing_cols = [col for col in required_cols if col not in runs_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in runs_today.csv: {missing_cols}")
    
    # Convert numeric columns, excluding rows with NaN/null
    runs_df = runs_df.copy()
    runs_df["CR01 @ Tight Bid"] = pd.to_numeric(runs_df["CR01 @ Tight Bid"], errors="coerce")
    
    # Filter: CR01 @ Tight Bid > 2000
    # Exclude rows with NaN/null values
    filtered_runs = runs_df[
        (runs_df["CR01 @ Tight Bid"] > 2000) &
        runs_df["CR01 @ Tight Bid"].notna()
    ].copy()
    
    if filtered_runs.empty:
        raise ValueError(
            "No CUSIPs found in runs_today.csv matching criteria: "
            "CR01 @ Tight Bid > 2000"
        )
    
    # Extract and normalize CUSIPs from runs_today
    runs_cusips = filtered_runs["CUSIP"].dropna().unique()
    normalized_runs_cusips = {normalize_cusip(c) for c in runs_cusips if normalize_cusip(c)}
    
    if not normalized_runs_cusips:
        raise ValueError("No valid CUSIPs found after normalization from runs_today.csv")
    
    # Read historical_portfolio.parquet
    if not portfolio_path.exists():
        raise FileNotFoundError(f"historical_portfolio.parquet not found at: {portfolio_path}")
    
    portfolio_df = pd.read_parquet(portfolio_path)
    
    if portfolio_df.empty:
        raise ValueError(f"historical_portfolio.parquet is empty at: {portfolio_path}")
    
    # Check required columns exist
    if "Date" not in portfolio_df.columns or "CUSIP" not in portfolio_df.columns:
        raise ValueError("Missing required columns (Date, CUSIP) in historical_portfolio.parquet")
    
    # Get last date
    last_date = portfolio_df["Date"].max()
    last_date_df = portfolio_df[portfolio_df["Date"] == last_date].copy()
    
    if last_date_df.empty:
        raise ValueError(f"No data found for last date ({last_date}) in historical_portfolio.parquet")
    
    # Get unique CUSIPs from last date (any account/portfolio)
    portfolio_cusips = last_date_df["CUSIP"].dropna().unique()
    normalized_portfolio_cusips = {normalize_cusip(c) for c in portfolio_cusips if normalize_cusip(c)}
    
    if not normalized_portfolio_cusips:
        raise ValueError(f"No valid CUSIPs found in historical_portfolio.parquet for last date ({last_date})")
    
    # Match CUSIPs: intersection of runs_today filtered CUSIPs and portfolio CUSIPs
    matched_cusips = normalized_runs_cusips & normalized_portfolio_cusips
    
    if not matched_cusips:
        raise ValueError(
            f"No matching CUSIPs found between runs_today.csv (filtered) and "
            f"historical_portfolio.parquet (last date: {last_date})"
        )
    
    return sorted(matched_cusips)


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
    # Compute spreads (cusip_1 - cusip_2)
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
    universe_path: Path = UNIVERSE_PARQUET_PATH,
    runs_today_path: Path = RUNS_TODAY_CSV_PATH,
    portfolio_path: Path = PORTFOLIO_PARQUET_PATH,
    output_dir: Path = OUTPUT_DIR,
    top_n: int = TOP_N_PAIRS,
    target_ticker: Optional[str] = TARGET_BOND_TICKER,
    target_security: Optional[str] = TARGET_BOND_SECURITY,
) -> pd.DataFrame:
    """
    Execute the custom bond vs holdings pair analytics workflow.

    Args:
        bql_path: Path to the BQL parquet file.
        historical_path: Path to historical_bond_details.parquet file.
        universe_path: Path to universe.parquet file.
        runs_today_path: Path to runs_today.csv file.
        portfolio_path: Path to historical_portfolio.parquet file.
        output_dir: Directory for CSV export.
        top_n: Number of top pairs to return (default 80).
        target_ticker: Ticker of target bond(s) - takes precedence (default None).
        target_security: Security name of target bond (default None).

    Returns:
        DataFrame containing top N pair analytics sorted by Z Score.
    """
    # Get CAD CUSIPs from historical data
    cad_cusips = get_cad_cusips(historical_path)
    
    if not cad_cusips:
        raise ValueError("No CAD CUSIPs found in historical data")
    
    # Find target bond CUSIP(s) from universe.parquet (filtered to CAD CUSIPs)
    target_cusips, cusip_to_security = find_target_bond_cusips(
        universe_path, cad_cusips=cad_cusips, ticker=target_ticker, security_name=target_security
    )
    
    # Load CR01 holdings CUSIPs
    print("Loading CR01 holdings CUSIPs from data sources...")
    holdings_cusips_raw = load_cr01_holdings(runs_today_path, portfolio_path)
    print(f"Found {len(holdings_cusips_raw)} CR01 holdings CUSIPs")
    
    # Load BQL data
    print("Loading BQL data...")
    bql_data = pd.read_parquet(bql_path)
    
    # Check that target bond(s) have BQL data
    target_bql_data = bql_data[bql_data["CUSIP"].isin(target_cusips)].copy()
    if target_bql_data.empty:
        raise ValueError(f"Target bond(s) (CUSIPs {', '.join(target_cusips)}) have no BQL data")
    
    target_cusips_with_data = target_bql_data["CUSIP"].unique().tolist()
    if len(target_cusips_with_data) < len(target_cusips):
        missing = set(target_cusips) - set(target_cusips_with_data)
        print(f"WARNING: {len(missing)} target bond(s) have no BQL data: {', '.join(missing)}")
        target_cusips = target_cusips_with_data
    
    print(f"Target bond(s) have {len(target_bql_data)} BQL data points across {target_bql_data['Date'].nunique()} dates")
    
    # Filter BQL data to target bonds and holdings CUSIPs
    combined_cusips = set(target_cusips) | set(holdings_cusips_raw)
    print(f"Filtering BQL data to {len(combined_cusips)} CUSIPs (target bonds + holdings)...")
    filtered_bql = bql_data[bql_data["CUSIP"].isin(combined_cusips)].copy()
    
    if filtered_bql.empty:
        raise ValueError("No BQL data found for target bonds and holdings")
    
    # Ensure all columns have complete data - drop any rows with missing values
    print("Filtering for complete data...")
    filtered_bql = filtered_bql.dropna()
    
    if filtered_bql.empty:
        raise ValueError("No data remaining after filtering for complete rows")
    
    print(f"Total BQL rows: {len(filtered_bql)}, Unique CUSIPs: {filtered_bql['CUSIP'].nunique()}, Unique Dates: {filtered_bql['Date'].nunique()}")
    
    # Filter to CUSIPs present in most recent dates (for both target bonds and holdings)
    print(f"Filtering to CUSIPs present in most recent {RECENT_DATE_PERCENT*100:.0f}% of dates...")
    filtered_bql = filter_recent_cusips(filtered_bql, RECENT_DATE_PERCENT)
    
    if filtered_bql.empty:
        raise ValueError("No data remaining after filtering for recent CUSIPs")
    
    print(f"After filtering: {len(filtered_bql)} rows, {filtered_bql['CUSIP'].nunique()} CUSIPs")
    
    # Verify target bond(s) are still in filtered data
    target_cusips_in_data = [c for c in target_cusips if c in filtered_bql["CUSIP"].values]
    if not target_cusips_in_data:
        raise ValueError(f"Target bond(s) (CUSIPs {', '.join(target_cusips)}) not present in recent dates")
    if len(target_cusips_in_data) < len(target_cusips):
        missing = set(target_cusips) - set(target_cusips_in_data)
        print(f"WARNING: {len(missing)} target bond(s) not in recent dates: {', '.join(missing)}")
        target_cusips = target_cusips_in_data
    
    # Filter holdings to those present in recent BQL data
    holdings_cusips = [c for c in holdings_cusips_raw if c in filtered_bql["CUSIP"].values]
    if not holdings_cusips:
        raise ValueError("No holdings CUSIPs found in recent BQL data")
    
    print(f"Found {len(holdings_cusips)} holdings CUSIPs in recent BQL data")
    
    # Build name lookup
    print("Building name lookup...")
    name_lookup = build_name_lookup(filtered_bql)
    
    # Create pivoted table (Date × CUSIP)
    print("Creating pivoted table...")
    wide_values = filtered_bql.pivot_table(
        index="Date",
        columns="CUSIP",
        values="Value",
        aggfunc="last",
    ).sort_index()
    
    # Verify target bonds are in pivoted table
    missing_targets = [c for c in target_cusips if c not in wide_values.columns]
    if missing_targets:
        raise ValueError(f"Target bond(s) (CUSIPs {', '.join(missing_targets)}) not found in pivoted table")
    
    # Verify holdings are in pivoted table
    holdings_cusips = [c for c in holdings_cusips if c in wide_values.columns]
    if not holdings_cusips:
        raise ValueError("No holdings CUSIPs found in pivoted table")
    
    print(f"Creating pairs: {len(target_cusips)} target bond(s) paired with {len(holdings_cusips)} holdings...")
    
    # Convert to numpy array for faster operations
    values_array = wide_values.values  # Shape: (num_dates, num_cusips)
    cusip_to_index = {cusip: i for i, cusip in enumerate(wide_values.columns)}
    
    # Pre-allocate results list
    summaries: List[PairSummary] = []
    
    # Process each target bond paired with each holding
    for target_cusip in target_cusips:
        target_idx = cusip_to_index[target_cusip]
        target_values = values_array[:, target_idx]  # Shape: (num_dates,)
        
        # Get target bond Security name for display
        target_security_name = cusip_to_security.get(target_cusip, "")
        target_bond_name = name_lookup.get(target_cusip, target_security_name)
        
        # Process each holding paired with this target bond
        for holdings_cusip in holdings_cusips:
            holdings_idx = cusip_to_index[holdings_cusip]
            holdings_values = values_array[:, holdings_idx]  # Shape: (num_dates,)
            
            # Find dates where both have values
            both_valid = ~(np.isnan(target_values) | np.isnan(holdings_values))
            
            if both_valid.sum() < 2:  # Need at least 2 overlapping dates
                continue
            
            # Extract valid values
            valid_1 = target_values[both_valid]
            valid_2 = holdings_values[both_valid]
            
            # Compute statistics (cusip_1 - cusip_2)
            stats = compute_pair_stats_vectorized(valid_1, valid_2)
            if stats is None:
                continue
            
            last_value, average_value, vs_average, z_score, percentile = stats
            
            # Only include pairs with valid Z scores (for sorting)
            if z_score is None:
                continue
            
            summaries.append(
                PairSummary(
                    Bond_1=target_bond_name,
                    Bond_2=name_lookup.get(holdings_cusip, holdings_cusip),
                    last_value=last_value,
                    average_value=average_value,
                    vs_average=vs_average,
                    z_score=z_score,
                    percentile=percentile,
                    cusip_1=target_cusip,
                    cusip_2=holdings_cusip,
                )
            )
    
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
    output_path = output_dir / "custom_bond_vs_holdings.csv"
    results_df.to_csv(output_path, index=False)
    
    # Display top N to console
    top_results = results_df.head(top_n).copy()
    print(f"\nTop {top_n} Pair Analytics (by Z Score):")
    print(top_results.to_string(index=False))
    print(f"\nCSV written to: {output_path} (all {len(results_df):,} rows)")
    
    return top_results


def main() -> None:
    """Entry point for running the custom bond vs holdings pair analytics script."""
    run_analysis(
        target_ticker=TARGET_BOND_TICKER,
        target_security=TARGET_BOND_SECURITY,
    )


if __name__ == "__main__":
    main()

