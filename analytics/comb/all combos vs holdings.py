"""
All combinations vs holdings pair analytics script.

This module dynamically loads universe and holdings CUSIPs from:
- historical_portfolio.parquet (last date only, all unique CUSIPs)

Then reads `bond_data/parquet/bql.parquet`, filters the universe and
holdings CUSIPs, computes all pairwise spreads (universe minus holdings), and
exports summary statistics for monitoring relative value opportunities.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# Get script directory and build paths relative to it
SCRIPT_DIR = Path(__file__).parent.resolve()
BQL_PARQUET_PATH = SCRIPT_DIR.parent.parent / "bond_data" / "parquet" / "bql.parquet"
PORTFOLIO_PARQUET_PATH = SCRIPT_DIR.parent.parent / "bond_data" / "parquet" / "historical_portfolio.parquet"
OUTPUT_DIR = SCRIPT_DIR.parent / "processed_data"


@dataclass
class PairSummary:
    """Container with pair analytics metrics."""

    universe_name: str
    holdings_name: str
    last_value: float
    average_value: float
    vs_average: float
    z_score: Optional[float]
    percentile: float
    universe_cusip: str
    holdings_cusip: str


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


def clean_cusip(raw: str) -> str:
    """
    Strip the trailing descriptor and return the normalized 9-character CUSIP.

    Args:
        raw: Raw CUSIP string that may contain trailing descriptors like " Corp".

    Returns:
        A cleaned 9-character uppercase CUSIP.
    """
    return raw.strip().upper().replace(" CORP", "")[:9]


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


def load_holdings(
    portfolio_path: Path = PORTFOLIO_PARQUET_PATH,
) -> List[str]:
    """
    Load holdings CUSIPs from historical_portfolio.parquet.
    
    Gets all unique CUSIPs from the last date (any account/portfolio).
    
    Args:
        portfolio_path: Path to historical_portfolio.parquet file.
    
    Returns:
        Sorted list of normalized CUSIPs from portfolio holdings.
    
    Raises:
        FileNotFoundError: If portfolio file doesn't exist.
        ValueError: If portfolio file is empty or no CUSIPs found.
    """
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
    
    return sorted(normalized_portfolio_cusips)


def load_universe(
    portfolio_path: Path = PORTFOLIO_PARQUET_PATH,
) -> List[str]:
    """
    Load universe CUSIPs from historical_portfolio.parquet.
    
    Gets all unique CUSIPs from the last date (any account/portfolio).
    
    Args:
        portfolio_path: Path to historical_portfolio.parquet file.
    
    Returns:
        Sorted list of normalized CUSIPs from portfolio universe.
    
    Raises:
        FileNotFoundError: If portfolio file doesn't exist.
        ValueError: If portfolio file is empty or no CUSIPs found.
    """
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
    
    return sorted(normalized_portfolio_cusips)


def normalize_cusip_list(cusips: Iterable[str]) -> List[str]:
    """
    Normalize and deduplicate a collection of raw CUSIP strings.

    Args:
        cusips: Iterable of raw CUSIP strings with potential duplicates.

    Returns:
        Sorted list of unique normalized CUSIPs.
    """
    cleaned = {clean_cusip(item) for item in cusips}
    return sorted(cleaned)


def build_name_lookup(data: pd.DataFrame) -> Dict[str, str]:
    """
    Generate a mapping from CUSIP to security name.

    Args:
        data: Filtered BQL DataFrame containing `CUSIP` and `Name`.

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


def compute_pair_summary(
    wide_values: pd.DataFrame,
    universe_cusip: str,
    holdings_cusip: str,
    name_lookup: Dict[str, str],
) -> Optional[PairSummary]:
    """
    Compute summary statistics for a single universe/holdings pair.

    Args:
        wide_values: Pivoted Date × CUSIP table of spreads.
        universe_cusip: Universe side of the pair.
        holdings_cusip: Holdings side of the pair.
        name_lookup: Mapping of CUSIP to security name.

    Returns:
        PairSummary with analytics, or None if the pair has no overlapping data.
    """
    if universe_cusip not in wide_values or holdings_cusip not in wide_values:
        return None

    pair_frame = wide_values[[universe_cusip, holdings_cusip]].dropna()
    if pair_frame.empty:
        return None

    # Handle case where universe_cusip == holdings_cusip (spread will be all zeros)
    if universe_cusip == holdings_cusip:
        spreads = pd.Series(0.0, index=pair_frame.index)
    else:
        spreads = pair_frame[universe_cusip] - pair_frame[holdings_cusip]
    
    # Ensure scalar extraction
    last_value = float(spreads.iloc[-1]) if len(spreads) > 0 else 0.0
    average_value = float(spreads.mean())
    vs_average = last_value - average_value
    spread_array = spreads.to_numpy(dtype=float, copy=False)
    if spread_array.size > 1:
        spread_std = float(np.std(spread_array, ddof=1))
    else:
        spread_std = math.nan
    if not math.isnan(spread_std) and spread_std > 0:
        z_score = vs_average / spread_std
    else:
        z_score = None
    percentile = float(spreads.rank(pct=True).iloc[-1] * 100.0)

    return PairSummary(
        universe_name=name_lookup.get(universe_cusip, universe_cusip),
        holdings_name=name_lookup.get(holdings_cusip, holdings_cusip),
        last_value=last_value,
        average_value=average_value,
        vs_average=vs_average,
        z_score=z_score,
        percentile=percentile,
        universe_cusip=universe_cusip,
        holdings_cusip=holdings_cusip,
    )


def format_results(pairs: List[PairSummary]) -> pd.DataFrame:
    """
    Convert a list of PairSummary objects into a sorted DataFrame.

    Args:
        pairs: List of PairSummary objects.

    Returns:
        DataFrame sorted by descending Z Score with formatted columns.
    """
    df = pd.DataFrame(
        [
            {
                "universe_name": pair.universe_name,
                "holdings_name": pair.holdings_name,
                "Last": pair.last_value,
                "Avg": pair.average_value,
                "vs Avg": pair.vs_average,
                "Z Score": pair.z_score,
                "Percentile": pair.percentile,
                "universe_cusip": pair.universe_cusip,
                "holdings_cusip": pair.holdings_cusip,
            }
            for pair in pairs
        ]
    )
    return df.sort_values("Z Score", ascending=False, na_position="last")


def run_analysis(
    bql_path: Path = BQL_PARQUET_PATH,
    output_dir: Path = OUTPUT_DIR,
    portfolio_path: Path = PORTFOLIO_PARQUET_PATH,
) -> pd.DataFrame:
    """
    Execute the all combinations vs holdings pair analytics workflow.

    Args:
        bql_path: Path to the BQL parquet file.
        output_dir: Directory for CSV export.
        portfolio_path: Path to historical_portfolio.parquet file.

    Returns:
        DataFrame containing all pair analytics.
    """
    # Load CUSIPs dynamically from data sources
    print("Loading holdings CUSIPs from portfolio...")
    holdings_cusips_raw = load_holdings(portfolio_path)
    print(f"Found {len(holdings_cusips_raw)} holdings CUSIPs")
    
    print("Loading universe CUSIPs from portfolio...")
    universe_cusips_raw = load_universe(portfolio_path)
    print(f"Found {len(universe_cusips_raw)} universe CUSIPs")
    
    # Normalize CUSIPs (already normalized, but ensure consistency)
    universe_cusips = normalize_cusip_list(universe_cusips_raw)
    holdings_cusips = normalize_cusip_list(holdings_cusips_raw)
    combined_cusips = sorted(set(universe_cusips) | set(holdings_cusips))

    data = pd.read_parquet(bql_path)
    filtered = data[data["CUSIP"].isin(combined_cusips)].copy()
    filtered = filtered.dropna(subset=["Value"])

    name_lookup = build_name_lookup(filtered)
    wide_values = filtered.pivot_table(
        index="Date",
        columns="CUSIP",
        values="Value",
        aggfunc="last",
    ).sort_index()

    summaries: List[PairSummary] = []
    for universe_cusip, holdings_cusip in product(universe_cusips, holdings_cusips):
        summary = compute_pair_summary(
            wide_values=wide_values,
            universe_cusip=universe_cusip,
            holdings_cusip=holdings_cusip,
            name_lookup=name_lookup,
        )
        if summary is not None:
            summaries.append(summary)

    results = format_results(summaries)
    for column in ["universe_name", "holdings_name"]:
        results[column] = results[column].map(ensure_ascii)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all combos vs holdings.csv"
    results.to_csv(output_path, index=False)

    top_display = results.head(80)
    print("All Combinations vs Holdings Pair Analytics (Top 80 by Z Score):")
    print(top_display.to_string(index=False))
    print(f"\nCSV written to: {output_path}")
    return results


def main() -> None:
    """Entry point for running the all combinations vs holdings pair analytics script."""
    run_analysis()


if __name__ == "__main__":
    main()

