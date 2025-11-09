"""
CR01 pair analytics script generated on 2025-11-09 00:00 ET.

This module reads `bond_data/parquet/bql.parquet`, filters the CR01 universe and
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
BQL_PARQUET_PATH = SCRIPT_DIR.parent / "bond_data" / "parquet" / "bql.parquet"
OUTPUT_DIR = SCRIPT_DIR / "processed_data"

CR01_HOLDINGS_RAW: List[str] = [
    "13607PXH2 Corp",
    "44810ZCS7 Corp",
    "29251ZCJ4 Corp",
    "06418YXB9 Corp",
    "83179XAL2 Corp",
    "13607HR79 Corp",
    "06418MM43 Corp",
    "766910BT9 Corp",
    "779926FY5 Corp",
    "63306AHT6 Corp",
    "83179XAH1 Corp",
    "25675TAP2 Corp",
    "07813ZCL6 Corp",
    "89116CST5 Corp",
    "780086XL3 Corp",
    "7800867G3 Corp",
    "89117FPG8 Corp",
    "87971MCC5 Corp",
    "13607PBA1 Corp",
    "89156VAC0 Corp",
    "387427AM9 Corp",
    "34527ACW8 Corp",
    "064164QM1 Corp",
    "26153WAJ8 Corp",
    "92938WAD5 Corp",
    "31430W3J1 Corp",
    "375916AA1 Corp",
    "759480AN6 Corp",
    "375916AC7 Corp",
    "16141AAG8 Corp",
    "759480AM8 Corp",
    "019456AK8 Corp",
    "667495AN5 Corp",
    "15135UAT6 Corp",
    "949746TJ0 Corp",
    "780086WG5 Corp",
    "019456AM4 Corp",
    "06369ZCL6 Corp",
    "55279QAE0 Corp",
    "891102AE5 Corp",
    "136765BX1 Corp",
    "12658MAD3 Corp",
    "02138ZAQ6 Corp",
    "63306AHF6 Corp",
    "16141AAF0 Corp",
    "775109BT7 Corp",
    "06415GDJ6 Corp",
    "89117GX51 Corp",
    "375916AE3 Corp",
    "89353ZCF3 Corp",
    "31430WU44 Corp",
    "11291ZAM9 Corp",
    "190330AQ3 Corp",
    "172967MJ7 Corp",
    "31943BBY5 Corp",
    "918423BJ2 Corp",
    "56501RAQ9 Corp",
    "86682ZAT3 Corp",
    "12658MAC5 Corp",
]

CR01_UNIVERSE_RAW: List[str] = [
    "70632ZAV3 Corp",
    "89353ZCQ9 Corp",
    "65339KDA5 Corp",
    "349553AR8 Corp",
    "86682ZAU0 Corp",
    "293365AH5 Corp",
    "17039AAX4 Corp",
    "775109DA6 Corp",
    "539481AP6 Corp",
    "116705AM6 Corp",
    "74340XCR0 Corp",
    "29251ZCH8 Corp",
    "136681AG8 Corp",
    "86682ZAS5 Corp",
    "06369ZCL6 Corp",
    "06369ZCK8 Corp",
    "13607PBA1 Corp",
    "01626PAS5 Corp",
    "29251ZCD7 Corp",
    "44810ZCS7 Corp",
    "663307AS5 Corp",
    "14046ZAR0 Corp",
    "670018AA8 Corp",
    "116705AQ7 Corp",
    "06418MZ49 Corp",
    "29251ZBW6 Corp",
    "779926HR8 Corp",
    "663307AJ5 Corp",
    "7800867G3 Corp",
    "89116CWY9 Corp",
    "11291ZAH0 Corp",
    "45075EAG9 Corp",
    "07813ZCK8 Corp",
    "11271ZAA9 Corp",
    "07813ZCL6 Corp",
    "609207BD6 Corp",
    "37482ZAF8 Corp",
    "663307AM8 Corp",
    "19046FAP7 Corp",
    "29251ZBZ9 Corp",
    "74340XCP4 Corp",
]


@dataclass
class PairSummary:
    """Container with CR01 pair analytics metrics."""

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
) -> pd.DataFrame:
    """
    Execute the CR01 pair analytics workflow.

    Args:
        bql_path: Path to the BQL parquet file.
        output_dir: Directory for CSV export.

    Returns:
        DataFrame containing all pair analytics.
    """
    universe_cusips = normalize_cusip_list(CR01_UNIVERSE_RAW)
    holdings_cusips = normalize_cusip_list(CR01_HOLDINGS_RAW)
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
    output_path = output_dir / "cr01_pair_analytics.csv"
    results.to_csv(output_path, index=False)

    top_display = results.head(80)
    print("CR01 Pair Analytics (Top 80 by Z Score):")
    print(top_display.to_string(index=False))
    print(f"\nCSV written to: {output_path}")
    return results


def main() -> None:
    """Entry point for running the CR01 pair analytics script."""
    run_analysis()


if __name__ == "__main__":
    main()

