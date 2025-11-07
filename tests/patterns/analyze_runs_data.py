"""
Deep data analysis script for RUNS Excel files.
Operates exclusively in patterns/ folder for data engineering analysis.

Created: January 2025
Purpose: Analyze Historical Runs data structure, quality, and patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime

# Path to Historical Runs folder
RUNS_DIR = Path(r"C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Support Files\Historical Runs")

def analyze_file(filename: Path) -> dict:
    """Analyze a single RUNS Excel file."""
    df = pd.read_excel(filename)
    
    # Basic stats
    stats = {
        'filename': filename.name,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'unique_dates': df['Date'].nunique(),
        'unique_times': df['Time'].nunique(),
        'unique_dealers': df['Dealer'].nunique(),
        'unique_cusips': df['CUSIP'].nunique(),
        'unique_securities': df['Security'].nunique() if 'Security' in df.columns else 0,
    }
    
    # Check for duplicate Date+Dealer+CUSIP combinations
    duplicates = df.duplicated(subset=['Date', 'Dealer', 'CUSIP'], keep=False)
    stats['duplicate_date_dealer_cusip'] = duplicates.sum()
    
    # Check Date+Time uniqueness
    duplicate_datetime = df.duplicated(subset=['Date', 'Time', 'Dealer', 'CUSIP'], keep=False)
    stats['duplicate_datetime_dealer_cusip'] = duplicate_datetime.sum()
    
    # Data types
    stats['date_dtype'] = str(df['Date'].dtype)
    stats['time_dtype'] = str(df['Time'].dtype)
    
    # Sample values
    stats['date_sample'] = df['Date'].iloc[0] if len(df) > 0 else None
    stats['time_sample'] = df['Time'].iloc[0] if len(df) > 0 else None
    stats['dealer_list'] = sorted(df['Dealer'].unique().tolist())
    
    # Missing values
    stats['missing_dates'] = df['Date'].isna().sum()
    stats['missing_times'] = df['Time'].isna().sum()
    stats['missing_dealers'] = df['Dealer'].isna().sum()
    stats['missing_cusips'] = df['CUSIP'].isna().sum()
    
    return stats, df

def analyze_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Find and analyze duplicate Date+Dealer+CUSIP combinations."""
    dupes = df[df.duplicated(subset=['Date', 'Dealer', 'CUSIP'], keep=False)].copy()
    
    if len(dupes) == 0:
        return pd.DataFrame()
    
    # Sort by Date, Dealer, CUSIP, Time
    dupes = dupes.sort_values(['Date', 'Dealer', 'CUSIP', 'Time'])
    
    return dupes

def main():
    """Main analysis function."""
    print("=" * 100)
    print("DEEP DATA ANALYSIS: Historical Runs Excel Files")
    print("=" * 100)
    print()
    
    # Get all RUNS files
    files = sorted(RUNS_DIR.glob("RUNS *.xlsx"))
    print(f"Found {len(files)} RUNS files")
    print()
    
    # Analyze each file
    all_stats = []
    all_dupes = []
    
    for file in files:
        print(f"Analyzing: {file.name}")
        stats, df = analyze_file(file)
        all_stats.append(stats)
        
        # Analyze duplicates
        dupes = analyze_duplicates(df)
        if len(dupes) > 0:
            print(f"  WARNING: Found {len(dupes)} rows with duplicate Date+Dealer+CUSIP")
            all_dupes.append((file.name, dupes))
    
    print()
    print("=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    print()
    
    # Overall statistics
    stats_df = pd.DataFrame(all_stats)
    
    print(f"Total files analyzed: {len(stats_df)}")
    print(f"Total rows across all files: {stats_df['total_rows'].sum():,}")
    print(f"Average rows per file: {stats_df['total_rows'].mean():.0f}")
    print(f"Total unique CUSIPs across all files: {len(set().union(*[pd.read_excel(f)['CUSIP'].unique() for f in files])):,}")
    print()
    
    print("File-by-file breakdown:")
    print("-" * 100)
    for _, row in stats_df.iterrows():
        print(f"{row['filename']:30s} | Rows: {row['total_rows']:6d} | "
              f"CUSIPs: {row['unique_cusips']:5d} | Dealers: {row['unique_dealers']:2d} | "
              f"Duplicates: {row['duplicate_date_dealer_cusip']:4d}")
    
    print()
    print("=" * 100)
    print("DATA QUALITY ISSUES")
    print("=" * 100)
    print()
    
    # Check for files with duplicates
    files_with_dupes = stats_df[stats_df['duplicate_date_dealer_cusip'] > 0]
    if len(files_with_dupes) > 0:
        print(f"Files with duplicate Date+Dealer+CUSIP combinations: {len(files_with_dupes)}")
        for _, row in files_with_dupes.iterrows():
            print(f"  - {row['filename']}: {row['duplicate_date_dealer_cusip']} duplicate rows")
    else:
        print("OK: No duplicate Date+Dealer+CUSIP combinations found")
    
    print()
    print("Missing values:")
    missing_summary = stats_df[['filename', 'missing_dates', 'missing_times', 'missing_dealers', 'missing_cusips']]
    missing_summary = missing_summary[missing_summary[['missing_dates', 'missing_times', 'missing_dealers', 'missing_cusips']].sum(axis=1) > 0]
    if len(missing_summary) > 0:
        print(missing_summary.to_string(index=False))
    else:
        print("OK: No missing values in Date, Time, Dealer, or CUSIP columns")
    
    print()
    print("=" * 100)
    print("SAMPLE DUPLICATE ANALYSIS")
    print("=" * 100)
    print()
    
    # Show sample duplicates if any
    if len(all_dupes) > 0:
        sample_file, sample_dupes = all_dupes[0]
        print(f"Sample duplicates from: {sample_file}")
        print()
        
        # Get first duplicate group
        first_group = sample_dupes.groupby(['Date', 'Dealer', 'CUSIP']).first().iloc[0]
        date, dealer, cusip = first_group.name
        
        group_rows = sample_dupes[
            (sample_dupes['Date'] == date) & 
            (sample_dupes['Dealer'] == dealer) & 
            (sample_dupes['CUSIP'] == cusip)
        ]
        
        print(f"Duplicate group: Date={date}, Dealer={dealer}, CUSIP={cusip}")
        print()
        print("All rows in this group:")
        cols = ['Date', 'Time', 'Dealer', 'CUSIP', 'Security', 'Bid Price', 'Ask Price', 'Bid Spread', 'Ask Spread']
        available_cols = [c for c in cols if c in group_rows.columns]
        print(group_rows[available_cols].to_string(index=False))
    else:
        print("No duplicates found to analyze")
    
    print()
    print("=" * 100)
    print("COLUMN STRUCTURE")
    print("=" * 100)
    print()
    
    # Check column consistency
    first_df = pd.read_excel(files[0])
    print(f"Columns in first file ({files[0].name}): {len(first_df.columns)}")
    print(first_df.columns.tolist())
    print()
    
    # Check if all files have same columns
    cols_sets = [set(pd.read_excel(f).columns) for f in files]
    if len(set(tuple(s) for s in cols_sets)) == 1:
        print("OK: All files have identical column structure")
    else:
        print("WARNING: Files have different column structures!")
        for i, (f, cols) in enumerate(zip(files, cols_sets)):
            if cols != cols_sets[0]:
                print(f"  DIFFERENT: {f.name}: {len(cols)} columns (different from first file)")
    
    print()
    print("=" * 100)
    print("DATE/TIME FORMAT ANALYSIS")
    print("=" * 100)
    print()
    
    sample_df = pd.read_excel(files[0])
    print(f"Date column dtype: {sample_df['Date'].dtype}")
    print(f"Time column dtype: {sample_df['Time'].dtype}")
    print(f"Sample Date values: {sample_df['Date'].head(5).tolist()}")
    print(f"Sample Time values: {sample_df['Time'].head(5).tolist()}")
    print()
    
    # Check date format consistency
    date_formats = []
    for f in files[:5]:  # Sample first 5 files
        df = pd.read_excel(f)
        date_formats.extend(df['Date'].unique()[:3])
    
    print(f"Unique date formats found: {len(set(date_formats))}")
    print(f"Sample date values: {list(set(date_formats))[:5]}")
    
    print()
    print("=" * 100)
    print("DEALER ANALYSIS")
    print("=" * 100)
    print()
    
    # Analyze dealers across all files
    all_dealers = []
    for f in files:
        df = pd.read_excel(f)
        all_dealers.extend(df['Dealer'].unique())
    
    dealer_counts = Counter(all_dealers)
    print(f"Unique dealers across all files: {len(dealer_counts)}")
    print("Dealer frequency:")
    for dealer, count in dealer_counts.most_common():
        print(f"  {dealer:30s}: {count} files")
    
    print()
    print("=" * 100)
    print("Analysis complete!")
    print("=" * 100)

if __name__ == "__main__":
    main()

