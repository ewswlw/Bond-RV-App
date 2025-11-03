"""
Deep analysis of duplicate Date+Dealer+CUSIP combinations.
Understanding the pattern and determining "most recent" selection logic.

Created: January 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# Path to Historical Runs folder
RUNS_DIR = Path(r"C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Support Files\Historical Runs")

def parse_time(time_str):
    """Parse time string to datetime.time for comparison."""
    if pd.isna(time_str):
        return None
    
    # Try different time formats
    time_str = str(time_str).strip()
    
    # Format: HH:MM
    if re.match(r'^\d{1,2}:\d{2}$', time_str):
        parts = time_str.split(':')
        hour = int(parts[0])
        minute = int(parts[1])
        return (hour, minute)
    
    return None

def analyze_duplicate_groups(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Analyze duplicate Date+Dealer+CUSIP groups in detail."""
    
    # Find all duplicate groups
    dupes = df[df.duplicated(subset=['Date', 'Dealer', 'CUSIP'], keep=False)].copy()
    
    if len(dupes) == 0:
        return pd.DataFrame()
    
    # Analyze each duplicate group
    groups_analysis = []
    
    # Group by Date+Dealer+CUSIP
    grouped = dupes.groupby(['Date', 'Dealer', 'CUSIP'])
    
    for (date, dealer, cusip), group_df in grouped:
        group_analysis = {
            'filename': filename,
            'date': date,
            'dealer': dealer,
            'cusip': cusip,
            'count': len(group_df),
            'times': sorted(group_df['Time'].unique().tolist()),
            'unique_times': len(group_df['Time'].unique()),
            'first_time': group_df['Time'].iloc[0],
            'last_time': group_df['Time'].iloc[-1],
        }
        
        # Check if rows are truly identical (all columns same)
        numeric_cols = ['Bid Price', 'Ask Price', 'Bid Spread', 'Ask Spread']
        text_cols = ['Security', 'Ticker', 'Currency']
        
        # Check if all numeric columns are identical
        numeric_identical = True
        for col in numeric_cols:
            if col in group_df.columns:
                if len(group_df[col].unique()) > 1:
                    numeric_identical = False
                    break
        
        # Check if all text columns are identical
        text_identical = True
        for col in text_cols:
            if col in group_df.columns:
                if len(group_df[col].unique()) > 1:
                    text_identical = False
                    break
        
        group_analysis['numeric_identical'] = numeric_identical
        group_analysis['text_identical'] = text_identical
        group_analysis['all_identical'] = numeric_identical and text_identical
        
        # Parse times and find latest
        times_parsed = [parse_time(t) for t in group_df['Time'].unique() if parse_time(t) is not None]
        if times_parsed:
            latest_parsed = max(times_parsed)
            latest_time_str = f"{latest_parsed[0]:02d}:{latest_parsed[1]:02d}"
            group_analysis['latest_time'] = latest_time_str
        else:
            group_analysis['latest_time'] = None
        
        groups_analysis.append(group_analysis)
    
    return pd.DataFrame(groups_analysis)

def analyze_sample_duplicates(df: pd.DataFrame, filename: str, n_samples: int = 5):
    """Analyze sample duplicate groups in detail."""
    
    dupes = df[df.duplicated(subset=['Date', 'Dealer', 'CUSIP'], keep=False)].copy()
    
    if len(dupes) == 0:
        print(f"\nNo duplicates found in {filename}")
        return
    
    # Get sample groups
    grouped = dupes.groupby(['Date', 'Dealer', 'CUSIP'])
    sample_groups = list(grouped)[:n_samples]
    
    print(f"\n{'='*100}")
    print(f"SAMPLE DUPLICATE GROUPS FROM: {filename}")
    print(f"{'='*100}\n")
    
    for idx, ((date, dealer, cusip), group_df) in enumerate(sample_groups, 1):
        print(f"Duplicate Group #{idx}: Date={date}, Dealer={dealer}, CUSIP={cusip}")
        print(f"  Number of rows: {len(group_df)}")
        print(f"  Unique times: {sorted(group_df['Time'].unique().tolist())}")
        print()
        
        # Show all rows in this group
        display_cols = ['Date', 'Time', 'Dealer', 'CUSIP', 'Security', 'Bid Price', 'Ask Price', 'Bid Spread', 'Ask Spread']
        available_cols = [c for c in display_cols if c in group_df.columns]
        
        print("  All rows in group:")
        print(group_df[available_cols].to_string(index=False))
        print()
        
        # Check if rows are identical
        if group_df.drop(['Date', 'Dealer', 'CUSIP', 'Time'], axis=1).duplicated().all():
            print("  >>> ROWS ARE IDENTICAL (only Time differs)")
        else:
            print("  >>> ROWS DIFFER IN OTHER COLUMNS TOO")
            
            # Show which columns differ
            for col in group_df.columns:
                if col not in ['Date', 'Dealer', 'CUSIP', 'Time']:
                    if len(group_df[col].unique()) > 1:
                        print(f"    - {col}: {group_df[col].unique().tolist()}")
        
        print()
        print("-" * 100)
        print()

def main():
    """Main analysis function."""
    
    print("="*100)
    print("DEEP DUPLICATE ANALYSIS: Understanding Date+Dealer+CUSIP Duplicates")
    print("="*100)
    print()
    
    files = sorted(RUNS_DIR.glob("RUNS *.xlsx"))
    print(f"Analyzing {len(files)} files...")
    print()
    
    all_groups_analysis = []
    
    for file in files:
        print(f"Processing: {file.name}")
        df = pd.read_excel(file)
        
        # Analyze duplicate groups
        groups_df = analyze_duplicate_groups(df, file.name)
        if len(groups_df) > 0:
            all_groups_analysis.append(groups_df)
            
            # Show sample duplicates
            analyze_sample_duplicates(df, file.name, n_samples=3)
    
    if len(all_groups_analysis) > 0:
        combined_analysis = pd.concat(all_groups_analysis, ignore_index=True)
        
        print("="*100)
        print("COMBINED DUPLICATE GROUP ANALYSIS")
        print("="*100)
        print()
        
        print(f"Total duplicate groups across all files: {len(combined_analysis):,}")
        print()
        
        # Statistics
        print("Statistics:")
        print(f"  Average rows per duplicate group: {combined_analysis['count'].mean():.2f}")
        print(f"  Max rows in single group: {combined_analysis['count'].max()}")
        print(f"  Min rows in single group: {combined_analysis['count'].min()}")
        print()
        
        print("Time pattern analysis:")
        print(f"  Groups with single unique time: {len(combined_analysis[combined_analysis['unique_times'] == 1]):,}")
        print(f"  Groups with multiple unique times: {len(combined_analysis[combined_analysis['unique_times'] > 1]):,}")
        print(f"  Average unique times per group: {combined_analysis['unique_times'].mean():.2f}")
        print(f"  Max unique times in group: {combined_analysis['unique_times'].max()}")
        print()
        
        print("Identity analysis:")
        print(f"  Groups with identical rows (only Time differs): {combined_analysis['all_identical'].sum():,}")
        print(f"  Groups with different data values: {(~combined_analysis['all_identical']).sum():,}")
        print()
        
        # Sample groups with multiple times
        multi_time_groups = combined_analysis[combined_analysis['unique_times'] > 1].head(10)
        if len(multi_time_groups) > 0:
            print("Sample groups with multiple times:")
            print(multi_time_groups[['filename', 'date', 'dealer', 'cusip', 'count', 'unique_times', 'times', 'latest_time']].to_string(index=False))
            print()
        
        # Sample groups with different data
        diff_data_groups = combined_analysis[~combined_analysis['all_identical']].head(10)
        if len(diff_data_groups) > 0:
            print("Sample groups with different data values:")
            print(diff_data_groups[['filename', 'date', 'dealer', 'cusip', 'count', 'unique_times', 'numeric_identical', 'text_identical']].to_string(index=False))
            print()
    
    print("="*100)
    print("Analysis complete!")
    print("="*100)

if __name__ == "__main__":
    main()

