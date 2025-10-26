"""
Outlook RUNS Email Data Miner

Parses Outlook Data CSV files (created by monitor_outlook.py) and extracts clean
bond pricing time series data into Parquet format.

Workflow:
    1. User runs: python monitor_outlook.py --days 7
    2. User runs: python runs_miner.py
    3. Output: bond_timeseries_clean.parquet

Created: October 26, 2025
"""

import pandas as pd
import re
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input: Outlook Data CSV files location
INPUT_DIR = r"C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Support Files\Outlook Runs"

# Output: Clean parquet file location
OUTPUT_DIR = r"C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Eddy\Python Projects\Bond-RV-App\bond_data\parquet"
OUTPUT_FILENAME = "bond_timeseries_clean.parquet"

# Options
SHOW_PROGRESS = True  # Display progress during processing
SAVE_LOG = True  # Save processing log file

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_valid_cusip(value):
    """
    Check if value is a valid 9-character CUSIP.

    Args:
        value: String or value to validate

    Returns:
        bool: True if valid CUSIP format
    """
    if pd.isna(value) or not value:
        return False
    value_str = str(value).strip().upper()
    return bool(re.match(r'^[A-Z0-9]{9}$', value_str))


def is_likely_numeric(value):
    """
    Check if value is numeric or could be converted to numeric.

    Args:
        value: Value to check

    Returns:
        bool: True if numeric
    """
    try:
        float(str(value).replace(',', ''))
        return True
    except:
        return False


def parse_maturity(date_str):
    """
    Parse maturity date from MM/DD/YY format to YYYY-MM-DD.

    Handles 2-digit years:
    - 00-49 → 2000-2049
    - 50-99 → 1950-1999

    Args:
        date_str: Date string in MM/DD/YY format

    Returns:
        str: Date in YYYY-MM-DD format or None
    """
    if pd.isna(date_str):
        return None
    try:
        parts = date_str.split('/')
        if len(parts) == 3:
            month, day, year = parts
            # Handle 2-digit years
            if len(year) == 2:
                year_int = int(year)
                year = '20' + year if year_int < 50 else '19' + year
            return f'{year}-{month.zfill(2)}-{day.zfill(2)}'
    except:
        pass
    return None


def fix_benchmark(bench_str):
    """
    Fix Unicode fraction characters in benchmark strings.

    Converts: ¼ → .25, ½ → .5, ¾ → .75
    Example: "CAN 2 ¾ 12/01/55" → "CAN 2.75 12/01/55"

    Args:
        bench_str: Benchmark string

    Returns:
        str: Fixed benchmark string
    """
    if pd.isna(bench_str):
        return bench_str

    fixed = str(bench_str)

    # Replace Unicode fractions with decimals
    # Pattern: "CAN X <frac> date" → "CAN X.decimal date"
    fixed = re.sub(r'(\d+)\s+\u00BC\s+', r'\g<1>.25 ', fixed)  # ¼
    fixed = re.sub(r'(\d+)\s+\u00BD\s+', r'\g<1>.5 ', fixed)   # ½
    fixed = re.sub(r'(\d+)\s+\u00BE\s+', r'\g<1>.75 ', fixed)  # ¾

    return fixed


def extract_bank_and_sender(dealer_str):
    """
    Extract bank and sender name from dealer string.

    Splits "FirstName LastName (BANK NAME)" into:
    - Bank: "BANK NAME"
    - Sender: "FirstName LastName"

    Args:
        dealer_str: Dealer string from email

    Returns:
        tuple: (bank, sender)
    """
    if pd.isna(dealer_str):
        return None, None

    match = re.match(r'(.+?)\s*\((.+?)\)', dealer_str)
    if match:
        sender = match.group(1).strip()
        bank = match.group(2).strip()
        return bank, sender
    else:
        return dealer_str, None


# ============================================================================
# CORE PARSER
# ============================================================================

def parse_email_body_dynamic(body, received_date, received_datetime, sender_name):
    """
    Parse email body with dynamic column detection.

    Strategy:
    - Don't rely on fixed column positions (emails vary)
    - Use data patterns as anchors:
      * CUSIP validation (9 alphanumeric chars)
      * '/' separator for bid/ask spreads
      * 'x' separator for bid/ask sizes
      * Pattern matching for benchmarks (CAN, UST)

    Args:
        body: Email body text
        received_date: Date email received
        received_datetime: DateTime email received
        sender_name: Name of sender

    Returns:
        list: List of record dictionaries
    """
    lines = body.strip().split('\n')

    # Find header end (look for CUSIP line or last header indicator)
    header_end_idx = None
    has_cusip_column = False

    for i, line in enumerate(lines):
        line_upper = line.upper()
        if 'CUSIP' in line_upper:
            header_end_idx = i
            has_cusip_column = True
            break
        elif any(header in line_upper for header in ['B YIELD', 'B YTM', 'B PX', 'A PX']):
            header_end_idx = i
            break

    if header_end_idx is None:
        return []

    # Data lines start after header
    data_lines = lines[header_end_idx + 1:]

    records = []
    for data_line in data_lines:
        # Skip empty lines and section headers
        if not data_line.strip() or data_line.startswith('*'):
            continue

        # Split by tabs
        values = data_line.replace('\r', '').split('\t')
        values = [v.strip() for v in values]

        # Remove trailing empty strings
        while values and not values[-1]:
            values.pop()

        if len(values) < 3:
            continue

        # Security is always first
        security = values[0]
        if not security or len(security) < 3:
            continue

        # Find CUSIP by validation (search backwards from end)
        cusip_idx = None
        cusip_value = ''

        if has_cusip_column:
            for idx in range(len(values) - 1, max(0, len(values) - 5), -1):
                if is_valid_cusip(values[idx]):
                    cusip_idx = idx
                    cusip_value = values[idx]
                    break

        # Skip if CUSIP expected but not found
        if has_cusip_column and cusip_idx is None:
            continue

        # If no CUSIP column in email, use all fields
        if cusip_idx is None:
            cusip_idx = len(values)
            cusip_value = ''

        # Data fields are between security and CUSIP
        data_fields = values[1:cusip_idx]

        # Build record using pattern matching
        record = {
            'Date': received_date,
            'ReceivedDateTime': received_datetime,
            'Dealer': sender_name,
            'Security': security,
            'CUSIP': cusip_value
        }

        # Find separators as anchors
        slash_indices = [i for i, v in enumerate(data_fields) if v == '/']
        x_indices = [i for i, v in enumerate(data_fields) if v.lower() == 'x']
        bench_indices = [i for i, v in enumerate(data_fields)
                        if (v.startswith('CAN') or v.startswith('UST') or '¾' in v)
                        and not is_likely_numeric(v.replace('¾', '').replace('CAN', '').strip())]

        # Extract spreads (around '/')
        if slash_indices:
            slash_idx = slash_indices[0]
            if slash_idx > 0:
                record['B_Spd'] = data_fields[slash_idx - 1]
            if slash_idx + 1 < len(data_fields):
                record['A_Spd'] = data_fields[slash_idx + 1]

        # Extract sizes (around 'x')
        if x_indices:
            x_idx = x_indices[0]
            if x_idx > 0:
                record['B_Sz_MM'] = data_fields[x_idx - 1]
            if x_idx + 1 < len(data_fields):
                record['A_Sz_MM'] = data_fields[x_idx + 1]

        # Extract benchmark
        if bench_indices:
            record['Bench'] = data_fields[bench_indices[0]]

        # Extract G-Spread (numeric value 30-300 after sizes)
        if x_indices and len(x_indices) > 0:
            for i in range(x_indices[-1] + 2, len(data_fields)):
                val = data_fields[i]
                if is_likely_numeric(val) and 'B_GSpd' not in record:
                    try:
                        num_val = float(val)
                        if 30 <= num_val <= 300:
                            record['B_GSpd'] = val
                            break
                    except:
                        pass

        # Extract yield (numeric value 1-10 at end)
        for i in range(len(data_fields) - 1, max(0, len(data_fields) - 4), -1):
            val = data_fields[i]
            if is_likely_numeric(val) and val != record.get('B_GSpd', ''):
                try:
                    num_val = float(val)
                    if 1.0 <= num_val <= 10.0:
                        record['B_YTNC'] = val
                        break
                except:
                    pass

        records.append(record)

    return records


def process_all_emails(input_dir):
    """
    Process all Outlook Data CSV files in directory.

    Args:
        input_dir: Directory containing Outlook Data*.csv files

    Returns:
        pandas.DataFrame: Raw extracted records
    """
    input_path = Path(input_dir)
    files = sorted(input_path.glob('Outlook Data*.csv'))

    if not files:
        raise FileNotFoundError(f"No 'Outlook Data*.csv' files found in {input_dir}")

    print(f"Found {len(files)} Outlook Data CSV files")
    print()

    all_records = []

    for file in files:
        if SHOW_PROGRESS:
            print(f"Processing: {file.name}")

        df = pd.read_csv(file, encoding='utf-8-sig')
        file_records = 0

        for idx, row in df.iterrows():
            body = str(row['Body'])
            received_date = row['ReceivedDate']
            received_datetime = row['ReceivedDateTime']
            sender_name = row['SenderName']

            records = parse_email_body_dynamic(body, received_date, received_datetime, sender_name)
            all_records.extend(records)
            file_records += len(records)

        if SHOW_PROGRESS:
            print(f"  Emails: {len(df)}, Records extracted: {file_records}")

    print()
    return pd.DataFrame(all_records)


# ============================================================================
# DATA CLEANING & FORMATTING
# ============================================================================

def deduplicate_records(df):
    """
    Remove duplicate records keeping most recent quote.

    Deduplication key: Date + CUSIP + Dealer
    Strategy: Keep last (most recent by ReceivedDateTime)

    Args:
        df: DataFrame with records

    Returns:
        pandas.DataFrame: Deduplicated records
    """
    # Convert to datetime for sorting
    df['ReceivedDateTime'] = pd.to_datetime(df['ReceivedDateTime'])

    # Sort by timestamp (ascending) to ensure 'last' is most recent
    df = df.sort_values('ReceivedDateTime', ascending=True)

    # Count duplicates before removal
    duplicates_before = df.duplicated(subset=['Date', 'CUSIP', 'Dealer'], keep='last').sum()

    # Remove duplicates
    df_clean = df.drop_duplicates(subset=['Date', 'CUSIP', 'Dealer'], keep='last')

    # Sort for output
    df_clean = df_clean.sort_values(['Date', 'CUSIP', 'ReceivedDateTime'])

    if SHOW_PROGRESS:
        print(f"Deduplication: Removed {duplicates_before:,} duplicates")
        print(f"  Records before: {len(df):,}")
        print(f"  Records after:  {len(df_clean):,}")
        print()

    return df_clean


def clean_bond_data(df):
    """
    Apply data cleaning transformations.

    - Normalize security names (remove extra whitespace)
    - Clean size fields (remove "MM" suffix)
    - Extract ticker from security name
    - Extract coupon from security name
    - Parse maturity dates

    Args:
        df: DataFrame with raw data

    Returns:
        pandas.DataFrame: Cleaned data
    """
    if SHOW_PROGRESS:
        print("Cleaning data...")

    # 1. Normalize security names
    df['Security'] = df['Security'].str.replace(r'  +', ' ', regex=True).str.strip()

    # 2. Clean size fields
    df['B_Sz_MM'] = df['B_Sz_MM'].astype(str).str.replace('MM', '', regex=False).replace('nan', '')
    df['A_Sz_MM'] = df['A_Sz_MM'].astype(str).str.replace('MM', '', regex=False).replace('nan', '')
    df.loc[df['B_Sz_MM'] == '', 'B_Sz_MM'] = None
    df.loc[df['A_Sz_MM'] == '', 'A_Sz_MM'] = None

    # 3. Extract ticker
    df['Ticker'] = df['Security'].str.split(' ').str[0]

    # 4. Extract coupon
    coupon_pattern = r'(\d+\.?\d*)\s+\d{2}/\d{2}/\d{2}'
    df['Coupon'] = df['Security'].str.extract(coupon_pattern, expand=False)
    df['Coupon'] = pd.to_numeric(df['Coupon'], errors='coerce')

    # 5. Extract maturity
    maturity_pattern = r'(\d{2}/\d{2}/\d{2,4})'
    df['Maturity'] = df['Security'].str.extract(maturity_pattern, expand=False)
    df['Maturity_Date'] = df['Maturity'].apply(parse_maturity)
    df['Maturity_Date'] = pd.to_datetime(df['Maturity_Date'], errors='coerce')
    df = df.drop('Maturity', axis=1)

    if SHOW_PROGRESS:
        print(f"  Extracted ticker for all {len(df)} records")
        print(f"  Extracted coupon for {df['Coupon'].notna().sum()} records")
        print(f"  Parsed maturity for {df['Maturity_Date'].notna().sum()} records")
        print()

    return df


def apply_final_formatting(df):
    """
    Apply final formatting changes.

    - Split ReceivedDateTime into Date and Time
    - Split Dealer into Bank code and Sender name
    - Fix benchmark fractions (Unicode → decimal)

    Args:
        df: DataFrame with cleaned data

    Returns:
        pandas.DataFrame: Final formatted data
    """
    if SHOW_PROGRESS:
        print("Applying final formatting...")

    # 1. Split datetime
    df['ReceivedDateTime'] = pd.to_datetime(df['ReceivedDateTime'])
    df['Date'] = pd.to_datetime(df['ReceivedDateTime'].dt.date)
    df['Time'] = df['ReceivedDateTime'].dt.time

    # 2. Split dealer
    df[['Bank', 'Sender']] = df['Dealer'].apply(
        lambda x: pd.Series(extract_bank_and_sender(x))
    )

    # Map to codes
    bank_mapping = {
        'BMO CAPITAL MARKETS': 'BMO',
        'NATIONAL BANK FINANC': 'NBF',
        'FINANCIERE BANQUE NA': 'NBF',
        'RBC DOMINION SECURIT': 'RBC'
    }
    df['Bank'] = df['Bank'].map(bank_mapping)
    df = df.rename(columns={'Bank': 'Dealer'})

    # 3. Fix benchmark fractions
    df['Bench'] = df['Bench'].apply(fix_benchmark)

    # Final column order
    final_columns = [
        'Date', 'Time',
        'Dealer', 'Sender',
        'Ticker', 'Security', 'CUSIP', 'Coupon', 'Maturity_Date',
        'B_Spd', 'A_Spd', 'B_Sz_MM', 'A_Sz_MM',
        'Bench', 'B_GSpd', 'B_YTNC'
    ]

    if SHOW_PROGRESS:
        print(f"  Split datetime into Date and Time")
        print(f"  Split dealer into {df['Dealer'].nunique()} unique dealers")
        print(f"  Fixed {df['Bench'].str.contains(r'\d+\.\d+', na=False, regex=True).sum()} benchmark fractions")
        print()

    return df[final_columns]


# ============================================================================
# VALIDATION
# ============================================================================

def validate_output(df):
    """
    Validate output data quality.

    Checks:
    - Column misalignment
    - Invalid CUSIPs
    - Inverted spreads (bid < ask)
    - Unicode encoding issues

    Args:
        df: Final DataFrame

    Returns:
        dict: Validation results
    """
    if SHOW_PROGRESS:
        print("Validating output quality...")
        print("-" * 80)

    results = {
        'total_records': len(df),
        'issues': []
    }

    # 1. Check for column misalignment
    misaligned = df[df['B_Spd'].astype(str).str.startswith('CAN', na=False)]
    if len(misaligned) > 0:
        results['issues'].append(f"Column misalignment: {len(misaligned)} records")
    else:
        if SHOW_PROGRESS:
            print("✓ Zero column misalignment")

    # 2. Validate CUSIPs
    valid_cusip_pattern = re.compile(r'^[A-Z0-9]{9}$')
    df['CUSIP_Valid'] = df['CUSIP'].apply(
        lambda x: bool(valid_cusip_pattern.match(str(x).upper()))
        if pd.notna(x) and x != '' else False
    )
    invalid = df[(df['CUSIP'] != '') & ~df['CUSIP_Valid']]
    if len(invalid) > 0:
        results['issues'].append(f"Invalid CUSIPs: {len(invalid)} records")
    else:
        if SHOW_PROGRESS:
            print("✓ All CUSIPs valid (excluding empty)")

    # 3. Check for inverted spreads
    df['B_Spd_num'] = pd.to_numeric(df['B_Spd'], errors='coerce')
    df['A_Spd_num'] = pd.to_numeric(df['A_Spd'], errors='coerce')
    inverted = df[(df['B_Spd_num'] < df['A_Spd_num']) &
                  df['B_Spd_num'].notna() & df['A_Spd_num'].notna()]
    if len(inverted) > 0:
        results['issues'].append(f"Inverted spreads: {len(inverted)} records")
    else:
        if SHOW_PROGRESS:
            print("✓ Zero inverted spreads")

    # 4. Check for Unicode fractions
    has_unicode = df['Bench'].str.contains(r'[\u00BC\u00BD\u00BE]', na=False, regex=True).sum()
    if has_unicode > 0:
        results['issues'].append(f"Unicode fractions remaining: {has_unicode} records")
    else:
        if SHOW_PROGRESS:
            print("✓ All Unicode fractions converted")

    # 5. Spread statistics
    spread_width = df['B_Spd_num'] - df['A_Spd_num']
    if SHOW_PROGRESS:
        print(f"\nSpread Statistics:")
        print(f"  Average: {spread_width.mean():.2f} bps")
        print(f"  Median:  {spread_width.median():.2f} bps")
        print(f"  Range:   {spread_width.min():.2f} - {spread_width.max():.2f} bps")

    results['spread_avg'] = spread_width.mean()
    results['spread_median'] = spread_width.median()

    if SHOW_PROGRESS:
        print()

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    print("=" * 80)
    print("OUTLOOK RUNS EMAIL DATA MINER")
    print("=" * 80)
    print()
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output location:  {Path(OUTPUT_DIR) / OUTPUT_FILENAME}")
    print()
    print("=" * 80)
    print()

    start_time = datetime.now()

    try:
        # 1. Process all emails
        print("STEP 1: PARSING EMAIL DATA")
        print("-" * 80)
        df_raw = process_all_emails(INPUT_DIR)
        print(f"Total records extracted: {len(df_raw):,}")
        print()

        # 2. Deduplicate
        print("STEP 2: DEDUPLICATION")
        print("-" * 80)
        df_dedup = deduplicate_records(df_raw)

        # 3. Clean data
        print("STEP 3: DATA CLEANING")
        print("-" * 80)
        df_clean = clean_bond_data(df_dedup)

        # 4. Final formatting
        print("STEP 4: FINAL FORMATTING")
        print("-" * 80)
        df_final = apply_final_formatting(df_clean)

        # 5. Validate
        print("STEP 5: VALIDATION")
        print("-" * 80)
        validation_results = validate_output(df_final)

        # 6. Save to Parquet
        print("STEP 6: SAVING TO PARQUET")
        print("-" * 80)
        output_path = Path(OUTPUT_DIR) / OUTPUT_FILENAME

        df_final.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Saved to: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print()

        # 7. Summary
        print("=" * 80)
        print("PROCESSING COMPLETE!")
        print("=" * 80)
        print()
        print(f"Total records:        {len(df_final):,}")
        print(f"Unique CUSIPs:        {df_final['CUSIP'][df_final['CUSIP'] != ''].nunique():,}")
        print(f"Unique dealers:       {df_final['Dealer'].nunique()}")
        print(f"Date range:           {df_final['Date'].min()} to {df_final['Date'].max()}")
        print()

        if validation_results['issues']:
            print("⚠️  WARNINGS:")
            for issue in validation_results['issues']:
                print(f"  - {issue}")
            print()
        else:
            print("✓ All validation checks passed!")
            print()

        print(f"Duration: {(datetime.now() - start_time).total_seconds():.1f}s")
        print("=" * 80)

    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR!")
        print("=" * 80)
        print(f"An error occurred: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
