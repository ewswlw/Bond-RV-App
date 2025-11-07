# Historical Runs Data - Deep Analysis & Questions

**Analysis Date**: January 2025  
**Data Location**: `C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Support Files\Historical Runs`  
**Files Analyzed**: 15 Excel files (RUNS MM.DD.YY.xlsx pattern)

---

## Executive Summary

### Data Overview
- **Total Files**: 15 files spanning December 2022 to October 2025
- **Total Rows**: 182,231 rows across all files
- **Average Rows per File**: ~12,149 rows
- **Unique CUSIPs**: 1,702 unique CUSIPs across all files
- **Unique Dealers**: 11 dealers (RBC, TD, CIBC, BMO, NBF, SCM, MS, YTMC, BBLP, IAS, TDS)

### Critical Finding: Massive Duplicate Problem
- **Duplicate Groups**: 53,048 groups with duplicate Date+Dealer+CUSIP combinations
- **Files with Duplicates**: 14 out of 15 files (only RUNS 12.31.22.xlsx has zero duplicates)
- **Average Rows per Duplicate Group**: 2.60 rows
- **Max Rows in Single Group**: 5 rows
- **Groups with Multiple Times**: 50,628 (95% of duplicate groups have different times)
- **Groups with Different Data Values**: 49,319 (93% - data actually changes, not just time)

---

## Data Structure Analysis

### Column Schema
**30 columns** consistently across files (except RUNS 12.31.22.xlsx which has 28):

1. Reference Security
2. **Date** ⭐
3. **Time** ⭐
4. Bid Workout Risk
5. Ticker
6. **Dealer** ⭐
7. Source
8. Security
9. Bid Price
10. Ask Price
11. Bid Spread
12. Ask Spread
13. Benchmark
14. Reference Benchmark
15. Bid Size
16. Ask Size
17. Sector
18. Bid Yield To Convention
19. Ask Yield To Convention
20. Bid Discount Margin
21. Ask Discount Margin
22. **CUSIP** ⭐
23. Sender Name
24. Currency
25. Subject
26. Keyword
27. Bid Interpolated Spread to Government
28. Ask Interpolated Spread to Government
29. Bid Contributed Yield
30. Bid Z-spread

### Key Columns for Primary Key
- **Date**: MM/DD/YY format (object/string type)
- **Time**: HH:MM format (object/string type)
- **Dealer**: String (e.g., "BMO", "RBC", "TD")
- **CUSIP**: 9-character identifier (object/string type)

### Data Quality
- ✅ **No Missing Values**: All Date, Time, Dealer, CUSIP columns are populated
- ⚠️ **Schema Evolution**: One old file (12.31.22.xlsx) has 28 columns vs 30 in others
- ⚠️ **Duplicate Issue**: 95% of duplicate groups have different times AND different data values

---

## Duplicate Pattern Analysis

### Pattern 1: Same Time, Different Data (True Duplicates)
- **Count**: 2,420 groups (5% of duplicates)
- **Pattern**: Same Date+Dealer+CUSIP+Time but different pricing data
- **Example**: Two rows with 08:12 time, same dealer/CUSIP, but different Bid/Ask prices
- **Question**: Are these data quality issues or legitimate differences?

### Pattern 2: Different Times, Identical Data
- **Count**: 3,729 groups (7% of duplicates)
- **Pattern**: Same Date+Dealer+CUSIP with different times but identical pricing
- **Example**: Same bond quoted at 07:28 and 07:33 with identical prices
- **Question**: Should we keep the latest time or the first occurrence?

### Pattern 3: Different Times, Different Data (Price Updates) ⭐ MOST COMMON
- **Count**: 49,319 groups (93% of duplicates)
- **Pattern**: Same Date+Dealer+CUSIP with different times AND different pricing
- **Example**: 
  - 07:17: Bid Price = 98.314, Ask Price = 98.504
  - 11:58: Bid Price = 98.429, Ask Price = 98.620 (price updated during day)
- **Question**: This appears to be intraday price updates - should we keep the latest time?

---

## Sample Duplicate Examples

### Example 1: Multiple Times, Price Changes
```
Date: 10/31/25, Dealer: BMO, CUSIP: 00208DAB7
- 07:17: Bid=98.314, Ask=98.504, Spread=106.0
- 11:58: Bid=98.429, Ask=98.620, Spread=106.0 (2 rows at this time!)
```
**Observation**: Latest time (11:58) has 2 identical rows - this is Pattern 1 within Pattern 3!

### Example 2: Multiple Updates Throughout Day
```
Date: 01/14/25, Dealer: BMO, CUSIP: 015857AG0
- 07:52: Spread=271.0
- 09:01: Spread=272.0
- 16:00: Spread=269.0
```
**Observation**: Prices change throughout the day, latest is 16:00

### Example 3: Missing Data in Some Rows
```
Date: 10/29/25, Dealer: BMO, CUSIP: 00889YAE1
- 07:36: Complete data (Bid=99.568, Ask=99.607)
- 08:34: Missing Bid data (Bid=NaN, Ask=99.608)
- 12:44: Complete data (Bid=99.537, Ask=99.576)
```
**Observation**: Some rows have NaN values - how should we handle missing data?

---

## CRITICAL QUESTIONS FOR DATA ENGINEERING DESIGN

### 1. PRIMARY KEY & DEDUPLICATION STRATEGY

**Q1.1: Primary Key Confirmation**
- You stated: "For every Date, Dealer, CUSIP there should be only 1 entry (pick most recent)"
- **Question**: Does "most recent" mean:
  - A) Latest Time value within the day (e.g., 16:00 vs 07:00)?
  - B) Latest row position in the file?
  - C) Something else?

**Q1.2: Handling Same-Time Duplicates**
- **Finding**: 2,420 groups have the SAME time but different data
- **Question**: If Date+Dealer+CUSIP+Time has multiple rows, which should we keep?
  - A) Last row (by position)?
  - B) Row with most complete data (fewest NaNs)?
  - C) First row?
  - D) Aggregate/average the values?

**Q1.3: Data Quality vs. Business Logic**
- **Finding**: Some duplicate groups have identical data except for minor columns (Subject, Keyword)
- **Question**: Should we deduplicate based on:
  - A) Date+Dealer+CUSIP+Time only?
  - B) Date+Dealer+CUSIP+Time+Key pricing columns (Bid Price, Ask Price)?
  - C) All columns (truly identical rows only)?

---

### 2. DATE/TIME HANDLING

**Q2.1: Date Format Parsing**
- **Current**: Dates stored as object/string in MM/DD/YY format (e.g., "10/31/25")
- **Question**: 
  - A) Parse to datetime64[ns] with timezone?
  - B) Parse to date-only (no time component)?
  - C) Keep as string and add separate datetime column?

**Q2.2: Time Format Parsing**
- **Current**: Times stored as object/string in HH:MM format (e.g., "15:45")
- **Question**:
  - A) Combine Date+Time into single datetime64 column?
  - B) Keep Date and Time separate?
  - C) Add computed DateTime column in addition to Date/Time?

**Q2.3: DateTime Column Strategy**
- **Question**: Should `runs_timeseries.parquet` have:
  - A) `Date` (date only) + `Time` (time only) + `DateTime` (combined) columns?
  - B) Only `DateTime` (combined) column?
  - C) Only `Date` and `Time` separate columns?

**Q2.4: Timezone Handling**
- **Question**: What timezone should we assume for times?
  - A) EST/EDT (Eastern)?
  - B) UTC?
  - C) No timezone (naive datetime)?

---

### 3. CUSIP NORMALIZATION & VALIDATION

**Q3.1: CUSIP Format**
- **Question**: Do CUSIPs need normalization like in the bond_pipeline?
  - A) Uppercase conversion?
  - B) Length validation (must be 9 characters)?
  - C) Extra text removal?
  - D) Keep as-is from Excel?

**Q3.2: Invalid CUSIPs**
- **Question**: If a CUSIP is invalid (wrong length, missing, etc.), should we:
  - A) Log warning and include in data?
  - B) Log warning and exclude from data?
  - C) Fix/clean automatically?

**Q3.3: CUSIP Linking**
- **You mentioned**: "CUSIP column will eventually be very important as it will be the link with the other parquet files"
- **Question**: Should we:
  - A) Use the same CUSIP normalization as `historical_bond_details.parquet`?
  - B) Create a CUSIP mapping table?
  - C) Handle separately for now?

---

### 4. MISSING DATA HANDLING

**Q4.1: NaN Values**
- **Finding**: Some rows have NaN in Bid Price, Ask Price, Spreads, etc.
- **Question**: How should we handle NaN values?
  - A) Keep NaN as-is (NULL in parquet)?
  - B) Fill with 0 or -999?
  - C) Exclude rows with NaN in key columns?
  - D) Forward-fill from previous time for same Date+Dealer+CUSIP?

**Q4.2: Incomplete Rows**
- **Finding**: Some rows have complete data, others have many NaNs
- **Question**: When selecting "most recent", should we:
  - A) Always prefer row with most complete data?
  - B) Always prefer latest time regardless of data completeness?
  - C) Use completeness as tiebreaker when times are equal?

---

### 5. SCHEMA & COLUMNS

**Q5.1: Column Selection**
- **Finding**: 30 columns in files (28 in one old file)
- **Question**: Should `runs_timeseries.parquet` contain:
  - A) All 30 columns?
  - B) Subset of key columns only?
  - C) All columns + additional computed columns?

**Q5.2: Schema Evolution**
- **Finding**: One file (12.31.22.xlsx) has 28 columns vs 30
- **Question**: Should we:
  - A) Align to 30-column master schema (fill missing with NaN)?
  - B) Use dynamic schema (different columns per file)?
  - C) Exclude the old file?

**Q5.3: Additional Columns**
- **Question**: Should we add computed columns like:
  - A) `DateTime` (Date+Time combined)?
  - B) `FileDate` (date extracted from filename)?
  - C) `ProcessedAt` (timestamp when row was processed)?
  - D) `RowNumber` (original row number in file)?

---

### 6. FILE PROCESSING

**Q6.1: Filename Date Extraction**
- **Pattern**: `RUNS MM.DD.YY.xlsx` (e.g., `RUNS 10.31.25.xlsx`)
- **Question**: Should we:
  - A) Extract date from filename and validate against Date column?
  - B) Use filename date if Date column is missing?
  - C) Use Date column exclusively?

**Q6.2: File Processing Order**
- **Question**: Should files be processed:
  - A) Alphabetically (by filename)?
  - B) Chronologically (by date in filename)?
  - C) Does order matter for deduplication?

**Q6.3: Append vs Override Mode**
- **Question**: Similar to bond_pipeline, should we support:
  - A) Append mode (skip dates already in parquet)?
  - B) Override mode (rebuild everything)?
  - C) Both?

---

### 7. DATA QUALITY & VALIDATION

**Q7.1: Data Validation Rules**
- **Question**: What validation should we perform?
  - A) Date within reasonable range?
  - B) Time in valid format (HH:MM)?
  - C) Prices positive?
  - D) Spreads reasonable?
  - E) Dealer in known list?

**Q7.2: Data Quality Logging**
- **Question**: Should we log:
  - A) All duplicate removals?
  - B) Data quality issues (NaN, invalid values)?
  - C) Schema mismatches?
  - D) Processing summary statistics?

---

### 8. OUTPUT PARQUET STRUCTURE

**Q8.1: Primary Key Structure**
- **Question**: What should be the primary key in `runs_timeseries.parquet`?
  - A) `Date + Dealer + CUSIP` (after deduplication)?
  - B) `DateTime + Dealer + CUSIP`?
  - C) No enforced primary key, just deduplicated?

**Q8.2: Parquet Format**
- **Question**: Should we:
  - A) Partition by Date for performance?
  - B) Single monolithic file?
  - C) Daily files?
  - D) Partition by Dealer?

**Q8.3: Column Order**
- **Question**: Should Date/DateTime columns be:
  - A) First columns (like bond_pipeline)?
  - B) Any order?
  - C) Grouped together (Date, Time, DateTime)?

---

### 9. INTEGRATION WITH EXISTING PIPELINE

**Q9.1: Code Reuse**
- **Question**: Should we:
  - A) Reuse bond_pipeline utilities (date parsing, CUSIP validation, logging)?
  - B) Create separate runs_pipeline module?
  - C) Extend existing bond_pipeline?

**Q9.2: Configuration**
- **Question**: Should we:
  - A) Add runs-specific config to `bond_pipeline/config.py`?
  - B) Create separate `runs_config.py`?
  - C) Use environment variables?

**Q9.3: Logging**
- **Question**: Should we:
  - A) Use same logging structure as bond_pipeline?
  - B) Separate logs for runs processing?
  - C) Combined logs?

---

### 10. BUSINESS LOGIC QUESTIONS

**Q10.1: "Most Recent" Definition**
- **Critical**: When you say "pick most recent", you mean:
  - A) Latest Time of day (e.g., 16:00 is more recent than 07:00)?
  - B) Latest file processed (if same date appears in multiple files)?
  - C) Latest row order in file?

**Q10.2: Intraday Price Updates**
- **Finding**: 93% of duplicates show price changes throughout the day
- **Question**: Are these legitimate updates you want to track, or should we only keep end-of-day snapshots?

**Q10.3: Multiple Quotes Same Time**
- **Finding**: Some groups have multiple rows with identical Date+Dealer+CUSIP+Time
- **Question**: Is this:
  - A) Data quality issue (should be deduplicated)?
  - B) Legitimate (multiple quotes at same time from different sources)?
  - C) Should we aggregate/average?

---

## RECOMMENDATIONS (Pending Your Answers)

### High-Level Approach
1. **Deduplication Strategy**: Keep latest Time for each Date+Dealer+CUSIP group
2. **DateTime Handling**: Combine Date+Time into datetime64 column, keep separate columns too
3. **CUSIP Normalization**: Reuse bond_pipeline CUSIP utilities for consistency
4. **Schema Alignment**: Use 30-column master schema, fill missing columns
5. **Modular Design**: Create `runs_pipeline/` module extending bond_pipeline patterns

### Processing Flow
1. Read Excel files (extract date from filename)
2. Normalize CUSIPs
3. Parse Date/Time to datetime objects
4. Deduplicate: Keep most recent Time per Date+Dealer+CUSIP
5. Handle same-time duplicates (need your input)
6. Align to master schema
7. Write to `runs_timeseries.parquet`

---

## NEXT STEPS

**Please answer the questions above (especially Q1.1, Q2.3, Q10.1) so I can:**
1. Design the exact deduplication logic
2. Create the data pipeline architecture
3. Implement the `runs_timeseries.parquet` generation

**Priority Questions to Answer:**
1. **Q10.1**: "Most recent" definition (latest time? latest file? latest row?)
2. **Q1.2**: How to handle same-time duplicates
3. **Q2.3**: DateTime column strategy
4. **Q5.1**: Which columns to include in output
5. **Q8.1**: Primary key structure

---

**Analysis Files Created (in patterns/ folder):**
- `analyze_runs_data.py` - Overall data statistics
- `deep_duplicate_analysis.py` - Duplicate pattern analysis
- `data_analysis_and_questions.md` - This document

