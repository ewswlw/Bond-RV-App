# Database Migration Guide

**Version**: 1.0  
**Last Updated**: 2025-01-21 16:30 ET  
**Purpose**: Guide for migrating from Parquet files to SQLite database (.db)

---

## Overview

This guide provides step-by-step instructions for migrating the Bond RV App from Parquet file storage to SQLite database format. The migration preserves all existing functionality while providing SQL query capabilities and better integration with database tools.

---

## Prerequisites

- Python 3.11+
- Poetry environment configured
- All parquet files generated and validated
- SQLite3 installed (comes with Python)

---

## Database Schema

See `Data_Architecture.md` section "Database Schema Design" for complete schema definitions.

### Key Tables

1. `historical_bond_details` - Bond time series (Date+CUSIP primary key)
2. `universe` - Current universe (CUSIP primary key)
3. `bql` - BQL spreads (Date+CUSIP primary key)
4. `runs_timeseries` - Runs time series (Date+Dealer+CUSIP primary key)
5. `historical_portfolio` - Portfolio time series (Date+CUSIP+ACCOUNT+PORTFOLIO primary key)
6. `runs_today` - Materialized runs today view (CUSIP+Benchmark primary key)

---

## Migration Script

### Step 1: Create Migration Script

Create `utils/migrate_to_db.py`:

```python
"""
Migration script to convert Parquet files to SQLite database.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PARQUET_DIR = PROJECT_ROOT / "bond_data" / "parquet"
DB_PATH = PROJECT_ROOT / "bond_data" / "bond_rv.db"

def create_tables(conn: sqlite3.Connection):
    """Create all tables with proper schema."""
    cursor = conn.cursor()
    
    # historical_bond_details
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_bond_details (
            date DATE NOT NULL,
            cusip VARCHAR(9) NOT NULL,
            security TEXT,
            benchmark TEXT,
            g_sprd REAL,
            yrs_cvn REAL,
            vs_bi REAL,
            vs_bce REAL,
            mtd_equity REAL,
            ytd_equity REAL,
            retracement REAL,
            yrs_since_issue REAL,
            z_score REAL,
            retracement2 REAL,
            rating TEXT,
            custom_sector TEXT,
            currency VARCHAR(3),
            ticker TEXT,
            PRIMARY KEY (date, cusip)
        )
    """)
    
    # universe
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS universe (
            cusip VARCHAR(9) PRIMARY KEY,
            benchmark_cusip VARCHAR(9),
            custom_sector TEXT,
            bloomberg_cusip VARCHAR(9),
            security TEXT,
            benchmark TEXT,
            pricing_date DATE,
            pricing_date_bench DATE,
            worst_date DATE,
            yrs_worst REAL,
            ticker TEXT,
            currency VARCHAR(3),
            equity_ticker TEXT
        )
    """)
    
    # bql
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bql (
            date DATE NOT NULL,
            name TEXT,
            cusip VARCHAR(9) NOT NULL,
            value REAL,
            PRIMARY KEY (date, cusip)
        )
    """)
    
    # runs_timeseries
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs_timeseries (
            date DATE NOT NULL,
            time TIME NOT NULL,
            dealer VARCHAR(10) NOT NULL,
            cusip VARCHAR(9) NOT NULL,
            security TEXT,
            benchmark TEXT,
            bid_spread REAL,
            ask_spread REAL,
            bid_size REAL,
            ask_size REAL,
            bid_workout_risk REAL,
            bid_price REAL,
            ask_price REAL,
            ticker TEXT,
            PRIMARY KEY (date, dealer, cusip)
        )
    """)
    
    # historical_portfolio
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_portfolio (
            date DATE NOT NULL,
            cusip VARCHAR(9) NOT NULL,
            account TEXT NOT NULL,
            portfolio TEXT NOT NULL,
            security TEXT,
            quantity REAL,
            position_cr01 REAL,
            PRIMARY KEY (date, cusip, account, portfolio)
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bond_details_date ON historical_bond_details(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bond_details_cusip ON historical_bond_details(cusip)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bql_date ON bql(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bql_cusip ON bql(cusip)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_date ON runs_timeseries(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_cusip ON runs_timeseries(cusip)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_date_cusip ON runs_timeseries(date, cusip)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_date ON historical_portfolio(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_cusip ON historical_portfolio(cusip)")
    
    conn.commit()

def migrate_parquet_to_db():
    """Migrate all parquet files to SQLite database."""
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Create tables
        print("Creating database tables...")
        create_tables(conn)
        
        # Migrate historical_bond_details
        parquet_path = PARQUET_DIR / "historical_bond_details.parquet"
        if parquet_path.exists():
            print(f"Migrating {parquet_path.name}...")
            df = pd.read_parquet(parquet_path)
            df.to_sql('historical_bond_details', conn, if_exists='replace', index=False)
            print(f"  Migrated {len(df):,} rows")
        else:
            print(f"  Warning: {parquet_path.name} not found")
        
        # Migrate universe
        parquet_path = PARQUET_DIR / "universe.parquet"
        if parquet_path.exists():
            print(f"Migrating {parquet_path.name}...")
            df = pd.read_parquet(parquet_path)
            df.to_sql('universe', conn, if_exists='replace', index=False)
            print(f"  Migrated {len(df):,} rows")
        else:
            print(f"  Warning: {parquet_path.name} not found")
        
        # Migrate bql
        parquet_path = PARQUET_DIR / "bql.parquet"
        if parquet_path.exists():
            print(f"Migrating {parquet_path.name}...")
            df = pd.read_parquet(parquet_path)
            df.to_sql('bql', conn, if_exists='replace', index=False)
            print(f"  Migrated {len(df):,} rows")
        else:
            print(f"  Warning: {parquet_path.name} not found")
        
        # Migrate runs_timeseries
        parquet_path = PARQUET_DIR / "runs_timeseries.parquet"
        if parquet_path.exists():
            print(f"Migrating {parquet_path.name}...")
            df = pd.read_parquet(parquet_path)
            df.to_sql('runs_timeseries', conn, if_exists='replace', index=False)
            print(f"  Migrated {len(df):,} rows")
        else:
            print(f"  Warning: {parquet_path.name} not found")
        
        # Migrate historical_portfolio
        parquet_path = PARQUET_DIR / "historical_portfolio.parquet"
        if parquet_path.exists():
            print(f"Migrating {parquet_path.name}...")
            df = pd.read_parquet(parquet_path)
            df.to_sql('historical_portfolio', conn, if_exists='replace', index=False)
            print(f"  Migrated {len(df):,} rows")
        else:
            print(f"  Warning: {parquet_path.name} not found")
        
        print("\nMigration completed successfully!")
        print(f"Database location: {DB_PATH}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_parquet_to_db()
```

### Step 2: Run Migration

```bash
poetry run python utils/migrate_to_db.py
```

### Step 3: Verify Migration

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("bond_data/bond_rv.db")

# Check row counts
tables = ['historical_bond_details', 'universe', 'bql', 'runs_timeseries', 'historical_portfolio']
for table in tables:
    count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn)['count'][0]
    print(f"{table}: {count:,} rows")

conn.close()
```

---

## Code Modifications for Database Support

### Option 1: Dual Support (Parquet + Database)

Modify pipeline loaders to support both formats:

```python
# In bond_pipeline/load.py
def load_historical_append(self, transformed_data, use_db=False):
    if use_db:
        return self._load_to_db_append(transformed_data)
    else:
        return self._load_to_parquet_append(transformed_data)
```

### Option 2: Database-Only

Replace all `pd.read_parquet()` calls with `pd.read_sql()`:

```python
# Old
df = pd.read_parquet("bond_data/parquet/historical_bond_details.parquet")

# New
import sqlite3
conn = sqlite3.connect("bond_data/bond_rv.db")
df = pd.read_sql("SELECT * FROM historical_bond_details", conn)
conn.close()
```

---

## Query Examples

### Latest Bond Details Per CUSIP

```sql
SELECT DISTINCT ON (cusip) *
FROM historical_bond_details
ORDER BY cusip, date DESC;
```

### Runs Today Aggregation

```sql
SELECT 
    cusip,
    benchmark,
    MIN(bid_spread) FILTER (WHERE bid_size > 3000000) AS tight_bid_3mm,
    MAX(ask_spread) FILTER (WHERE ask_size > 3000000) AS wide_offer_3mm,
    COUNT(DISTINCT dealer) FILTER (WHERE bid_size > 3000000) AS num_bids_3mm
FROM runs_timeseries
WHERE date = (SELECT MAX(date) FROM runs_timeseries)
GROUP BY cusip, benchmark;
```

### Portfolio Holdings by Date

```sql
SELECT 
    date,
    cusip,
    SUM(quantity) AS total_quantity,
    SUM(position_cr01) AS total_cr01
FROM historical_portfolio
WHERE date = '2025-11-21'
GROUP BY date, cusip;
```

### Join Bond Details with Runs

```sql
SELECT 
    b.date,
    b.cusip,
    b.security,
    b.custom_sector,
    r.tight_bid_3mm,
    r.wide_offer_3mm
FROM (
    SELECT 
        cusip,
        MIN(bid_spread) FILTER (WHERE bid_size > 3000000) AS tight_bid_3mm,
        MAX(ask_spread) FILTER (WHERE ask_size > 3000000) AS wide_offer_3mm
    FROM runs_timeseries
    WHERE date = (SELECT MAX(date) FROM runs_timeseries)
    GROUP BY cusip
) r
JOIN (
    SELECT DISTINCT ON (cusip) *
    FROM historical_bond_details
    ORDER BY cusip, date DESC
) b ON r.cusip = b.cusip;
```

---

## Performance Considerations

### Indexing Strategy

**Essential Indexes**:
- Date columns (for time-series queries)
- CUSIP columns (for joins)
- Composite indexes (date, cusip) for common queries

**Additional Indexes** (if needed):
- Ticker, Custom_Sector (for filtering)
- Dealer (for runs queries)

### Query Optimization

1. **Use WHERE clauses** to filter before joins
2. **Limit result sets** with LIMIT clause
3. **Use EXPLAIN QUERY PLAN** to analyze query performance
4. **Consider materialized views** for frequently accessed aggregations

### Maintenance

```sql
-- Analyze tables for query optimization
ANALYZE;

-- Vacuum database to reclaim space
VACUUM;

-- Check database integrity
PRAGMA integrity_check;
```

---

## Backup and Recovery

### Backup Database

```bash
# Simple copy
cp bond_data/bond_rv.db bond_data/bond_rv.db.backup

# SQL dump
sqlite3 bond_data/bond_rv.db .dump > bond_data/bond_rv_backup.sql
```

### Restore Database

```bash
# From copy
cp bond_data/bond_rv.db.backup bond_data/bond_rv.db

# From SQL dump
sqlite3 bond_data/bond_rv.db < bond_data/bond_rv_backup.sql
```

---

## Migration Checklist

- [ ] Create migration script
- [ ] Backup existing parquet files
- [ ] Run migration script
- [ ] Verify row counts match
- [ ] Test sample queries
- [ ] Update code to use database (if migrating fully)
- [ ] Update documentation
- [ ] Test all analytics scripts
- [ ] Performance test queries
- [ ] Set up backup schedule

---

## Rollback Plan

If migration fails:

1. **Keep Parquet Files**: Don't delete parquet files until migration verified
2. **Dual Support**: Maintain both formats during transition
3. **Version Control**: Tag code before migration
4. **Testing**: Test on copy of database first

---

**End of Migration Guide**

