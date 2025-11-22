# Bond RV App - Documentation Index

**Version**: 1.0  
**Last Updated**: 2025-01-21 16:30 ET

---

## Documentation Overview

This documentation provides comprehensive coverage of the Bond RV App data architecture, business logic, and system design. It is designed for developers, data engineers, and AI agents who need to understand, maintain, or extend the system.

---

## Documentation Files

### 1. [Data Architecture](./Data_Architecture.md) - **START HERE**

**Comprehensive system documentation** covering:
- Executive summary and system overview
- Complete data flow architecture
- Data sources and input formats
- All three data processing pipelines (Bond, Runs, Portfolio)
- Parquet file specifications and schemas
- Analytics layer (runs_today, runs_views, comb)
- Output tables and views
- Business logic documentation
- Database schema design (SQLite)
- Developer guide

**Use this when**:
- Understanding the overall system architecture
- Learning how data flows through the system
- Understanding file structures and schemas
- Planning modifications or extensions
- Onboarding new developers

---

### 2. [Business Logic Reference](./Business_Logic_Reference.md)

**Quick reference guide** for all calculations and formulas:
- Spread calculations (Tight Bid, Wide Offer, Bid/Offer)
- CR01 calculations
- Change metrics (DoD, MTD, YTD, Custom Date)
- Aggregation logic
- Filtering logic
- Pair analytics statistics
- Deduplication logic
- CUSIP normalization

**Use this when**:
- Need to understand a specific calculation
- Debugging calculation issues
- Implementing new metrics
- Verifying business logic

---

### 3. [Database Migration Guide](./Database_Migration_Guide.md)

**Guide for migrating to SQLite database**:
- Database schema definitions
- Migration script template
- Code modification examples
- Query examples
- Performance considerations
- Backup and recovery

**Use this when**:
- Planning migration to database format
- Need SQL query examples
- Understanding database schema
- Setting up database backups

---

## Quick Navigation

### For Data Engineers

**Understanding Data Flow**:
1. Start with [Data Architecture](./Data_Architecture.md) → "Data Flow Architecture"
2. Review [Data Architecture](./Data_Architecture.md) → "Data Processing Pipelines"
3. Check [Business Logic Reference](./Business_Logic_Reference.md) for calculation details

**Understanding Business Logic**:
1. Start with [Business Logic Reference](./Business_Logic_Reference.md)
2. Cross-reference with [Data Architecture](./Data_Architecture.md) → "Business Logic Documentation"

**Modifying Pipelines**:
1. Review [Data Architecture](./Data_Architecture.md) → "Data Processing Pipelines"
2. Check [Data Architecture](./Data_Architecture.md) → "Developer Guide"
3. Reference [Business Logic Reference](./Business_Logic_Reference.md) for calculation changes

### For Developers

**Getting Started**:
1. Read [Data Architecture](./Data_Architecture.md) → "Executive Summary"
2. Review [Data Architecture](./Data_Architecture.md) → "System Overview"
3. Follow [Data Architecture](./Data_Architecture.md) → "Developer Guide"

**Understanding Calculations**:
1. Start with [Business Logic Reference](./Business_Logic_Reference.md)
2. Cross-reference with code in `analytics/runs/runs_today.py`

**Database Work**:
1. Review [Database Migration Guide](./Database_Migration_Guide.md)
2. Check [Data Architecture](./Data_Architecture.md) → "Database Schema Design"

### For AI Agents

**System Understanding**:
1. Read [Data Architecture](./Data_Architecture.md) completely
2. Reference [Business Logic Reference](./Business_Logic_Reference.md) for specific calculations
3. Use [Database Migration Guide](./Database_Migration_Guide.md) for database-related tasks

**Code Modifications**:
1. Review [Data Architecture](./Data_Architecture.md) → "Developer Guide"
2. Check [Business Logic Reference](./Business_Logic_Reference.md) for calculation formulas
3. Reference existing code patterns in pipeline modules

---

## Key Concepts

### Data Flow

```
Excel Files → Extract → Transform → Load → Parquet Files → Analytics → Output Tables
```

### Three Pipelines

1. **Bond Pipeline**: Processes bond details (75 columns)
2. **Runs Pipeline**: Processes dealer quotes (30 columns)
3. **Portfolio Pipeline**: Processes portfolio holdings (82 columns)

### Five Parquet Files

1. `historical_bond_details.parquet` - Bond time series
2. `universe.parquet` - Current universe
3. `bql.parquet` - BQL spreads
4. `runs_timeseries.parquet` - Runs time series
5. `historical_portfolio.parquet` - Portfolio time series

### Analytics Layer

1. `runs_today.py` - Daily runs analytics with change metrics
2. `runs_views.py` - Formatted portfolio and universe tables
3. `comb.py` - Pair analytics (11 analysis types)

---

## Common Tasks

### Running Pipelines

```bash
# Unified orchestrator (recommended)
poetry run python run_pipeline.py

# Individual pipelines
poetry run python -m bond_pipeline.pipeline -m append --process-bql
poetry run python -m runs_pipeline.pipeline -m append
poetry run python -m portfolio_pipeline.pipeline -m append
```

### Running Analytics

```bash
# Runs today analytics
poetry run python analytics/runs/runs_today.py

# Runs views (portfolio and universe)
poetry run python analytics/runs/runs_views.py

# Pair analytics
poetry run python analytics/comb/comb.py
```

### Understanding Calculations

See [Business Logic Reference](./Business_Logic_Reference.md) for:
- Spread calculations
- CR01 calculations
- Change metrics (DoD, MTD, YTD, Custom Date)
- Aggregation logic

### Database Migration

See [Database Migration Guide](./Database_Migration_Guide.md) for:
- Schema definitions
- Migration scripts
- Query examples

---

## File Locations

**Documentation**: `documentation/` (lowercase folder name)
- `Data_Architecture.md` - Main documentation
- `Business_Logic_Reference.md` - Calculation reference
- `Database_Migration_Guide.md` - Database guide

**Code**:
- Pipelines: `bond_pipeline/`, `runs_pipeline/`, `portfolio_pipeline/`
- Analytics: `analytics/runs/`, `analytics/comb/`
- Configuration: `bond_pipeline/config.py`

**Data**:
- Parquet: `bond_data/parquet/`
- Logs: `bond_data/logs/`
- Analytics Output: `analytics/processed_data/`

---

## Support

For questions or issues:
1. Check relevant documentation section
2. Review code comments and docstrings
3. Check logs in `bond_data/logs/`
4. Review `CHANGELOG.md` for recent changes

---

**Last Updated**: 2025-01-21 16:30 ET

