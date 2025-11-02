# Portfolio123 Ticker and Identifier System

## When to Use

- Reference this document when mapping securities, handling delistings, or ensuring identifier consistency across Portfolio123 workflows.
- Use it before building integrations or importing data to choose the proper identifier (StockID, FIGI, CUSIP, etc.).
- Apply it when troubleshooting mismatched securities or survivorship bias issues.
- Consult it to educate collaborators about identifier capabilities in screens, ranking systems, and API endpoints.
- For lighter reminders, refer to the quick reference card; rely on this guide for rigorous identifier management.

## Overview

Portfolio123 supports multiple stock identification methods to handle ticker changes, delistings, and cross-border listings. This is critical for accurate backtesting and data mapping.

## Supported Identifiers

### 1. **P123 StockID** (Internal)
- **Description**: Portfolio123's internal unique identifier for each security
- **Characteristics**:
  - Never changes, even if ticker changes
  - Assigned when stock is added to P123 database
  - Persistent across corporate actions
- **Usage**: Primary identifier for internal operations
- **Example**: `774` (for a specific stock)

### 2. **Ticker Symbol**
- **Description**: Standard stock ticker/symbol
- **Characteristics**:
  - Can change over time due to corporate actions
  - May be reused for different companies
  - Country suffix supported (e.g., `:USA`, `:CAN`)
- **Limitations**: Not reliable for historical data or data merging
- **Example**: `AAPL:USA`, `BDT:CAN`

### 3. **FIGI (Financial Instrument Global Identifier)**
- **Description**: Bloomberg's open standard for security identification
- **Types Available**:
  - **Share Class FIGI (scFIGI)**: Identifies specific share class
  - **Country Composite FIGI (ccFIGI)**: Identifies security at country level
  - **Exchange/Venue FIGI**: Identifies security at specific exchange
- **Characteristics**:
  - Open standard (free to use)
  - Globally unique
  - Handles cross-listings
- **Best Practice**: Use ccFIGI for most applications
- **Usage**: Available in screen reports, ranking systems, API endpoints
- **Example**: `BBG000B9XRY4` (Apple Inc.)

### 4. **CUSIP**
- **Description**: Committee on Uniform Securities Identification Procedures number
- **Characteristics**:
  - 9-character alphanumeric code
  - Proprietary (requires licensing)
  - US and Canada only
- **Usage**: Supported for data imports and API calls
- **Example**: `037833100`

### 5. **CIK (Central Index Key)**
- **Description**: SEC's identifier for companies filing with the SEC
- **Characteristics**:
  - 10-digit number
  - Free and public
  - US companies only
- **Usage**: Supported for data imports and API calls
- **Example**: `0000320193` (Apple Inc.)

### 6. **GVKey (Global Company Key)**
- **Description**: Compustat's unique identifier
- **Characteristics**:
  - Used in Compustat database
  - Requires data license
- **Usage**: Supported for data retrieval with Compustat license
- **Example**: `001690`

## Delisted Stocks

### Ticker Notation for Delisted Stocks
- **Format**: `TICKER^YEAR`
- **Example**: `ENRON^2001`
- **Purpose**: Distinguishes delisted stocks from current tickers
- **Usage**: Automatically handled in CSV uploads and data imports

### Handling in Portfolio123
- Delisted stocks remain in database with historical data
- Point-in-time data ensures accurate backtesting
- Survivorship bias is avoided by including delisted stocks

## Identifier Usage by Function

### Screen Reports
- ✅ Ticker
- ✅ P123 StockID
- ✅ scFIGI
- ✅ ccFIGI

### Ranking Systems
- ✅ Ticker
- ✅ P123 StockID
- ✅ scFIGI
- ✅ ccFIGI

### API Data Retrieval (`/data` endpoint)
- ✅ Ticker
- ✅ P123 StockID (p123Uids)
- ✅ FIGI
- ✅ CIK
- ✅ GVKey (with Compustat license)

### Imported Stock Factors
- ✅ Ticker
- ✅ P123 StockID
- ✅ FIGI
- ✅ CUSIP
- ✅ CIK
- ✅ GVKey

### API `/rank/ranks` Endpoint
- ✅ Ticker
- ✅ FIGI (via additionalData parameter)

## Best Practices

### For Backtesting
1. **Use P123 StockID or FIGI** for historical accuracy
2. **Avoid ticker-only identification** due to ticker changes
3. **Include delisted stocks** to avoid survivorship bias

### For Data Integration
1. **Prefer FIGI** for cross-system integration (free, open standard)
2. **Use ccFIGI** to handle cross-listings properly
3. **Map using multiple identifiers** for robustness

### For Imported Data
1. **Use FIGI or CUSIP** as key column for uploads
2. **Avoid ticker-only uploads** for historical data
3. **Validate mappings** before importing

## Country Suffixes

Portfolio123 supports country-specific ticker notation:
- `:USA` - United States
- `:CAN` - Canada
- `:CHE` - Switzerland
- `:GBR` - United Kingdom
- `:DEU` - Germany
- And others...

**Default**: If no country suffix is specified, `:USA` is assumed.

## Functions for Identifier Filtering

### FIGI() Function
```
FIGI("BBG000B9XRY4,BBG000BVPV84")
```
Returns 1 (TRUE) if the stock's FIGI is in the list, 0 (FALSE) otherwise.

### Ticker() Function
```
Ticker("AAPL,MSFT,GOOGL")
```
Returns 1 (TRUE) if the stock's ticker is in the list.

## Cross-Listing Handling

### Share Class FIGI (scFIGI)
- Different for each share class
- Example: BRK.A and BRK.B have different scFIGIs

### Country Composite FIGI (ccFIGI)
- Different for each country listing
- Example: BDT:CAN and BIRDF:USA have different ccFIGIs
- **Recommended** for most use cases to differentiate liquidity

## Data Mining Considerations

When creating training/holdout sets:
- **Use ccFIGI** to prevent data leakage between cross-listings
- **Avoid having BRK.A in training and BRK.B in holdout**
- **Ensure same company isn't in both sets** via different listings

## API Integration Example

```python
from p123api import Client

client = Client(api_id='YOUR_ID', api_key='YOUR_KEY')

# Using tickers
data = client.data({
    'tickers': ['AAPL:USA', 'MSFT:USA'],
    'formulas': ['Close(0)', 'Volume(0)'],
    'startDt': '2024-01-01',
    'endDt': '2024-12-31'
})

# Using FIGI
data = client.data({
    'figi': ['BBG000B9XRY4', 'BBG000BPH459'],
    'formulas': ['Close(0)', 'Volume(0)'],
    'startDt': '2024-01-01',
    'endDt': '2024-12-31'
})
```

## Summary Table

| Identifier | Unique | Persistent | Free | Global | Best For |
|------------|--------|------------|------|--------|----------|
| P123 StockID | ✅ | ✅ | ✅ | ❌ | Internal P123 operations |
| Ticker | ❌ | ❌ | ✅ | ✅ | Human readability |
| FIGI | ✅ | ✅ | ✅ | ✅ | Data integration, API |
| CUSIP | ✅ | ✅ | ❌ | ❌ | US/Canada data |
| CIK | ✅ | ✅ | ✅ | ❌ | SEC filings |
| GVKey | ✅ | ✅ | ❌ | ✅ | Compustat data |

## References

- OpenFIGI API: https://www.openfigi.com/
- Bloomberg FIGI Documentation: https://www.bloomberg.com/figi
- Portfolio123 Community: https://community.portfolio123.com/

