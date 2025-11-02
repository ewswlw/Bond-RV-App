# Portfolio123 Industry Classification System (RBICS)

## When to Use

- Use this guide when filtering screens or ranking systems by sectors/industries and you need accurate RBICS codes or mnemonics.
- Apply it before importing or reconciling external datasets so you can map RBICS classifications correctly.
- Reference it when users ask for industry-specific insights; it explains hierarchy levels and example codes.
- Consult it when verifying historical classification changes or handling deprecated industries.
- For quick mnemonic lookups you can skim the tables; rely on this document for authoritative context and usage patterns.

## Overview

Portfolio123 uses the **Revere Business Industry Classification System (RBICS)** for sector, subsector, industry, and subindustry classifications.

### Key Features:
- **Hierarchical Structure**: Sector → SubSector → Industry → SubIndustry
- **Mnemonic Codes**: Portfolio123 provides easy-to-remember codes (e.g., `ENERGY`, `WOODPAPER`, `TECH`)
- **RBICS Numeric Codes**: Standard numeric codes for precise classification
- **Historical Tracking**: Deprecated industries are marked with dates
- **Usage in Screens**: Can filter using `Industry!=WOODPAPER` or `Sector=ENERGY`

## Classification Levels

### 1. Sector (Top Level)
Major economic sectors like:
- 10 - Business Services (BIZSVCE)
- 15 - Consumer Services (CONSUMERSVCE)
- 20 - Consumer Cyclicals (CYCLICALS)
- 25 - Energy (ENERGY)
- 30 - Financials (FINANCIALS)
- 35 - Healthcare (HEALTHCARE)
- 40 - Industrials (INDUSTRIALS)
- 45 - Materials (MATERIALS)
- 50 - Real Estate (REALESTATE)
- 55 - Technology (TECH)
- 60 - Utilities (UTILITIES)

### 2. SubSector (Second Level)
More specific groupings within sectors

### 3. Industry (Third Level)
Detailed industry classifications

### 4. SubIndustry (Fourth Level)
Most granular classification level

## Example RBICS Codes

| RBICS Code | Description | P123 Mnemonic |
|------------|-------------|---------------|
| 10 | Business Services | BIZSVCE |
| 101010 | Marketing and Printing Services | MKTINGPRINTING |
| 10101010 | Marketing and Advertising Services | ADVERTISING |
| 25 | Energy | ENERGY |
| 251015 | Fossil Fuel Exploration and Production | FOSSILFUEL |
| 55 | Technology | TECH |
| 551010 | Computer Hardware | COMPUTERHW |

## Usage in Portfolio123

### In Screens:
```
Industry!=WOODPAPER  # Exclude forestry/paper products
Sector=ENERGY        # Only energy stocks
SubSector=TECH       # Technology subsector
```

### In Ranking Systems:
- Can rank within industry: "Rank Factor vs Stocks in its Industry"
- Can rank within sector: "Rank Factor vs Stocks in its Sector"

## Important Notes

1. **Point-in-Time Data**: Industry classifications are tracked historically
2. **Deprecated Industries**: Shown in gray with deprecation dates
3. **Industry Changes**: Companies can change industries over time
4. **Mapping**: Both numeric RBICS codes and mnemonic codes are supported

## Data Source

Industry classifications are provided by FactSet through the RBICS system, which is more granular and flexible than the older GICS (Global Industry Classification Standard) system.

