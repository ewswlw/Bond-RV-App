# Portfolio123 Complete Factor Reference

## When to Use

- Use this catalog when you need authoritative information on Portfolio123 factors, functions, and data fields for screening or ranking.
- Apply it before designing new models to confirm factor availability, naming conventions, and category coverage.
- Reference it during data-mapping tasks so agents can translate Portfolio123 mnemonics to business descriptions and vice versa.
- Consult it when troubleshooting missing fields or ensuring compliance with licensing requirements (e.g., RBICS, Compustat).
- For quick reminders of a handful of factors, the quick reference card may suffice; rely on this document for exhaustive lookup.

**Total Factors Documented: 1126**

This comprehensive reference contains all available data fields, factors, and functions in Portfolio123 for creating screening and ranking rules.

## Table of Contents

- [Macro Economic Data](#macro-economic-data) - 94 items
- [Fundamental Data](#fundamental-data) - 36 items
- [Technical Indicators](#technical-indicators) - 1 items
- [Analyst Estimates](#analyst-estimates) - 127 items
- [Institutional Data](#institutional-data) - 51 items
- [Constants & IDs](#constants--ids) - 57 items
- [Functions](#functions) - 251 items
- [Price & Volume](#price--volume) - 14 items
- [Ratios & Metrics](#ratios--metrics) - 103 items
- [Other](#other) - 392 items

---

## Macro Economic Data

**Count: 94**

| Factor Code | Category/Name | Description |
|-------------|---------------|-------------|
| `##ADJMBASE` | Macro - Money, M1, M2 | Monetary Base; Total (monthly) |
| `##CAB3MO` | Macro - Interbank Rate | 3-Month Interbank Rate for Canada (monthly) |
| `##CAPUTIL` | Macro - Other Business Activity | Capacity Utilization: Total Industry (monthly) |
| `##CAT10YR` | Macro - Treasury notes (T-notes) | 10-Year Government Bond Yield for Canada (monthly) |
| `##CHT10YR` | Macro - Treasury notes (T-notes) | 10-Year Government Bond Yield for Switzerland (monthly) |
| `##CIVLABOR` | Macro - Labor Force | Civilian Labor Force (monthly) |
| `##CLAIMSCONTINUE` | Macro - Unemployment | Continued Claims (Insured Unemployment) (weekly) |
| `##CLAIMSNEW` | Macro - Unemployment | Initial Claims (weekly) |
| `##CONSTR` | Macro - Other Business Activity | Total Construction Spending (monthly) |
| `##CORPAAA` | Macro - Corporate Bonds | BofA Merrill Lynch US Corporate AAA Effective Yield© (daily) |
| `##CORPB` | Macro - Corporate Bonds | BofA Merrill Lynch US High Yield B Effective Yield© (daily) |
| `##CORPBB` | Macro - Corporate Bonds | BofA Merrill Lynch US High Yield BB Effective Yield© (daily) |
| `##CORPBBB` | Macro - Corporate Bonds | BofA Merrill Lynch US Corporate BBB Effective Yield© (daily) |
| `##CORPBBBOAS` | Macro - Corporate Bonds | BofA Merrill Lynch US Corporate BBB Option-Adjusted Spread (daily) |
| `##CORPBBOAS` | Macro - Corporate Bonds | BofA Merrill Lynch US High Yield Master II Opt-Adj Spread (daily) |
| `##CORPJNK` | Macro - Corporate Bonds | BofA Merrill Lynch US High Yield CCC or Below Effective Yield© (daily) |
| `##CPI` | Macro - Price Index, CPI, PPI, HPI | CPI All Urban Consumers: All Items (monthly) |
| `##DBTGDP` | Macro - Other Economic Activity | Debt as Percent of GDP (quarterly) |
| `##DELINQCC` | Macro - Delinquency Rate | Delinquency Rate On Credit Card Loans, All Commercial Banks (quarterly) |
| `##DELINQMORT` | Macro - Delinquency Rate | Delinquency Rate On Single-Family Residential Mortgages (quarterly) |
| `##DOMINV` | Macro - Other Economic Activity | Gross Private Domestic Investment (quarterly) |
| `##EUB3MO` | Macro - Interbank Rate | 3-Month Interbank Rate for the Euro Area (monthly) |
| `##EUT10YR` | Macro - Treasury notes (T-notes) | 10-Year Government Bond Yield for the Euro Area (monthly) |
| `##FEDFUNDS` | Macro - Interest, Mortgage, Prime, TED | Effective Federal Funds Rate (daily) |
| `##GBB3MO` | Macro - Interbank Rate | 3-Month Interbank Rate for the United Kingdom (monthly) |
| `##GBT10YR` | Macro - Treasury notes (T-notes) | 10-Year Government Bond Yield for the United Kingdom (monthly) |
| `##GNP` | Macro - GDP/GNP | Gross National Product (quarterly) |
| `##HDEBTSERV` | Macro - Income | Household Debt Service Payments as a Percent of Disposable Personal Income (quarterly) |
| `##HPRICES` | Macro - Price Index, CPI, PPI, HPI | S&P Case-Shiller 20-City Home Price Index© (monthly) |
| `##HSTARTS` | Macro - Other Business Activity | Housing Starts (Total) (monthly) |
| `##HVACANCY` | Macro - Vacancy Rates | Home Vacancy Rate for the United States (annual) |
| `##INDPRO` | Macro - Other Business Activity | Industrial Production Index (monthly) |
| `##INFLEXP` | Macro - Sentiment | University of Michigan Inflation Expectation© (monthly) |
| `##INV2SHIP` | Macro - Manufacturing | Ratio of Total Inventories to Shipments for All Manufacturing Industries (monthly) |
| `##INV2SLS` | Macro - Other Business Activity | Total Business: Inventories to Sales Ratio (monthly) |
| `##INV2SLSAUTO` | Macro - Vehicle, Automobile | Auto Inventory/Sales Ratio (monthly) |
| `##INVTOT` | Macro - Other Business Activity | Total Business Inventories (monthly) |
| `##LABORPARTIC` | Macro - Labor Force | Civilian Labor Force Participation Rate (monthly) |
| `##M1` | Macro - Money, M1, M2 | M1 Money Stock (monthly) |
| `##M2` | Macro - Money, M1, M2 | M2 Money Stock (monthly) |
| `##MORT30Y` | Macro - Interest, Mortgage, Prime, TED | 30-Year Fixed Rate Mortgage Average in the United States (weekly) |
| `##NBDI` | Macro - Currency | Nominal Broad U.S. Dollar Index (daily) |
| `##NOB3MO` | Macro - Interbank Rate | 3-Month Interbank Rates for Norway (monthly) |
| `##NONFARMEMPL` | Macro - Labor Force | All Employees: Total nonfarm (monthly) |
| `##OIL` | Macro - Oil, Gold Price | Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, OK (daily) |
| `##ORDERSCAP` | Macro - Manufacturing | Manufacturers' New Orders: Nondefense Capital Goods ex.Aircraft (monthly) |
| `##ORDERSDUR` | Macro - Manufacturing | Manufacturers' New Orders: Durable Goods (monthly) |
| `##ORDERSUNFILL` | Macro - Manufacturing | Value of Unfilled Orders for All Manufacturing ex. Transportation (monthly) |
| `##PCE` | Macro - Other Economic Activity | Personal Consumption Expenditures (monthly) |
| `##PLB3MO` | Macro - Interbank Rate | 3-Month Interbank Rates for Poland (monthly) |
| `##POPUL` | Macro - Labor Force | Total Population: All Ages including Armed Forces Overseas (monthly) |
| `##PPI` | Macro - Price Index, CPI, PPI, HPI | Producer Price Index: Finished Goods (monthly) |
| `##PRIME` | Macro - Interest, Mortgage, Prime, TED | Bank Prime Loan Rate (monthly) |
| `##PRODAUTO` | Macro - Vehicle, Automobile | Domestic Auto Production (monthly) |
| `##RBDI` | Macro - Currency | Real Broad Dollar Index (monthly) |
| `##RDISPINC` | Macro - Income | Real Disposable Personal Income (monthly) |
| `##RECPROB` | Macro - Sentiment | Smoothed U.S. Recession Probabilities (monthly) |
| `##RGDP` | Macro - GDP/GNP | Real Gross Domestic Product (quarterly) |
| `##RINCPERCAP` | Macro - Income | Real Disposable Personal Income: Per capita (monthly) |
| `##RMINCOME` | Macro - Income | Real Median Income (annual) |
| `##RPCE` | Macro - Other Economic Activity | Real Personal Consumption Expenditures (monthly) |
| `##RVACANCY` | Macro - Vacancy Rates | Rental Vacancy Rate for the United States (annual) |
| `##SALESALLVEH` | Macro - Vehicle, Automobile | Total Vehicle Sales (monthly) |
| `##SALESAUTO` | Macro - Vehicle, Automobile | Light Weight Vehicle Sales: Autos & Light Trucks (monthly) |
| `##SALESRET` | Macro - Other Business Activity | Retail Sales: Total (Excluding Food Services) (monthly) |
| `##SALESRETFD` | Macro - Other Business Activity | Real Retail and Food Services Sales (monthly) |
| `##SAVING` | Macro - Other Economic Activity | Personal Saving Rate (monthly) |
| `##SEB3MO` | Macro - Interbank Rate | 3-Month Interbank Rates for Sweden (monthly) |
| `##SOFR3MO` | Macro - Interest, Mortgage, Prime, TED | 90-Day Average SOFR (daily) |
| `##STRESS` | Macro - Stress Index | St. Louis Fed Financial Stress Index© (weekly) |
| `##SURPLUS` | Macro - Other Economic Activity | Federal Surplus or Deficit [-] (annual) |
| `##UMCSENT` | Macro - Sentiment | University of Michigan: Consumer Sentiment© (monthly) |
| `##UNDURATION` | Macro - Unemployment | Median Duration of Unemployment (monthly) |
| `##UNRATE` | Macro - Unemployment | Civilian Unemployment Rate (monthly) |
| `##UNTEEN` | Macro - Unemployment | Unemployment Rate - 16 to 19 years (monthly) |
| `##UNTOT` | Macro - Unemployment | Total unemployed, including under-employed (monthly) |
| `##UNWANT` | Macro - Labor Force | Not in Labor Force, Want a Job Now (monthly) |
| `##USCURRACCT` | Macro - Money, M1, M2 | Balance on Current Account (quarterly) |
| `##USR10YR` | Macro - Interest, Mortgage, Prime, TED | 10-Year US Real Interest Rate (monthly) |
| `##USSLIND` | Macro - Other Economic Activity | Leading Index for the United States (monthly) |
| `##UST10YR` | Macro - Treasury notes (T-notes) | 10-Year Treasury Constant Maturity Rate (USD) (daily) |
| `##UST1MO` | Macro - Treasury bills (T-bills) | 1-Month Treasury Constant Maturity Rate (USD) (daily) |
| `##UST1YR` | Macro - Treasury bills (T-bills) | 1-Year Treasury Constant Maturity Rate (USD) (daily) |
| `##UST20YR` | Macro - Treasury bonds (T-bonds) | 20-Year Treasury Constant Maturity Rate (USD) (daily) |
| `##UST2YR` | Macro - Treasury notes (T-notes) | 2-Year Treasury Constant Maturity Rate (USD) (daily) |
| `##UST30YR` | Macro - Treasury bonds (T-bonds) | 30-Year Treasury Constant Maturity Rate (USD) (daily) |
| `##UST3MO` | Macro - Treasury bills (T-bills) | 3-Month Treasury Constant Maturity Rate (USD) (daily) |
| `##UST3YR` | Macro - Treasury notes (T-notes) | 3-Year Treasury Constant Maturity Rate (USD) (daily) |
| `##UST5YR` | Macro - Treasury notes (T-notes) | 5-Year Treasury Constant Maturity Rate (USD) (daily) |
| `##UST6MO` | Macro - Treasury notes (T-notes) | 6-Month Treasury Constant Maturity Rate (USD) (daily) |
| `##UST7YR` | Macro - Treasury notes (T-notes) | 7-Year Treasury Constant Maturity Rate (USD) (daily) |
| `##VELM1` | Macro - Money, M1, M2 | Velocity of M1 Money Stock (quarterly) |
| `##VELM2` | Macro - Money, M1, M2 | Velocity of M2 Money Stock (quarterly) |
| `##WAGES` | Macro - Manufacturing | Wages in manufacturing (monthly) |

## Fundamental Data

**Count: 36**

| Factor Code | Category/Name | Description |
|-------------|---------------|-------------|
| `AstTurnTTMInd` | Asset Turnover, Industry | Asset Turnover Industry, TTM |
| `CurFYDnRev4WkAgo` | EPS Revisions Current Year | Current Fiscal Year Down Revisions, 4 Weeks ago |
| `CurFYDnRevLastWk` | EPS Revisions Current Year | Current Fiscal Year Down Revisions, Last Week |
| `CurFYUpRev4WkAgo` | EPS Revisions Current Year | Current Fiscal Year Up Revisions, 4 Weeks ago |
| `CurFYUpRevLastWk` | EPS Revisions Current Year | Current Fiscal Year Up Revisions, Last Week |
| `Div%ChgTTM` | Dividend Growth | Dividend Percent Change, TTM (%) |
| `DivPSTTM` | Dividends in a Filing Period | Sum of all regular dividends with ex-date in the past 4 quarters. It is equivalent to DivPS(0,TTM,#Regular,#ExDate) |
| `EBITDAActualGr%TTM` | EBITDA Actual | EBITDA Actual growth trailing twelve months (TTM) |
| `EBITDAActualTTM` | EBITDA Actual | EBITDA Actual trailing twelve months |
| `FCFYield` | Free Cash Flow Yield | The ratio is calculated by dividing the most recent trailing twelve months free cash flow by market cap. |
| `GMgn%TTMInd` | Gross Profit Margin, Industry | Gross Margin Industry, TTM (%) |
| `IncPerEmpTTMInd` | Income per Employee, Industry | Income Per Employee Industry, TTM |
| `IntCovTTMInd` | Interest Coverage, Industry | Interest Coverage Industry, TTM |
| `InvTurnTTMInd` | Inventory Turnover, Industry | Inventory Turnover Industry, TTM |
| `NPMgn%TTMInd` | Net Profit Margin, Industry | Net Profit Margin Industry, TTM (%) |
| `NextFYDnRev4WkAgo` | EPS Revisions Next Year | Next Fiscal Year Down Revisions, 4 Weeks ago |
| `NextFYDnRevLastWk` | EPS Revisions Next Year | Next Fiscal Year Down Revisions, Last Week |
| `NextFYUpRev4WkAgo` | EPS Revisions Next Year | Next Fiscal Year Up Revisions, 4 Weeks ago |
| `NextFYUpRevLastWk` | EPS Revisions Next Year | Next Fiscal Year Up Revisions Last Week |
| `OCFYield` | Operating Cash Flow Yield | Operating Cash Flow Yield |
| `OpMgn%TTMInd` | Operating Margin, Industry | Operating Margin Industry, TTM (%) |
| `PEExclXorTTMInd` | Price To Earnings (PE), Industry | Price To Earnings Ratio Industry, Excluding Extraordinary Items, TTM |
| `PTMgn%TTMInd` | Pretax Margin, Industry | Pretax Margin Industry, TTM (%) |
| `PayRatioTTMInd` | Payout Ratio, Industry | Payout Ratio Industry, TTM (%) |
| `Pr2CashFlTTMInd` | Price To Cash Flow, Industry | Price to Cash Flow Per Share Ratio Industry, TTM |
| `Pr2FrCashFlTTMInd` | Price To Free Cash Flow, Industry | Price To Free Cash Flow Per Share Ratio Industry, TTM |
| `ProjPECurFY` | Price To Earnings (PE) Projected | Current Year Projected P/E Ratio |
| `ProjPENextFY` | Price To Earnings (PE) Projected | Next Year Projected P/E Ratio |
| `ProjPENextFYInd` | Price To Earnings (PE) Projected, Industry | Next Year Projected P/E Ratio Industry |
| `ROA%TTMInd` | Return On Assets, Industry | Return on Assets Industry, TTM (%) |
| `ROE%TTMInd` | Return On Equity, Industry | Return on Average Common Equity Industry, TTM (%) |
| `ROI%TTMInd` | Return On Investment, Industry | Return on Investment Industry, TTM (%) |
| `RecTurnTTMInd` | Receivable Turnover, Industry | Receivables Turnover Industry, TTM |
| `Retn%TTMInd` | Retention Rate, Industry | Retention Rate Industry, TTM (%) |
| `TaxRate%TTMInd` | Tax Rate, Industry | Tax Rate Industry, Effective, TTM (%) |
| `ValROETTM` | Value of Return On Equity | Value of Return On Equity, TTM {ratio} |

## Technical Indicators

**Count: 1**

| Factor Code | Category/Name | Description |
|-------------|---------------|-------------|
| `Vol3MAvg` | Average Volume | Monthly average total volume (in millions). |

## Analyst Estimates

**Count: 127**

| Factor Code | Category/Name | Description |
|-------------|---------------|-------------|
| `#AnalystsCurFY` | EPS Estimate Current Year | Number of analysts for current fiscal year EPS. Note: if a security has no analysts, this factor returns NA, not 0. |
| `#AnalystsCurFYSales` | Sales Estimate Current Year | Number of analysts for current fiscal year sales. Note: if a security has no analysts, this factor returns NA, not 0. |
| `#AnalystsCurQ` | EPS Estimate Current Quarter | Number of analysts for current quarter EPS estimate. Note: if a security has no analysts, this factor returns NA, not 0. |
| `#AnalystsNextFY` | EPS Estimate Next Year | Number of analysts for next fiscal year EPS. Note: if a security has no analysts, this factor returns NA, not 0. |
| `#AnalystsNextFYSales` | Sales Estimate Next Year | Number of analysts for next fiscal year sales. Note: if a security has no analysts, this factor returns NA, not 0. |
| `#AnalystsNextQ` | EPS Estimate Next Quarter | Number of analysts for next fiscal quarter EPS. Note: if a security has no analysts, this factor returns NA, not 0. |
| `CapExEstCY` | CapEx Estimate Mean | CapEx Estimate Mean Current Year |
| `CapExEstNY` | CapEx Estimate Mean | CapEx Estimate Mean Next Year |
| `CapExEstY2` | CapEx Estimate Mean | CapEx Estimate Mean 2 Years |
| `CapExEstY3` | CapEx Estimate Mean | CapEx Estimate Mean 3 Years |
| `ConsEstCnt()` | Consensus Count | Number of analysts providing estimates for an item |
| `ConsEstDn()` | Consensus Down | Number of analysts revising estimate down in past 75 days |
| `ConsEstHi()` | Consensus High | Consensus highest estimate |
| `ConsEstLow()` | Consensus Low | Consensus lowest estimate |
| `ConsEstMean()` | Consensus Mean | Average of analyst estimates |
| `ConsEstMedian()` | Consensus Median | Median of analyst estimates |
| `ConsEstRSD()` | Consensus Relative Standard Deviation | Relative Standard Deviation of analysts estimates in percentage |
| `ConsEstStdDev()` | Consensus Standard Deviation | Standard Deviation of analysts estimates |
| `ConsEstUp()` | Consensus Up | Number of analysts revising estimate up in past 75 days |
| `CurFYEPSStdDev` | EPS Estimate Current Year | Current fiscal year EPS |
| `CurFYSalesStdDev` | Sales Estimate Current Year | Current fiscal year sales estimate |
| `CurQEPSStdDev` | EPS Estimate Current Quarter |  |
| `CurrYRevRatio4W` | Estimate Revision Direction | Ratio ranging from -1 to +1 indicating the direction of revisions for the stock's industry. The extreme values 1 and -1 would indicate that every analyst's CurrY estimate in the industry have been rev |
| `EBITDAEstCY` | EBITDA Estimate Mean | Average of analyst estimates for EBITDA for the Current Year |
| `EBITDAEstNY` | EBITDA Estimate Mean | Average of analyst estimates for EBITDA for the Next Year |
| `EBITDAEstY2` | EBITDA Estimate Mean | Average of analyst estimates for EBITDA for Year 2 |
| `EBITDAEstY3` | EBITDA Estimate Mean | Average of analyst estimates for EBITDA for the Year 3 |
| `EPS#Positive` | Income Trend | Consecutive years of positive EPS |
| `EPSActual()` | EPS Actual | Historical (Actual) EPS |
| `EPSActualGr%PYQ` | EPS Actual | EPS Actual Growth Previous Year Quarter (PYQ) |
| `EPSActualGr%TTM` | EPS Actual | EPS Actual Growth Trailing Twelve Months (TTM) |
| `EPSActualPTM` | EPS Actual | EPS Actual Previous Twelve Months (PTM) |
| `EPSActualTTM` | EPS Actual | EPS Actual Trailing Twelve Months (TTM) |
| `EPSEst()` | Historical EPS Estimate | Historical mean EPS estimate. |
| `EPSExclXor5YAvg` | Earnings Per Share (EPS) Excl Xor | Earnings per share without extraordinary expenses. |
| `EPSExclXorGr%3YInd` | Industry Growth | EPS Growth Rate Industry, 3 Years (%) |
| `EPSExclXorGr%5YInd` | Industry Growth | Earnings Per Share Industry, 5 Year Growth Rate (%) |
| `EPSExclXorGr%AInd` | Industry Growth | EPS Percent Change Industry, Year Over Year (%) |
| `EPSExclXorGr%PYQInd` | Industry Growth | EPS Percent Change Industry, Most Recent Quarter vs. Quarter 1 Year Ago (%) |
| `EPSExclXorGr%TTMInd` | Industry Growth | EPS Percent Change Industry, TTM Over TTM (%) |
| `EPSHistEstCnt()` | Historical Estimate Number of Analysts | Historical EPS Estimate Number of Analysts |
| `EPSHistEstSD()` | Historical Estimate Standard Deviation | Historical EPS Estimate Standard Deviation |
| `EPSInclXor5YAvg` | Earnings Per Share (EPS) Incl Xor | Earnings per share with all expenses including extraordinary items. |
| `EPSSUE()` | Unexpected Earnings (SUE) | Standardized Unexpected Earnings general formula |
| `EPSStableQ` | Earnings Per Share (EPS) Stability | EPS Stability (standard deviation of the past 20 quarters) |
| `EPSSurprise()` | Analyst EPS Surprise | EPS surprise (Est vs. Actual) |
| `EV2SalesPY` | Enterprise value to Sales | Enterprise value to sales. |
| `EstimateXY()` | Regression Estimate | Returns the Y estimate for a previously computed XY regression. |
| `EstimateY()` | Regression Estimate |  |
| `FCFEstCY` | FCF Estimate Mean | Free cash flow estimate mean current year |
| `FCFEstNY` | FCF Estimate Mean | Free cash flow estimate mean next year |
| `FCFEstY2` | FCF Estimate Mean | Free cash flow estimate mean 2 years |
| `FCFEstY3` | FCF Estimate Mean | Free cash flow estimate mean 3 years |
| `FY2EPSMean` | EPS Estimate Other Years | EPS mean estimate 2 years |
| `FY2SalesMean` | Sales Estimate Other Years | Sales estimate 2 years |
| `FY3EPSMean` | EPS Estimate Other Years | EPS estimate 3 years |
| `FY3SalesMean` | Sales Estimate Other Years | Sales estimate 3 years |
| `HistQ1EPSActual` | EPS Actual | Historical EPS (Actual), 1 Quarter ago |
| `HistQ1EPSEst` | Historical EPS Estimate | Historical EPS Mean Estimate, 1 Quarter ago |
| `HistQ2EPSActual` | EPS Actual | Historical EPS (Actual), 2 Quarters ago |
| `HistQ2EPSEst` | Historical EPS Estimate | Historical EPS Mean Estimate, 2 Quarters ago |
| `HistQ3EPSActual` | EPS Actual | Historical EPS (Actual), 3 Quarters ago |
| `HistQ3EPSEst` | Historical EPS Estimate | Historical EPS Mean Estimate, 3 Quarters ago |
| `HistQ4EPSActual` | EPS Actual | Historical EPS (Actual), 4 Quarters ago |
| `HistQ4EPSEst` | Historical EPS Estimate | Historical EPS Mean Estimate, 4 Quarters ago |
| `HistQ5EPSActual` | EPS Actual | Historical EPS (Actual), 5 Quarters ago |
| `HistQ5EPSEst` | Historical EPS Estimate | Historical EPS Mean Estimate, 5 Quarters ago |
| `NTMSalesMean` | Sales Estimate Other Years | Sales Estimate Next Twelve Months |
| `NextFYEPSStdDev` | EPS Estimate Next Year |  |
| `NextFYSalesStdDev` | Sales Estimate Next Year |  |
| `NextQEPSStdDev` | EPS Estimate Next Quarter |  |
| `NextYRevRatio4W` | Estimate Revision Direction | Ratio ranging from -1 to +1 indicating the direction of revisions for the stock's industry. The extreme values 1 and -1 would indicate that every analyst's NextY estimate in the industry have been rev |
| `NoPosEPS5Q` | Income Trend | Number of quarter with positive EPS of the previous 5 |
| `Pr2SalesNTM` | Price to Sales Projected | Price to Sales Next Twelve Months |
| `Pr2SalesNTMInd` | Price To Sales, Industry | Price To Sales Next Twelve Months |
| `Pr2SalesPY` | Price to Sales | Market cap to total revenues. |
| `Pr2SalesTTMInd` | Price To Sales, Industry | Price to Sales Ratio Industry, TTM |
| `SE` | Standard Error of the Y Estimate (Sy.x) | Returns the SE of the Y estimate for a previously computed regression |
| `SGA2Sales%5Y` | Selling,Gen,Admin to Sales % | Operating Margin, 5 Year Factor (%) |
| `SGA2Sales%5YAvg` | Selling,Gen,Admin to Sales % | Selling, general and administrative expenses (including R&D) as percentage of total sales. |
| `Sales5YAvg` | Sales (Revenues) | Total value of goods/services sold in a period. |
| `SalesActual()` | Sales Actual | Historical (Actual) Sales |
| `SalesActualGr%PYQ` | Sales Actual | Sales Actual Growth Previous Year Q (PYQ) |
| `SalesActualGr%TTM` | Sales Actual | Sales Actual Growth Trailing Twelve Months (TTM) |
| `SalesActualPTM` | Sales Actual | Sales Actual Previous Twelve Months (PTM) |
| `SalesActualQ1` | Sales Actual | Sales Actual, 1 Quarter Ago |
| `SalesActualQ2` | Sales Actual | Sales Actual, 2 Quarters Ago |
| `SalesActualQ3` | Sales Actual | Sales Actual, 3 Quarters Ago |
| `SalesActualQ4` | Sales Actual | Sales Actual, 4 Quarters Ago |
| `SalesActualQ5` | Sales Actual | Sales Actual, 5 Quarters Ago |
| `SalesActualTTM` | Sales Actual | Sales Actual Trailing Twelve Months (TTM) |
| `SalesEst()` | Historical Sales Estimate | Historical Sales Estimate |
| `SalesEstQ1` | Historical Sales Estimate | Sales Estimate, 1 Quarter Ago |
| `SalesEstQ2` | Historical Sales Estimate | Sales Estimate, 2 Quarters Ago |
| `SalesEstQ3` | Historical Sales Estimate | Sales Estimate, 3 Quarters Ago |
| `SalesEstQ4` | Historical Sales Estimate | Sales Estimate, 4 Quarters Ago |
| `SalesEstQ5` | Historical Sales Estimate | Sales Estimate, 5 Quarters Ago |
| `SalesGr%3YInd` | Industry Growth | Sales Growth Rate Industry, 3 Years (%) |
| `SalesGr%5YInd` | Industry Growth | Sales Industry, 5 Year Growth Rate (%) |
| `SalesGr%AInd` | Industry Growth | Sales percent change for the industry, recent Y vs prior Y |
| `SalesGr%PYQInd` | Industry Growth | Sales percent change for the industry, recent Q vs Q 1 year ago |
| `SalesGr%TTMInd` | Industry Growth | Sales percent change for the industry, recent TTM vs prior TTM |
| `SalesHistEstCnt()` | Historical Estimate Number of Analysts | Historical Sales Estimate Number of Analysts |
| `SalesHistEstSD()` | Historical Estimate Standard Deviation | Historical Sales Estimate Standard Deviation |
| `SalesPS5YAvg` | Sales (Revenues) Per Share | Total revenue divided by fully-diluted average shares outstanding. |
| `SalesPerEmp5YAvg` | Sales Per Employee | Sales Per Employee. |
| `SalesPerEmpTTMInd` | Sales per Employee, Industry | Sales Per Employee Industry, TTM |
| `SalesSUS()` | Unexpected Sales (SUS) | Standardized Unexpected Sales general formula |
| `SalesSurp%Q1` | Analyst Sales Surprise | Sales Surprise (Estimated vs. Actual), 1 Quarter Ago (%) |
| `SalesSurp%Q2` | Analyst Sales Surprise | Sales Surprise (Estimated vs. Actual), 2 Quarters Ago (%) |
| `SalesSurp%Q3` | Analyst Sales Surprise | Sales Surprise (Estimated vs. Actual), 3 Quarters Ago (%) |
| `SalesSurp%Q4` | Analyst Sales Surprise | Sales Surprise (Estimated vs. Actual), 4 Quarters Ago (%) |
| `SalesSurp%Q5` | Analyst Sales Surprise | Sales Surprise (Estimated vs. Actual), 5 Quarters Ago (%) |
| `SalesSurp%Y1` | Analyst Sales Surprise | Sales Surprise (Estimated vs. Actual), Most recent year (%) |
| `SalesSurp%Y2` | Analyst Sales Surprise | Sales Surprise (Estimated vs. Actual), 2 Years Ago (%) |
| `SalesSurp%Y3` | Analyst Sales Surprise | Sales Surprise (Estimated vs. Actual), 3 Years Ago (%) |
| `SalesSurp%Y4` | Analyst Sales Surprise | Sales Surprise (Estimated vs. Actual), 4 Years Ago (%) |
| `SalesSurprise()` | Analyst Sales Surprise | Sales Surprise in % |
| `Surprise%Q1` | Analyst EPS Surprise | Earnings Surprise (Estimated vs. Actual), 1 Quarter Ago (%) |
| `Surprise%Q2` | Analyst EPS Surprise | Earnings Surprise (Estimated vs. Actual), 2 Quarters Ago (%) |
| `Surprise%Q3` | Analyst EPS Surprise | Earnings Surprise (Estimated vs. Actual), 3 Quarters Ago (%) |
| `Surprise%Q4` | Analyst EPS Surprise | Earnings Surprise (Estimated vs. Actual), 4 Quarters Ago (%) |
| `Surprise%Q5` | Analyst EPS Surprise | Earnings Surprise (Estimated vs. Actual), 5 Quarters Ago (%) |
| `Surprise%Y1` | Analyst EPS Surprise | Earnings Surprise (Estimated vs. Actual), Most recent year (%) |
| `Surprise%Y2` | Analyst EPS Surprise | Earnings Surprise (Estimated vs. Actual), 2 Year Ago (%) |
| `Surprise%Y3` | Analyst EPS Surprise | Earnings Surprise (Estimated vs. Actual), 3 Year Ago (%) |
| `Surprise%Y4` | Analyst EPS Surprise | Earnings Surprise (Estimated vs. Actual), 4 Year Ago (%) |

## Institutional Data

**Count: 51**

| Factor Code | Category/Name | Description |
|-------------|---------------|-------------|
| `#Institution` | Institutional Factors | The total number of investors who own shares of the company |
| `Ins#ShrPurch` | Insider Factors | Insider total shares purchased past 6 months (positive number in millions) |
| `Ins#ShrSold` | Insider Factors | Insider total shares sold in the past 6 months (negative number in millions) |
| `InsBuyTrans` | Insider Factors | Insider number of BUY transactions in the past 6 months |
| `InsNetShrPurch` | Insider Factors | Insider total shares net in the past 6 months (positive or negative number in millions) |
| `InsNetTrans` | Insider Factors | Insider NET number of transactions in the past 6 months |
| `InsSelTrans` | Insider Factors | Insider number of SELL transactions in the past 6 months |
| `Insider#Own` | Insider Factors | Common stock in millions held by the officers and directors of the company plus beneficial owners who own more than 5 percent |
| `Insider%Own` | Insider Factors | Percent of common stock held by the officers and directors of the company plus beneficial owners who own more than 5 percent |
| `InsiderBuySh12M()` | Insider Functions | Shares bought by insiders past 12 months (in millions) |
| `InsiderBuySh1M()` | Insider Functions | Shares bought by insiders past 1 month (in millions) |
| `InsiderBuySh3M()` | Insider Functions | Shares bought by insiders past 3 months (in millions) |
| `InsiderBuySh6M()` | Insider Functions | Shares bought by insiders past 6 months (in millions) |
| `InsiderBuyTran12M()` | Insider Functions | Buy transactions by insiders past 12 months (in millions) |
| `InsiderBuyTran1M()` | Insider Functions | Buy transactions by insiders past 1 month (in millions) |
| `InsiderBuyTran3M()` | Insider Functions | Buy transactions by insiders past 3 months (in millions) |
| `InsiderBuyTran6M()` | Insider Functions | Buy transactions by insiders past 6 months (in millions) |
| `InsiderSellSh12M()` | Insider Functions | Shares sold by insiders past 12 months (in millions) |
| `InsiderSellSh1M()` | Insider Functions | Shares sold by insiders past 1 month (in millions) |
| `InsiderSellSh3M()` | Insider Functions | Shares sold by insiders past 3 months (in millions) |
| `InsiderSellSh6M()` | Insider Functions | Shares sold by insiders past 6 months (in millions) |
| `InsiderSellTran12M()` | Insider Functions | Sell transactions by insiders past 12 months |
| `InsiderSellTran1M()` | Insider Functions | Sell transactions by insiders past 1 month |
| `InsiderSellTran3M()` | Insider Functions | Sell transactions by insiders past 3 months |
| `InsiderSellTran6M()` | Insider Functions | Sell transactions by insiders past 6 months |
| `InsiderUniqBuy1M()` | Insider Functions | Unique number of insiders buying past 1 month |
| `InsiderUniqBuy3M()` | Insider Functions | Unique number of insiders buying past 3 months |
| `InsiderUniqSell1M()` | Insider Functions | Unique number of insiders selling past 1 month |
| `InsiderUniqSell3M()` | Insider Functions | Unique number of insiders selling past 3 months |
| `Inst#ShsOwn` | Institutional Factors | The number of shares of the company held by investors in the latest period (in millions) |
| `Inst#ShsOwnPQ` | Institutional Factors | The number of shares of the company held by investors in the previous period (in millions) |
| `Inst#ShsPurch` | Institutional Factors | The number of shares of the company purchased by investors during the latest period (in millions) |
| `Inst#ShsPurchPQ` | Institutional Factors | The number of shares of the company purchased by investors during the previous period (in millions) |
| `Inst#ShsSold` | Institutional Factors | The number of shares of the company sold by investors during the latest period (in millions) |
| `Inst#ShsSoldPQ` | Institutional Factors | The number of shares of the company sold by investors during the previous period (in millions) |
| `Inst%Own` | Institutional Factors | The percentage of shares outstanding of the company owned by institutional shareholders in the latest period |
| `Inst%OwnInd` | Institutional % Owned, Industry | Institutional Percent Owned Industry, (%) average |
| `Inst%OwnPQ` | Institutional Factors | The percentage of shares outstanding of the company owned by institutional shareholders in the previous period |
| `InstNetPurch` | Institutional Factors | The net number of shares of the company transacted by investors during the latest period (in millions) |
| `InstNetPurchPQ` | Institutional Factors | The net number of shares of the company transacted by investors during the previous period (in millions) |
| `InstitutionalBuyers()` | Institutional Functions | The number of investors who purchased shares of the company during the period |
| `InstitutionalClosed()` | Institutional Functions | The number of investors who sold all shares and closed their holding position in the company during the period |
| `InstitutionalHolders()` | Institutional Functions | The total number of investors who own shares of the company |
| `InstitutionalNewBuyers()` | Institutional Functions | The number of investors who opened a new position in the company by purchasing shares during the period |
| `InstitutionalPctChg()` | Institutional Functions | The net shares changed as a percent of shares outstanding |
| `InstitutionalPctOwn()` | Institutional Functions | The percentage of shares outstanding of the company owned by institutional shareholders |
| `InstitutionalSellers()` | Institutional Functions | The number of investors who sold shares of the company during the period |
| `InstitutionalShsBought()` | Institutional Functions | The number of shares of the company purchased by investors during the period (in millions) |
| `InstitutionalShsHeld()` | Institutional Functions | The number of shares of the company held by investors at the end of the period (in millions) |
| `InstitutionalShsNet()` | Institutional Functions | The net number of shares of the company transacted by investors during the period (in millions) |
| `InstitutionalShsSold()` | Institutional Functions | The number of shares of the company sold by investors during the period (in millions) |

## Constants & IDs

**Count: 57**

| Factor Code | Category/Name | Description |
|-------------|---------------|-------------|
| `#APeriods` | Number of Periods | Number of Historical Periods, Annual |
| `#AnalystsLTGrthRt` | Long Term EPS Growth | This is the number of analysts who are reporting a long term earnings per share growth rate. If a security has no analysts, this factor returns NA, not 0. |
| `#AnalystsPriceTarget` | Price Target | Number of analysts giving price target estimates. Note: if a security has no analysts, this factor returns NA, not 0. |
| `#BOND20YR` | Compustat Macro - TO BE DEPRECATED | 20-Year Treasury Yield (monthly) |
| `#BOND30YR` | Compustat Macro - TO BE DEPRECATED | 30-Year Treasury Yield (monthly) |
| `#CABGDP2` | Compustat Macro - TO BE DEPRECATED | Total Current Account Balance of the United States as a percentage of GDP (*quarterly data, 3 equal monthly values) |
| `#CPI` | Compustat Macro - TO BE DEPRECATED | Consumer Price Index for Urban Consumers, All Items (monthly) |
| `#EMPLOY` | Compustat Macro - TO BE DEPRECATED | Total Non-Farm Employment, Thousands of People (monthly) |
| `#FEDFUNDS` | Compustat Macro - TO BE DEPRECATED | Fed Funds Rate, in percentage points (monthly) |
| `#GDP` | Compustat Macro - TO BE DEPRECATED | Real Gross Domestic Product, Billions of Chained 2009 Dollars (*quarterly data, 3 equal monthly values) |
| `#HOUSE` | Compustat Macro - TO BE DEPRECATED | Total New Housing Starts, Privately Owned Residences, in thousands of units (monthly) |
| `#M1` | Compustat Macro - TO BE DEPRECATED | M1 Money Stock, in billions of dollars (monthly) |
| `#M2` | Compustat Macro - TO BE DEPRECATED | M2 Money Stock, in billions of dollars (monthly) |
| `#NOTE10YR` | Compustat Macro - TO BE DEPRECATED | 10-Year Treasury Yield (monthly) |
| `#NOTE2YR` | Compustat Macro - TO BE DEPRECATED | 2-Year Treasury Yield (monthly) |
| `#NOTE3YR` | Compustat Macro - TO BE DEPRECATED | 3-Year Treasury Yield (monthly) |
| `#NOTE5YR` | Compustat Macro - TO BE DEPRECATED | 5-Year Treasury Yield (monthly) |
| `#NOTE7YR` | Compustat Macro - TO BE DEPRECATED | 7-Year Treasury Yield (monthly) |
| `#POPT` | Compustat Macro - TO BE DEPRECATED | Total U.S. Population (*annual data, 12 equal monthly values) |
| `#PPI` | Compustat Macro - TO BE DEPRECATED | Producer Price Index (PPI), Finished Goods (monthly) |
| `#PRIME` | Compustat Macro - TO BE DEPRECATED | Prime Rate, in percentage points (monthly) |
| `#QPeriods` | Number of Periods | Number of Historical Periods, Quarterly |
| `#RTLSALES` | Compustat Macro - TO BE DEPRECATED | Retail Sales, excluding Food Services, in millions of dollars (monthly) |
| `#TBILL12M` | Compustat Macro - TO BE DEPRECATED | 12-Month Treasury Yield (monthly) |
| `#TBILL3M` | Compustat Macro - TO BE DEPRECATED | 3-Month Treasury Yield (monthly) |
| `#TBILL6M` | Compustat Macro - TO BE DEPRECATED | 6-Month Treasury Yield (monthly) |
| `#UNEMP` | Compustat Macro - TO BE DEPRECATED | Unemployment Rate, in percentage points (monthly) |
| `#USDAUD` | Macro - FX Rates | USD to AUD |
| `#USDBAM` | Macro - FX Rates | USD to BAM |
| `#USDBGN` | Macro - FX Rates | USD to BGN |
| `#USDCAD` | Macro - FX Rates | USD to CAD |
| `#USDCHF` | Macro - FX Rates | USD to CHF |
| `#USDCZK` | Macro - FX Rates | USD to CZK |
| `#USDDKK` | Macro - FX Rates | USD to DKK |
| `#USDEUR` | Macro - FX Rates | USD to EUR |
| `#USDGBP` | Macro - FX Rates | USD to GBP |
| `#USDHKD` | Macro - FX Rates | USD to HKD |
| `#USDHRK` | Macro - FX Rates | USD to HRK |
| `#USDHUF` | Macro - FX Rates | USD to HUF |
| `#USDILS` | Macro - FX Rates | USD to ILS |
| `#USDISK` | Macro - FX Rates | USD to ISK |
| `#USDJPY` | Macro - FX Rates | USD to JPY |
| `#USDLVL` | Macro - FX Rates | USD to LVL |
| `#USDMKD` | Macro - FX Rates | USD to MKD |
| `#USDMXN` | Macro - FX Rates | USD to MXN |
| `#USDNOK` | Macro - FX Rates | USD to NOK |
| `#USDNZD` | Macro - FX Rates | USD to NZD |
| `#USDPLN` | Macro - FX Rates | USD to PLN |
| `#USDRON` | Macro - FX Rates | USD to RON |
| `#USDRSD` | Macro - FX Rates | USD to RSD |
| `#USDRUB` | Macro - FX Rates | USD to RUB |
| `#USDSEK` | Macro - FX Rates | USD to SEK |
| `#USDSGD` | Macro - FX Rates | USD to SGD |
| `#USDSKK` | Macro - FX Rates | USD to SKK |
| `#USDTRY` | Macro - FX Rates | USD to TRY |
| `#USDUAH` | Macro - FX Rates | USD to UAH |
| `#USDZAR` | Macro - FX Rates | USD to ZAR |

## Functions

**Count: 251**

| Factor Code | Category/Name | Description |
|-------------|---------------|-------------|
| `ADX()` | Average Directional Movement | Returns the Average Directional movement Index as defined by Welles Wilder. An offset can be specified to find trends in the ADX. For example to screen for stocks whose ADX increased in the past 10 ba |
| `AIFactor()` | Predictions | Returns the inferences (predictions) of your trained Predictor. Mainly for use to rebalance but it can also be used in backtests with some restrictions. |
| `AIFactorValidation()` | Validation Predictions | Returns the saved inference (prediction) of your model's validation |
| `ATR()` | Average True Range | Returns the Average True Range as defined by Welles Wilder over the specified period. |
| `ATRN()` | Average True Range Normalized | Returns ATR as a percentage of the closing price. ATRN provides a better way to compare stocks with different price magnitude. |
| `Abs()` | Absolute value | Evaluates to the absolute value of expr |
| `Account()` | Account Holdings | Return TRUE if stock being analyzed is an open position. For the id parameters use either the name in quotes, or the numerical id. |
| `AccountClose()` | Account Holdings | Returns the number of calendar days since the position was last closed for the stock being analyzed, -1 if currently held, or NA if not held within the past 6 months. |
| `AccountCloseBar()` | Account Holdings | Returns the number of bars since the position was last closed for the stock being analyzed, -1 if currently held, or NA if not held within the past 6 months. |
| `AccountOpen()` | Account Holdings | Returns the number of calendar days since the position was first opened for the stock being analyzed, otherwise NA. |
| `AccountOpenBar()` | Account Holdings | Returns the number of bars since the position was first opened for the stock being analyzed, otherwise NA. |
| `Actual()` | Actual Functions | Retrieve the actual item for a specific period indicated by the offset and period type |
| `ActualA()` | Actual Functions | Retrieve the actual annual item for a specific year indicated by the offset with zero being the most recent |
| `ActualGr%A()` | Actual Functions | Calculates the most recent year growth for a particular actual item |
| `ActualGr%PQ()` | Actual Functions | Calculates the most recent quarter growth for a particular actual item |
| `ActualGr%PYQ()` | Actual Functions | Calculates the most recent quarter vs. prior year quarter growth for a particular actual item |
| `ActualGr%TTM()` | Actual Functions | Calculates the most recent twelve months (TTM) growth for a particular actual item |
| `ActualInterimDays()` | Actual Interim Days | Days in the filing period for the given interim offset. Usually 91 or 92 for quarterly, 182 or 183 for semiannual. |
| `ActualPQ()` | Actual Functions | Retrieve the actual item for the previous quarter |
| `ActualPTM()` | Actual Functions | Retrieve the actual item for the previous twelve months (PTM) |
| `ActualPY()` | Actual Functions | Retrieve the actual item for the previous year |
| `ActualPYQ()` | Actual Functions | Retrieve the actual item for the previous year quarter |
| `ActualQ()` | Actual Functions | Retrieve the actual item for the most recent quarter |
| `ActualTTM()` | Actual Functions | Retrieve the actual item for the most recent twelve months (TTM) |
| `Aggregate()` | Group Average | Returns the average or cap-weighted average for each scope |
| `Avg()` | Average of a set of values | Returns the average value in list. Up to 20 parameters are allowed and NAs are discarded |
| `AvgDailyTot()` | Liquidity | Average daily total amount traded (price * volume) for the past number of bars. You can use the offset to calculate rising avg. For ex: |
| `AvgVol()` | Average Volume | Daily average volume of the past number of bars. |
| `BBLower()` | Bollinger Band | Lower Bollinger band value |
| `BBUpper()` | Bollinger Band | Upper Bollinger band value |
| `BarsSince()` | Date | Returns the number of bars since the given date. Date is a number in this format YYYYMMDD. |
| `BenchClose()` | Benchmark Price | Historical Close price of the benchmark. Ex: |
| `BenchHi()` | Benchmark Price | Historical high price of the benchmark. To get the last high enter |
| `BenchLow()` | Benchmark Price | Historical low price of the benchmark. To get the last low enter |
| `BenchOpen()` | Benchmark Price | Historical open price of the benchmark. To get the last open enter: |
| `BetaFunc()` | Beta |  |
| `Between()` | Between | Returns TRUE (1) if value is between min and max (inclusive), FALSE(0) otherwise |
| `Bound()` | Constrains to a max and/or a min | Constrains a value to a maximum or a minimum. When returnNA is set to TRUE the function returns NA if expression exceeds the minimum or maximum. |
| `CCI()` | Commodity Channel Index | Commodity Channel Index. |
| `ChaikinAD()` | Chaikin Accumulation Distribution | The Chaikin Accumulation Distribution Indicator is a measure of the money flowing into and out of a stock over the period |
| `ChaikinMFP()` | Chaikin Money Flow | This indicator calculates the percentage of days in the previous lookback window that the ChaikinAD is > 0 |
| `ChaikinTrend()` | Chaikin Trend | The ChaikinTrend Indicator is a special purpose double smoothed exponential average |
| `Close()` | Price OHLC | Historical close price 'bars' or 'trading days' ago (the length of a bar can be determined by the series used). Use negative values to peek into the future. |
| `CloseAdj()` | Price Special | Historical Close adjusted for splits even when used in the past (only availalble in the Series Tool). |
| `CloseExDiv()` | Price Special | Historical close unadjusted by dividends. When this function is evaluated in the past it reverses out future splits (if any). |
| `Close_D()` | Price OHLC | Historical daily close price 'days' ago including holidays. Holidays are filled in with previous day close. |
| `Close_W()` | Price OHLC | Historical weekly close price. Use 0 for the most recent week, 1 for the week before, etc. |
| `CoName()` | Company Name | Returns 1(TRUE) if the company name is in the list, 0 (FALSE) otherwise. You can also use wildcards * to match any string or ? for any character |
| `ConsRec()` | Recommendation Function | Consensus recommendation |
| `Correl()` | Correlation coefficient | This function returns the correlation coefficient between the specified series |
| `Country()` | Country of domicile |  |
| `CrossOver()` | Moving Avg Cross Over/Under | Returns TRUE(1) when the MA(period1) is over MA(period2) and the cross happened 'bars' ago or less. Otherwise it returns FALSE(0). |
| `CrossUnder()` | Moving Avg Cross Over/Under | Returns TRUE(1) when the MA(period1) is below MA(period2) and the cross happened 'bars' ago or less. Otherwise it returns FALSE(0). |
| `DMICrossOver()` | Directional Movement Cross | Returns TRUE (or 1) if the DMI+ has crossed over the DMI- in the previous bars. DMI+ and DMI- are component of Welles Wilder DMI. To find stocks whose DMI+ crossed over the DMI- within the last 2 bars |
| `DMICrossUnder()` | Directional Movement Cross | Returns TRUE (or 1) if the DMI- has crossed over the DMI+ in the previous bars. DMI+ and DMI- are component of Welles Wilder DMI. To find stocks whose DMI- crossed over the DMI+ within the last 2 bars |
| `DMIMinus()` | Directional Movement Indicator | Returns the DMI- component of the Directional Movement System developed by Welles Wilder. An offset can be specified to calculate past values. To find stocks with strong negative directional movement  |
| `DMIPlus()` | Directional Movement Indicator | Returns the DMI+ component of the Directional Movement System developed by Welles Wilder. An offset can be specified to calculate past values. To find stocks with strong positive directional movement  |
| `DaysDiff()` | Date | Returns the number of days between the given dates. Dates are numbers in this format YYYYMMDD. |
| `DaysSince()` | Date | Returns the number of days since the given date. Date is a number in this format YYYYMMDD. |
| `DivPS()` | Dividends in a Filing Period | Returns the sum (or count) of the dividends per share that were paid in the period specified. By default it returns the sum of REGULAR dividends using EX DATES. See Full description for details. |
| `DivPSDays()` | Dividends in a Time Period | Sum or count of the dividends paid in the past number of days. By default it sums regular dividends with ex-dates in the period. See the full description for more details. |
| `EBITDAActual()` | EBITDA Actual | EBITDA Actual function |
| `EMA()` | Exponential Moving Average | Exponential moving average of a time series. Period is in 'bars' or 'trading days ago' (can vary depending on the series you choose). |
| `EMA_D()` | Exponential Moving Average | Exponential moving average of a time series. Period is in 'days' which include holidays. |
| `EMA_W()` | Exponential Moving Average | Exponential moving average of the weekly time series. |
| `ETFAssetClassSet()` | ETF Asset Class | Return 1 (TRUE) if the ETF Asset Class is one of the parameters |
| `ETFCountrySet()` | ETF Country | Return 1 (TRUE) if the ETF Country is one of the parameters |
| `ETFFamilySet()` | ETF Family | Return 1 (TRUE) if the ETF Family is one of the parameters |
| `ETFMethodSet()` | ETF Method | Return 1 (TRUE) if the ETF Method is one of the parameters |
| `ETFRegionSet()` | ETF Region | Return 1 (TRUE) if the ETF Region is one of the parameters |
| `ETFSectorSet()` | ETF Sector | Return 1 (TRUE) if the ETF Sector is one of the parameters |
| `ETFSizeSet()` | ETF Size | Return 1 (TRUE) if the ETF Size is one of the parameters |
| `ETFStyleSet()` | ETF Style | Return 1 (TRUE) if the ETF Style is one of the parameters |
| `Eval()` | Evaluate condition and return one of two values |  |
| `ExchCountry()` | Exchange Country |  |
| `FCount()` | Group Count | Counts the number of stocks in each scope where formula is true (non 0) |
| `FHist()` | Point In Time Value | Returns the value of 'formula' calculated in the past (or future when using negative values). Split/dividend sensitive values adjusted to the as-of date (observation date). |
| `FHistAvg()` | Historical Average | Returns the average of 'formula' sampled multiple times in the past. Split/dividend sensitive values adjusted to the as-of date (observation date). |
| `FHistMax()` | Historical Max | Returns the maximum value of 'formula' sampled multiple times in the past. Split/dividend sensitive values adjusted to the as-of date (observation date). |
| `FHistMed()` | Historical Median | Returns the median of 'formula' sampled multiple times in the past. Split/dividend sensitive values adjusted to the as-of date (observation date). |
| `FHistMin()` | Historical Min | Returns the minimum value of 'formula' sampled multiple times in the past. Split/dividend sensitive values adjusted to the as-of date (observation date). |
| `FHistRank()` | FHist Rank | Returns the percentile rank from 0 to 100 of most recent value vs. previous PIT values. |
| `FHistRel()` | FHist Relative | Returns the relative value from 0 to 1 of most recent value vs. previous minimum and maximum PIT values. |
| `FHistRelStdDev()` | Historical Relative Standard Deviation | Returns the RSD of 'formula' sampled multiple times in the past. Split/dividend sensitive values adjusted to the as-of date (observation date). |
| `FHistStdDev()` | Historical Standard Deviation | Returns the SD of 'formula' sampled multiple times in the past. Split/dividend sensitive values adjusted to the as-of date (observation date). |
| `FHistSum()` | Historical Sum | Returns the sum of 'formula' sampled multiple times in the past. Split/dividend sensitive values adjusted to the as-of date (observation date). |
| `FHistZScore()` | FHist ZScore | Returns the zscore of most recent value vs. previous PIT values. |
| `FIGI()` | FIGI | Returns 1(TRUE) if the FIGI is in the list, 0 (FALSE) otherwise. This searches the global share class level FIGIs. To add FIGIs to the screen report visit Screen Reports. |
| `FMedian()` | Group Median | Returns the median value of the formula in each scope |
| `FOrder()` | Order in a Group | Sorts stocks based on formula and returns the position in the array. |
| `FRank()` | Rank in a Group | Ranks stocks based on a formula and returns the percentile. |
| `FSum()` | Group Sum | Returns the sum value of the formula in each scope |
| `FXPerf()` | Currency Adjustment Ratio | Ratio to normalize a stock performance to the currency of your system |
| `FlipFlop()` | Flip Flop | Returns TRUE or FALSE depending on the last threshold that was exceeded |
| `Future%Chg()` | Future Return | Future total return |
| `Future%Chg_D()` | Future Return | Future total return |
| `FutureRel%Chg()` | Future Return | Future total return relative to the benchmark or other series |
| `FutureRel%Chg_D()` | Future Return | Future total return relative to the benchmark or other series |
| `GapDown()` | Gap Down | This function returns TRUE (1) when a stock has a Gap Down pattern in the period specified by the 'bars' parameters, with the start bar being specified by 'offset'. |
| `GapUp()` | Gap Up | This function returns TRUE (1) when a stock has a Gap Up pattern in the period specified by the 'bars' parameters, with the start bar being specified by 'offset'. |
| `GetRank()` | Latest Rank | Returns the rank of the ticker |
| `GetRankPos()` | Latest Rank | Returns the rank position of the ticker |
| `GetSeries()` | GetSeries | Use this function in functions that have a 'series' parameter. You can use any stock, ETF, index ticker, or a custom series. If you use a stock ticker, your function may stop working if the ticker cha |
| `Gr%()` | Growth Rate Annualized | Calculates the annual growth rate as 100 * (pow(1 + (a - b) / abs(b), 1/years) -1). When years is 1 or unspecified, calculates growth rate as 100 * ((a - b) / abs(b)). |
| `Hi()` | Price OHLC | Historical High price 'bars' or 'trading days' ago (the length of a bar can be determined by the series used). Use negative values to peek into the future. |
| `Hi_D()` | Price OHLC | Historical High price 'days' ago including holidays. Holidays are filled in with previous day close. |
| `Hi_W()` | Price OHLC | Historical weekly High price. Use 0 for the most recent week, 1 for the week before, etc. |
| `HighPct()` | Percent From Hi/Lo | Percent from high in the period (incl div) |
| `HighPct_W()` | Percent From Hi/Lo | Percent from high in the period using the weekly series (incl div) |
| `HighVal()` | Price Highest/Lowest | This function returns the highest value of the series within the specified lookback period. The close prices are used by default. |
| `HighValBar()` | Price Highest/Lowest | This function returns bar where highest value of the series occurred within the specified lookback period. The close prices are used by default. |
| `Higher()` | Count higher/lower values in a set | Returns the number of times xi > xi+1. For example to find companies that price increased at least 2 out of 3 days enter: |
| `HoldingsCnt()` | Running count of holdings | Returns a count of stocks in the list that are current holdings |
| `Holiday()` | Holiday | Returns TRUE (1) if weekday is a holiday, FALSE (0) otherwise. Use negative values to check for upcoming holidays. (Note: It doesn't support checking future dates.) |
| `InList()` | Stock is in Custom List | Returns 1 if the stock is in your custom list, 0 otherwise. To enter a custom list go to TOOLS->Lists->Custom. Ex: to only buy stocks in your list "MyList" enter the following Buy rule: |
| `InSet()` | Check if an expression is in a set | Returns TRUE (1) if 'expression' is found in the list of parameters. For example, to screen for stocks in one of three industries you can write: |
| `InterimMonths()` | Interim Months | Months in the filing period for the given interim offset (the offset parameter is the number of interim periods you want to look back). Always returns either 3 or 6: 3 for stocks that report quarterly |
| `IsNA()` | Replace NA Number (Not Available) | Returns the value of expr1 if it's not NA, otherwise it returns expr2 |
| `IsNeg()` | Replace Negative Number | Returns the value of expr1 if it's not negative, otherwise it returns expr2 |
| `IsNegOrNA()` | Replace Negative or NA Number | Returns the value of expr1 if it's not negative and not NA, otherwise it returns expr2 |
| `IsZero()` | Replace Zero Number | Returns the value of expr1 if it's not zero, otherwise it returns expr2 |
| `LBound()` | Constrains to a max and/or a min | Constrains a value to a minimum. When returnNA is set to TRUE the function returns NA if expression exceeds the minimum. |
| `LN()` | Natural Log | Returns the natural log of val |
| `LastSellDaysLT()` | Recently sold | LastSellDaysLT(days): Returns TRUE or 1 if the stock was sold within the last no. of days, otherwise it returns FALSE or 0 (LT stands for Less Than). |
| `LinReg()` | Linear Regression using Loop | Executes the "formula" parameter a number of times and operates a regression on the set of values. A special CTR variable must be used inside "formula". |
| `LinRegVals()` | Linear Regressions of Values | Operates a regression on a set of Y values. |
| `LinRegXY()` | Linear XY Regression using Loop | Executes the "formula" parameter a number of times and operates a bivariate regression on the set of values. A special CTR variable must be used inside "formula". |
| `LinRegXYVals()` | Linear Regression of XY Values | Operates a regression on a set of XY values. |
| `Log10()` | Base 10 log | Returns the base-10 log of val |
| `LoopAvg()` | Loop Average | Evaluates the "formula" parameter a number of times and return the average. A special CTR variable must be used inside "formula". |
| `LoopMax()` | Loop Max | Evaluates the "formula" parameter a number of times and return the maximum value. A special CTR variable must be used inside "formula". |
| `LoopMedian()` | Loop Median | Evaluates the "formula" parameter a number of times and returns the median value. A special CTR variable must be used inside "formula". |
| `LoopMin()` | Loop Min | Evaluates the "formula" parameter a number of times and returns the minimum value. A special CTR variable must be used inside "formula". |
| `LoopProd()` | Loop Product | Evaluates the "formula" parameter a number of times and returns the product of the values. A special CTR variable must be used inside "formula". |
| `LoopRank()` | Loop Rank | Returns the rank of most recent value of the Loop formula vs. previous values calculated by iterating the CTR variable. |
| `LoopRel()` | Loop Relative | Returns the relative value from 0 to 1 of most recent value of the Loop formula vs. previous minimum and maximum values calculated by iterating the CTR variable. |
| `LoopRelStdDev()` | Loop Rel Standard Deviation | Evaluates the "formula" parameter a number of times and returns the relative standard deviation. A special CTR variable must be used inside "formula". |
| `LoopStdDev()` | Loop Standard Deviation | Evaluates the "formula" parameter a number of times and returns the standard deviation. A special CTR variable must be used inside "formula". |
| `LoopStreak()` | Loop Streak | Evaluates the "formula" parameter a number of times and returns the streak count. By default it returns the most recent streak of positive values. A special CTR variable must be used inside "formula". |
| `LoopSum()` | Loop Sum | Evaluates the "formula" parameter a number of times and returns the sum of the values. A special CTR variable must be used inside "formula". |
| `LoopZScore()` | Loop ZScore | Returns the zscore of most recent value of the Loop formula vs. previous values calculated by iterating the CTR variable. |
| `Low()` | Price OHLC | Historical Low price 'bars' or 'trading days' ago (the length of a bar can be determined by the series used). Use negative values to peek into the future. |
| `LowPct()` | Percent From Hi/Lo | Percent from low in the period (incl div) |
| `LowPct_W()` | Percent From Hi/Lo | Percent from low in the period using the weekly series (incl div) |
| `LowVal()` | Price Highest/Lowest | This function returns the lowest value of the series within the specified lookback period. The close prices are used by default. |
| `LowValBar()` | Price Highest/Lowest | This function returns bar where lowest value of the series occurred within the specified lookback period. The close prices are used by default. |
| `Low_D()` | Price OHLC | Historical Low price 'days' ago including holidays. Holidays are filled in with previous day close. |
| `Low_W()` | Price OHLC | Historical weekly Low price. Use 0 for the most recent week, 1 for the week before, etc. |
| `Lower()` | Count higher/lower values in a set | Returns the number of times xi < xi+1. For example to find companies that price dropped at least 2 out of 3 days in a row enter: |
| `MACD()` | Moving Avg Converge/Diverge | Moving Average Convergence/Divergence. Returns the difference between a 26-bar and a 12-bar exponential moving average. |
| `MACDD()` | Moving Avg Converge/Diverge | Difference of the MACD with its EMA average (the signal line) when period>1 otherwise returns MACD |
| `Max()` | Calculate the max/min in a set | Returns the largest value in list. Up to 20 parameters are allowed and NAs are discarded |
| `MaxCorrel()` | Maximum Position Correlation | Calculates the maximum correlation coefficient of the stock being evaluated vs. the current holdings. You can use this to avoid buying or holding highly correlated stocks. |
| `Median()` | Median of a set of values | Returns the median value in list. Up to 20 parameters are allowed and NAs are discarded |
| `MedianDailyTot()` | Liquidity | Median daily total amount traded (price * volume) for the past number of bars. |
| `MedianVol()` | Average Volume | Median volume of the past number of bars |
| `Min()` | Calculate the max/min in a set | Returns the smallest value in list. Up to 20 parameters are allowed and NAs are discarded |
| `MinLiquidity()` | Liquidity | Returns the lowest total amount traded (price * volume) for the past number of bars. |
| `Mod()` | Modulus Operator | Modulus operation that returns the remainder of val/modulo |
| `Momentum()` | Momentum | Measures the amount that a security's closing price has changed over a given time span. |
| `MonthBars()` | Trading day in month | Returns 1 (TRUE) if the as-of date is the nth trading day of the month. Negative offsets are evaluated relative to end of month (e.g. -2 is the second-to-last trading day of the month). In multi-count |
| `NAVAvg()` | Net Asset Value (NAV) | Monthly NAV average. |
| `NAVDiscAvg()` | Net Asset Value (NAV) Discount | Monthly NAV Discount average. |
| `NAVDiscHist()` | Net Asset Value (NAV) Discount | Historical monthly NAV Discount. |
| `NAVHist()` | Net Asset Value (NAV) | Historical monthly NAV. |
| `Negate()` | Negate | Changes the sign of the expression. |
| `NodeRank()` | Node Rank |  |
| `OBV()` | On Balance Volume (OBV) | On Balance Volume |
| `OBVSlopeN()` | On Balance Volume (OBV) Slope | Normalized rate of change of the OBV |
| `Open()` | Price OHLC | Historical Open price 'bars' or 'trading days' ago (the length of a bar can be determined by the series used). Use negative values to peek into the future. |
| `Open_D()` | Price OHLC | Historical Open price 'days' ago including holidays. Holidays are filled in with previous day close. |
| `Open_W()` | Price OHLC | Historical weekly Open price. Use 0 for the most recent week, 1 for the week before, etc. |
| `PastCorpAct()` | Corporate Actions (ICE Data) | Returns TRUE (1) if at least one completed or expired corporate action is found, or FALSE(0) |
| `PctAvg()` | Average Percent Change | Calculates the average of the percentage moves for selected period. To calculate the average of the past 20 weeks enter PctAvg(20,5). Note: using a bar length of 5 returns weekly percentage moves if t |
| `PctAvgDailyTot()` | Trade amount as % of liquidity |  |
| `PctDev()` | Standard Deviation (Volatility) | Calculates the standard deviation of the percentage moves of the closing prices. For example to calculate the SD of 50 weekly percentage moves enter: PctDev(50,5) |
| `PendingCorpAct()` | Corporate Actions (ICE Data) | Returns TRUE (1) if at least one corporate action is found, or FALSE(0) |
| `Portfolio()` | Portfolio Holdings | Return TRUE if stock being analyzed is an open position. For the id parameters use either the name in quotes, or the numerical id. |
| `PortfolioClose()` | Portfolio Holdings | Returns the number of calendar days since the position was last closed for the stock being analyzed, -1 if currently held, or NA if not held within the past 6 months. |
| `PortfolioCloseBar()` | Portfolio Holdings | Returns the number of bars since the position was last closed for the stock being analyzed, -1 if currently held, or NA if not held within the past 6 months. |
| `PortfolioOpen()` | Portfolio Holdings | Returns the number of calendar days since the position was first opened for the stock being analyzed, otherwise NA. |
| `PortfolioOpenBar()` | Portfolio Holdings | Returns the number of bars since the position was first opened for the stock being analyzed, otherwise NA. |
| `Pow()` | Power | Returns a number raised to a power |
| `PrcRegEst()` | Price Regression End Value | Returns the ending value of a Regression Line of the prices (incl div) |
| `PrcRegEst_W()` | Price Regression End Value | Returns the ending value of a Regression Line of the weekly prices (incl div) |
| `RBICS()` | Revere Industry Classification (RBICS) | Evaluates to TRUE if any combination of RBICS Sector, Sub-sector, Industry, and Sub-industry matches. |
| `ROC()` | Rate of Change (ROC) | Rate of Change. |
| `RSI()` | Relative Strength Indicator (RSI) | Welles Wilder's Relative Strength Index. Period is in 'bar' or 'trading days' (can depend on the series you specify) |
| `RSI_D()` | Relative Strength Indicator (RSI) | Welles Wilder's Relative Strength Index. Period is in 'days' which include holidays. |
| `RSI_W()` | Relative Strength Indicator (RSI) | Welles Wilder's Relative Strength Index. Period is in 'weeks' which include holidays. |
| `RankPosPrev()` | Previous Rank | Historical weekly rank position based on the selected ranking system. Weekly ranks are updated every Saturday. |
| `RankPrev()` | Previous Rank | Historical weekly rank based on the selected ranking system. Weekly ranks are updated every Saturday. |
| `Rating()` | Other Rank | Returns the rank for the named Ranking System. You can specify one of your systems or one of our pre-built ranking systems. Example: |
| `RatingPos()` | Other Rank | Returns the rank position for the named Ranking System. You can specify one of your systems or one of our pre-built ranking systems. Example: |
| `RegGr%()` | Regression Growth | Returns the growth for a previously computed time series regression. The period parameter can be used to annualize the growth |
| `Rel%Chg()` | Relative Return Incl Div | Total return relative to the benchmark or other series |
| `Rel%Chg_D()` | Relative Return Incl Div | Total return relative to the benchmark or other series |
| `RelStdDev()` | Standard deviation of a set of values | Returns the Relative Standard Deviation (SD divided by the mean) of the parameters. |
| `Ret%Chg()` | Return Incl Div (Total Return) | Total return (includes dividends) in the period specified |
| `Ret%Chg_D()` | Return Incl Div (Total Return) | Total return (includes dividends) in the period specified |
| `SMA()` | Simple Moving Average | Simple moving average of a time series. Period is in 'bars' or 'trading days ago' (can vary depending on the series you choose). |
| `SMAPct()` | Percent From Average | Percent from SMA (incl div) |
| `SMAPct_W()` | Percent From Average | Percent from SMA using the weekly series (incl div) |
| `SMA_D()` | Simple Moving Average | Simple moving average of a time series. Period is in 'days' which include holidays. |
| `SMA_W()` | Simple Moving Average | Simple moving average of the weekly time series. |
| `Screen()` | Screen Holdings | Runs the screen specified in the first parameter. If top>0 then only the specified top stocks are returned. With this you can create screen-of-screens. |
| `SetVar()` | Set Variable | Sets the variable @myvar to the expression and returns TRUE. You can use the variable in subsequent rules. |
| `SharesCur()` | Common Shares Most Recent |  |
| `Sharpe()` | Sharpe Ratio | Returns a sharpe-like ratio. Unlike the normal Sharpe ratio, it is not adjusted for the risk-free return. |
| `ShowCorrel()` | Show Correlation Matrix in report | This function produces a correlation matrix in the screen report |
| `ShowVar()` | Show/Set Variable function | Sets the variable @myvar to expression, returns TRUE, and displays @myvar in the screen report. |
| `Sortino()` | Sortino Ratio | Returns a sortino-like ratio. Unlike the normal Sortino ratio, it is not adjusted for the risk-free return. |
| `SplitCount()` | Splits | Returns the number of splits in the past or future number of days |
| `SplitFactor()` | Splits | Returns the compounded split ratio in the past or future number of days |
| `Splits()` | Splits | Returns the compounded split ratio in the past or future number of days. Set the second parameter to TRUE to return the number of splits. |
| `Spread()` | Bid Ask Spread | Returns the closing spread (ask-bid) for the bar. Data is from ICE Data (formerly Interactive Data Corp). |
| `Spread_D()` | Bid Ask Spread | Returns the closing spread (ask-bid) for the day including holidays. Holidays are filled in with the spread from the previous day. Data is from ICE Data (formerly Interactive Data Corp). |
| `StdDev()` | Standard deviation of a set of values | Returns the Standard Deviation of the parameters. |
| `StochD()` | Stochastic Oscillator | Returns the Stochastic %D value |
| `StochK()` | Stochastic Oscillator | Returns the Stochastic %K value |
| `SurpriseY()` | Regression Surprise |  |
| `Ticker()` | Ticker | Returns 1(TRUE) if the ticker is in the list, 0 (FALSE) otherwise. Use commas or spaces to separate your list. You can also use wildcards * to match any string or ? for any character |
| `Trunc()` | Truncate number | Rounds val toward zero |
| `UBound()` | Constrains to a max and/or a min | Constrains a value to a maximum. When returnNA is set to TRUE the function returns NA if expression exceeds the maximum. |
| `ULTOSC()` | Ultimate Oscillator | Ultimate Oscillator. |
| `UnivAvg()` | Universe Average | Calculate the simple average of the values of "formula" for the stocks that pass "criteria" |
| `UnivCapAvg()` | Universe Cap Average | Calculate the cap-weighted average of the values of "formula" for the stocks that pass "criteria" |
| `UnivCnt()` | Universe Count | Count stocks that pass "criteria" |
| `UnivExclude()` | Exclude by ticker | Exclude stocks in universe with specific tickers |
| `UnivMax()` | Universe Maximum | Calculate the maximum value of "formula" for the stocks that pass "criteria" |
| `UnivMedian()` | Universe Median | Calculate the median of the values of "formula" for the stocks that pass "criteria" |
| `UnivMin()` | Universe Minimum | Calculate the minimum value of "formula" for the stocks that pass "criteria" |
| `UnivRBICS()` | Filter by RBICS | Only use stocks in universe with specific RBICS |
| `UnivStdDev()` | Universe Standard Deviation | Calculate the standard deviation of the values of "formula" for the stocks that pass "criteria" |
| `UnivSubset()` | Filter by ticker | Only use stocks in universe with specific tickers |
| `UnivSum()` | Universe Sum | Calculate the sum of the values of "formula" for the stocks that pass "criteria" |
| `Universe()` | Universe filter | Returns 1 if the stock is in the universe, 0 otherwise. Ex: to only buy S&P 500 stocks enter the following Buy rule: |
| `UpDownRatio()` | Up/Down Volume Ratio | Calculates the Up/Down Volume ratio for a stock. If the ratio exceeds 0.5, it means volume on a stock's up days outweighed downside volume in the specified period. It is calculated as: |
| `VMA()` | Volume weighted average | Volume weighted moving average of a time series. Period is in 'bars' or 'trading days ago' (can vary depending on the series you choose). |
| `Vol()` | Volume | Historical volume for a day in the past. |
| `Vol_D()` | Volume | Historical volume 'days' ago including holidays. Holidays are filled in with previous day volume. |
| `Vol_W()` | Volume | Historical weekly volume. Use 0 for the most recent week. 1 for the week before, etc. |
| `WMA()` | Weighted moving average | Weighted moving average of a time series. Period is in 'bars' or 'trading days ago' (can vary depending on the series you choose). |
| `WMA_D()` | Weighted moving average | Weighted moving average of a time series. Period is in 'days' which include holidays. |
| `Watchlist()` | Watchlist Holdings | Return TRUE if stock being analyzed is in the watchlist on the date in question. For the id parameters use either the name in quotes, or the numerical id. |
| `WatchlistClose()` | Watchlist Holdings | Returns the number of calendar days since the stock was last removed, -1 if currently in the watchlist, or NA if not added within the past 6 months. |
| `WatchlistCloseBar()` | Watchlist Holdings | Returns the number of bars since the stock was last removed, -1 if currently in the watchlist, or NA if not added within the past 6 months. |
| `WatchlistCurrent()` | Watchlist Holdings | Returns TRUE if the stock is currently in your watchlist. |
| `WatchlistOpen()` | Watchlist Holdings | Returns the number of calendar days since the stock was first added, otherwise NA. |
| `WatchlistOpenBar()` | Watchlist Holdings | Returns the number of bars since the stock was first added, otherwise NA. |
| `ZScore()` | ZScore in a Group | Calculates how many standard deviations the value from the formula is from the mean |

## Price & Volume

**Count: 14**

| Factor Code | Category/Name | Description |
|-------------|---------------|-------------|
| `LastSellPrice` | Previous sell price for new position | Price the stock was last sold at, or NA. Example: To check that the current price is either above the last sell price, or that the stock was never bought, enter: |
| `PEHigh` | Price to Earnings (other) | Price Earnings Ratio, 5 year High |
| `PEHighInd` | Price To Earnings (PE), Industry | Price Earnings Ratio Industry, 5 year High |
| `PELow` | Price to Earnings (other) | Price Earnings Ratio, 5 year Low |
| `PELowInd` | Price To Earnings (PE), Industry | Price Earnings Ratio Industry, 5 year Low |
| `Price` | Price | Price ($) unadjusted for future splits. Same as Close(0) |
| `PriceH` | Price Highest/Lowest | Price, 12 Month High ($) adjusted for splits and dividends. |
| `PriceL` | Price Highest/Lowest | Price, 12 Month Low ($) adjusted for splits and dividends. |
| `PricePY` | Price | Price, Year Ago ($) adjusted for splits and dividends. |
| `PriceTarget4WkAgo` | Price Target | Analyst mean Price Target 4 weeks ago |
| `PriceTargetHi` | Price Target | High analyst mean Price Target |
| `PriceTargetLo` | Price Target | Low analyst mean Price Target |
| `PriceTargetMean` | Price Target | Analyst mean Price Target 0-18 months out |
| `PriceTargetStdDev` | Price Target | Analyst Price Target standard deviation |

## Ratios & Metrics

**Count: 103**

| Factor Code | Category/Name | Description |
|-------------|---------------|-------------|
| `BenchPct` | Benchmark Return | Benchmark return since the position was opened. |
| `BenchPctFromPosHi` | Benchmark Return From Hi | Benchmark percentage return from highest close of the position. Use this to prevent a stop-loss in case the market is also dropping. For example to enter a stoploss only if the market outperformed the |
| `CapCount` | Market Cap Concentration |  |
| `CapSp5YCGr%Ind` | Industry Growth | Capital Spending Industry, 5 Year Growth Rate (%) |
| `CapWeight` | Market Cap Concentration |  |
| `CurRatio5YAvg` | Current Ratio | Measures company's ability to pay short-term obligations using short-term assets. |
| `CurRatioQInd` | Current Ratio, Industry | Current Ratio Industry, Quarterly |
| `Div%ChgA` | Dividend Growth | Dividend Percent Change, Year Over Year (%) |
| `Div%ChgAInd` | Industry Growth | Dividend Percent Change Industry, Year Over Year (%) |
| `Div%ChgPYQ` | Dividend Growth | Dividend Percent Change, Quarter vs Quarter a year ago (%) |
| `Div3YCGr%` | Dividend Growth | Dividend Growth Rate, 3 Years (uses corporate action data) |
| `Div3YCGr%Ind` | Industry Growth | Dividend Growth Rate Industry, 3 Years (uses corporate action data) |
| `Div5YCGr%` | Dividend Growth | Dividend, 5 Year Growth Rate (uses corporate action data) |
| `Div5YCGr%Ind` | Industry Growth | Dividend Industry, 5 Year Growth Rate (uses corporate action data) |
| `EBITDAMgn%5Y` | EBITDA Margin | EBITDA Margin 5 year (%) |
| `EBITDAMgn%5YAvg` | EBITDA Margin | EBITDA as percentage of revenue. |
| `EBITDAMgn%5YAvgInd` | EBITDA Margin, Industry | EBITDA Margin Industry, 5 year average (%) |
| `EBITDAMgn%5YInd` | EBITDA Margin, Industry | EBITDA Margin Industry, 5 year (%) |
| `EBITDAYield` | EBITDA Yield | The ratio is calculated by dividing the most recent 12 months EBITDA by the current enterprise value. |
| `EarnYield` | Earnings Yield | The earnings per share for the most recent 12 months divided by market price per share. |
| `FCFLMgn%5YAvg` | Free Cash Flow Margin | Free cash flow as a percentage of total revenues. |
| `FundsFromOp5YAvg` | Funds From Operations (FFO) | Earnings adjusted for real estate performance. Non-GAAP metric reported only by REITs. |
| `GMgn%5Y` | Gross Profit Margin | Gross Margin, 5 Year Factor (%) |
| `GMgn%5YAvg` | Gross Profit Margin | Gross profit relative to revenue as a percentage. GMgn% = (Gross Profit ÷ Sales) × 100. |
| `GMgn%5YAvgInd` | Gross Profit Margin, Industry | Gross Margin Industry, 5 Year Average (%) |
| `GMgn%5YInd` | Gross Profit Margin, Industry | Gross Profit Margin Industry, 5 year (%) |
| `GMgn%_GAAP5YAvg` | Gross Profit Margin GAAP | Gross profit relative to revenue as a percentage. GMgn%_GAAP = ((Gross Profit - D&A) ÷ Sales) × 100. |
| `Gain` | Dollar Return of Position | Return of an existing position in dollars |
| `GainPct` | Percent return of Position | Return of an existing position in % (dividends are not included). Ex: to sell a position if it gained 50% enter: |
| `LTGrthStdDev` | Long Term EPS Growth | Long Term EPS Growth Rate (%) |
| `LargeCount` | Market Cap Concentration |  |
| `LargeWeight` | Market Cap Concentration |  |
| `MicroCount` | Market Cap Concentration |  |
| `MicroWeight` | Market Cap Concentration |  |
| `MidCount` | Market Cap Concentration |  |
| `MidWeight` | Market Cap Concentration |  |
| `NAV%Chg12M` | NAV Growth | NAV percent change 12 months |
| `NAV%Chg1M` | NAV Growth | NAV percent change 1 month |
| `NAV%Chg3M` | NAV Growth | NAV percent change 3 months |
| `NAV%Chg6M` | NAV Growth | NAV percent change 6 months |
| `NPMgn%5Y` | Net Profit Margin | Net Margin, 5 Year Factor (%) |
| `NPMgn%5YAvg` | Net Profit Margin | Income after taxes as a percentage of total revenue. |
| `NPMgn%5YAvgInd` | Net Profit Margin, Industry | Net Profit Margin Industry, 5 Year Average (%) |
| `NPMgn%5YInd` | Net Profit Margin, Industry | Net Profit Margin Industry, 5 year (%) |
| `NetFCFLMgn%5YAvg` | Net Free Cash Flow Margin | Net free cash flow as a percentage of total revenues. |
| `OpIncYield` | Operating Income Yield | Operating Income Yield |
| `OpMgn%5Y` | Operating Margin | Operating Margin, 5 Year Factor (%) |
| `OpMgn%5YAvg` | Operating Margin | Percent of revenues after operating expenses. |
| `OpMgn%5YAvgInd` | Operating Margin, Industry | Operating Margin Industry, 5 Year Average (%) |
| `OpMgn%5YInd` | Operating Margin, Industry | Operating Margin Industry, 5 year (%) |
| `OperCashFl5YAvg` | Operating Cash Flow from Operations | Total cash change from operating activities. |
| `PEGLT` | PEG Ratio (long term) | Projected Price/Earnings to Long Term Growth Rate |
| `PEGLTInd` | PEG Ratio, Industry | Projected Price/Earnings to Long Term Growth Rate Industry |
| `PEGLTY` | PEG Ratio (long term) | Projected Price/Earnings to Long Term Growth Rate including Yield |
| `PEGST` | PEG Ratio (short term) | Price/Earnings to Next Year Growth Rate |
| `PEGSTInd` | PEG Ratio, Industry | Price/Earnings to Next Year Growth Rate Industry |
| `PEGSTY` | PEG Ratio (short term) | Price/Earnings to Next Year Growth Rate including Yield |
| `PTMgn%5Y` | Pretax Margin | Pretax Margin, 5 Year Factor (%) |
| `PTMgn%5YAvg` | Pretax Margin | Income before taxes as a percentage of total revenue. |
| `PTMgn%5YAvgInd` | Pretax Margin, Industry | Pretax Margin Industry, 5 Year Average (%) |
| `PTMgn%5YInd` | Pretax Margin, Industry | Pretax Margin Industry, 5 year (%) |
| `PayRatio5Y` | Payout Ratio | Payout Ratio, 5 Year Factor (%) |
| `PayRatio5YAvg` | Payout Ratio | Pividends paid relative to net income. |
| `PayRatio5YAvgInd` | Payout Ratio, Industry | Payout Ratio Industry, 5 Year Average (%) |
| `PayRatio5YInd` | Payout Ratio, Industry | Payout Ratio Industry, 5 Year (%) |
| `Pr13W%Chg` | Return Excl Div | Price percent change during the period. Dividends are not included (see Ret%Chg if you want dividends included) |
| `Pr13WRel%Chg` | Relative Return Excl Div | Price percent change during the period relative to the regional benchmark. Dividends are not included |
| `Pr26W%Chg` | Return Excl Div | Price percent change during the period. Dividends are not included (see Ret%Chg if you want dividends included) |
| `Pr26WRel%Chg` | Relative Return Excl Div | Price percent change during the period relative to the regional benchmark. Dividends are not included |
| `Pr4W%Chg` | Return Excl Div | Price percent change during the period. Dividends are not included (see Ret%Chg if you want dividends included) |
| `Pr4WRel%Chg` | Relative Return Excl Div | Price percent change during the period relative to the regional benchmark. Dividends are not included |
| `Pr52W%Chg` | Return Excl Div | Price percent change during the period. Dividends are not included (see Ret%Chg if you want dividends included) |
| `Pr52WRel%Chg` | Relative Return Excl Div | Price percent change during the period relative to the regional benchmark. Dividends are not included |
| `QuickRatio5YAvg` | Quick Ratio | Immediate liquidity excluding inventory. |
| `QuickRatioQInd` | Quick Ratio, Industry | Quick Ratio Industry, Quarterly |
| `ROA%5YAvg` | Return on Assets | Net income before extraordinary items as a percentage of average total assets. |
| `ROA%5YAvgInd` | Return On Assets, Industry | Return on Average Assets Industry, 5 Year Average (%) |
| `ROE%5YAvg` | Return on Equity | Net income before extraordinary items as a percentage of average common equity. |
| `ROE%5YAvgInd` | Return On Equity, Industry | Return on Average Common Equity Industry, 5 Year Average (%) |
| `ROI%5YAvg` | Return on Investment | Net income plus after-tax interest expense as a percentage of average total capital. |
| `ROI%5YAvgInd` | Return On Investment, Industry | Return on Investment Industry, 5 Year Average (%) |
| `Ret1W%Chg` | Return Incl Div (Total Return) | Total return 1 week |
| `Ret1Y%Chg` | Return Incl Div (Total Return) | Total return 1 year |
| `Ret2Y%Chg` | Return Incl Div (Total Return) | Total return 2 years |
| `Ret3M%Chg` | Return Incl Div (Total Return) | Total return 3 months |
| `Ret4W%Chg` | Return Incl Div (Total Return) | Total return in the period specified |
| `Ret6M%Chg` | Return Incl Div (Total Return) | Total return 6 months |
| `SIRatio` | Short Interest Ratio | Short Interest Ratio |
| `SIRatioPM` | Short Interest Ratio | Short Interest Ratio, 1 Month Ago |
| `SIRatioPM2` | Short Interest Ratio | Short Interest Ratio - 2 Months Ago |
| `SIRatioPM3` | Short Interest Ratio | Short Interest Ratio - 3 Months Ago |
| `ShareholderYield` | Shareholder Yield | Share holder Yield |
| `Sharpe1Y` | Sharpe Ratio | Sharpe1Y: returns the 1 year Sharpe-like ratio for the stock. Unlike the normal Sharpe ratio, it is not adjusted for the risk-free return. Weekly returns are used in the calculations. |
| `Sharpe2Y` | Sharpe Ratio | Sharpe2Y: returns the 2 year Sharpe-like ratio for the stock. Unlike the normal Sharpe ratio, it is not adjusted for the risk-free return. Weekly returns are used in the calculations. |
| `SmallCount` | Market Cap Concentration |  |
| `SmallWeight` | Market Cap Concentration |  |
| `Sortino1Y` | Sortino Ratio | Sortino1Y: returns the 1 year Sortino-like ratio for the stock. Unlike the normal Sortino ratio, it is not adjusted for the risk-free return. Weekly returns are used in the calculations. |
| `Sortino2Y` | Sortino Ratio | Sortino2Y: returns the 2 year Sortino-like ratio for the stock. Unlike the normal Sortino ratio, it is not adjusted for the risk-free return. Weekly returns are used in the calculations. |
| `SusGr%` | Sustainable Growth | Growth Sustainable (%) |
| `Yield` | Dividend Yield | Dividend Yield (%) using the Indicated Annual Dividend (IAD). IAD is a projection of the dividends that will be payed in the next 12 months. |
| `Yield5YAvg` | Dividend Yield | Dividend Yield, 5 Year Average (%) |
| `Yield5YAvgInd` | Yield, Industry | Dividend Yield Industry, 5 Year Average (%) |
| `YieldInd` | Yield, Industry | Dividend Yield Industry (%) |

## Other

**Count: 392**

| Factor Code | Category/Name | Description |
|-------------|---------------|-------------|
| `AccruedExp5YAvg` | Change to Accrued Liabilities (Accrued Expenses) | Change in accrued liabilities on a company's balance sheet. |
| `AccumDep5YAvg` | Accumulated Depreciation | Total depreciation recognized on gross property, plant and equipment (GrossPlant). |
| `Acquis5YAvg` | Acquisitions | Cash outflows for purchasing companies or equity stakes. |
| `AltmanX1` | Altman X Component | Working Capital to Assets |
| `AltmanX2` | Altman X Component | Retained Earnings to Total Assets |
| `AltmanX3` | Altman X Component | Earnings before Interest and Taxes/Total Assets |
| `AltmanX4` | Altman X Component | Market Value of Equity / Book Value of Total Liabilities |
| `AltmanX4Rev` | Altman X Component | Book Value of Equity / Book Value of Total Liabilities |
| `AltmanX5` | Altman X Component | Sales / Total Assets |
| `AltmanZNonManu` | Altman Z-Score | Z-score for non-manufacturers & emerging markets, recommended cutoff is 1.1 or higher |
| `AltmanZOrig` | Altman Z-Score | Original z-score, recommended cutoff is 1.81 or higher |
| `AltmanZPriv` | Altman Z-Score | Z-score estimated for private firms, recommended cutoff is 1.23 or higher |
| `Amort5YAvg` | Amortization of Intangibles | Write down value of non-physical assets like copyrights, patents, and brands. |
| `AnnounceDaysPYQ` | Latest in Database | Returns the number of days it took the company to announce the Prev Year Q filing (announce date - period end) |
| `AnnounceDaysQ` | Latest in Database | Returns the number of days it took the company to announce the latest filing (announce date - period end) |
| `AsOfDate` | As-Of Date | Returns the current trading day as a number in the following format YYYYMMDD |
| `AstCur5YAvg` | Current Assets Total | Sum of all assets expected to convert to cash within 12 months. |
| `AstCurOther5YAvg` | Current Assets Other | Sum of all current assets that are not included in cash, cash equivalents, short-term investments, receivables or inventory. |
| `AstIntan5YAvg` | Intangible Assets | Sum of assets that are not included in the tangible assets of property, plant and equipment. |
| `AstNonCurOther5YAvg` | Non-Current Assets Other | Value of assets that do not fit into either current assets or property plant and equipment. |
| `AstTot5YAvg` | Total Assets | Total value of assets as reported on the balance sheet. |
| `AstTurn5YAvg` | Asset Turnover | Revenue divided by the average total assets for the same period. |
| `AvgRec` | Average Recommendation | Average Recommendation on a 1-3 linear scale, where 1 is a strong buy, 3 a sell. For CapitalIQ the range is 1-5. |
| `AvgRec13WkAgo` | Average Recommendation | Average Recommendation 13 Weeks ago on a 1-3 linear scale, where 1 is a strong buy, 3 a sell. For CapitalIQ the range is 1-5. |
| `AvgRec1WkAgo` | Average Recommendation | Average Recommendation 1 Weeks ago on a 1-3 linear scale, where 1 is a strong buy, 3 a sell. For CapitalIQ the range is 1-5. |
| `AvgRec4WkAgo` | Average Recommendation | Average Recommendation 4 Weeks ago on a 1-3 linear scale, where 1 is a strong buy, 3 a sell. For CapitalIQ the range is 1-5. |
| `AvgRec8WkAgo` | Average Recommendation | Average Recommendation 8 Weeks ago on a 1-3 linear scale, where 1 is a strong buy, 3 a sell. For CapitalIQ the range is 1-5. |
| `AvgVol10` | Average Volume | Daily average volume past 10 bars. Equivalent to AvgVol(10) |
| `AvgVol1M` | Average Volume | Daily average volume past 21 bars. Equivalent to AvgVol(#Month) |
| `AvgVol3M` | Average Volume | Daily average volume past 62 bars. Equivalent to AvgVol(#Month3) |
| `AvgVol5` | Average Volume | Daily average volume past 5 bars. Equivalent to AvgVol(5) |
| `AvgVol6M` | Average Volume | Average volume past 125 bars. Equivalent to AvgVol(#Month6) |
| `BVPS5YAvg` | Book Value Per Share | The per-share value of common equity. |
| `BeneishMScore` | Beneish M-Score | Beneish M-Score finds companies that are associated with increased probability of manipulations. Companies with M-score of -1.89 or lower are safest, while -1.49 or higher are riskiest. |
| `Beta1Y` | Beta | Returns Beta using up to 1 year of weekly returns of the stock and the country's main benchmark. |
| `Beta1YInd` | Beta, Industry | Beta1Y for the industry |
| `Beta3Y` | Beta | Returns Beta using up to 3 years of weekly returns of the stock and the country's main benchmark. Requires a minimum of 70 weekly returns. |
| `Beta3YInd` | Beta, Industry | Beta3Y for the industry |
| `Beta5Y` | Beta | Returns Beta using up to 5 years of weekly returns of the stock and the country's main benchmark. Requires a minimum of 100 weekly returns. |
| `Beta5YInd` | Beta, Industry | Beta5Y for the industry |
| `BookVal5YAvg` | Book Value | Value of common equity excluding preferred shares and minority interests. |
| `BuyAmount` | Buy amount for new position | Amount which will be used to buy a stock before any commissions or slippage. You can use this to set up a liquidity filter, for ex.: |
| `CapEx5YAvg` | Capital Expenditures (CapEx) | Expenditures on property, plant, and equipment investments. |
| `CapExPS5YAvg` | Capital Expenditures (CapEx) Per Share | Capital expenditures divided by shares outstanding. |
| `CapSurplus5YAvg` | Capital Surplus | Amount shareholders paid above par value for shares. |
| `Cash5YAvg` | Cash | Amount of cash and equivalents not including short-term investments. |
| `CashEquiv5YAvg` | Cash and Equivalents | Total cash available. It includes both cash and short-term investments. |
| `CashFl5YAvg` | Cash Flow | Income After Taxes - Preferred Dividends + Depreciation & Amortization |
| `CashFlPS5YAvg` | Cash Flow Per Share | Cash Flow ÷ Fully-Diluted Average Shares Outstanding. |
| `CashFrFin5YAvg` | Cash From Financing | Sum of all financing activity cash flows. |
| `CashFrInvest5YAvg` | Cash from Investing | Net cash flows from investing activities. |
| `CashPS5YAvg` | Cash Per Share | Total cash plus short-term investments divided by fully-diluted shares outstanding. |
| `CashPct` | Cash percentage | Returns the % of the cash in the port/sim vs. the total market value |
| `ChangeDebt5YAvg` | Change to Debt | Net debt activity during the period. |
| `ChangeEq5YAvg` | Change to Equity | Net equity activity during the period. |
| `ComEq5YAvg` | Common Equity | Common shareholders' ownership interest. |
| `CompleteStmt` | Complete Flag | Complete Statement. Set to TRUE (1) if the latest filing is final and, typically, filed with SEC. FALSE (0) if it contains pre-announcement data. When CompleteStmt is FALSE our''fallback'' mechanism k |
| `CostG5YAvg` | Cost of Goods Sold | Direct production expenses. CostG = Direct Materials + Direct Labor (excludes D&A). |
| `CostG_GAAP5YAvg` | Cost of Goods Sold GAAP | Direct production expenses. CostG_GAAP = Direct Materials + Direct Labor + D&A. |
| `CountryCode` | Country Code | Country of domicile |
| `CountryCount` | Country Weight | Number of positions in the country of domicile (could be misleading for companies domiciled in tax havens). For a buy rule, the count is checked assuming the stock is purchased. For a sell rule the co |
| `CurQDnRev4WkAgo` | EPS Revisions Current Quarter | Current Quarter Down Revisions, 4 Weeks ago |
| `CurQDnRevLastWk` | EPS Revisions Current Quarter | Current Quarter Down Revisions, Last Week |
| `CurQUpRev4WkAgo` | EPS Revisions Current Quarter | Current Quarter Up Revisions, 4 Weeks ago |
| `CurQUpRevLastWk` | EPS Revisions Current Quarter | Current Quarter Up Revisions, Last Week |
| `DaysFromDivEx` | Dividends | Days since the last dividend ex-date. Always a positive number or NA |
| `DaysFromDivPay` | Dividends | Days since the last dividend payment date. Always a positive number or NA. If in-between ex-date and pay-date NA is returned. |
| `DaysFromMergerAnn` | Corporate Actions (ICE Data) | Returns the days since a M&A (incl spinoff) corporate action has been announced or NA. Equivalent to PendingCorpAct(#MANDA, #ANNCEDAYS) |
| `DaysLate` | Earnings Release |  |
| `DaysToDivEx` | Dividends | Days until the next dividend ex-date. Always a positive number or NA. If in-between ex-date and pay-date NA is returned. |
| `DaysToDivPay` | Dividends | Days until next dividend payment date. Always a positive number or NA. |
| `DbtLT2Ast5YAvg` | Long Term Debt to Total Assets | Long-term debt relative to total assets. |
| `DbtLT2Cap5YAvg` | Long Term Debt to Total Capital | Long-term debt relative to total capital. |
| `DbtLT2Eq5YAvg` | Long Term Debt to Total Equity | Long-term debt relative to common equity. |
| `DbtLT2EqQInd` | Long Term Debt To Total Equity, Industry | Long Term Debt To Total Equity Industry, Quarterly |
| `DbtLT5YAvg` | Long Term Debt | Debt due more than 12 months from balance sheet date. |
| `DbtLTIssued5YAvg` | Long Term Debt Issued | Cash inflow from issuing debt with maturity over 12 months. |
| `DbtLTReduced5YAvg` | Long Term Debt Reduced | Cash outflow from retiring debt with maturity over 12 months. |
| `DbtS2NI5YAvg` | Debt Service to Net Income | Ratio of interest expense relative to earnings. |
| `DbtST5YAvg` | Short Term Debt | Debt due within 12 months, reported as current liability on balance sheet. |
| `DbtTot2Ast5YAvg` | Total Debt To Total Assets | Total debt relative to total assets. |
| `DbtTot2Cap5YAvg` | Total Debt to Total Capital | Total debt relative to total capital. |
| `DbtTot2Eq5YAvg` | Total Debt to Total Equity | Total debt relative to common equity. |
| `DbtTot2EqQInd` | Total Debt To Total Equity, Industry | Total Debt To Total Equity Industry, Quarterly |
| `DbtTot5YAvg` | Total Debt | Sum of all debt obligations as reported on the balance sheet. |
| `DepAmort2GP5YAvg` | Depreciation And Amort to Gross Profit | D&A expenses relative to gross profit |
| `DepAmort5YAvg` | Depreciation and Amortization | The sum of D&A expenses as reported specifically on income statement. |
| `DepAmortCF5YAvg` | Depreciation and Amort, Cash Flow | Depreciation and Amortization as reported on cash flow statement. |
| `DivPS52W` | Dividends in a Time Period | Returns the sum of all regular dividends with ex-dates in the past calendar year. It's equivalent to DivPSDays(365,0,#Regular,#ExDate) |
| `DivPS5YAvg` | Dividends in a Filing Period | Annual average of regular dividends per share for the past 5 fiscal years using ex-dates. |
| `DivPSNextQ` | Dividends in a Filing Period | Sum of all regular dividends with ex-dates falling in the ongoing quarter. If the company has not yet announced the dividends it returns 0 |
| `DivPSNextQCnt` | Dividends in a Filing Period | Count of all regular dividends with ex-dates in the ongoing quarter. If the company has not yet announced the dividends it returns 0 |
| `DivPSQ` | Dividends in a Filing Period | Sum of all regular dividends with ex-dates in the most recent quarter. This is equivalent to DivPS(0,QTR,#Regular,#ExDate) |
| `DivPaid5YAvg` | Dividends Paid (Total Cash Flow) | Total dividends paid across all share classes and preferred shares. |
| `Divest5YAvg` | Divestitures | Cash inflow from selling companies/subsidiaries. |
| `EBIT5YAvg` | Earnings Before Interest and Taxes (EBIT) | Operating income including depreciation and amortization. |
| `EBITDA5YAvg` | Earnings Before Interest, Taxes, Dep and Amort (EBITDA) | Operating income excluding non-cash expenses D&A. |
| `EBITDAActualGr%PYQ` | EBITDA Actual | EBITDA Actual growth previous year quarter (PYQ) |
| `EBITDAActualPTM` | EBITDA Actual | EBITDA Actual previous twelve months |
| `EBITDAPS5YAvg` | EBITDA Per Share | Earnings before interest, taxes, depreciation and amortization on a per-share basis. |
| `ETFAssetClass` | ETF Asset Class | ETFAssetClass returns the ETF Asset Class. |
| `ETFClassCount` | ETF Taxonomy Weight | Number of positions in the ETF category. |
| `ETFClassWeight` | ETF Taxonomy Weight | Weight of the ETF category as % of total market value. |
| `ETFCountry` | ETF Country | ETFCountry returns the ETF Country. |
| `ETFCountryCount` | ETF Taxonomy Weight | Number of positions in the ETF category. |
| `ETFCountryWeight` | ETF Taxonomy Weight | Weight of the ETF category as % of total market value. |
| `ETFFamily` | ETF Family | ETFFamily returns the ETF Family. |
| `ETFFamilyCount` | ETF Taxonomy Weight | Number of positions in the ETF category. |
| `ETFFamilyWeight` | ETF Taxonomy Weight | Weight of the ETF category as % of total market value. |
| `ETFMethod` | ETF Method | ETFMethod returns the ETF Method. |
| `ETFMethodCount` | ETF Taxonomy Weight | Number of positions in the ETF category. |
| `ETFMethodWeight` | ETF Taxonomy Weight | Weight of the ETF category as % of total market value. |
| `ETFRegion` | ETF Region | ETFRegion returns the ETF Region. |
| `ETFRegionCount` | ETF Taxonomy Weight | Number of positions in the ETF category. |
| `ETFRegionWeight` | ETF Taxonomy Weight | Weight of the ETF category as % of total market value. |
| `ETFSecCount` | ETF Taxonomy Weight | Number of positions in the ETF category. |
| `ETFSecWeight` | ETF Taxonomy Weight | Weight of the ETF category as % of total market value. |
| `ETFSector` | ETF Sector | ETFSector returns the ETFSector. |
| `ETFSize` | ETF Size | ETFSize returns the ETF Size. |
| `ETFSizeCount` | ETF Taxonomy Weight | Number of positions in the ETF category. |
| `ETFSizeWeight` | ETF Taxonomy Weight | Weight of the ETF category as % of total market value. |
| `ETFStyle` | ETF Style | ETFStyle returns the ETF Style. |
| `ETFStyleCount` | ETF Taxonomy Weight | Number of positions in the ETF category. |
| `ETFStyleWeight` | ETF Taxonomy Weight | Weight of the ETF category as % of total market value. |
| `EV` | Enterprise Value | Enterprise Value is an estimated measure of the total value of a corporation. |
| `EV2EBITDAPY` | Enterprise value to EBITDA | Enterprise value to operating income plus depreciation and amortization. |
| `EVPS` | Enterprise Value | Enterprise Value per share is Enterprise Value divided by the number of outstanding shares |
| `EqIssued5YAvg` | Equity Issued | Cash received from issuing equity instruments. |
| `EqPurch5YAvg` | Equity Purchased | Cash outflow from equity repurchases. |
| `EqTot5YAvg` | Shareholder's Equity Total | Total assets minus total liabilities on the balance sheet. |
| `EvenID` | Unique ID Odd or Even | Returns 1 (TRUE) if the internal ID of the stock or ETF is Even, 0 (FALSE) otherwise. Can be used for data-mining to create two samples from the universe. |
| `ExchangeCode` | Exchange Code | Returns the point in time exchange code where the listing trades |
| `ExpNonOp5YAvg` | Non-Operating Expenses | Net result from secondary activities unrelated to core business. |
| `FCF5YAvg` | Free Cash Flow | Cash generated after accounting for operational support and capital asset maintenance. |
| `FCFPS5YAvg` | Free Cash Flow Per Share | Free cash flow divided by fully-diluted average shares outstanding. |
| `FXRate` | Foreign exchange rate | FXRate gives you the foreign exchange rate used to convert the currency of the stock (or ETF) to the currency in the screen/ranking system/simulation. |
| `Float` | Float | Float (millions). This is the number of the common shares outstanding that are freely available for public trading. |
| `FloatPct` | Float | This is the percent of the common shares outstanding that are freely available for public trading. |
| `FutureDivFactor` | Future Dividend Factor | Returns the product of all the dividends in the future of the observation date (As-Of Date) |
| `FutureDivSplitFactor` | Future Dividend and Split Factor | Returns the product of all the splits and dividends in the future of the observation date (As-Of Date) |
| `FutureSplitFactor` | Future Split Factor | Returns the product of all the splits in the future of the observation date (As-Of Date) |
| `Goodwill5YAvg` | Goodwill | Excess cost over equity in acquisitions. |
| `GrossPlant5YAvg` | Gross Property Plant and Equipment | Total value of physical assets owned before depreciation adjustments. |
| `GrossProfit5YAvg` | Gross Profit | Profit after deducting production/service costs. GrossProfit = Sales - CostG (excludes D&A). |
| `GrossProfit_GAAP5YAvg` | Gross Profit GAAP | Profit after deducting production/service costs. GrossProfit_GAAP = Sales - CostG_GAAP (includes D&A). |
| `HistQ1Difference` | Historical EPS Difference | Historical Quarter Difference (Actual - Estimate), 1 Quarter Ago |
| `HistQ2Difference` | Historical EPS Difference | Historical Quarter Difference (Actual - Estimate), 2 Quarters Ago |
| `HistQ3Difference` | Historical EPS Difference | Historical Quarter Difference (Actual - Estimate), 3 Quarters Ago |
| `HistQ4Difference` | Historical EPS Difference | Historical Quarter Difference (Actual - Estimate), 4 Quarters Ago |
| `HistQ5Difference` | Historical EPS Difference | Historical Quarter Difference (Actual - Estimate), 5 Quarters Ago |
| `IAC5YAvg` | Income Available to Common | Income before extraordinary items and discontinued operations less preferred dividends. |
| `IAD` | Indicated Annual Dividend | Indicated Annual Dividend. This is a forward looking number used to calculate yield. It can also be used to find companies increasing their dividends. |
| `IAD13W` | Indicated Annual Dividend | Indicated Annual Dividend 3 months ago. It can be used to find companies increasing their dividends. |
| `IAD26W` | Indicated Annual Dividend | Indicated Annual Dividend 6 months ago. It can be used to find companies increasing their dividends. |
| `IAD52W` | Indicated Annual Dividend | Indicated Annual Dividend 1 year ago. It can be used to find companies increasing their dividends. |
| `IncAftTax5YAvg` | Income After Taxes | Pre-tax income excluding extraordinary items less tax expense. |
| `IncBTax5YAvg` | Income Before Taxes | Income Before Taxes includes all expenses except extraordinary and discontinued items. |
| `IncBXorAdjCSE5YAvg` | Income Before Xor Items Adj for Common | Net income available to common divided by fully diluted shares. |
| `IncBXorForCom5YAvg` | Income Before Xor Items Avail for Common | Income before extraordinary/discontinued items minus preferred dividends. |
| `IncPerEmp5YAvg` | Income Per Employee | Income per Employee. |
| `IncTaxExp5YAvg` | Income Tax Expense | Net amount paid by the company during the period. |
| `IndCode` | Industry Code |  |
| `IndCount` | Industry Count | Running count of stocks in the industry. This rule must be the last rule in the screen. |
| `IndDescr` | Industry Description |  |
| `IndWeight` | Industry Weight | Weight of the industry as % of total market value. For a buy rule, the weight is checked assuming the stock is purchased. For a sell rule the weight is checked before the sell. |
| `Industry` | Industry | Which industry the stock belongs to. For example, to screen for stocks in the leisure products industry use: Industry=LEISURE |
| `IntCov5YAvg` | Interest Coverage | Measure of ability to pay interest expenses. |
| `IntExp5YAvg` | Interest Expense | Amount paid to service all debt during the period. |
| `IntInc5YAvg` | Interest Income | Amount earned from loans during the period. |
| `IntanOther5YAvg` | Other Intangibles | All non-classifiable assets. |
| `Intercept` | Y Intercept | Returns the Y-Intercept for a previously computed regression |
| `InterceptSE` | Y Intercept Standard Error | Returns the Y-Intercept Standard Error for a previously computed regression |
| `InvTurn5YAvg` | Inventory Turnover | Measure of how quickly inventory is sold. |
| `Inventory5YAvg` | Inventory | Current asset representing merchandise or materials held for sale or revenue generation. |
| `InvstAdvOther5YAvg` | Investments and Advances Other | Long-term receivables including investments in unconsolidated companies. |
| `InvstEq5YAvg` | Equity Investments | Long-term equity investments. |
| `InvstOther5YAvg` | Other Investing Cash Flow | Miscellaneous investing cash flows not classified elsewhere. |
| `InvstST5YAvg` | Short Term Investments | Current asset representing marketable securities due or expected to be traded within 12 months. |
| `InvtyChg5YAvg` | Change to Inventory | Change in current inventory account from cash flow statement. |
| `IsADR` | Is ADR | Returns 1 (TRUE) if the stock is an American Depository Receipt (ADR) |
| `IsMLP` | Is MLP | Returns 1 (TRUE) if the stock is a Master Limited Partnership (MLP) in USA or Canada. |
| `IsOTC` | Is Over The Counter (OTC) | Returns 1 (TRUE) if the stock trades OTC in the USA. |
| `IsPrimary` | Is Primary | Returns 1 (TRUE) if the stock is the primary listing for the company, i.e. not a foreign stock. |
| `LatestActualDays` | Actual Latest | Calendar days since analysts actuals for the most recent quarter were available from the data vendor. |
| `LatestActualPeriodDate` | Actual Latest | Period date of latest analysts actuals, represented as a number YYYYMMDD. |
| `LatestFilingDate` | Latest (Any Source) | The date the latest period was first filed by the company with the SEC. This date may be before any data appears for the as-of date of your analysis due to vendor delays in processing the filing. |
| `LatestNewsDate` | Latest (Any Source) | The earliest date of either the press release from the company or the date any data appears in the database |
| `LatestPeriodDate` | Latest (Any Source) | The latest period that has been announced by the company. This data may not have been processed yet by Compustat and/or may not have been filed with the SEC. |
| `LiabCur5YAvg` | Current Liabilities Total | Total liabilities due within 12 months of the balance sheet date. |
| `LiabCurOther5YAvg` | Current Liabilities Other | All non-debt, non-payables liabilities due within 12 months. |
| `LiabNonCurOther5YAvg` | Other Non-current Liabilities | All long-term liabilities excluding debt, deferred taxes, investment tax credits, and minority interest. |
| `LiabTot5YAvg` | Total Liabilities | Sum of all balance sheet obligations excluding shareholders equity. |
| `MScoreAQI` | Beneish M-Score | Beneish M-Score AQI: Asset Quality Index |
| `MScoreDEPAMI` | Beneish M-Score | Beneish M-Score DEPI: Depreciation Index (uses Dep&Amort) |
| `MScoreDEPI` | Beneish M-Score | Beneish M-Score DEPI: Depreciation Index (uses an estimate for Depreciation) |
| `MScoreDSRI` | Beneish M-Score | Beneish M-Score DSRI: Days Sales in Receivables Index |
| `MScoreGMI` | Beneish M-Score | Beneish M-Score GMI: Gross Margin Index |
| `MScoreLVGI` | Beneish M-Score | Beneish M-Score LVGI: Leverage Index |
| `MScoreSGAI` | Beneish M-Score | Beneish M-Score SGAI: Sales, General and Administrative expenses Index |
| `MScoreSGI` | Beneish M-Score | Beneish M-Score SGI: Sales Growth Index |
| `MScoreTATA` | Beneish M-Score | Beneish M-Score TATA: Total Accruals to Total Assets |
| `MktCap` | Market Capitalization | Market Capitalization ($ millions) - for stocks and closed-end funds (CEFs) |
| `Month` | Date | Returns the month 1-12 of the current trading day. |
| `MonthDay` | Date | Returns the day of the month 1-31 of the current trading day. |
| `NAV` | Net Asset Value (NAV) | The NAV per share. This is a MONTHLY item. This item applies to closed-end funds only. |
| `NAVDisc` | Net Asset Value (NAV) Discount | The most recent NAV discount. Funds trading below their NAV will show positive discounts. |
| `NAVDiscPM` | Net Asset Value (NAV) Discount | The NAV discount 1 month ago. Funds trading below their NAV will show positive discounts. |
| `NAVDiscPM2` | Net Asset Value (NAV) Discount | The NAV discount 2 months ago Funds trading below their NAV will show positive discounts. |
| `NAVDiscPM3` | Net Asset Value (NAV) Discount | The NAV discount 3 months ago. Funds trading below their NAV will show positive discounts. |
| `NAVPM` | Net Asset Value (NAV) | The NAV per share 1 month ago This is a MONTHLY item. This item applies to closed-end funds only. |
| `NAVPM2` | Net Asset Value (NAV) | The NAV per share 2 months ago. This is a MONTHLY item. This item applies to closed-end funds only. |
| `NAVPM3` | Net Asset Value (NAV) | The NAV per share 3 months ago. This is a MONTHLY item. This item applies to closed-end funds only. |
| `NI2CapEx5YAvg` | Net Income to Cap Expenditures | Profit is reinvested in capital expenditures. |
| `NetChgCash5YAvg` | Net Change to Cash Position | Total change in cash for the period. |
| `NetFCF5YAvg` | Net Free Cash Flow | Cash generated after supporting operations, maintaining capital assets, and paying dividends. |
| `NetFCFPS5YAvg` | Net Free Cash Flow Per Share | Net Free Cash Flow per Share is quarterly net free cash flow divided by fully-diluted average shares outstanding. |
| `NetIncBXor5YAvg` | Net Income Before Xor | Income after all expenses including taxes and minority interest. |
| `NetIncBXorNonC5YAvg` | Net Income Before Xor and Non-Control Interest | Total earnings excluding extraordinary items and minority interest. |
| `NetIncCFStmt5YAvg` | Net Income (Cash Flow Statement) | Top line of the cash flow statement using indirect method. |
| `NetPlant5YAvg` | Net Property Plant and Equipment | Total physical assets minus accumulated depreciation. |
| `NextQDnRev4WkAgo` | EPS Revisions Next Quarter | Next Quarter Down Revisions, 4 Weeks ago |
| `NextQDnRevLastWk` | EPS Revisions Next Quarter | Next Quarter Down Revisions, Last Week |
| `NextQUpRev4WkAgo` | EPS Revisions Next Quarter | Next Quarter Up Revisions, 4 Weeks ago |
| `NextQUpRevLastWk` | EPS Revisions Next Quarter | Next Quarter Up Revisions, Last Week |
| `NoBars` | Days (bars) since position was started | Number of bars since the position has been first opened. This is the number of tradings days which excludes weekends and holidays. |
| `NoConst` | Number of Constituents | Number of constituents in the industry |
| `NoDays` | Days (calendar) since position was started | Number of days since the position has been first opened. This is the actual number of days, not bars. Ex: sell rule to exit position if the return is less than the benchmark after 1 month: |
| `NoEmp5YAvg` | Number of employees | Number of employees. |
| `NoIncP4YN2Y` | Income Trend | Returns the number of yearly EPS increases using the past four years and the future two year estimates. Values range from 0-6 |
| `NoPosEBITDA5Y` | Income Trend | Number positive EBITDA past 5 Y |
| `NonControlInt5YAvg` | Non Controlling Interest | Formerly minority interest (pre-2009). |
| `OCFPS5YAvg` | Operating Cashflow Per Share | Operating Cashflow Per Share. |
| `OpInc5YAvg` | Operating Income | Revenues minus cost of goods sold, SG&A, and depreciation/amortization. |
| `OpIncAftDepr5YAvg` | Operating Income After Depreciation | Revenues minus COGS, SG&A, and depreciation/amortization. Identical to Operating Income. |
| `OpIncBDepr5YAvg` | Operating Income Before Depreciation | Revenues minus COGS and SG&A. |
| `OpIncPS5YAvg` | Operating Income Per Share | Operating income divided by fully-diluted average shares outstanding. |
| `Opinion` | Recent Opinion | Returns your most recent opinion for the stock or NA |
| `Opinion%Chg` | Recent Opinion | Returns the total return % since your most recent opinion for the stock or NA |
| `OpinionBars` | Recent Opinion | Returns the number of bars since your most recent opinion for the stock or NA |
| `OpinionDays` | Recent Opinion | Returns the number of days since your most recent opinion for the stock or NA |
| `OtherWCChg5YAvg` | Change to Other Working Capital Lines | All working capital changes not captured in specific line items (inventory, accounts payable, accounts receivable). |
| `PEExclXorPY` | Price to Earnings (PE) Excl Xor | Share price relative to earnings per share excluding extraordinary items. |
| `PEInclRDPY` | Price to Earnings Incl R&D (aka Innovation PE) | Adjusts P/E ratio by adding R&D expenses back to earnings. |
| `PEInclXorPY` | Price to Earnings (PE) Incl Xor | Share price relative to earnings per share including extraordinary items. |
| `PERelative` | Price to Earnings (other) | Historical Relative Price Earnings Ratio |
| `Payables5YAvg` | Accounts Payable | Money owed by the company due within 12 months. |
| `PayablesChg5YAvg` | Change to Payables (Accounts Payable) | Change in accounts payable balance for the period. |
| `Pct3MH` | Percent From Hi/Lo | Percent from 3 month high using the weekly series (incl div) |
| `Pct3ML` | Percent From Hi/Lo | Percent from 3 month low using the weekly series (incl div) |
| `Pct4WH` | Percent From Hi/Lo | Percent from 4 week high using the weekly series (incl div) |
| `Pct4WL` | Percent From Hi/Lo | Percent from 4 week low using the weekly series (incl div) |
| `Pct52WH` | Percent From Hi/Lo | Percent from 52 week high using the weekly series (incl div) |
| `Pct52WL` | Percent From Hi/Lo | Percent from 52 week low using the weekly series (incl div) |
| `PctFromHi` | Percent from high | Percentage from highest close since position started. Always 0 or negative. |
| `PeriodDateA` | Latest in Database | Latest Annual Period Date. Represented internally as a number YYYYMMDD and displays as YYYY-MM-DD in screen reports. |
| `PeriodDateQ` | Latest in Database | Latest Interim Period Date. Represented internally as a number YYYYMMDD and displays as YYYY-MM-DD in screen reports. |
| `PfdDiv5YAvg` | Preferred Dividends Paid, Total | Total amount paid to preferred shareholders across all preferred share issues during the period. |
| `PfdEquity5YAvg` | Preferred Equity | Net preferred shares multiplied by par/stated value per share. |
| `PiotFScore` | Piotroski F-Score | Piotroski F-Score assigns a score from 0 (more likely to go bankrupt) to 9 (strong financials) based on 9 PASS/FAIL tests |
| `PortBars` | Days (bars) since inception | Returns the number of bars since the inception of the portfolio/sim. Use this when calculating averages of the portfolio equity to make sure there are enough bars. For ex., this buy rules prevents buy |
| `PortCash` | Available cash | Returns the amount of cash in the port/sim |
| `PosCnt` | Running count of holdings | Returns the # of positions in the portfolio during a rebalance |
| `Pr13W%ChgInd` | Price Percent Change, Industry | 13 Week Price Percent Change Industry (%) |
| `Pr13WRel%ChgInd` | Relative Price Percent Change, Industry | Relative Price Percent Change Industry, 13 Weeks (%) |
| `Pr26W%ChgInd` | Price Percent Change, Industry | 26 Week Price Percent Change Industry (%) |
| `Pr26WRel%ChgInd` | Relative Price Percent Change, Industry | Relative Price Percent Change Industry, 26 Weeks (%) |
| `Pr2BookPY` | Price to Book Value | Market value relative to book value of equity. |
| `Pr2BookQInd` | Price To Book, Industry | Price to Book Ratio Industry, Quarterly |
| `Pr2CFInclRDPY` | Price to Cash Flow Incl R&D (aka Innovation Pr2CF) | Adjusts P/CF ratio by adding R&D expenses back to operating cash flow. |
| `Pr2CashFlPY` | Price to Cash Flow | Stock price relative to cash flow per share. |
| `Pr2FrCashFlPY` | Price to Free Cash Flow | Market price per share to free cash flow per share. |
| `Pr2NetFrCashFlPY` | Price to Net Free Cash Flow | Market price per share to net free cash flow per share. |
| `Pr2TanBkPY` | Price to Tangible Book Value | Stock price to tangible book value per share (common equity minus intangibles). |
| `Pr2TanBkQInd` | Price To Tangible Book, Industry | Price to Tangible Book Ratio Industry, Quarterly |
| `Pr4W%ChgInd` | Price Percent Change, Industry | 4 Week Price Percent Change Industry (%) |
| `Pr4WRel%ChgInd` | Relative Price Percent Change, Industry | Relative Price Percent Change Industry, 4 Weeks (%) |
| `Pr52W%ChgInd` | Price Percent Change, Industry | 52 Week Price Percent Change Industry (%) |
| `Pr52WRel%ChgInd` | Relative Price Percent Change, Industry | Relative Price Percent Change Industry, 52 Weeks (%) |
| `PrcRegEst10` | Price Regression End Value | Returns the ending value of a 10 bars Regression Line of the prices (incl div) |
| `PrcRegEst10W` | Price Regression End Value | Returns the ending value of a 10 week Regression Line of the weekly prices (incl div) |
| `PrcRegEst20` | Price Regression End Value | Returns the ending value of a 20 bars Regression Line of the prices (incl div) |
| `PrcRegEst20W` | Price Regression End Value | Returns the ending value of a 20 week Regression Line of the weekly prices (incl div) |
| `PrcRegEst50` | Price Regression End Value | Returns the ending value of a 50 bars Regression Line of the prices (incl div) |
| `PrcRegEst50W` | Price Regression End Value | Returns the ending value of a 50 week Regression Line of the weekly prices (incl div) |
| `PrevBarDaysAgo` | Date |  |
| `ProjPENTM` | Price To Earnings (PE) Projected | Next Twelve Months Projected P/E Ratio |
| `ProjPENTMInd` | Price To Earnings (PE) Projected, Industry | Next Twelve Months Projected P/E Ratio Industry |
| `QtrComplete` | Latest in Database | Latest Quarter Updated by SEC filings |
| `R` | Correlation Coefficient | Returns the R for a previously computed regression |
| `R2` | Coefficient of Determination | Returns the R² for a previously computed regression |
| `RandD5YAvg` | Research and Development Expense | Spending on future products/services during the period. |
| `Random` | Random Number | Random: Returns a random number uniformly distributed between 0 and 1 |
| `Rank` | Latest Rank | Latest stock rank updated daily Tuesday-Saturday. Usage examples: |
| `RankBars` | Rank Bars | Returns the number of bars of the last rank data. NOTE: A "bar" is a trading day. Therefore there are 5 bars in a week with no holidays. |
| `RankPos` | Latest Rank | Return the position within the ranked stocks array. The highest ranked stock is position 1, the next is 2, etc. |
| `RecTurn5YAvg` | Receivables Turnover | Measure of how efficiently a company collects receivables. |
| `Recvbl5YAvg` | Accounts Receivables | Amounts due to the company within 12 months, typically from credit sales. |
| `RecvblChg5YAvg` | Change to Accounts Receivables | Changes in receivables balance over a period. |
| `RetainedEarn5YAvg` | Retained Earnings | Net income kept rather than distributed as dividends. |
| `Retn%5YAvg` | Retention Rate | Percentage of earnings kept by the company for reinvestment rather than distributed as dividends. |
| `SAR` | Parabolic SAR | SAR: Parabolic SAR. |
| `SGA2GP5YAvg` | Sales, General and Administrative Expense to Gross Profit | Indirect costs as percentage of gross profit. |
| `SGandA5YAvg` | Sales, General and Admin Exp | Expenses include all general/administrative costs plus direct/indirect selling expenses - excludes R&D expenses. |
| `SGandA_GAAP5YAvg` | Sales, General and Admin Exp GAAP | Expenses include all general/administrative costs plus direct/indirect selling expenses - includes R&D expenses. |
| `SI%Float` | Short Interest Percent of Float | Short Interest, Percent of Float (%) |
| `SI%FloatPM` | Short Interest Percent of Float | Short Interest, Percent of Float, 1 Month Ago (%) |
| `SI%FloatPM2` | Short Interest Percent of Float | Short Interest, Percent of Float, 2 Months Ago (%) |
| `SI%FloatPM3` | Short Interest Percent of Float | Short Interest, Percent of Float, 3 Months Ago (%) |
| `SI%ShsOut` | Short Interest Percent of Shares Outstanding | Short Interest, Percent of Shares Outstanding (%) |
| `SI%ShsOutPM` | Short Interest Percent of Shares Outstanding | Short Interest, Percent of Shares Outstanding, 1 Month Ago (%) |
| `SI%ShsOutPM2` | Short Interest Percent of Shares Outstanding | Short Interest, Percent of Shares Outstanding, 2 Months Ago (%) |
| `SI%ShsOutPM3` | Short Interest Percent of Shares Outstanding | Short Interest, Percent of Shares Outstanding, 3 Months Ago (%) |
| `SI1Mo%Chg` | Short Interest, Percent Change | Short Interest, One Month Percent Change (%) |
| `SICM` | Short Interest | Short Interest, Current Month Position (millions) |
| `SIPM` | Short Interest | Short Interest, Previous Month (millions) |
| `SIPM2` | Short Interest | Short Interest, 2 Months Ago (millions) |
| `SIPM3` | Short Interest | Short Interest, 3 Months Ago (millions) |
| `SUEQ1` | Unexpected Earnings (SUE) | Standardized Unexpected Earnings Most Recent Quarter |
| `SUEQ2` | Unexpected Earnings (SUE) | Standardized Unexpected Earnings 2 Quarters Ago |
| `SUEQ3` | Unexpected Earnings (SUE) | Standardized Unexpected Earnings 3 Quarters Ago |
| `SUEQ4` | Unexpected Earnings (SUE) | Standardized Unexpected Earnings 4 Quarters Ago |
| `SUEY1` | Unexpected Earnings (SUE) | Standardized Unexpected Earnings Most Recent Year |
| `SUEY2` | Unexpected Earnings (SUE) | Standardized Unexpected Earnings 2 Years Ago |
| `SUEY3` | Unexpected Earnings (SUE) | Standardized Unexpected Earnings 3 Years Ago |
| `SUEY4` | Unexpected Earnings (SUE) | Standardized Unexpected Earnings 4 Years Ago |
| `SUSQ1` | Unexpected Sales (SUS) | Standardized Unexpected Sales Most Recent Quarter |
| `SUSQ2` | Unexpected Sales (SUS) | Standardized Unexpected Sales 2 Quarters Ago |
| `SUSQ3` | Unexpected Sales (SUS) | Standardized Unexpected Sales 3 Quarters Ago |
| `SUSQ4` | Unexpected Sales (SUS) | Standardized Unexpected Sales 4 Quarters Ago |
| `SUSY1` | Unexpected Sales (SUS) | Standardized Unexpected Sales Most Recent Year |
| `SUSY2` | Unexpected Sales (SUS) | Standardized Unexpected Sales 2 Years Ago |
| `SUSY3` | Unexpected Sales (SUS) | Standardized Unexpected Sales 3 Years Ago |
| `SUSY4` | Unexpected Sales (SUS) | Standardized Unexpected Sales 4 Years Ago |
| `Samples` | Regression Samples (Observations) | Returns the number of samples for a previously computed regression |
| `SecCount` | Sector Count | Running count of stocks in the sector. This rule must be the last rule in the screen. |
| `SecWeight` | Sector Weight | Weight of the sector as % of total market value. For a buy rule, the weight is checked assuming the stock is purchased. For a sell rule the weight is checked before the sell. |
| `Sector` | Sector | Which sector the stock belongs to. For example, to screen for stocks in the technology sector use: Sector=TECH |
| `SectorCode` | Sector Code |  |
| `SectorDescr` | Sector Description |  |
| `SecurityType` | Security Type |  |
| `Shares5YAvg` | Common Shares | Undiluted shares outstanding. |
| `SharesFD5YAvg` | Common Shares Fully Diluted | Fully diluted shares outstanding. |
| `SimWeeks` | Weeks since inception | Returns the number of weeks since inception. |
| `Slope` | Slope of the regression | Returns the Slope for a previously computed regression |
| `SlopeConf%` | Slope Confidence % | Confidence is defined as 100 * (1-SlopePVal) for a previously computed regression |
| `SlopePVal` | Slope p-value | Returns the P-value of the slope for a previously computed regression |
| `SlopeSE` | Slope Standard Error | Returns the Slope Standard Error for a previously computed regression |
| `SlopeTStat` | Slope t-statistic | Returns the t Stat of the slope for a previously computed regression |
| `SpcItems5YAvg` | Special Items | Total pre-tax non-recurring items. |
| `StaleStmt` | Stale Flag | Returns 1 (TRUE) when there's no data in the database for the latest period that is publicly available from a press release or SEC filing |
| `StkOptCF5YAvg` | Stock Option Compensation, Cash Flow | Cashflow statement SBC. Data begins around 2006, with full annual coverage starting in 2013, and interim in 2021 |
| `StkOptExp5YAvg` | Stock Option Compensation Expense | Income statement SBC. Data begins around 2003, with full annual coverage starting in 2008, and interim in 2017 |
| `StockID` | Unique Stock Identifier | Returns the internal ID of the current stock or ETF. Can be used to create any number samples in conjunction with modulus function Mod() |
| `SubIndCode` | SubIndustry Code |  |
| `SubIndCount` | Industry Count | Running count of stocks in the sub-industry. This rule must be the last rule in the screen. |
| `SubIndDescr` | SubIndustry Description |  |
| `SubIndWeight` | Industry Weight | Weight of the sub-industry as % of total market value. For a buy rule, the weight is checked assuming the stock is purchased. For a sell rule the weight is checked before the sell. |
| `SubIndustry` | SubIndustry | Which sub-industry the stock belongs to. For example, to screen for stocks in the diversified REITs sub-industry use: SubIndustry=REITDIV |
| `SubSecCount` | Sector Count | Running count of stocks in the sub-sector. This rule must be the last rule in the screen. |
| `SubSecWeight` | Sector Weight | Weight of the sub-sector as % of total market value. For a buy rule, the weight is checked assuming the stock is purchased. For a sell rule the weight is checked before the sell. |
| `SubSector` | SubSector | Which sub-sector the stock belongs to. |
| `SubSectorCode` | SubSector Code |  |
| `SubSectorDescr` | SubSector Description |  |
| `TRSD1YD` | Standard Deviation (Volatility) | Annualized Standard Deviation of Daily Total Return from last year |
| `TRSD30D` | Standard Deviation (Volatility) | Annualized Standard Deviation of Daily Total Return from last 30 days |
| `TRSD3YD` | Standard Deviation (Volatility) | Annualized Standard Deviation of Daily Total Return from last 3 years |
| `TRSD3YM` | Standard Deviation (Volatility) | Annualized Standard Deviation of Monthly Total Return from last 3 years |
| `TRSD5YD` | Standard Deviation (Volatility) | Annualized Standard Deviation of Daily Total Return from last 5 years |
| `TRSD5YM` | Standard Deviation (Volatility) | Annualized Standard Deviation of Monthly Total Return from last 5 years |
| `TRSD60D` | Standard Deviation (Volatility) | Annualized Standard Deviation of Daily Total Return from last 60 days |
| `TRSD90D` | Standard Deviation (Volatility) | Annualized Standard Deviation of Daily Total Return from last 90 days |
| `TanBV5YAvg` | Tangible Book Value (Net Tangible Assets) | Common equity minus intangible assets. |
| `TotMktVal` | Total value of portfolio | Returns the total value of the port/sim (incl. cash) |
| `TotRevisions4W` | Sum of EPS Revisions | Sum of (No Analysts Up Revisions) - (No Analysts Dn revisions) in the past 4 weeks. . Complete estimate data is available starting in 2000 |
| `TotRevisionsLastW` | Sum of EPS Revisions | Sum of (No Analysts Up Revisions) - (No Analysts Dn revisions) in the past week.. Complete estimate data is available starting in 2000 |
| `TxAcrudChg5YAvg` | Change in Accrued Income Taxes | Balance sheet changes in tax liabilities. |
| `TxDfd5YAvg` | Deferred Taxes | Expense from repaying deferred tax liabilities (on cash flow statement). |
| `TxDfdIC5YAvg` | Deferred Taxes and Investment Credits | Accumulated deferred taxes from timing differences plus investment tax credits. |
| `TxPayable5YAvg` | Taxes Payable | Current liability for taxes owed within 12 months of balance sheet date. |
| `TxRate%5YAvg` | Tax Rate | Tax expense as percentage of pre-tax income. |
| `Vol10DAvg` | Average Volume | Daily average volume (in millions). |
| `VolD%ShsOut` | Liquidity | Daily 10 day average volume as a % of Shares Outstanding. Equivalent to: |
| `VolM%ShsOut` | Liquidity | Monthly average total volume for past 3 months as a % of Shares Outstanding. Equivalent to: |
| `WCapPS2PrA` | Working Capital To Price | Working Capital Per Share To Price Ratio, Annual |
| `WCapPS2PrQ` | Working Capital To Price | Working Capital Per Share To Price Ratio, Quarterly |
| `WeekDay` | Date |  |
| `WeeksIntoQ` | Earnings Release | Number of weeks into the most recent quarter. A value of 0 indicates the last earnings report was less than a week ago. When it reaches values > 11 an earnings report should be announced soon. |
| `WeeksToQ` | Earnings Release | Number of weeks until the next earnings release. Typical values are 0-13. Stocks that return 0 are due to report anytime in the next 7 days. It's an estimate based on previous year date. |
| `WeeksToY` | Earnings Release | Number of weeks until the next annual earnings release. Typical values are 0-52. Stocks that return 0 are due to report anytime in the next 7 days. It's an estimate based on previous year date. |
| `Weight` | Weight of a position as % of total market value | Weight of an existing position in % of total portfolio market value. See the Eval() function for an interesting use of Weight. |
| `WorkCap5YAvg` | Working Capital | Current assets less current liabilities. |
| `Year` | Date | Returns the Year of the current trading day. |

