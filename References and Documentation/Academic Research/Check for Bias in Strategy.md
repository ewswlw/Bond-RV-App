# Topic: Check for Bias in Strategy

## When to Use

- Run through this checklist immediately after any promising backtest to pressure-test robustness before investing additional build time or capital.
- Use it when auditing external research or vendor signals to uncover hidden assumptions, missing data treatments, or fragile parameter choices.
- Apply it when strategy performance changes abruptly—especially after regime shifts—to identify whether structural bias or market microstructure drift is responsible.
- Pair it with live trading reviews whenever slippage, tax drag, or corporate action handling appears to diverge from expectations.
- Skip this document only when a separate, fully documented fragility review already exists for the same strategy variant; otherwise treat it as mandatory due diligence.

## Fragility Analysis

Comprehensive critique of the strategy's robustness, generalizability, and real-world viability. Address the following:

## Data & Methodology Biases

- **Survivorship Bias:** Does the strategy rely on current index constituents or ex-post knowledge?
- **Look-Ahead Leakage:** Are any signals, baskets, or parameters selected using future information?
- **Selection/Parameter Bias:** Are thresholds, lookbacks, or asset choices arbitrary or overfit to the sample?
- **Sample Overlap/Autocorrelation:** Are returns or signals highly autocorrelated, inflating statistical significance?
- **Out-of-Sample Breadth:** Is the strategy validated across multiple periods, markets, or asset classes?

## Structural & Economic Fragility

- **Macro Regime Dependence:** Is performance concentrated in a single macro regime (e.g., tech bull market)?
- **Asset Class Concentration:** Does the "defensive" side still carry equity beta or sector risk?
- **Crowding/Reflexivity:** Is the signal widely known or easily arbitraged, risking breakdown under crowding?
- **Duration/Rate Sensitivity:** Are the chosen assets exposed to macro factors (e.g., rates, inflation) not modeled in the backtest?

## Execution & Practical Constraints

- **Turnover & Slippage:** Are transaction costs, bid/ask spreads, and slippage realistically modeled?
- **Tax Drag:** Would real-world tax treatment (short-term gains) erode returns?
- **Corporate Actions:** Are splits, mergers, and delistings handled robustly?
- **Capacity:** Would scaling up the strategy impact execution or returns?

## Statistical Robustness

- **Drawdown & Tail Risk:** Are max drawdown, tail ratios, and other risk metrics reported and realistic?
- **Bayesian Priors:** Is there a plausible economic reason to expect the edge to persist, or is it sample noise?
- **Sensitivity Analysis:** Are results robust to small changes in parameters or asset selection?

## Market Structure & Regulatory Risks

- **Index Reconstitution:** Could future changes in index rules or membership impact results?
- **Market Microstructure:** Are there regime shifts in liquidity, tick size, or settlement that could affect execution?
- **Regulatory Overhang:** Are there sector-specific or macro risks (e.g., antitrust, capital controls) that could impair the strategy?

## Bottom Line

Summarize the most critical reasons the strategy's historical performance may not persist, and highlight any "red flags" that would require further robustness testing before live deployment.
