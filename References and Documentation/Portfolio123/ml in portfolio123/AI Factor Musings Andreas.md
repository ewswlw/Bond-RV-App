# AI Factor Musings Andreas

## When to Use

- Use this document for deep theory and practitioner commentary around Portfolio123’s AI Factor models, especially when justifying ML adoption to stakeholders.
- Apply it when designing validation frameworks or communicating robustness principles; it outlines Andreas’s systematic methodology.
- Reference it during model reviews to articulate rationale for LightGBM/ExtraTrees, training windows, and feature importance interpretations.
- Consult it when you need narrative explanations that complement the procedural AI Factor guides.
- For implementation specifics, pair it with the official AI capabilities guide; this file focuses on philosophy, validation, and nuanced best practices.

## Table of Contents
- [AI Factor Introduction](#ai-factor-introduction)
- [AI Factor Process](#ai-factor-process)
- [AI Factor Under the Hood](#ai-factor-under-the-hood)
- [Andreas's Systematic AI Framework - Complete Validation & Robustness Methodology](#andreass-systematic-ai-framework---complete-validation--robustness-methodology)

## AI Factor Introduction

We believe the greater risk lies in ignoring non-linearity — not in adopting AI tools designed to harness it. When used thoughtfully, machine learning enables us to capture complex, nonlinear relationships in a transparent and controlled way.

Our goal is not to replace human judgment, but to enhance it. We use machine learning to augment intuition and uncover patterns that traditional linear models often miss.

Many traditional mandates still avoid machine learning, often citing concerns about complexity or interpretability. But these concerns are increasingly outdated.

- Modern ML methods such as LightGBM and ExtraTrees are no longer "black boxes." Their tree-based structures provide transparency, making it possible to trace decisions and model logic.
- Feature importance and model structure are fully interpretable, allowing us to understand which factors drive predictive power—and why. This aligns machine learning with the transparency expectations of institutional investors.
- Long, uninterrupted training phases (2003–2020) are specifically designed to capture underlying causal relationships, not just historical correlations. We reinforce this by applying strict out-of-sample portfolio testing, completely separate from the training period, to ensure robustness and real-world validity.
- Our AI Factor models demonstrate very low sensitivity to hyperparameter settings, a strong indicator of stability and resilience across different configurations and market regimes.
- Each week, the AI Factor models generate updated performance predictions for thousands of stocks, which are then ranked to guide systematic stock selection.

**AI Factor / Machine Learning Ranking via ML Algo LightGBM or ExtraTrees.**

## AI Factor Process

### Clean Data

Our AI Factor Machine Learning process begins with a rigorously clean, point-in-time database — spanning over **1 billion timestamped data points** across the U.S., Canada, Europe, and the U.K. This infrastructure ensures bias-free modeling that reflects the real-world conditions faced by investors.

### Machine Learning: Machine Learning Algos, Universe, Target and Features

Unlike conventional factors such as value, momentum, or quality, the AI Factor dynamically adapts to changing market conditions and captures nonlinear relationships in the data that traditional methods may overlook.

## AI Factor Under the Hood (Excerpts)

### Conservative Training: Robustness by Design

The core model is trained on 2003–2020 data using a conservative point-in-time dataset. This period reflects realistic data availability, accounting for lags in earnings releases, estimate updates, and other signals — exactly as an investor would have experienced them at the time.

Post-2020, Portfolio123 data became faster and more granular due to a switch to FactSet. Rather than retraining the model to fit this new regime prematurely, a light rule overlay (e.g., "EPS revisions up") is applied to capture faster signals without disrupting the learned structure.

Annual retraining (e.g., 2003–2021 next year) allows measured adaptation, not overfitting.

### AI Factor Machine Learning Algorithm: LightGBM

At the heart of the AI Factor approach is an ensemble of decision trees, each functioning like a micro-strategy or conditional logic rule. These trees are not hand-designed — they are learned from decades of historical data.

- Each decision tree acts like a mini ranking system, tuned to a specific pattern in the data.
- Every stock is passed through all trees, each contributing a prediction based on how well the stock matches previously successful conditions.
- The final model score is the average of all tree predictions — a blend of perspectives that captures complex, nonlinear relationships.
- A stock that for example receives an AI Factor Rank of > 95 is triggering an unusually high number of learned alpha patterns across the entire model. This means it doesn't just look good on one dimension — it matches dozens of historically successful multi-factor combinations simultaneously. The result is high consensus, high confidence, and a ranking that reflects strong alignment with past winners. **This resonates with the alpha of "Overlap Stocks":** [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5244033](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5244033)

This architecture mimics how experienced traders think across different regimes and setups — only at scale, and with no emotional bias.

### An Example – When a Deep Value Stock Rallies

Imagine a small-cap stock with a Price/Cash ratio of 0.1 — a classic deep value candidate. Then it rallies +100% in a month.

- One tree may focus on value and still flag it as a buy.
- Another tree, trained to detect overbought conditions, may now downgrade it due to momentum exhaustion.
- A third tree might combine sentiment, insider activity, and sector conditions to remain neutral.

Each of these perspectives contributes to the final score. The model doesn't rely on one rule — it integrates many lenses, moderating extreme views and highlighting subtle risks or opportunities.

### Feature Importance – What It Means and What It Doesn't

In tree-based models, the most important features often appear near the top of trees. These are the splits that affect the largest number of stocks and provide the highest predictive gain.

However:

- Lower-ranked features still matter — often in rare but highly predictive combinations.
- A feature may only "fire" in specific regimes or contexts yet deliver outsized returns when it does.

The model captures both: frequent, generally predictive factors and rare but powerful interactions.

### Learning Rare but Powerful Setups

Machine learning models like LightGBM remember rare but effective combinations — especially during extreme market events.

For example, in past crash regimes, the model may have learned:

- When a stock drops sharply
- And shows positive earnings revisions
- And has insider buying or low short interest

…it tends to rebound. Even if this occurs in just 1% of training data, the model remembers it — and uses it when the setup reappears.

This makes AI Factor models highly responsive to market dislocations, while traditional models often overlook such edge cases.

### Adapting to Market Regimes – Without Explicit Labels

AI Factor models are not told what a bull or bear market is — they learn it implicitly.

During training, the model sees that:

- Certain combinations worked during rallies
- Others worked during drawdowns or sideways periods

At rebalance, it scores each stock based on how well it matches historical patterns that led to alpha under similar conditions. This built-in context awareness allows the model to adapt its ranking logic as the market evolves — without requiring macro inputs or regime definitions.

### Rebalancing – How the Model Selects Stocks

At each rebalance:

1. The model receives current feature data for all stocks.
2. Each stock is passed through all decision trees.
3. Each tree evaluates whether the stock matches one of its learned high-return paths.
4. The final score is the average across all trees.
5. Stocks are ranked by this score, and the top candidates are selected.

A stock that activates many "strong signal" paths gets a higher rank. One that matches few (or conflicting) paths scores lower. This ensures precision without over-reliance on any single factor.

### Why Machine Learning Outperforms Traditional Ranking Systems

**Traditional models:**
- Use linear or monotonic rankings
- Apply fixed weights (e.g., 40% value, 20% quality)
- Struggle to model nonlinear combinations and rare patterns

**AI Factor models:**
- Learn thousands of conditional rules from historical outcomes
- Adapt the signal weighting depending on context
- Recognize rare but predictive combinations that humans miss

In complex, nonlinear environments like small caps, machine learning provides a clear advantage.

### Small Fund, Big Edge – AI's Advantage for Low AUM Strategies

Large funds need scalable alpha. AI Factor strategies can monetize patterns that do **not** scale — exactly where smaller funds thrive.

Examples:

- Microcaps with insider buying and high short interest
- Niche R&D/sales patterns in $150M software companies
- Rebound patterns following sharp drawdowns

These signals are often unusable at large scale — but for family offices, SMAs, or emerging funds, they are pure alpha.

AI helps discover and act on them automatically.

### Summary – What Makes AI Factor Unique

- **Multi-perspective logic**: Each tree represents a valid investment lens.
- **Context awareness**: Patterns are regime-sensitive without being told.
- **Conservative foundation**: Trained on realistic point-in-time data.
- **Light retraining, heavy discipline**: Ensures stability and robustness.
- **Adaptive intelligence**: Captures value, momentum, sentiment — and combinations no traditional model can encode.

### How Many Trees Are Built – And What That Means for Predictive Power

Example: AI Factor model is configured with the following key hyperparameters:

```json
{
  "n_estimators": 500,
  "max_depth": 5,
  "learning_rate": 0.05,
  "num_leaves": 31,
  "subsample": 0.7,
  "colsample_bytree": 0.7,
  "min_child_samples": 50,
  "reg_alpha": 0.5,
  "reg_lambda": 0.5,
  "subsample_freq": 1
}
```

The most important for understanding how many trees are used:

- **n_estimators = 500** → This means the model builds 500 decision trees during training.

Each of these 500 trees:

- Is trained sequentially, each one improving on the residual errors of the previous.
- Has a maximum depth of 5, meaning it can create decision paths up to 5 feature-splits long.
- Can split into up to 31 leaves, allowing for complex but controlled decision structures.
- Uses a random subset of features (70%) and samples (70%) per tree to improve diversity and prevent overfitting.

#### What This Means in Practice

- The model builds 500 diverse, shallow-but-rich trees, each learning different interactions and combinations of features.
- Each tree represents a mini ranking system, focusing on different conditional logic (e.g., "cheap + insider buying," or "low float + rising revisions").
- Together, they form an ensemble (ranking each stock with that ensemble prediction!) that blends thousands of synthetic patterns.

**Total synthetic ranking paths created:**
Approximately 15,000–30,000 conditional paths
(500 trees × average 30 leaf splits × multiple interaction branches)

This gives our model(s) a vast internal playbook of "if-this-then-that" rules + ranks all those rules based on their predictive power — far more than a human could design by hand — but all based on real, historically successful patterns in our point-in-time data.

### Each Tree = A Ranking System

- Every LightGBM tree acts like a mini ranking model, scoring stocks based on a specific combination of factor thresholds.
- Your AI Factor model uses 500+ trees, each refining the logic of the ones before — starting with robust, global signals and moving to fine-tuned edge cases.
- Instead of manually stacking factor rankings, you automatically construct and weight hundreds of them.

### Boosting, Not Filtering

- LightGBM does not discard "bad" trees — it builds each one to reduce the remaining error from previous trees.
- All tree predictions are weighted using the learning rate (e.g. 0.05), meaning no single tree can dominate.
- Even "late" trees matter: they capture exceptions, special cases, or rare regimes.

### Every Stock Gets Scored by Every Tree

- There is no rejection — every tree produces a numeric prediction for every stock.
- Even if a stock doesn't match the top-level split (e.g., not MarketCap < 500M), it still follows a decision path and gets a final score from that tree.

### How a Tree Chooses a Split like MarketCap < 500M

- At every node, LightGBM:
  1. Tests every feature
  2. Evaluates many possible split thresholds
  3. Uses a greedy algorithm to select the split that most reduces prediction error (maximizes gain)

If MarketCap < 500M appears as a top node, it's because it was the single most powerful split at that point, based on the loss function.

- This process is repeated for every node of every tree — meaning all splits are data-validated, not arbitrary.

#### Regularization

- **reg_alpha = 0.5, reg_lambda = 0.5**: Add friction to leaf weight updates, penalizing complexity and encouraging generalizable patterns.

#### Subsampling

- **subsample = 0.7**: Each tree trains on a random 70% of stocks
- **colsample_bytree = 0.7**: Each tree sees only 70% of features

Together (Regularization & Subsampling), these techniques make models more robust and less prone to overfitting — even with 500 trees and 179 features.

### Tree Layers = Alpha Layers

#### Greedy Function Builds Base Trees (Tree #1–50)
*"The Base Trade"*

- LightGBM uses a greedy algorithm to test all feature/threshold combinations
- Chooses the ones with the highest gain to form robust base splits (e.g., MarketCap < 500M)
- Captures broad, high-confidence signals
- Fires on **many stocks** and builds **stable, repeatable alpha**

#### Conditional Refinement (Tree #100–400)
*"Refined Logic"*

- Adds conditional logic: sector effects, factor interactions, regime filters
- Learns nuances (e.g., value + EPS revisions + sentiment)
- Targets subsets of stocks for smart differentiation

#### Rare Edge Hunting (Tree #450–500) + Learning Rate Scaling
*"Special Situations"*

- Final trees identify rare cases with high alpha
- Often fire on very few stocks (e.g., insider buying + low float + rate cuts)
- Learning rate (0.05) ensures each tree contributes just a little — balancing broad signal and rare edge
- Final ensemble = weighted combination of all trees

Only the full ensemble delivers robust, high-coverage alpha.

### Traditional Ranking vs. AI Factor

| Traditional Models | AI Factor Trees |
| --- | --- |
| Manual feature selection | Automated from 179 factors |
| Static weighting | Dynamic learning via boosting |
| Few systems (5–30) | 500+ intelligent trees |
| Linear logic | Non-linear, conditional relationships |
| Manual retuning | Auto-adapts via error correction |

*To replicate AI Factor manually, you'd need to build and tune hundreds of separate ranking systems — which is practically impossible.*

### ExtraTrees + ZScore + Date in small caps

#### 1. How ExtraTrees works vs. LightGBM

**ExtraTrees:**
- Splits are chosen randomly among candidate thresholds.
- Trees are fully grown (deep), then averaged.
- Tends to lower variance because of the extra randomness, but at the cost of bias.
- Works best when the **features are already normalized/comparable** (like ZScore).

**LightGBM:**
- Greedy boosting, optimizes splits deterministically.
- Can chase subtle patterns (good in noisy data, but risk of late-tree overfit).
- Handles raw scales better, doesn't rely as much on pre-normalization.

#### 2. Why ExtraTrees + ZScore + Date fits small caps

- **ZScore + Date normalization** puts every feature on the same statistical footing at each point in time.
  → ExtraTrees' random split selection becomes more meaningful, because every feature is scaled comparably.
- **Small caps are noisy:**
  - Many idiosyncratic shocks, micro-structure quirks.
  - LightGBM can over-emphasize a few "loud" but spurious signals.
  - ExtraTrees, by being deliberately less greedy, avoids latching onto noise and instead finds more **robust cross-feature averages**.
- **Wide feature set (133–179):**
  - With lots of normalized features, randomization is not a bug but a feature.
  - ExtraTrees acts like a "factor bagging machine," combining hundreds of weak but stable z-scored hints.

#### 3. Why this shows up in small caps OOS

- In small caps, market structure changes less slowly than in large caps (fewer institutional players updating the rules every year).
- The **averaging of random splits** means ExtraTrees creates very robust rank distributions that don't overreact to short-term shifts.
- ZScore + Date ensures the comparisons are always "apples to apples" across time.

#### 4. The allocator pitch version

*"ExtraTrees with ZScore + Date shines in small caps because random splits on normalized features reduce noise sensitivity. It behaves like an ensemble of hundreds of factor rankers, each weak but stable — and when combined, they produce highly robust, regime-independent stock selection. This robustness is why the OOS curves look unusually clean."*

## AI Factor: Trim and Outlier

Here's the short, P123-specific explanation:

"If the feature is normalized using the dataset the mean and standard deviation are computed once when dataset is loaded. Predictions data is normalized using those statistics and z-scores clamped using the AI factor setting.

If the feature is normalized by date, then prediction data is normalized on the fly using the trim and outlier limit settings."

### Normalize by Dataset (global):

- μ and σ are computed **once** when the dataset loads (fixed across all dates).
- At prediction time, values are standardized with those fixed stats; **only the outlier limit (z-clamp)** is applied then.
- (No per-date trim during prediction. Any trim would have affected the one-time stats, if used at build time.)

### Normalize by Date (cross-sectional, what we described earlier):

- For each date, compute μ/σ **on the fly** using the **trim** setting.
- Standardize *all* names with those per-date stats, then **clamp** to the outlier limit.

### What "Trim" does (per date, per factor)

- Take all stocks' values at date *t*.
- Temporarily drop the lowest and highest *x%* **only to compute** mean and standard deviation.
- Compute μ and σ from this trimmed subset.
- **No stock is removed from scoring.** Trim affects **stats only**, not membership.

### What "Outlier limit" does

- Convert every stock's value to a z-score: z_raw = (x − μ) / σ (using the trimmed μ, σ).
- Cap ("neuter") extreme z's to the chosen limit, e.g., **±3**: z = clamp(z_raw, −3, +3).
- This prevents absurd values (e.g., 1000% FCF yield) from dominating.

### Live behavior

- The same steps run on the live cross-section each rebalance: compute trimmed μ/σ, z-score everyone, cap extremes.
- Outliers are **not deleted**, just **capped**.

### Why both matter

- **Trim** makes μ/σ robust by ignoring tails when estimating scale.
- **Outlier limit** keeps scoring stable by limiting how much any single datapoint can help/hurt.

### Typical settings

- Small caps: Trim ≈ **7.5%** per tail, Outlier limit **±3**.
- Going too low on trim (e.g., 1%) or too high on the cap (e.g., ±5 SD) can make results unstable.

## Building Feature Sets

For LightGBM/ExtraTrees a "lean & de-collinear" instinct can backfire because of how forests are built:

- Greedy early splits need strong anchors. If those few "obvious" features (rel-momentum, revisions, basic risk/liquidity) aren't present, the model grabs noisier price/vol signals → high beta, deeper DD.
- Later trees harvest small, orthogonal edges. Pruning too hard removes the weak-but-useful "nuggets" (accelerations, Up/Down, volume trend, profitability) that the ensemble squeezes in later stages.
- Redundancy helps stability. Having both absolute *and* industry/sector-relative windows across 3/6/12m gives multiple good candidates for the first split; bagging/colsampling then averages routes → lower variance and nicer OOS.
- Trees don't fear collinearity. With column subsampling, bagging, and regularization, extra correlated features rarely hurt—and often make early split choice robust across regimes. Over-pruning makes the split choice brittle.

**Practical takeaway:**

- Keep a balanced block set (rel/abs momentum, revisions, value/quality, risk/liquidity, flow/technical).
- Control complexity with regularization and subsampling, not aggressive feature removal.
- If you must trim, drop only obvious noise (1-week returns, duplicative Chaikin variants), not the anchors or the orthogonal squeezers.

So yes — your framing is right: forests win when they find strong stuff first and pick up scarce edges later.

## Hyperparameter Settings LightGBM

### LightGBM Hyperparameter Guide: Rank vs Z-Score Normalization

| # | Profile | Normalization | Why (1-liner) | Hyperparams (JSON) |
| --- | --- | --- | --- | --- |
| 1 | **lightgbm I – Seed III** | **Z-Score** | **Conservative baseline with full reproducibility** | `{ "n_estimators": 400, "max_depth": 5, "learning_rate": 0.04, "num_leaves": 31, "subsample": 0.75, "colsample_bytree": 0.65, "min_child_samples": 48, "reg_alpha": 0.50, "reg_lambda": 0.60, "random_state": 424242, "deterministic": true }` |
| 2 | **lightgbm II – GPT1** | **Z-Score** | **Maximum stability for noisy z-score data** | `{ "n_estimators": 350, "max_depth": 5, "learning_rate": 0.03, "num_leaves": 24, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_samples": 20, "reg_alpha": 0.4, "reg_lambda": 0.4 }` |
| 3 | **lightgbm medium 2/3/4** | **Z-Score** | **L1/L2 regularization variants for outlier protection** | `{ "n_estimators": 500, "max_depth": 6, "learning_rate": 0.025, "num_leaves": 32, "subsample": 0.75, "colsample_bytree": 0.75, "min_child_samples": 50, "reg_alpha": 0.1-1.0, "reg_lambda": 0.1-1.0 }` |
| 4 | **lightgbm II – deepseek** | **Both** | **Balanced torque with controlled depth** | `{ "n_estimators": 500, "max_depth": 5, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 50, "reg_alpha": 0.5, "reg_lambda": 0.5 }` |
| 5 | **lightgbm II** | **Rank Preferred** | **Balanced mid-capacity for rank patterns** | `{ "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "num_leaves": 32, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 25, "reg_alpha": 0.3, "reg_lambda": 0.3 }` |
| 6 | **Nonlinear-Ranks LGBM** | **RANK ONLY** | **Optimized for rank/bucket ordinal relationships** | `{ "n_estimators": 450, "max_depth": 6, "learning_rate": 0.035, "num_leaves": 48, "subsample": 0.75, "colsample_bytree": 0.65, "min_child_samples": 50, "reg_alpha": 0.5, "reg_lambda": 0.7 }` |
| 7 | **lightgbm II – SCGotlesp** | **RANK ONLY** | **Medium-deep for complex rank interactions** | `{ "n_estimators": 500, "max_depth": 8, "learning_rate": 0.01, "num_leaves": 64, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.1 }` |
| 8 | **lightgbm II – deepseek 3** | **RANK ONLY** | **High capacity for rich rank feature sets** | `{ "n_estimators": 1000, "max_depth": 6, "learning_rate": 0.02, "num_leaves": 63, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 40, "reg_alpha": 0.5, "reg_lambda": 0.7 }` |
| 9 | **lightgbm III** | **RANK ONLY** | **Deep architecture for stable rank patterns** | `{ "n_estimators": 500, "max_depth": 12, "learning_rate": 0.01, "num_leaves": 128, "subsample": 0.6, "colsample_bytree": 0.6, "min_child_samples": 10 }` |

| Name | Normalization | Why it's here (OOS rationale) | Hyperparams |
| --- | --- | --- | --- |
| **Pure Non-Linear LGBM** | **RANK ONLY** | **Specialized for rank-based non-linear patterns; too complex for z-score outliers** | `{ "n_estimators": 500, "max_depth": 7, "learning_rate": 0.03, "num_leaves": 64, "subsample": 0.8, "colsample_bytree": 0.7, "min_child_samples": 60, "reg_alpha": 0.6, "reg_lambda": 0.8, "subsample_freq": 1, "random_state": 424242 }` |

## Hyperparameter Settings ExtraTrees

### Extra Trees Hyperparameter Guide: Rank vs Z-Score Normalization

| # | Name | Normalization | Why it's here (OOS rationale) | Hyperparams |
| --- | --- | --- | --- | --- |
| 1 | **extra trees III – Deepseek** | **Z-Score** | **Maximum anti-overfit for noisy z-scores: bagging + low max_features + pruning** | `{"n_estimators": 800, "max_depth": 12, "min_samples_split": 8, "min_samples_leaf": 5, "max_features": 0.33, "bootstrap": true, "max_samples": 0.7, "min_impurity_decrease": 0.001, "ccp_alpha": 0.005}` |
| 2 | **extra trees III – TOF** | **Z-Score** | **Sparse features (0.3) + bagging for z-score outlier protection** | `{"n_estimators": 600, "max_depth": 12, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": 0.3, "bootstrap": true}` |
| 3 | **extra trees medium 4** | **Z-Score** | **Balanced bias/variance via max_features 0.75; z-score default** | `{"n_estimators": 250, "max_depth": 12, "min_samples_split": 8, "max_features": 0.75}` |
| 4 | **extra trees III – GPT** | **Z-Score** | **Stricter split thresholds for z-score stability** | `{"n_estimators": 500, "max_depth": 10, "min_samples_split": 12}` |
| 5 | **extra trees I – gotlessp 3** | **RANK ONLY** | **Huge forest + bagging for rank pattern smoothing** | `{"n_estimators": 2000, "max_depth": 12, "min_samples_split": 6, "min_samples_leaf": 4, "max_features": 0.5, "bootstrap": true, "max_samples": 0.8}` |
| 6 | **extra trees III** | **RANK ONLY** | **High-capacity for strong rank signals** | `{"n_estimators": 400, "max_depth": 16, "min_samples_split": 4}` |
| 7 | **extra trees slow 4** | **RANK ONLY** | **Very deep + many trees for complex rank relationships** | `{"n_estimators": 750, "max_depth": 30, "min_samples_split": 8, "max_features": 0.5}` |
| 8 | **extra trees II** | **Both** | **Simple baseline for quick comparisons** | `{"n_estimators": 200, "max_depth": 8, "min_samples_split": 2}` |
| 9 | **extra trees I – gotlessp** | **Z-Score** | **Very regularized shallow model for conservative z-score** | `{"n_estimators": 100, "max_depth": 4, "min_samples_split": 10, "min_samples_leaf": 10, "max_features": "sqrt"}` |
| 10 | **extra trees I** | **Z-Score** | **Fastest smoke test for z-score validation** | |

## AI Validation Settings

*Note: The original document contained images that were converted to base64 data URLs. These have been preserved in the conversion but are not displayed in this markdown version for readability.*

---

## Andreas's Systematic AI Framework - Complete Validation & Robustness Methodology

*Source: [Systematic AI Investing Portfolios - Substack Article](https://systematicportfolios.substack.com/p/turn-powerful-ai-into-actionable?utm_source=post-email-title&publication_id=3234494&post_id=177026641&utm_campaign=email-post-title&isFreemail=true&r=2dao57&triedRedirect=true&utm_medium=email)*

### Core Philosophy: Robust Over Optimal

Every design choice prioritizes future performance over perfect historical fit. The framework sacrifices in-sample perfection for out-of-sample robustness.

**Key Principles:**
1. **Simple over complex**: Basic holdout beats sophisticated walk-forward
2. **Honest over impressive**: Real gaps and realistic slippage
3. **Shallow over deep**: Conservative models over deep neural networks
4. **Stable over sensitive**: Low hyperparameter sensitivity is key
5. **Rich over lean**: For tree models, more features beat aggressive pruning
6. **Broad over narrow**: Train wide, apply specific filters
7. **Tested over theoretical**: Pseudo OOS and live validation required

---

### 🏛️ Validation: The "Basic Holdout" Gold Standard

**Methodology:**
- Rejects complex, potentially overfit walk-forward methods
- Uses a simple and honest test: train on long history, test on completely unseen multi-year period
- "What you see is what you get" - no iterative optimization games

**Why This Works:**
- Eliminates false confidence from repeated optimization on the same data
- Provides realistic estimate of future performance
- Reduces curve-fitting that plagues many backtesting approaches
- Honest assessment of model generalization

**Practical Implementation:**
- Train on maximum available history (e.g., 2003-2020)
- Test on completely separate period (e.g., 2020-2025)
- No parameter adjustments after training begins
- Accept lower in-sample perfection for better live results

---

### ⏳ The Critical "Gap": Respecting the Timeline

**Purpose:**
Enforces a gap between the prediction date and the target period to prevent look-ahead bias.

**What It Prevents:**
- Model accidentally using data that wouldn't have been available at decision time
- "Spillover" or look-ahead bias that inflates backtest results
- Subtle data leakages that dramatically inflate perceived performance

**Technical Details:**
- Creates buffer period between when predictions are made and period being predicted
- Ensures point-in-time data integrity
- Accounts for real-world data lags (earnings releases, estimate updates, etc.)

**Critical Insight:**
Even small data leakages can create unrealistic backtests. The gap ensures the model only uses truly available historical information.

---

### 🛡️ Conservative Machine Learning: Shallow is Beautiful

**Core Decision:**
Deliberately use shallow, regularized, non-linear tree-based models (LightGBM and ExtraTrees).

**The Trade-Off:**
- NOT seeking: Most complex pattern that perfectly fits the past
- SEEKING: Most robust pattern that will generalize to the future

**Example Configuration (LightGBM II - deepseek):**
```json
{
  "n_estimators": 500,
  "max_depth": 5,           // Shallow depth prevents overfitting
  "learning_rate": 0.05,    // Conservative learning rate
  "num_leaves": 31,         // Limited leaf nodes
  "subsample": 0.7,         // 70% data sampling per tree
  "colsample_bytree": 0.7,  // 70% feature sampling per tree
  "min_child_samples": 50,  // Minimum samples per leaf
  "reg_alpha": 0.5,         // L1 regularization
  "reg_lambda": 0.5         // L2 regularization
}
```

**Why These Settings Matter:**

**Shallow Depth (max_depth=5):**
- Prevents learning noise and overly specific patterns
- Forces focus on broad, generalizable relationships
- Reduces prediction variance

**Heavy Regularization (reg_alpha, reg_lambda):**
- Penalizes model complexity
- Prevents overfitting to training data quirks
- Smooths decision boundaries

**Subsampling (0.7):**
- Creates diversity among trees in ensemble
- Each tree learns from different data/feature perspectives
- Reduces correlation between trees → better ensemble robustness

---

### 🎯 Low Hyperparameter Sensitivity: The Ultimate Robustness Test

**Definition:**
Model performance remains strong and stable even when internal settings are adjusted.

**What You Want:**
All algorithm variations perform well, indicating robust underlying patterns.

**What You DON'T Want:**
A model that performs brilliantly with one configuration but collapses with slight changes.

**Why This Matters:**
- High sensitivity = model found a "lucky" configuration that won't generalize
- Low sensitivity = model discovered genuine market relationships
- Provides confidence the strategy will work in different market regimes

**How to Test:**
- Run multiple hyperparameter configurations
- Compare OOS performance across configurations
- Look for consistent performance, not perfect performance
- If all variants work reasonably well → robust signal

---

### 🧩 The "Sweet Spot" Feature Set: 87-180 Features

**Optimal Range:**
Through extensive testing, 87 to 180 features provides the optimal balance of informational depth and model stability.

**Feature Categories:**
1. **Valuation** - P/E, P/B, EV/EBITDA, price/cash, etc.
2. **Growth** - Revenue growth, earnings growth, sales acceleration
3. **Momentum** - Price momentum, earnings momentum, relative strength
4. **Estimates** - Analyst revisions, estimate trends, revision breadth
5. **Volatility** - Price volatility, earnings volatility, beta
6. **Quality** - ROE, margins, cash flow quality, profitability
7. **Sentiment** - Analyst sentiment, news sentiment, positioning

---

### 🌲 Counter-Intuitive Feature Philosophy: Rich Feature Sets for Tree Models

**Conventional Wisdom (WRONG for tree models):**
- Remove highly correlated features
- Aggressively prune to minimize collinearity
- Keep only the "best" features
- Lean and mean feature sets

**Andreas's Approach (RIGHT for LightGBM/ExtraTrees):**
- Build rich, diverse feature sets
- Intentionally avoid aggressive pruning based on collinearity
- Keep redundant but complementary features
- Let the model and regularization handle complexity

---

### 🎯 Four Core Principles for Feature Engineering

#### **Principle 1: Strong Feature Anchors First**

**The Problem with Lean Sets:**
- Initial splits in random forest need reliable, foundational features
- Without "obvious" features (rel-momentum, revisions, basic risk/liquidity), model grabs noisier signals
- Results: Higher volatility, deeper drawdowns, unstable predictions

**The Solution:**
- Keep multiple versions of strong signals
- Provide model with reliable first-split options
- Examples: Relative momentum, earnings revisions, basic risk metrics, liquidity measures

**Why It Works:**
- Greedy early splits need strong anchors
- If those few foundational features aren't present, model latches onto noise
- Strong anchors create stable base for entire tree structure

---

#### **Principle 2: Later Trees Harvest Small, Orthogonal Edges**

**How Ensemble Learning Works:**
- Early trees (1-50): Capture "obvious" broad patterns
- Middle trees (100-400): Add conditional logic and refinements
- Late trees (450-500): Identify rare cases with high alpha

**What Gets Lost with Aggressive Pruning:**
- Acceleration features (rate of change in momentum)
- Up/Down market behavior differences  
- Volume trends and patterns
- Profitability micro-signals
- Sector/industry interaction effects

**Impact:**
- Ensemble loses ability to squeeze out small edges
- Overall prediction quality degrades
- Model becomes less adaptive to regime changes
- Misses the "weak-but-useful nuggets" that later trees harvest

---

#### **Principle 3: Redundancy Helps Stability**

**Example: Comprehensive Momentum Windows**

Keep ALL of these:
- **Absolute momentum**: 3-month, 6-month, 12-month
- **Industry-relative momentum**: 3-month, 6-month, 12-month
- **Sector-relative momentum**: 3-month, 6-month, 12-month

**Why This "Redundancy" is Actually Good:**
- Gives multiple good candidates for first split decision
- Bagging and column subsampling average across these routes
- Results: Lower variance and better out-of-sample performance
- Different market regimes may favor different versions
- Model can adapt by using the most relevant version for each regime

**The Math:**
- With 70% column subsampling, not all features used in each tree
- Multiple correlated features → multiple pathways to same insight
- Averaging across pathways → more stable predictions
- More robust across regime changes

---

#### **Principle 4: Trees Don't Fear Collinearity**

**Why Linear Models Struggle:**
- Collinearity causes coefficient instability
- Makes interpretation difficult
- Can lead to overfitting
- Need to carefully manage feature correlations

**Why Tree Models Handle It:**
- Column subsampling (70% of features per tree) means not all features used together
- Bagging creates diverse trees naturally
- Regularization controls complexity without aggressive pruning
- Extra correlated features rarely hurt—often help by making split choices robust

**The Risk of Over-Pruning:**
- Makes split choice brittle
- Model becomes regime-dependent
- Loses adaptability to market condition changes
- Removes the insurance policy of alternative pathways

---

### ✅ Practical Feature Set Guidelines

**DO:**
- Keep balanced block sets (rel/abs momentum, revisions, value/quality, risk/liquidity, flow/technical)
- Include both absolute and relative versions of features
- Include multiple time windows (3m, 6m, 12m)
- Include both momentum and acceleration (change in momentum)
- Keep sector/industry interactions
- Control complexity with regularization and subsampling, NOT aggressive feature removal

**DON'T:**
- Aggressively remove correlated features
- Over-rely on feature importance for pruning
- Remove features just because they seem "redundant"
- Use linear model intuitions for tree model feature selection
- Drop the anchors or the orthogonal squeezers
- Go below ~87 features or above ~180 features

**Only Remove:**
- Obvious noise (1-week returns in monthly strategies)
- True duplicates (identical calculation with different name)
- Completely redundant technical variants (multiple Chaikin variants)

---

### 🌳 The "Activation" Secret: Train Wide, Apply Specific

This is the core architectural insight that enables massive flexibility and robustness.

#### **Training Phase: Build the Library**

**Approach:**
- Train models on BROAD universe (e.g., all stocks below $200M daily volume)
- Creates a rich "library of expertise" within the AI
- Model learns patterns across different market segments, sectors, conditions

**What Gets Learned:**
- Value patterns in deep value stocks
- Growth patterns in high-growth stocks
- Quality patterns across profitability spectrum
- Momentum patterns in different volatility regimes
- Special situation patterns (insider buying, low float, etc.)

---

#### **Application Phase: Activate Specific Patterns**

**Approach:**
- Apply SPECIFIC filters in portfolio strategies
- Don't retrain—just filter the universe the trained model sees

**Example Filters:**
- Industry exclusions (no financials)
- Valuation criteria (low P/B, low P/E)
- Market cap ranges (nano, micro, small, mid)
- Liquidity requirements (min daily volume)
- Quality screens (positive earnings, low debt)
- Special situations (insider buying, low float)

---

#### **The Counterintuitive Magic**

**What You Might Expect:**
- Applying filters different from training would "break" the model
- Model would fail because it wasn't trained on that specific subset
- Would need to retrain for each filtered universe

**What Actually Happens:**
- Filters **strategically activate specific tree patterns** within the AI
- Broad training created diverse set of decision trees
- Each tree specialized in different patterns/combinations
- Specific filters activate the trees most relevant to that segment
- Robustness is maintained or even improved

---

#### **Practical Examples from Live Trading**

**Example 1: No Buy Filters (Baseline)**
- Portfolio strategy runs with AI factor
- Exclude financials only
- No other buy rules
- Model uses full range of learned patterns
- Strong performance with broad diversification

**Example 2: Specific Buy Filter (Low Price-to-Book)**
- Same AI model as Example 1
- Add filter: Only stocks with P/B < 1.0
- Activates value-specific patterns in tree ensemble
- Different trees become more influential
- Performance remains strong with value tilt

**Example 3: Extreme Concentration (3-10 Stocks)**
- Heavy buy filters create very concentrated portfolios
- Example: Low P/B + Positive EPS revisions + Insider buying
- OOS live performance since 04/16/2020 "extremely stable"
- Demonstrates robustness even with radical filtering
- Only the most relevant trees fire strongly

---

#### **Why This Works: The Tree Architecture**

**Diverse Learning:**
- Each tree in 500-tree ensemble learned different patterns
- Tree #50 might specialize in: "Low P/B + Rising revisions"
- Tree #150 might specialize in: "Momentum + Quality"
- Tree #300 might specialize in: "Insider buying + Low float"

**Natural Activation:**
- When you filter for low P/B stocks, Tree #50 becomes more influential
- Its patterns match the filtered universe better
- Other trees contribute where relevant
- Final prediction = weighted average that naturally emphasizes relevant patterns

**Avoids Overfitting:**
- If you trained separately on each filtered subset → high overfitting risk
- Training broadly provides natural regularization
- Applying filters post-training = intelligent subset selection
- No danger of fitting to noise in small subsets

---

#### **Strategic Implications**

**Flexibility:**
- One AI model can power multiple strategies
- Different filters create different strategy characteristics
- No need to retrain for each market segment
- Rapid strategy development and testing

**Efficiency:**
- Leverage development work across multiple strategies
- Reduce overfitting risk from multiple training cycles
- Maintain consistency in modeling approach
- Lower computational costs

**Portfolio Construction:**
- Run multiple strategies from one AI model
- Creates diversification through different activation patterns
- Reduces correlation between strategies despite shared AI core
- Different filters = different factor exposures

**Risk Management:**
- Each strategy validates different aspects of the model
- If one filter type stops working, others continue
- Natural diversification across market segments
- Real-time validation of model robustness

---

### 🧪 The 5-Year Live Simulation (Pseudo OOS Testing)

#### **Methodology: Lock and Run**

**Step 1: Lock the Predictor**
- Train on data up to specific date (e.g., up to 2020)
- **Freeze the model completely**
- No further training or adjustments
- No parameter tweaking
- No feature changes

**Step 2: Run Forward Simulation**
- Simulate last 5 years completely out-of-sample
- Model has never seen this data
- No hindsight bias possible
- True test of generalization

**Step 3: Include Realistic Conditions**
- **AUM-based slippage models**
- Market impact based on position size
- Realistic bid-ask spreads
- Liquidity constraints
- Trading costs
- Rebalancing friction

---

#### **Purpose: True Out-of-Sample Validation**

**What This Tests:**
- Did the model learn genuine patterns or noise?
- Do patterns persist out-of-sample?
- How does performance degrade with real costs?
- Is the strategy viable with real capital?

**What This Prevents:**
- Overfitting to training period quirks
- Unrealistic performance expectations
- Underestimating real-world costs
- Deploying strategies that won't work live

---

#### **What to Look For: Success Criteria**

**Performance Metrics:**
- Positive alpha maintained out-of-sample
- Reasonable degradation from in-sample (not collapse)
- Risk-adjusted returns still attractive
- Drawdowns within expected ranges
- Volatility in expected ranges

**Stability Indicators:**
- Consistent factor exposures over time
- Predictable turnover patterns
- No regime-dependent blow-ups
- Smooth equity curve (no sudden breaks)
- Holdings make qualitative sense

**Cost Sensitivity:**
- Strategy survives realistic slippage
- Not too sensitive to small cost changes
- Sufficient alpha to overcome friction
- Appropriate turnover for strategy type

---

### 🧪 The Ultimate-Ultimate Test: Go Live

#### **Real-World Validation: The Final Proof**

**Deploy Multiple Systems:**
- Run "a ton of systems" simultaneously in live accounts
- Each represents different filters/activations of core AI
- Creates living laboratory of strategy performance
- Real money, real pressure, real results

**Why Multiple Systems:**
- Validates different aspects of the framework
- Tests robustness across market segments
- Creates natural diversification
- Provides real-time feedback on methodology

---

#### **Success Criteria: Two Key Comparisons**

**Criterion 1: Relative to Market Impact**

OOS live performance should hold up after accounting for:
- **Actual bid-ask spreads** (not theoretical)
- **Market impact from real orders** (price moves against you)
- **Timing delays** (can't always trade at exact close)
- **Liquidity constraints** (can't always get full position)
- **Adverse selection** (harder to trade winners than losers)

**What Success Looks Like:**
- Live returns close to simulation expectations
- Slippage in expected ranges
- No systematic adverse selection
- Can actually execute the strategy

---

**Criterion 2: Relative to Cap Curve (Pseudo-OOS Projection)**

Live results should compare well to the 5-year pseudo-OOS test:
- No major degradation from simulation to live
- Similar risk characteristics
- Similar return patterns
- Validates slippage modeling was realistic

**What Success Looks Like:**
- Live equity curve tracks pseudo-OOS curve
- Drawdowns in similar ranges
- Volatility as expected
- Factor exposures consistent

---

#### **The Feedback Loop: Rinse and Repeat**

**If Systems Hold Up Well:**
- Design choices are validated ("seem to be o.k.")
- Provides confidence in methodology
- Justifies deploying additional variations
- Increases allocation to proven strategies

**Continuous Improvement Process:**
1. **Learn**: Analyze what's working and why
2. **Refine**: Make small improvements to process
3. **Test**: Pseudo-OOS test new variations
4. **Deploy**: Put new strategies live
5. **Monitor**: Track performance vs expectations
6. **Iterate**: Feed learnings back into process

**Building Institutional Knowledge:**
- What types of filters work best
- Which market segments are most predictable
- How much concentration is optimal
- What turnover levels are sustainable
- Which hyperparameters are most robust

---

### 🎯 Complete Validation Pipeline: Summary

**The Full Testing Cascade:**

1. **Training Phase**: 2003-2020 with point-in-time data + gap enforcement
2. **Hyperparameter Testing**: Low sensitivity across configurations
3. **Basic Holdout Validation**: Test on unseen period
4. **5-Year Pseudo-OOS**: Locked model + realistic slippage
5. **Live Deployment**: Real money with multiple strategies
6. **Ongoing Monitoring**: Compare live vs expectations
7. **Annual Retraining**: Measured adaptation (e.g., add 2021 data)

**At Each Stage, Ask:**
- Does it work?
- Is it stable?
- Is it robust?
- Can I trust it with real money?

**Only Proceed If:**
- Each stage shows positive results
- No red flags or anomalies
- Performance degrades reasonably (not collapses)
- Results make qualitative sense

---

### 💡 Universal Lessons for Any Quantitative Investor

#### **Validation Principles:**
1. **Simple validation beats complex**: Basic holdout > walk-forward optimization
2. **Enforce temporal integrity**: Use gaps to prevent look-ahead bias
3. **Test with realistic costs**: Include slippage, impact, liquidity
4. **Lock before testing**: No adjustments during OOS period
5. **Go live to truly know**: Paper trading isn't enough

#### **Model Design Principles:**
1. **Conservative architecture wins**: Shallow, regularized models
2. **Stability matters more than fit**: Low hyperparameter sensitivity
3. **Rich features for forests**: Don't aggressively prune tree models
4. **Train broad, apply narrow**: Flexibility through activation
5. **Prioritize future over past**: Sacrifice in-sample perfection

#### **Deployment Principles:**
1. **Multiple strategies reduce risk**: Don't put all eggs in one basket
2. **Monitor continuously**: Compare live to expectations
3. **Adapt slowly**: Annual retraining, not daily tweaking
4. **Build institutional knowledge**: Learn what works over time
5. **Trust the process**: If validation works, trust the results

---

### 🔬 Why This Framework Works: Fighting Overfitting at Every Step

**Layer 1: Validation Method**
- Basic holdout prevents optimization games
- Gap enforcement prevents data leakage
- Realistic costs ground expectations

**Layer 2: Model Architecture**
- Shallow depth prevents memorization
- Heavy regularization penalizes complexity
- Subsampling creates ensemble diversity

**Layer 3: Training Approach**
- Broad universe training provides generalization
- Rich feature sets give strong anchors
- Long training period captures true patterns

**Layer 4: Testing Protocol**
- Locked model ensures honest test
- Multi-year OOS tests persistence
- Realistic costs test viability

**Layer 5: Live Validation**
- Real money reveals truth
- Multiple strategies validate robustness
- Continuous monitoring catches degradation

**Result:**
Each layer reinforces the others, creating a robust methodology that prioritizes real-world performance over backtested perfection.

---

### 📊 Comparison: Andreas's Approach vs. Common Pitfalls

| Aspect | Andreas's Approach | Common Pitfalls |
|--------|-------------------|-----------------|
| **Validation** | Simple basic holdout | Complex walk-forward (often overfit) |
| **Look-ahead** | Strict gap enforcement | Subtle data leakage |
| **Model depth** | Shallow (depth=5) | Deep networks (overfit) |
| **Regularization** | Heavy (0.5/0.5) | Light or none |
| **Features** | Rich sets (87-180) | Aggressively pruned |
| **Training** | Broad universe | Narrow, filtered |
| **Testing** | Locked model + costs | Adjustments allowed |
| **Deployment** | Multiple strategies | Single "best" strategy |
| **Adaptation** | Annual retraining | Constant tweaking |
| **Success metric** | Live performance | Backtest perfection |

---

### 🎓 Key Takeaways for Portfolio123 Users

#### **For AI Factor Training:**

1. **Use maximum available history**: More data = better pattern learning
2. **Train on broad universe**: Don't over-filter during training
3. **Implement proper gap**: Prevent look-ahead bias
4. **Use conservative settings**: LightGBM or ExtraTrees with shallow depth
5. **Rich feature sets**: 100-180 features across all factor categories
6. **Test multiple hyperparameters**: Look for low sensitivity

#### **For Portfolio Strategy Design:**

1. **Apply filters in strategy, not training**: Activate specific tree patterns
2. **Test multiple filter combinations**: One AI → multiple strategies
3. **Run both concentrated and diversified versions**: Test robustness
4. **Track which activations work best**: Build knowledge over time
5. **Monitor live vs. expectations**: Catch degradation early
6. **Be patient**: Give strategies time to work

#### **For Robust Strategy Development:**

1. **Start with validation design**: Plan OOS testing before training
2. **Build in realistic costs**: AUM-based slippage from day one
3. **Don't chase perfect backtests**: Prioritize OOS stability
4. **Deploy multiple variants**: Diversification across activations
5. **Learn from live results**: Feed insights back into process
6. **Annual retraining only**: Resist urge to constantly tweak

---

### 🏆 The Ultimate Success Pattern

**What Separates Winners from Losers:**

**Winners:**
- Robust processes that work imperfectly but consistently
- Simple validation that's hard to game
- Conservative models that generalize
- Multiple strategies that survive live trading
- Continuous learning and measured adaptation

**Losers:**
- Perfect backtests that collapse live
- Complex optimization that overfits
- Aggressive models that memorize noise
- Single strategies that eventually break
- Constant tweaking that destroys edge

**The Andreas Framework:**
Prioritizes being roughly right in the future over being perfectly right in the past.

---

This framework represents a battle-tested, pragmatic approach to AI-driven investing that has proven itself in live trading with real capital. The emphasis on robustness, proper validation, and conservative modeling makes it particularly suitable for actual deployment rather than academic research or marketing materials.