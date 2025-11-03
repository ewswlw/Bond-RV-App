## 4 % Rule

act like a machine learning, algo trading and data scientist expert, your goal, and algo trading expert, your goal is to generate trading strategy that would have beat the buy and hold  for the CAD-IG-ER index byt at least 2.5% on an annualized return basis (min 7 period holding period, no leverage, long only, binary positioning,no transaction costs). You do this by you'll find hidden patterns in the data, do iterate on data transformation, feature engineering, get very very creative, take a statistical approach. Keep iterating till you get to my target. Must do all statistical validation and bias checking as you iterate as to only provide me with strategies that suitable for live trading in the future. 

## Coding

- for all documentation, include date and time stamp of entry, and any time making new documentation figure out if any of the old documentation needs to be updated
- any virtual environment should be set up with poetry, and thus any libraries with poetry install
- everything project must be set up in a modular way (correct folder pattern etc) according to industry best practices
- after finishing a request that adds/edits files show a summary of what was changed in each file and their paths
- Unless already in another virtual enviroment, always use Poetry environment (`poetry run python ...`). All code execution must use the current Poetry environment. Dependencies are managed via `pyproject.toml` and `poetry.lock`
- Python 3.10-3.12 compatibility. Follow PEP 8 style guidelines. Use type hints where appropriate.
- avoid any unicode encoding issues for Windows. Use ASCII-compatible characters if needed. 

# Testing
- All tests should be in `tests/` directories within each module
- Use pytest for testing
- Test files should follow naming convention: `test_*.py`

# Path Handling
- Use `pathlib.Path` for path operations
- Scripts should detect their location and adjust paths accordingly
- Support running from multiple directory locations






## Genera AI

- whenever i give you a prompt you use think deeply about what's applicable given my end goal and current context. 
- Example if i am doing an algo trading project and in the prompt it has something about volume but the raw data only has prce you must ignore that
- when you ignore these types of things or use judgement in interpreting my prompts you must let me know
- always ask clariying questions in planning mode, end keep on asking questions before you enter plan mode until all possible questions are answered that you think cover virtually any possible ambiguity 
Update all my .cursorrules files before every git commit


## Claude Code
- always ask clariying questions in planning mode
- Update all my CLAUDE.md files before every git commit


## AI General Logic

# Asking Questions to Clarify Intent

- Ask me clarifying questions until you are 95% confident you can complete the task successfully 
- What would a top 0.1% person in this field think 
- Reframe this in a way that challenges how I see the problem 
- Roleplay as Nobel laureate solving this problem 
- Flesh out all possible edge cases
- Onece answered, list any follow on remaining questions and keep going till all answered
- Use ultrathink and think very deeply 


# Thinking of More Solutions

- For each query, please generate a set of five possible responses, and rank them on which is 
best and why
- 


# Getting To The Best Prompt
- Extract the user’s core intent and reframe it as a clear, targeted prompt  
- Structure inputs to optimize model reasoning, formatting, and creativity  
- Anticipate ambiguities and preemptively clarify edge cases  
- Incorporate relevant domain-specific terminology, constraints, and examples  
- Output prompt templates that are modular, reusable, and adaptable across domains
- When designing prompts, follow this protocol: Define the Objective: What is the outcome or deliverable? Be unambiguous. Understand the Domain: Use contextual cues (e.g., cooling tower paperwork, ISO curation, general context). You may provide code examples but never try to actually execute any of the code. Choose the Right Format: Narrative, JSON, bullet list, markdown, code-based on the use case.   Inject Constraints: Word limits, tone, persona, structure (e.g., headers for documents). Build Examples: Use “few-shot” learning by embedding examples if needed. Simulate a Test Run: Predict how the LLM will respond. Refine.  





## Algo Trading

# EDA
- Objective: build robust, anti-correlated trading signals based on deep statistal relationshps found in the data. Before each analysis, interrogate five pillars—statistical validity, regime stability, implementation feasibility, risk exposure, and overfitting defenses. Methodology highlights: bootstrapped confidence intervals, multiple testing corrections, regime-aware models (HMM/GARCH), time-series cross-validation, Monte Carlo tail probes, walk-forward and 20%+ out-of-sample testing, crisis stress tests, cross-asset robustness, and live-execution replication with monitoring/early warning systems. Required output format for every analysis area: define scope, give executive summary, document hypothesis testing (with corrections), attribution statistics, robustness checks, visual evidence, codable trading/risk rules, regime filters, automation protocols, parameter tuning (with walk-forward), full performance metrics plus confidence intervals, driver breakdowns, failure analysis, system architecture, data/production requirements, pitfalls, regime mapping, and model-risk assessment. Validation checklist enforces statistical power, bias prevention, realistic costs, and regime robustness. Success requires operational rigor, economic significance, robustness, implementability, risk mastery, and adaptability—backed by examples and constant breakthrough-level QA.

# 20 Ideas
- Guide = disciplined ideation engine for producing 20 differentiated strategies from one core concept. It refuses to run until you supply that concept (e.g., volatility regime transitions). Optional fields narrow platform, timeframe, trade count, markets, data, or banned approaches. Process: Phase 1 decomposes the concept into fundamental forces, edge persistence, and implementation constraints; Phase 2 applies top-tier quant perspective with first-principles market structure analysis; Phase 3 challenges assumptions to unlock unconventional angles.Output template for each idea: name, three-condition setup logic tied to the concept, best markets, strategy type, timeframe, first-principles edge note, testing needs (data, trade counts, constraints), and 0–10 scores for plausibility, testability, creativity. Quality gate ensures all 20 ideas are unique, testable by a single researcher, exploit distinct inefficiencies, and rely on standard data/tooling. Execution flow = validate core concept → process optional constraints → first-principles breakdown → generate 20 ideas → run uniqueness/testability checks → deliver final set with next steps.

# Genetic Algos:
- Why GA: multi-objective optimization (return, Sharpe, risk), broad parameter exploration, natural constraint handling, robustness through population diversity, ability to tackle non-linear market structures. Core pieces: define populations, chromosomes (binary or real), and multi-factor fitness with constraint penalties. Implement selection (tournament/rank), crossover (single-point/uniform), mutation (Gaussian/bit-flip) while managing bounds. Represent strategies via indicator parameters, weighted signal fusion, thresholds; support Pareto fronts for trade-off decisions and enforce caps on drawdown, turnover, position size. Implementation flow: data handler loads/validates prices, builds features; GA class initializes populations, runs evolution with elitism, tracks convergence, and stops when fitness plateaus. Backtest engine enforces temporal integrity, realistic execution, and calculates full metrics suite (returns, Sharpe, drawdown, Calmar, win rate, profit factor). Validation layer runs bootstrap, Monte Carlo, and walk-forward analyses with proper in/out-of-sample splits. Best practices: design sensible parameter ranges (log scaling, microstructure-aware), preserve diversity, adapt mutation, include transaction costs, and benchmark fitness. Common pitfalls to avoid: overfitting, look-ahead, survivorship bias, data snooping, ignoring costs. Outcome: a rigorous GA framework delivering optimized strategies plus validation artifacts ready for institutional deployment.

# ML Trading
- Core benefits: detect non-linear patterns, and integrate risk controls; core hazards: non-stationary markets, low signal-to-noise, regime shifts, overfitting, latency. Pipeline: formulate problem (classification/regression targets), engineer features (price, volume, volatility, momentum, alternative data), run feature selection (correlation filters, mutual information, RFE, Lasso). Model menu spans linear, trees, ensembles, LSTM/transformers; ensembles optimize weights via validation. Validation framework centers on time-series CV, walk-forward analysis, and randomized hyperparameter search keyed to information coefficient. Risk layer covers Kelly-style sizing, portfolio risk limits, dynamic stops, and ATR-based exits. Implementation skeleton builds reusable ML trading system classes for backtesting and real-time execution with monitoring hooks. Optimization sections address latency, memory, quantization, and dimensionality reduction; deployment guidance covers REST serving, health checks, drift monitoring, and alerting. Best practices: clean/validate data, avoid look-ahead, adjust for corporate actions, start simple, watch overfitting, retrain regularly, apply robust risk management, stress test. Pitfalls: survivorship bias, data snooping, regime ignorance, latency/scaling failures. Performance evaluation template computes RMSE/MAE, correlation, strategy returns, Sharpe, drawdown, hit rate and prints reports. Outcome: modular, institution-ready ML systems with end-to-end QA and production readiness.

# Strategy Improver
- Strategy Improver = disciplined enhancement loop for an already good strategy. Before running filters you must capture current setup (rules, metrics, objectives), market context, data sources, plus any optional constraints (timeframe, risk tolerance, complexity, banned tweaks). Philosophy: abandon Holy Grail hunts; improve a logical, modestly profitable base by testing one economically justified filter at a time, using binary on/off questions to avoid curve-fitting. Three-step framework: Vet base strategy: positive total return, Sharpe >0.5, drawdown <30%, profit factor >1.2, ≥50 trades, solid economic story. Test filters individually with clean isolation; library spans trend, momentum, volatility, volume, time-based, price-action, each backed by reasoning. Compute performance deltas on total return, Sharpe, drawdown, profit factor, win rate, etc.
Promote filters only if they deliver multi-metric gains, statistical significance, regime persistence, and clear economics; rank via weighted improvement scores.
Supporting modules (not shown in summary) cover economic logic validation, implementation procedures, performance reporting, code templates, best practices, curve-fit traps, and advanced optimization. Overall goal: incremental, evidence-backed upgrades while documenting rationale and outcomes

# Validator
- Pipeline stages: data quality → temporal integrity (look-ahead, PIT checks, survivorship) → statistical significance (t-tests, bootstrap, IC, Sharpe tests, multiple-testing corrections) → performance validation (returns, Sharpe, drawdown, Calmar, Sortino, IR, sanity checks) → risk validation (VaR/ES, drawdowns, tail metrics, stress scenarios) → overfitting detection (in/out-sample decay, parameter stability, CV, random-data runs, regime analysis) → stress/regime testing. Implementation scaffold provides reusable validator classes, walk-forward engines, and automated report generation so every finding is auditable. Best practices emphasize clean point-in-time data, comprehensive documentation, multiple validation methods, regime awareness, stress tests, and Monte Carlo. Pitfall list flags look-ahead, survivorship, data snooping, weak sample sizes, p-hacking, unrealistic assumptions, omitted costs, and operational gaps—ensuring institutional-grade QA before strategies ship.

# VectorBT/Backtesting
- All backtests must be done with the VectorBT library. For every backtest make sure the total return is validated against a manual backtest total return, to make sure everything went smoothly with the implementation.
VectorBT basics (portfolio objects, signal formats, execution timing), configuration recipes (single/multi-asset, fees/slippage, sizing modes), signal generation with temporal integrity, and execution/position management classes. Validation stack enforces point-in-time data, anti-look-ahead, manual return comparison, Performance analysis section wraps stats, trade/position diagnostics, visualizations, and HTML reporting. Advanced modules handle multi-asset cash sharing, signal filtering/enhancement, risk constraints (drawdown/volatility limits, VaR/ES), plus end-to-end implementation example with integrated validation and QA. Troubleshooting lists fixes for timing, sizing, alignment, validation mismatches, and adds optimization/debug tips so production backtests stay audit-ready. Make sure that every backtest includes the following (in both csv's and txt.files, and for the .txt files must show all the data from all the methods in a very nicely formated way) pf.stats(), pf.returns.stats(), pf.drawdowns.stats(), pf.trades.stats(), pf.trades.records_readable. Must always format the numbers and data nicely and logically. 
- All strategies must inherit from `BaseStrategy`. Use VectorBT for backtesting operations. Configuration via YAML files in `configs/` directory. Outputs should go to `outputs/` directory



# Pattern Discovery
Universal Pattern Discovery Framework = mandatory playbook before any pattern mining. Core rule: AI must trigger Precision Clarification Engine and capture every required spec—target CAGR, asset, leverage/shorting permissions, positioning style, holding limits, signal-trade lag, data frequency/range/source, volume/alt data, risk tolerance, language, real-time intent, ML complexity, pattern focus, sig thresholds, OOS/walk-forward prefs, output format/detail/interactivity/assumptions (suggest default parameters based on context). Only after answers are confirmed does the workflow progress through elite analysis, paradigm challenge, visualization, and Nobel-level rigor to test hypotheses, surface edge cases, and design experiments.Implementation stack: information-driven bars (dollar/volume/tick/imbalance) for data engineering; triple-barrier labeling with volatility/regime-aware thresholds; meta-labeling; purged CV; fractal analysis; HMM regime detection; wavelet multi-scale patterns; pattern lifecycle tracking; pattern synthesis; multi-objective GA optimization; VectorBT integration with precision validation. Decision tree and priority matrix guide which modules to deploy first (Phase 1 essentials → Phase 4 advanced). Process ends with detailed implementation plan and user confirmation before code or discovery runs.




## XBBG

- python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi
