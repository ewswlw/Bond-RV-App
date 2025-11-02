# Portfolio123 Expert Knowledge Base for AI Agents

## When to Use

- Use this document as the master index and onboarding guide for AI agents or teammates who support Portfolio123 workflows.
- Apply it before delegating tasks so everyone knows which supporting module to consult for factors, syntax, AI, automation, and identifiers.
- Reference it when building or auditing end-to-end workflows; the example agent process demonstrates the expected sequence.
- Consult it when updating the knowledge base to ensure new modules are linked appropriately.
- Once you know the specific topic, jump to the referenced guide for detailed instructions.

**Version**: 1.0  
**Date**: October 11, 2025  
**Author**: Manus AI

## 1. Introduction

This document serves as a comprehensive knowledge base about the Portfolio123 platform, designed to provide an expert-level understanding for an AI agent. The goal is to enable the agent to assist users in creating sophisticated stock screens, ranking systems, and machine learning models by translating plain English requests into the specific syntax and workflows of Portfolio123.

The knowledge base is structured into several detailed guides, each covering a critical component of the platform. By referencing these guides, the AI agent will gain a deep understanding of data fields, identifier systems, rule syntax, and advanced AI capabilities.

## 2. Knowledge Base Structure

This knowledge base is organized into the following modules. Each module is a self-contained guide that provides in-depth information on a specific topic.

### Table of Contents

1.  **[Complete Factor and Data Field Reference](./factors_complete.md)**
    *   A complete dictionary of all 1,126+ data fields, factors, functions, and their definitions available in Portfolio123. This is the foundational data mapping required to build any rule.

2.  **[Screening and Ranking Rules Guide](./screening_ranking_guide.md)**
    *   A detailed guide on how to construct screening and ranking rules. It covers syntax, operators, functions, and provides numerous examples of translating plain English investment concepts into Portfolio123 formulas.

3.  **[Machine Learning & AI Capabilities Guide](./ml_ai_capabilities_guide.md)**
    *   A comprehensive overview of Portfolio123's AI Factor module. This guide details the supported machine learning algorithms, data preprocessing options, model training workflows, and best practices for creating and deploying AI-driven factors.

4.  **[API and Automation Guide](./api_automation_guide.md)**
    *   A complete reference for the Portfolio123 REST API and the `p123api` Python wrapper. It includes examples for data retrieval, automated screening, backtesting, and integration with other platforms.

5.  **[Ticker and Identifier System Guide](./ticker_identifier_system.md)**
    *   An essential guide to understanding how Portfolio123 handles stock identification. It covers the various identifiers (Ticker, FIGI, CUSIP, P123 StockID), how delisted stocks are managed, and best practices for ensuring data accuracy.

6.  **[Industry Classification (RBICS) Guide](./rbics_classification.md)**
    *   An explanation of the Revere Business Industry Classification System (RBICS) used by Portfolio123 for sector and industry analysis, including the hierarchy and mnemonic codes.

## 3. How to Use This Knowledge Base

An AI agent should use this knowledge base as its primary reference for all Portfolio123-related tasks. When a user makes a request, the agent should:

1.  **Deconstruct the Request**: Identify the key concepts in the user's plain English request (e.g., "cheap stocks," "high growth," "low risk").

2.  **Map to Factors**: Use the **Factor and Data Field Reference** to find the corresponding Portfolio123 factor codes for the concepts (e.g., `PEExclXorTTM` for "P/E ratio," `SalesGr%TTM` for "revenue growth").

3.  **Construct the Rules**: Refer to the **Screening and Ranking Rules Guide** to correctly formulate the syntax for screens or ranking systems. For example, to find stocks with a P/E below 15, the rule is `PEExclXorTTM < 15`.

4.  **Leverage AI/ML**: If the user requests predictive modeling or wants to "find hidden patterns," consult the **Machine Learning & AI Capabilities Guide** to formulate a plan for creating an AI Factor.

5.  **Automate Workflows**: For requests involving data export, automated backtesting, or integration with other tools, use the **API and Automation Guide** to generate the necessary Python code using the `p123api` library.

6.  **Ensure Data Integrity**: When dealing with historical data, ticker changes, or data imports, consult the **Ticker and Identifier System Guide** to select the correct identifier (e.g., FIGI for external mapping, P123 StockID for internal consistency).

### Example Agent Workflow

**User Request**: "*Create a screen for me that finds profitable, small-cap technology stocks with low valuation and recent insider buying.*"

**Agent's Internal Steps**:

1.  **Deconstruct**: 
    *   *Profitable*: Look for profitability metrics (e.g., ROE, Net Income).
    *   *Small-cap*: Define market cap range (e.g., $300M - $2B).
    *   *Technology stocks*: Filter by sector.
    *   *Low valuation*: Look for value factors (e.g., P/E, P/B).
    *   *Insider buying*: Find insider trading factors.

2.  **Map to Factors** (using `factors_complete.md`):
    *   Profitability: `ROE > 0.15` or `NetIncTTM > 0`
    *   Small-cap: `MktCap > 300 AND MktCap < 2000`
    *   Technology: `Sector = TECH` (from `rbics_classification.md`)
    *   Low valuation: `PEExclXorTTM < 20`
    *   Insider buying: `InsiderBuying6M > 0`

3.  **Construct the Screen Rule** (using `screening_ranking_guide.md`):
    ```
    # Technology Sector
    Sector = TECH
    
    # Small Cap
    MktCap > 300 AND MktCap < 2000
    
    # Profitable
    ROE > 0.15
    
    # Low Valuation
    PEExclXorTTM < 20
    
    # Insider Buying
    InsiderBuying6M > 0
    ```

4.  **Respond to User**: Present the generated screen rules and explain how they match the user's request, potentially offering to run the screen via the API (using `api_automation_guide.md`).

By following this structured approach, the AI agent can effectively act as a Portfolio123 expert, providing accurate, actionable, and context-aware assistance.

