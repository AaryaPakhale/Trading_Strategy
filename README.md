# Nomura Quant Challenge 2025 - Trading Strategies

Welcome to our repository for the **Nomura Quant Challenge 2025**. This project implements quantitative trading strategies with a focus on market making and risk mitigation, developed during the challenge.

## Contents

- [Overview](#overview)
- [Strategy Summary](#strategy-summary)
- [Methodology](#methodology)
- [Model Design](#model-design)
- [Risk Management](#risk-management)
- [Performance Evaluation](#performance-evaluation)
- [Usage Instructions](#usage-instructions)
- [Team](#team)

---

## Overview

This project was built as part of the Nomura Quant Challenge 2025, where the objective was to develop and implement trading strategies on equities data. The main goals were:

- To build a market making strategy with inventory and risk constraints.
- To implement a risk mitigation strategy to minimize portfolio Value at Risk (VaR).
- To simulate real-world trading dynamics using historical order book and trade data.

## Strategy Summary

We developed two major components:

1. **Market Making Strategy**:
   - Generates bid-ask quotes using alpha signals, trend detection, and inventory management.
   - Adheres to tick size and inventory constraints with penalties factored into P&L.

2. **Risk Mitigation Strategy**:
   - Constructs hedges for a portfolio using liquid equities.
   - Minimizes 95% historical VaR and capital cost using historical return covariance.

## Methodology

- **Data**: Historical L2 order book data and trade data were used.
- **Signal Generation**: Alpha signals were generated using statistical momentum indicators and recent price changes.
- **Execution**: Orders were simulated under strict tick size and timing constraints.
- **Backtesting**: Strategies were tested over multiple time periods and evaluated on P&L, Sharpe ratio, and VaR reduction.

## Model Design

- **Market Maker**:
  - Quote prices were adjusted based on recent price trends and inventory exposure.
  - Used trend-following alpha with mean reversion tendencies.

- **Risk Hedge**:
  - Portfolio covariance matrix estimated from historical returns.
  - Optimization problem solved to minimize portfolio VaR given a capital budget.

## Risk Management

- **Market Making**:
  - Real-time inventory control with quote-skew adjustments.
  - Penalties imposed for large inventory holdings and adverse fills.

- **Portfolio Hedging**:
  - 95% historical VaR calculated using rolling window.
  - Capital usage constraints enforced to simulate real-world risk limits.

## Performance Evaluation

- **Metrics**:
  - Net P&L, inventory penalty, execution cost.
  - Risk-adjusted returns and VaR reduction effectiveness.

- **Results**:
  - Consistent profits under normal volatility conditions.
  - Effective hedging of portfolio tail risk using a sparse hedge basket.

## Usage Instructions

To run the simulations:

```bash
# Clone the repository
git clone https://github.com/yourusername/trading_strategy.git
cd trading_strategy

# Install dependencies
pip install -r requirements.txt

# Run the market making simulation
python simulate_market_maker.py

# Run the portfolio risk mitigation module
python hedge_portfolio.py

```

