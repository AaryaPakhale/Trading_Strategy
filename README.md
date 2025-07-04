# 📈 Trading Challenge – Multi-Strategy Ensemble with ML Selection

This repository contains the complete solution for a multi-strategy trading challenge involving 20 stocks over 4000 trading days. The solution includes 5 alpha-generating strategies and a machine learning-based ensemble model to select the best strategy per day.

---

## 🧩 Problem Overview

The goal is to:
1. Design **5 individual trading strategies** that assign weights to stocks based on historical price data.
2. Develop a **daily strategy selector** that picks the best strategy using only past information.

---

## 📊 Task Breakdown

### ✅ Task 1: Individual Trading Strategies

Implemented 5 strategies using price-based technical indicators:

| Strategy | Description |
|----------|-------------|
| **Strategy 1** | Weekly returns mean-reversion (long losers, short winners) |
| **Strategy 2** | Short vs Long-term Moving Average Divergence |
| **Strategy 3** | Rate of Change (ROC) momentum reversal |
| **Strategy 4** | Support/Resistance proximity-based breakout logic |
| **Strategy 5** | Stochastic %K oscillator (14-day high/low band positioning) |

Each strategy outputs a weight vector of size 20 per day (weights sum to 1 and -1 for long and short sides).

---

### ✅ Task 2: Machine Learning-Based Strategy Selector

A robust ensemble model dynamically selects one of the five strategies each day using engineered market features.

#### 📌 Feature Engineering
12 features designed to capture market regimes:
- Short-term and long-term volatility
- Trend acceleration and mean returns
- Momentum skew and dispersion
- Market-wide average correlation
- Volume metrics and efficiency measures

#### 🧠 Model Ensemble
- **Random Forest (Stable)**: shallow, high-regularization
- **Logistic Regression (L1)**: interpretable, sparse
- **Random Forest (Variant)**: different seed and depth

A **voting mechanism** selects the final strategy daily, unless a single model clearly outperforms on the CV set.

---


## 🧪 Reproducibility

To reproduce the results:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Task 1: Generate weights for all 5 strategies
python main.py --task 1

# Run Task 2: Apply ML selector and generate final weights/performance
python main.py --task 2

# Run Task 3: Apply turnover-aware strategy selection
python main.py --task 3
```

---

## 📈 Evaluation Metrics

- **CAGR**
- **Sharpe Ratio**
- **Max Drawdown**
- **Volatility**
- **Strategy selection consistency**
- **Net vs Gross return analysis under transaction costs**

---

## 📝 Documentation

Full methodology, feature definitions, model choices, and performance visualizations are available in `PS/documentation_3.pdf`.

---

## 🎓 Author

- **Name:** Aarya Pakhale  
- **Challenge:** [Nomura Global Markets Quant Challenge]  
- **Status:** Finalist  

---

## 📫 Contact

For inquiries or collaboration opportunities:

- 📧 [aaryapakhale01@gmail.com]  
- 🌐 [[LinkedIn URL](https://www.linkedin.com/in/aarya-pakhale-0b9788217/)]

---

## 🏁 License

This project is provided under the MIT License.



