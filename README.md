# Bitcoin Volatility Spike Trading Strategy

This project implements and evaluates a systematic trading strategy on historical Bitcoin price data using Python. The strategy identifies volatility-adjusted price spikes combined with abnormal trading volume and tests whether fading these spikes produces statistically significant returns.

The backtesting framework includes out-of-sample evaluation, parameter search, and Monte Carlo permutation testing to distinguish true signal from noise.

---

## Strategy Overview

The strategy operates on high-frequency Bitcoin data and follows three core steps:

1. **Feature Engineering**
   - Log returns over multiple horizons
   - Rolling volatility estimates
   - Rolling volume sums and volume z-scores

2. **Signal Generation**
   - Detects large positive or negative price moves relative to recent volatility
   - Requires elevated trading volume to confirm market stress
   - Generates contrarian signals that fade extreme moves

3. **Position Management**
   - Signals are entered on the following bar to avoid lookahead bias
   - Positions are held for a fixed number of bars using signal persistence logic

---

## Backtesting Methodology

- Data is split chronologically into train, validation, and test sets
- Strategy parameters are selected using validation Sharpe ratio
- Final performance is evaluated on an unseen test set
- Performance metrics include:
  - Total return
  - CAGR
  - Sharpe and Sortino ratios
  - Max drawdown
  - Profit factor
  - Win rate and exposure

To assess statistical significance, a Monte Carlo permutation test is applied by randomly shuffling returns while holding signals fixed.

---

