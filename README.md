# Quantitative-Portfolio-Optimization


## Executive Summary
This project is an advanced algorithmic portfolio management tool designed to optimize asset allocation across global equities and local mutual funds. By integrating Natural Language Processing (NLP) for sentiment-driven return predictions and the Black-Litterman model for portfolio optimization, the engine calculates the most efficient asset distribution to maximize the Sharpe ratio. It also features a robust Monte Carlo simulation module for forward-looking risk assessment.

## Technologies Used
* **Programming Language:** Python
* **Data Acquisition:** `yfinance` (Yahoo Finance API), `tefas` (Turkish Electronic Fund Trading Platform Crawler)
* **Data Processing & Mathematics:** `pandas`, `numpy`, `scipy.optimize`
* **Natural Language Processing (NLP):** `nltk` (VADER Sentiment Analysis)
* **Data Visualization:** `matplotlib`

## Scope & Methodology
1. **Automated Data Synchronization:** Simultaneously fetches, cleans, and aligns historical price data from international stocks/ETFs and local mutual funds, ensuring a unified time-series dataset.
2. **NLP-Based Sentiment Analysis:** Scans recent global financial news headlines for the selected assets and dynamically adjusts expected return targets and confidence intervals based on market sentiment.
3. **Black-Litterman Optimization:** Incorporates the NLP-driven views into historical market data. It uses the SLSQP algorithm to determine the optimal asset weights, strictly constrained to maximize the Sharpe ratio while applying mathematical stabilization to prevent covariance matrix singularity (SVD crashes).
4. **Monte Carlo Risk Simulation:** Projects 1,000 simulated portfolio paths over a 1-year (252 trading days) horizon to evaluate potential drawdowns, calculate expected median returns, and determine the 5% Value at Risk (VaR) metric.
