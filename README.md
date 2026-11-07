Stock Price & Trend Predictor using Machine Learning

A machine learning application that predicts stock prices and trends for any publicly traded company entered by the user (US stocks, Indian NSE/BSE stocks, ETFs, crypto, global indices) using historical financial market data and quantitative feature engineering.

🚀 Features
Any Global Company / Ticker Support: Enter any ticker from Yahoo Finance (e.g., AAPL, NVDA, TSLA, MSFT, GOOGL, RELIANCE.NS, TCS.NS, BTC-USD).
Multiple Companies at Once: Pass multiple tickers separated by commas to compare predictions across an entire portfolio.
Quantitative Feature Engineering:
Trend Indicators: SMA 5, 10, 20, 50 | EMA 12, 26
Momentum Indicators: MACD, MACD Signal, MACD Histogram, 14-day RSI (Relative Strength Index)
Volatility Indicators: Bollinger Bands (Upper, Lower, %B, Bandwidth), 10-day & 20-day Rolling Volatility
Volume & Interaction: Volume / 20-day Volume SMA ratio, High-Low spread, Close-Open return
Lagged Signals: 1-day, 2-day, 3-day, 5-day historical returns
Machine Learning Regressors:
Random Forest Regressor: Non-linear ensemble model capturing complex interactions
Gradient Boosting Regressor: Sequentially optimized decision trees
Ridge Regression: Regularized linear model with standard scaling
Ensemble Model: Combined consensus prediction across all models
Stationary Target Modeling: Instead of naively fitting models to non-stationary raw prices (which causes tree-based models to fail when extrapolating beyond historical price ranges), the system models Next-Day Percentage Return 

Strict Time-Series Validation: Chronological train/test split (80/20) with no lookahead bias or data leakage.
Actionable Evaluation Metrics:
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
Mean Absolute Percentage Error (MAPE %)
Directional Accuracy (%): How often the model correctly foresaw whether the stock would go UP or DOWN
Visual Analytics: Automatically generates high-resolution charts displaying actual vs. predicted curves, prediction error
distributions, top 10 feature importances, and next-day forecast targets.
📦 Installation

Ensure Python 3.10+ is installed:

bash
pip install -r requirements.txt
💻 How to Use
1. Interactive Mode (Prompt-driven)

Simply run:

bash
python main.py

You will be prompted to enter the stock ticker(s) and timeframe:

text
Enter Stock Ticker(s): AAPL, NVDA, RELIANCE.NS
Enter historical timeframe (6mo / 1y / 2y / 5y) [Default: 2y]: 2y
2. Command-Line Arguments Mode

Run directly with arguments for automated scripts or quick runs:

bash
python main.py --tickers AAPL,MSFT,TSLA --period 2y

Options:

--tickers, -t: One or more tickers separated by commas.
--period, -p: Historical data duration (6mo, 1y, 2y, 5y, max).
--no-plot: Skip chart generation.
--output-dir, -o: Custom directory for saving charts (default: current directory).
📊 Sample Output
text
==========================================================================================
 ANALYSIS & PREDICTION REPORT: Apple Inc. (AAPL)
==========================================================================================
 Latest Recorded Market Close (2026-11-06): $224.23
------------------------------------------------------------------------------------------
Model Name           | RMSE       | MAE        | MAPE (%)   | Dir. Acc (%)   | Next-Day Pred  | Expected %
------------------------------------------------------------------------------------------
Random Forest *      | 2.81       | 2.15       | 0.98%      | 56.8%          | $225.41        | +0.53%   
Gradient Boosting    | 2.94       | 2.24       | 1.02%      | 54.5%          | $224.95        | +0.32%   
Ridge Regression     | 2.89       | 2.20       | 1.00%      | 53.4%          | $225.10        | +0.39%   
Ensemble Model       | 2.83       | 2.17       | 0.99%      | 55.7%          | $225.15        | +0.41%   
------------------------------------------------------------------------------------------
 * Lowest RMSE Model on unseen holdout test data.
##########################################################################################
 SUMMARY FORECAST FOR NEXT TRADING SESSION (Random Forest):
 Target Price:   $225.41
 Expected Move:  +0.53% (BULLISH ▲)
 Test Accuracy:  Directional: 56.8% | MAPE: 0.98%
##########################################################################################
[✓] Evaluation plot saved successfully to 'AAPL_prediction_chart.png'.
📈 Ticker Cheatsheet
Market / Asset	Example Symbols
US Tech Giants	AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
US Blue Chips	JPM, BRK-B, V, WMT, DIS, KO
Indian NSE	RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, TATAMOTORS.NS
Indian BSE	RELIANCE.BO, TCS.BO, INFY.BO
Indices / ETFs	SPY, QQQ, ^GSPC (S&P 500), ^NSEI (Nifty 50), ^BSESN (Sensex)
Crypto Assets	BTC-USD, ETH-USD, SOL-USD
