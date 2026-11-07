"""
=============================================================================
STOCK PRICE & TREND PREDICTOR USING MACHINE LEARNING
Rectified and improved version
=============================================================================

Requirements:
    pip install numpy pandas scikit-learn yfinance matplotlib

Run:
    python stock_predictor.py

Features:
    - Yahoo Finance historical data
    - SMA / EMA
    - MACD
    - RSI
    - Bollinger Bands
    - Volatility
    - Lagged returns
    - Random Forest
    - Gradient Boosting
    - Ridge Regression
    - Ensemble prediction
    - Chronological train/test split
    - Directional accuracy
    - RMSE / MAE / MAPE / R²
    - Prediction chart
    - Feature importance
=============================================================================
"""

import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ============================================================================
# UTF-8 OUTPUT
# ============================================================================

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


# ============================================================================
# FETCH STOCK DATA
# ============================================================================

def fetch_stock_data(ticker: str, period: str = "2y"):
    """
    Download historical market data from Yahoo Finance.

    Returns:
        df             : OHLCV DataFrame
        company_name   : Company name
        currency       : Currency symbol
    """

    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker symbol cannot be empty.")

    print(f"\n[*] Fetching market data for '{ticker}' from Yahoo Finance...")

    try:
        t_obj = yf.Ticker(ticker)

        df = t_obj.history(
            period=period,
            auto_adjust=True
        )

        # Try to get company information.
        try:
            info = t_obj.info

            company_name = (
                info.get("shortName")
                or info.get("longName")
                or ticker
            )

            curr = info.get("currency", "USD")

            curr_map = {
                "USD": "$",
                "INR": "₹",
                "EUR": "€",
                "GBP": "£",
                "JPY": "¥",
            }

            currency = curr_map.get(curr, curr)

        except Exception:
            company_name = ticker
            currency = "$"

    except Exception as e:
        raise RuntimeError(
            f"Error fetching data for '{ticker}': {e}"
        ) from e

    if df is None or len(df) < 60:
        rows = len(df) if df is not None else 0

        raise ValueError(
            f"Insufficient data for '{ticker}' "
            f"({rows} rows found). "
            f"Please verify the ticker symbol."
        )

    # Flatten MultiIndex columns if Yahoo Finance returns them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns from Yahoo Finance: {missing}"
        )

    df = (
        df[required_columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    if len(df) < 60:
        raise ValueError(
            f"Not enough clean trading data for '{ticker}'."
        )

    print(
        f"[OK] Retrieved {len(df)} trading days of data "
        f"for {company_name} ({ticker})."
    )

    return df, company_name, currency


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame):
    """
    Calculate technical indicators and supervised learning targets.
    """

    data = df.copy()

    # ------------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------------

    data["Daily_Return"] = data["Close"].pct_change()

    data["Log_Return"] = np.log(
        data["Close"] / data["Close"].shift(1)
    )

    # ------------------------------------------------------------------------
    # Moving averages
    # ------------------------------------------------------------------------

    data["SMA_5"] = data["Close"].rolling(window=5).mean()
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()

    data["EMA_12"] = data["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    data["EMA_26"] = data["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    # ------------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------------

    data["MACD"] = (
        data["EMA_12"] -
        data["EMA_26"]
    )

    data["MACD_Signal"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    data["MACD_Hist"] = (
        data["MACD"] -
        data["MACD_Signal"]
    )

    # ------------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------------

    delta = data["Close"].diff()

    gain = delta.where(
        delta > 0,
        0.0
    ).rolling(window=14).mean()

    loss = (-delta.where(
        delta < 0,
        0.0
    )).rolling(window=14).mean()

    rs = gain / (loss + 1e-9)

    data["RSI_14"] = 100 - (
        100 / (1 + rs)
    )

    # ------------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------------

    data["BB_Middle"] = data["SMA_20"]

    data["BB_Std"] = (
        data["Close"]
        .rolling(window=20)
        .std()
    )

    data["BB_Upper"] = (
        data["BB_Middle"] +
        2 * data["BB_Std"]
    )

    data["BB_Lower"] = (
        data["BB_Middle"] -
        2 * data["BB_Std"]
    )

    data["BB_PctB"] = (
        (data["Close"] - data["BB_Lower"])
        /
        (
            data["BB_Upper"] -
            data["BB_Lower"] +
            1e-9
        )
    )

    # ------------------------------------------------------------------------
    # Historical volatility
    # ------------------------------------------------------------------------

    data["Volatility_10"] = (
        data["Daily_Return"]
        .rolling(window=10)
        .std()
    )

    data["Volatility_20"] = (
        data["Daily_Return"]
        .rolling(window=20)
        .std()
    )

    # ------------------------------------------------------------------------
    # Price / volume ratios
    # ------------------------------------------------------------------------

    data["HL_Ratio"] = (
        (data["High"] - data["Low"])
        /
        (data["Close"] + 1e-9)
    )

    data["CO_Ratio"] = (
        (data["Close"] - data["Open"])
        /
        (data["Open"] + 1e-9)
    )

    data["Price_to_SMA20"] = (
        data["Close"]
        /
        (data["SMA_20"] + 1e-9)
    )

    data["Price_to_SMA50"] = (
        data["Close"]
        /
        (data["SMA_50"] + 1e-9)
    )

    data["Vol_SMA_10"] = (
        data["Volume"]
        .rolling(window=10)
        .mean()
    )

    data["Vol_Ratio"] = (
        data["Volume"]
        /
        (data["Vol_SMA_10"] + 1e-9)
    )

    # ------------------------------------------------------------------------
    # Lagged returns
    # ------------------------------------------------------------------------

    data["Lag_Return_1"] = (
        data["Daily_Return"].shift(1)
    )

    data["Lag_Return_2"] = (
        data["Daily_Return"].shift(2)
    )

    data["Lag_Return_3"] = (
        data["Daily_Return"].shift(3)
    )

    data["Lag_Return_5"] = (
        data["Daily_Return"].shift(5)
    )

    # Day of week
    data["DayOfWeek"] = data.index.dayofweek

    # ------------------------------------------------------------------------
    # TARGETS
    # ------------------------------------------------------------------------

    # Tomorrow's return
    data["Target_Return"] = (
        data["Daily_Return"].shift(-1)
    )

    # Tomorrow's closing price
    data["Target_Next_Close"] = (
        data["Close"].shift(-1)
    )

    # ------------------------------------------------------------------------
    # FEATURE COLUMNS
    # ------------------------------------------------------------------------

    feature_cols = [
        "Daily_Return",
        "Log_Return",

        "SMA_5",
        "SMA_10",
        "SMA_20",
        "SMA_50",

        "EMA_12",
        "EMA_26",

        "MACD",
        "MACD_Signal",
        "MACD_Hist",

        "RSI_14",

        "BB_PctB",
        "BB_Std",

        "Volatility_10",
        "Volatility_20",

        "HL_Ratio",
        "CO_Ratio",

        "Price_to_SMA20",
        "Price_to_SMA50",

        "Vol_Ratio",

        "Lag_Return_1",
        "Lag_Return_2",
        "Lag_Return_3",
        "Lag_Return_5",

        "DayOfWeek",
    ]

    return data, feature_cols


# ============================================================================
# SAFE MAPE
# ============================================================================

def calculate_mape(actual, predicted):
    """
    Calculate MAPE while avoiding division by zero.
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    denominator = np.where(
        np.abs(actual) < 1e-9,
        1e-9,
        np.abs(actual)
    )

    return np.mean(
        np.abs(actual - predicted) / denominator
    ) * 100


# ============================================================================
# TRAIN AND EVALUATE
# ============================================================================

def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list,
    test_size: float = 0.20
):
    """
    Train ML models on next-day returns.

    Uses chronological train/test split to avoid look-ahead
    caused by random shuffling.
    """

    # Remove infinite values first.
    clean_df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=feature_cols + [
            "Target_Return",
            "Target_Next_Close"
        ])
        .copy()
    )

    if len(clean_df) < 100:
        raise ValueError(
            "Not enough clean observations for ML training. "
            "Try using a longer historical period such as 2y or 5y."
        )

    # IMPORTANT:
    # Take latest row AFTER cleaning so the model never receives NaN.
    latest_row = clean_df.iloc[[-1]].copy()

    X = clean_df[feature_cols]
    y_return = clean_df["Target_Return"]

    current_close = clean_df["Close"]
    actual_next_close = clean_df["Target_Next_Close"]

    # ------------------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------------------

    split_idx = int(
        len(clean_df) * (1 - test_size)
    )

    if split_idx <= 0 or split_idx >= len(clean_df):
        raise ValueError("Invalid train/test split.")

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y_return.iloc[:split_idx]
    y_test = y_return.iloc[split_idx:]

    test_dates = clean_df.index[split_idx:]

    test_current = (
        current_close
        .iloc[split_idx:]
        .values
    )

    test_actual = (
        actual_next_close
        .iloc[split_idx:]
        .values
    )

    # ------------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------------

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            random_state=42
        ),

        "Ridge Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "regressor",
                Ridge(alpha=10.0)
            )
        ])
    }

    print(
        f"[*] Training models on {len(X_train)} historical days, "
        f"testing on {len(X_test)} days..."
    )

    results = {}

    # Actual market direction.
    actual_dir = np.sign(
        test_actual - test_current
    )

    # ------------------------------------------------------------------------
    # Train individual models
    # ------------------------------------------------------------------------

    for name, model in models.items():

        print(f"    [*] Training {name}...")

        model.fit(
            X_train,
            y_train
        )

        # Predict next-day return.
        pred_return = model.predict(X_test)

        # Convert predicted return to predicted price.
        pred_close = (
            test_current *
            (1.0 + pred_return)
        )

        # --------------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------------

        rmse = np.sqrt(
            mean_squared_error(
                test_actual,
                pred_close
            )
        )

        mae = mean_absolute_error(
            test_actual,
            pred_close
        )

        mape = calculate_mape(
            test_actual,
            pred_close
        )

        r2 = r2_score(
            test_actual,
            pred_close
        )

        pred_dir = np.sign(
            pred_close - test_current
        )

        dir_acc = (
            np.mean(
                actual_dir == pred_dir
            ) * 100.0
        )

        # --------------------------------------------------------------------
        # Latest prediction
        # --------------------------------------------------------------------

        latest_features = latest_row[
            feature_cols
        ]

        latest_price = float(
            latest_row["Close"].iloc[0]
        )

        next_ret = float(
            model.predict(
                latest_features
            )[0]
        )

        next_price = (
            latest_price *
            (1.0 + next_ret)
        )

        results[name] = {
            "model": model,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "directional_accuracy": dir_acc,
            "test_dates": test_dates,
            "actual_close": test_actual,
            "predicted_close": pred_close,
            "latest_price": latest_price,
            "next_price": next_price,
            "next_return_pct": next_ret * 100.0,
        }

    # =========================================================================
    # ENSEMBLE
    # =========================================================================

    model_names = list(models.keys())

    ensemble_predictions = np.mean(
        [
            results[name]["predicted_close"]
            for name in model_names
        ],
        axis=0
    )

    ens_rmse = np.sqrt(
        mean_squared_error(
            test_actual,
            ensemble_predictions
        )
    )

    ens_mae = mean_absolute_error(
        test_actual,
        ensemble_predictions
    )

    ens_mape = calculate_mape(
        test_actual,
        ensemble_predictions
    )

    ens_r2 = r2_score(
        test_actual,
        ensemble_predictions
    )

    ens_dir = np.sign(
        ensemble_predictions - test_current
    )

    ens_dir_acc = (
        np.mean(
            actual_dir == ens_dir
        ) * 100.0
    )

    latest_price = float(
        latest_row["Close"].iloc[0]
    )

    ensemble_next_price = np.mean(
        [
            results[name]["next_price"]
            for name in model_names
        ]
    )

    ensemble_next_return = (
        (
            ensemble_next_price -
            latest_price
        )
        /
        latest_price
    ) * 100.0

    results["Ensemble Model"] = {
        "model": None,
        "rmse": ens_rmse,
        "mae": ens_mae,
        "mape": ens_mape,
        "r2": ens_r2,
        "directional_accuracy": ens_dir_acc,
        "test_dates": test_dates,
        "actual_close": test_actual,
        "predicted_close": ensemble_predictions,
        "latest_price": latest_price,
        "next_price": ensemble_next_price,
        "next_return_pct": ensemble_next_return,
    }

    return results, latest_row


# ============================================================================
# PRINT SUMMARY
# ============================================================================

def print_summary_table(
    ticker: str,
    company: str,
    curr: str,
    results: dict,
    latest_row: pd.DataFrame
):
    """
    Print model performance and next-day prediction.
    """

    # Do not let ensemble compete with individual models
    # when identifying the best individual ML model.
    individual_models = [
        "Random Forest",
        "Gradient Boosting",
        "Ridge Regression"
    ]

    best_name = min(
        individual_models,
        key=lambda k: results[k]["rmse"]
    )

    latest_date = latest_row.index[0]

    # Remove timezone if present for simple formatting.
    if hasattr(latest_date, "strftime"):
        latest_date_text = latest_date.strftime(
            "%Y-%m-%d"
        )
    else:
        latest_date_text = str(latest_date)

    latest_price = results[
        best_name
    ]["latest_price"]

    print("\n" + "=" * 110)
    print(
        f" ML PREDICTION REPORT: {company} ({ticker})"
    )
    print("=" * 110)

    print(
        f" Latest Market Close ({latest_date_text}): "
        f"{curr}{latest_price:,.2f}"
    )

    print("-" * 110)

    header = (
        f"{'Model Name':<22} | "
        f"{'RMSE':<10} | "
        f"{'MAE':<10} | "
        f"{'MAPE (%)':<10} | "
        f"{'Dir. Acc (%)':<14} | "
        f"{'Next-Day Pred':<15} | "
        f"{'Expected %':<12}"
    )

    print(header)
    print("-" * 110)

    for name, res in results.items():

        is_best = (
            " *"
            if name == best_name
            else ""
        )

        ret_pct = res[
            "next_return_pct"
        ]

        sign = (
            "+"
            if ret_pct >= 0
            else ""
        )

        row = (
            f"{name + is_best:<22} | "
            f"{res['rmse']:<10.2f} | "
            f"{res['mae']:<10.2f} | "
            f"{res['mape']:<9.2f}% | "
            f"{res['directional_accuracy']:<13.1f}% | "
            f"{curr}{res['next_price']:<14.2f} | "
            f"{sign}{ret_pct:<10.2f}%"
        )

        print(row)

    print("-" * 110)

    print(
        " * Best individual model based on lowest "
        "RMSE on the chronological holdout test set."
    )

    # ------------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------------

    best_res = results[best_name]

    sentiment = (
        "BULLISH (+)"
        if best_res["next_return_pct"] >= 0
        else "BEARISH (-)"
    )

    print("\n" + "#" * 110)

    print(
        f" NEXT TRADING DAY FORECAST ({best_name})"
    )

    print(
        f" Projected Price: "
        f"{curr}{best_res['next_price']:.2f}"
    )

    print(
        " Expected Move:   "
        f"{'+' if best_res['next_return_pct'] >= 0 else ''}"
        f"{best_res['next_return_pct']:.2f}% "
        f"({sentiment})"
    )

    print(
        " Holdout Accuracy: "
        f"Directional "
        f"{best_res['directional_accuracy']:.1f}% | "
        f"MAPE "
        f"{best_res['mape']:.2f}%"
    )

    print("#" * 110)


# ============================================================================
# PLOT RESULTS
# ============================================================================

def plot_predictions(
    ticker: str,
    company: str,
    curr: str,
    results: dict,
    latest_row: pd.DataFrame,
    feature_cols: list
):
    """
    Plot:
        1. Actual vs predicted prices
        2. Residual distribution
        3. Random Forest feature importance
    """

    individual_models = [
        "Random Forest",
        "Gradient Boosting",
        "Ridge Regression"
    ]

    best_name = min(
        individual_models,
        key=lambda k: results[k]["rmse"]
    )

    best_res = results[best_name]

    # ------------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------------

    fig = plt.figure(
        figsize=(14, 9)
    )

    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.8, 1.2]
    )

    # =========================================================================
    # 1. ACTUAL VS PREDICTED
    # =========================================================================

    ax1 = fig.add_subplot(
        gs[0, :]
    )

    dates = best_res[
        "test_dates"
    ]

    ax1.plot(
        dates,
        best_res["actual_close"],
        label="Actual Market Close",
        linewidth=2.5,
        alpha=0.9
    )

    line_styles = {
        "Random Forest": "--",
        "Gradient Boosting": "--",
        "Ridge Regression": "--",
        "Ensemble Model": "-."
    }

    for model_name, res in results.items():

        style = (
            "-"
            if model_name == best_name
            else line_styles.get(
                model_name,
                "--"
            )
        )

        width = (
            2.0
            if model_name == best_name
            else 1.2
        )

        label = (
            f"{model_name} "
            f"(RMSE: {res['rmse']:.2f})"
            if model_name == best_name
            else model_name
        )

        ax1.plot(
            dates,
            res["predicted_close"],
            linestyle=style,
            linewidth=width,
            alpha=0.8,
            label=label
        )

    # ------------------------------------------------------------------------
    # Next-day forecast marker
    # ------------------------------------------------------------------------

    latest_date = latest_row.index[0]

    # Approximate next calendar trading day.
    # This is only for visualization.
    if latest_date.weekday() == 4:
        next_date = latest_date + timedelta(days=3)
    elif latest_date.weekday() == 5:
        next_date = latest_date + timedelta(days=2)
    else:
        next_date = latest_date + timedelta(days=1)

    ax1.scatter(
        [next_date],
        [best_res["next_price"]],
        s=130,
        zorder=5,
        edgecolors="black",
        label=(
            f"Next Day Target: "
            f"{curr}{best_res['next_price']:.2f}"
        )
    )

    ax1.set_title(
        f"{company} ({ticker}) - "
        f"Actual vs Machine Learning Forecast",
        fontsize=13,
        fontweight="bold",
        pad=10
    )

    ax1.set_ylabel(
        f"Price ({curr})",
        fontsize=11
    )

    ax1.xaxis.set_major_formatter(
        mdates.DateFormatter("%b %Y")
    )

    ax1.grid(
        True,
        linestyle=":",
        alpha=0.6
    )

    ax1.legend(
        loc="upper left"
    )

    # =========================================================================
    # 2. RESIDUAL ERROR DISTRIBUTION
    # =========================================================================

    ax2 = fig.add_subplot(
        gs[1, 0]
    )

    residuals = (
        best_res["actual_close"] -
        best_res["predicted_close"]
    )

    ax2.hist(
        residuals,
        bins=25,
        edgecolor="black",
        alpha=0.7
    )

    ax2.axvline(
        0,
        linestyle="--"
    )

    ax2.set_title(
        f"Prediction Error Distribution ({best_name})",
        fontsize=11,
        fontweight="bold"
    )

    ax2.set_xlabel(
        f"Error ({curr})",
        fontsize=10
    )

    ax2.set_ylabel(
        "Frequency",
        fontsize=10
    )

    ax2.grid(
        True,
        linestyle=":",
        alpha=0.5
    )

    # =========================================================================
    # 3. FEATURE IMPORTANCE
    # =========================================================================

    ax3 = fig.add_subplot(
        gs[1, 1]
    )

    rf_model = results.get(
        "Random Forest",
        {}
    ).get("model")

    if (
        rf_model is not None
        and hasattr(
            rf_model,
            "feature_importances_"
        )
    ):

        top_feat = (
            pd.Series(
                rf_model.feature_importances_,
                index=feature_cols
            )
            .sort_values(
                ascending=False
            )
            .head(10)
            [::-1]
        )

        ax3.barh(
            np.arange(len(top_feat)),
            top_feat.values,
            alpha=0.8,
            edgecolor="black"
        )

        ax3.set_yticks(
            np.arange(len(top_feat))
        )

        ax3.set_yticklabels(
            top_feat.index,
            fontsize=9
        )

        ax3.set_title(
            "Top 10 Technical Indicators "
            "(Random Forest)",
            fontsize=11,
            fontweight="bold"
        )

        ax3.set_xlabel(
            "Relative Importance",
            fontsize=10
        )

        ax3.grid(
            True,
            linestyle=":",
            alpha=0.5,
            axis="x"
        )

    else:
        ax3.text(
            0.5,
            0.5,
            "Feature importance unavailable",
            ha="center",
            va="center"
        )

        ax3.set_axis_off()

    # ------------------------------------------------------------------------
    # Save chart
    # ------------------------------------------------------------------------

    plt.tight_layout()

    safe_ticker = (
        ticker
        .replace("^", "")
        .replace(".", "_")
        .replace("/", "_")
    )

    chart_file = (
        f"{safe_ticker}_prediction_chart.png"
    )

    plt.savefig(
        chart_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"\n[OK] Chart saved to disk: "
        f"{os.path.abspath(chart_file)}"
    )

    # Display if GUI is available.
    try:
        plt.show()
    except Exception:
        pass

    plt.close(fig)


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():

    print(
        """
============================================================
        STOCK PRICE & TREND PREDICTOR
             MACHINE LEARNING SYSTEM
============================================================
"""
    )

    print(
        "Enter company stock ticker symbol(s)."
    )

    print(
        "Examples:"
    )

    print(
        "  - US Stocks:     AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN"
    )

    print(
        "  - Indian Stocks: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS"
    )

    print(
        "  - Indices/Crypto: ^GSPC (S&P 500), BTC-USD, ETH-USD\n"
    )

    # ------------------------------------------------------------------------
    # Ticker input
    # ------------------------------------------------------------------------

    user_input = input(
        "Enter Stock Ticker(s) [e.g. AAPL, NVDA]: "
    ).strip()

    if not user_input:

        user_input = "AAPL"

        print(
            "No ticker entered. "
            "Defaulting to 'AAPL'."
        )

    tickers = [
        ticker.strip().upper()
        for ticker in user_input.split(",")
        if ticker.strip()
    ]

    if not tickers:
        print(
            "[!] No valid ticker symbols entered."
        )
        return

    # ------------------------------------------------------------------------
    # Historical period
    # ------------------------------------------------------------------------

    period = input(
        "Enter historical timeframe "
        "(6mo / 1y / 2y / 5y / max) "
        "[Default: 2y]: "
    ).strip().lower()

    valid_periods = [
        "6mo",
        "1y",
        "2y",
        "5y",
        "max"
    ]

    if period not in valid_periods:

        period = "2y"

        print(
            "Invalid timeframe. "
            "Defaulting to '2y'."
        )

    # ------------------------------------------------------------------------
    # Process tickers
    # ------------------------------------------------------------------------

    for idx, ticker in enumerate(
        tickers,
        start=1
    ):

        print(
            f"\n{'=' * 35} "
            f"Company {idx}/{len(tickers)}: {ticker} "
            f"{'=' * 35}"
        )

        try:

            # Fetch data.
            df, company, curr = fetch_stock_data(
                ticker,
                period=period
            )

            # Engineer technical indicators.
            featured_df, feat_cols = engineer_features(
                df
            )

            # Train and evaluate.
            results, latest_row = train_and_evaluate(
                featured_df,
                feat_cols,
                test_size=0.20
            )

            # Print report.
            print_summary_table(
                ticker,
                company,
                curr,
                results,
                latest_row
            )

            # Plot results.
            plot_predictions(
                ticker,
                company,
                curr,
                results,
                latest_row,
                feat_cols
            )

        except KeyboardInterrupt:

            print(
                "\n[!] Program interrupted by user."
            )
            return

        except Exception as e:

            print(
                f"[!] Error processing '{ticker}': {e}",
                file=sys.stderr
            )


# ============================================================================
# PROGRAM ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()