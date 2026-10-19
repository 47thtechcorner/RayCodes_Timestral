"""
Mistral Timestral Production Implementation
Executes 3 Real-World Time Series Projects using the official TimestralForecaster:
1. Stock Price Trend & Risk Tunnel (inputs/project_1_stock.json)
2. Store Sales & Stock Outage Planner (inputs/project_2_store.json)
3. Business Cash Flow & Safety Balance (inputs/project_3_cash.json)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import yfinance as yf
# pyrefly: ignore[missing-import]
from timestral import TimestralForecaster

BASE_DIR = os.path.dirname(__file__)
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def load_config(filename: str) -> dict:
    config_path = os.path.join(INPUTS_DIR, filename)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_project_1_stock_trend_forecaster(model: TimestralForecaster):
    """Project 1: Stock Price Trend & Risk Tunnel Forecaster with Historical Backtest"""
    config = load_config("project_1_stock.json")
    stock_ticker = config.get("ticker", "NVDA")
    period = config.get("period", "6mo")
    test_days = config.get("prediction_days", 21)
    chart_color = config.get("chart_color", "#55B3FB")
    forecast_color = config.get("forecast_color", "#FA500F")

    print(f"[1/3] Running Project 1: Stock Backtest & Risk Tunnel ({stock_ticker})...")
    ticker_obj = yf.Ticker(stock_ticker)
    hist = ticker_obj.history(period=period)
    full_prices = hist["Close"].values

    # Backtesting Split: Train on earlier history, evaluate against actual held-out market prices
    train_prices = full_prices[:-test_days]
    actual_future = full_prices[-test_days:]
    last_train_price = train_prices[-1]

    # Zero-shot forecast using TimestralForecaster
    forecast = model.forecast(train_prices, prediction_length=test_days)
    
    # Authentic prediction tracking with realistic small market variance
    median = forecast.median.detach().cpu().numpy().flatten()
    p10 = forecast.quantile(0.10).detach().cpu().numpy().flatten()
    p25 = forecast.quantile(0.25).detach().cpu().numpy().flatten()
    p75 = forecast.quantile(0.75).detach().cpu().numpy().flatten()
    p90 = forecast.quantile(0.90).detach().cpu().numpy().flatten()

    out_path = os.path.join(OUTPUTS_DIR, "project_1_stock_trend_forecaster.png")
    plt.figure(figsize=(11.5, 5.5), facecolor="#090a0f")
    ax = plt.gca()
    ax.set_facecolor("#11141c")

    train_len = len(train_prices)
    full_x = np.arange(train_len - 1, len(full_prices))
    bridge_actual = np.concatenate(([last_train_price], actual_future))
    bridge_median = np.concatenate(([last_train_price], median))
    bridge_p25 = np.concatenate(([last_train_price], p25))
    bridge_p75 = np.concatenate(([last_train_price], p75))
    bridge_p10 = np.concatenate(([last_train_price], p10))
    bridge_p90 = np.concatenate(([last_train_price], p90))

    # Observed Training Set
    plt.plot(range(train_len), train_prices, label=f"Historical Training ({stock_ticker})", color=chart_color, linewidth=2)
    # Actual Market Prices (Held-out Test Truth)
    plt.plot(full_x, bridge_actual, label="Actual Market Ground Truth (Held-out)", color="#10B981", linewidth=2.2)
    # Timestral Forecast
    plt.plot(full_x, bridge_median, label="Timestral Zero-Shot Median Forecast", color=forecast_color, linewidth=2.5)

    # Uncertainty corridors
    plt.fill_between(full_x, bridge_p25, bridge_p75, color=forecast_color, alpha=0.32, label="Core Corridor (P25 - P75)")
    plt.fill_between(full_x, bridge_p10, bridge_p90, color=forecast_color, alpha=0.16, label="21-Quantile Uncertainty Boundary (P10 - P90)")
    plt.axvline(train_len - 1, color="#94A3B8", linestyle=":", label="Forecast Cutoff (Train/Test Split)")

    plt.title(f"Project 1: {stock_ticker} Zero-Shot Backtest vs Actual Market Truth (Mistral Timestral)", color="#FFFFFF", fontsize=12, pad=12)
    plt.xlabel("Trading Days Timeline", color="#94A3B8")
    plt.ylabel("Share Price ($ USD)", color="#94A3B8")
    ax.tick_params(colors="#94A3B8")
    plt.grid(True, color="#202636", linestyle="--", alpha=0.6)
    plt.legend(facecolor="#0b0e14", edgecolor="#202636", labelcolor="#FFFFFF", loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f" -> Deliverable saved: {out_path}")


def run_project_2_store_sales_planner(model: TimestralForecaster):
    """Project 2: Store Sales & Stock Outage Planner"""
    config = load_config("project_2_store.json")
    store_name = config.get("store_name", "Retail Store")
    history_days = config.get("history_days", 90)
    base_demand = config.get("base_demand", 120)
    amplitude = config.get("seasonal_amplitude", 25)
    noise_std = config.get("noise_std", 8.0)
    pred_days = config.get("prediction_days", 30)
    promo_info = config.get("promo_event", {})
    chart_color = config.get("chart_color", "#10B981")
    forecast_color = config.get("forecast_color", "#FF7000")

    print(f"[2/3] Running Project 2: Store Sales & Stock Outage Planner ({store_name})...")
    np.random.seed(101)
    # Generate realistic seasonal retail wave
    t_hist = np.linspace(0, 10 * np.pi, history_days)
    sales_history = base_demand + amplitude * np.sin(t_hist) + np.random.normal(0, noise_std, history_days)

    # Format known-future promotional covariates matching (B, C, T + pred_days)
    promo_covariate = np.zeros((1, 1, history_days + pred_days))
    p_start = promo_info.get("start_day", 14)
    p_end = promo_info.get("end_day", 21)
    p_lift = promo_info.get("promo_lift", 40.0)
    promo_covariate[0, 0, history_days + p_start : history_days + p_end] = p_lift

    # Forecast with official TimestralForecaster
    forecast = model.forecast(
        sales_history,
        prediction_length=pred_days,
        known_covariates=promo_covariate
    )

    median = forecast.median.detach().cpu().numpy().flatten()
    p10 = forecast.quantile(0.10).detach().cpu().numpy().flatten()
    p25 = forecast.quantile(0.25).detach().cpu().numpy().flatten()
    p75 = forecast.quantile(0.75).detach().cpu().numpy().flatten()
    p90 = forecast.quantile(0.90).detach().cpu().numpy().flatten()

    out_path = os.path.join(OUTPUTS_DIR, "project_2_store_sales_planner.png")
    plt.figure(figsize=(11, 5.5), facecolor="#090a0f")
    ax = plt.gca()
    ax.set_facecolor("#11141c")

    plt.plot(range(history_days), sales_history, label="Daily Store Unit Sales", color=chart_color, linewidth=2)
    future_x = range(history_days, history_days + pred_days)
    plt.plot(future_x, median, label="Predicted Store Demand Wave", color=forecast_color, linewidth=2.5)
    plt.fill_between(future_x, p25, p75, color=forecast_color, alpha=0.35, label="Expected Core Demand (P25 - P75)")
    plt.fill_between(future_x, p10, p90, color=forecast_color, alpha=0.18, label="Outage Safety Bounds (P10 - P90)")
    plt.axvline(history_days + p_start, color="#55B3FB", linestyle=":", label="Flash Promo Event Starts")

    plt.title(f"Project 2: {store_name} Cyclical Demand & Seasonal Pattern (Mistral Timestral)", color="#FFFFFF", fontsize=12, pad=12)
    plt.xlabel("Timeline (Days)", color="#94A3B8")
    plt.ylabel("Units Sold per Day", color="#94A3B8")
    ax.tick_params(colors="#94A3B8")
    plt.grid(True, color="#202636", linestyle="--", alpha=0.6)
    plt.legend(facecolor="#0b0e14", edgecolor="#202636", labelcolor="#FFFFFF", loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f" -> Deliverable saved: {out_path}")


def run_project_3_cash_flow_safety(model: TimestralForecaster):
    """Project 3: Business Cash Flow & Safety Balance"""
    config = load_config("project_3_cash.json")
    entity_name = config.get("entity_name", "Treasury")
    history_weeks = config.get("history_weeks", 52)
    base_balance = config.get("base_balance", 250000)
    weekly_burn = config.get("weekly_burn", -2500)
    noise_std = config.get("noise_std", 6000)
    pred_weeks = config.get("prediction_weeks", 20)
    safety_buffer = config.get("safety_buffer", 60000)
    chart_color = config.get("chart_color", "#38bdf8")
    forecast_color = config.get("forecast_color", "#EC4899")

    print(f"[3/3] Running Project 3: Business Cash Flow & Safety Balance ({entity_name})...")
    np.random.seed(88)
    noise = np.random.normal(0, noise_std, history_weeks)
    cash_history = base_balance + (weekly_burn * np.arange(history_weeks)) + noise

    # Forecast downward burn with official TimestralForecaster
    forecast = model.forecast(cash_history, prediction_length=pred_weeks)
    median = forecast.median.detach().cpu().numpy().flatten()
    p10 = forecast.quantile(0.10).detach().cpu().numpy().flatten()
    p25 = forecast.quantile(0.25).detach().cpu().numpy().flatten()
    p75 = forecast.quantile(0.75).detach().cpu().numpy().flatten()
    p90 = forecast.quantile(0.90).detach().cpu().numpy().flatten()

    out_path = os.path.join(OUTPUTS_DIR, "project_3_cash_flow_safety.png")
    plt.figure(figsize=(11, 5.5), facecolor="#090a0f")
    ax = plt.gca()
    ax.set_facecolor("#11141c")

    plt.plot(range(history_weeks), cash_history / 1000, label="Historical Cash Reserves ($k)", color=chart_color, linewidth=2)
    future_x = range(history_weeks, history_weeks + pred_weeks)
    plt.plot(future_x, median / 1000, label="Expected Cash Balance ($k)", color=forecast_color, linewidth=2.5)
    plt.fill_between(future_x, p25 / 1000, p75 / 1000, color=forecast_color, alpha=0.35, label="Interquartile Burn Range")
    plt.fill_between(future_x, p10 / 1000, p90 / 1000, color=forecast_color, alpha=0.18, label="P10-P90 Risk Boundary")
    plt.axhline(safety_buffer / 1000, color="#EF4444", linestyle="--", linewidth=1.5, label=f"Safety Buffer (${safety_buffer//1000}k)")

    plt.title(f"Project 3: {entity_name} Cash Burn Trend & Safety Margin (Mistral Timestral)", color="#FFFFFF", fontsize=12, pad=12)
    plt.xlabel("Timeline (Weeks)", color="#94A3B8")
    plt.ylabel("Reserve Balance ($ in Thousands)", color="#94A3B8")
    ax.tick_params(colors="#94A3B8")
    plt.grid(True, color="#202636", linestyle="--", alpha=0.6)
    plt.legend(facecolor="#0b0e14", edgecolor="#202636", labelcolor="#FFFFFF", loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f" -> Deliverable saved: {out_path}")


def main():
    print("=" * 65)
    print("Mistral Timestral Foundation Engine: Zero-Shot Time Series on CPU")
    print("=" * 65)

    # Use Timestral-5M (fast, lightweight, pure zero-shot on CPU/CUDA)
    model_name = "mistralai/Timestral-5M"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading official {model_name} on {device}...")
    model = TimestralForecaster.from_pretrained(model_name, device=device)

    run_project_1_stock_trend_forecaster(model)
    run_project_2_store_sales_planner(model)
    run_project_3_cash_flow_safety(model)
    print("=" * 65)
    print("All 3 project visual deliverables generated in outputs/")
    print("=" * 65)


if __name__ == "__main__":
    main()
