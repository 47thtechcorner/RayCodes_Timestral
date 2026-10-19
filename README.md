<div align="center">
  <a href="https://youtu.be/aaKlUfosjwM">
    <img src="https://img.youtube.com/vi/aaKlUfosjwM/0.jpg" alt="Mistral Timestral: Tiny 209M AI Predicts Stocks Locally on CPU!">
  </a>
  <h3>📺 <a href="https://youtu.be/aaKlUfosjwM">Watch the full tutorial on YouTube: Mistral Timestral: Tiny 209M AI Predicts Stocks Locally on CPU!</a></h3>
</div>

# ⏱️ Mistral Timestral: Zero-Shot Time Series Foundation Models

> **Mistral AI's official family of time-series foundation models (5M to 209M parameters) delivering state-of-the-art probabilistic forecasting on ordinary CPUs without needing a GPU.**

---

## ⚡ 3-Step Visual Workflow

| Step 1: Input Telemetry | ➔ | Step 2: Zero-Shot Attention | ➔ | Step 3: Quantile Deliverables |
|:---|:---:|:---|:---:|:---|
| Ingest past series & future calendar covariates up to 32,768 steps | ➔ | Decoupled horizon tokens process sequence on standard CPU in 10-140ms | ➔ | Receive 21 probabilistic quantiles (P01 to P99) with risk thresholds |

---

## 🚀 Quickstart: Single-Command Setup

Install the official `timestral` library from Mistral AI's GitHub repository along with dependencies, and execute the production pipeline:

```powershell
pip install git+https://github.com/mistralai/timestral.git yfinance matplotlib torch; python main.py
```

Or install in your active virtual environment:

```powershell
pip install git+https://github.com/mistralai/timestral.git
pip install yfinance matplotlib torch
python main.py
```

---

## 📊 Benchmark & Accuracy Showdown

Timestral-209M ranks **#1 globally** across all three major public time-series benchmarks, outperforming models more than 10x its size:

| Model | Parameters | GIFT-Eval MASE (↓) | TIME MASE (↓) | fev-bench SQL (↑) | Inference Device |
|---|---|---|---|---|---|
| **Timestral-209M** | **209M** | **0.658** | **0.626** | **50.39** | **Standard CPU (140ms)** |
| t0-gamma | ~250M | 0.661 | 0.637 | 49.42 | GPU Recommended |
| TimesFM-3 (Google) | 330M | 0.667 | 0.640 | 48.99 | GPU Recommended |
| Toto-2.0-2.5B (Databricks) | 2,500M | 0.696 | 0.642 | 44.42 | Heavy GPU Required |
| Chronos-2 (Amazon) | 120M | 0.698 | 0.662 | 47.28 | GPU Standard |

---

## 🛠️ 3 Practical Projects Included

1. **Stock Price Trend & Risk Tunnel (`outputs/project_1_stock_trend_forecaster.png`):**
   - Automatically loads historical price data (e.g. NVDA) via `yfinance`.
   - Forecasts the upcoming 21 trading days with macro momentum curve and dual-layer P10-P90 uncertainty tunnels.

2. **Store Sales & Stock Outage Planner (`outputs/project_2_store_sales_planner.png`):**
   - Forecasts daily retail item demand conditioned on upcoming discount promotions using `future_covariates`.
   - Generates confidence bands to guarantee shelves never run out of inventory.

3. **Business Cash Flow & Safety Balance (`outputs/project_3_cash_flow_safety.png`):**
   - Projects weekly business cash balances and incoming revenue trends with quarterly billing oscillations.
   - Highlights worst-case P10 cash dips against critical emergency cash reserve lines.

---

## 🌟 5 Real-World Use Cases

- **Smart IoT & Edge Telemetry:** Run the 5M model on low-power devices for instant anomaly alerts in 10ms.
- **Renewable Energy Grid Balancing:** Predict power demand using known future weather forecasts as covariates.
- **Supply Chain Lead Time Planning:** Forecast warehouse inventory bottlenecks weeks in advance.
- **Financial Treasury Reserves:** Project cash inflows and outflows across multiple bank accounts.
- **SRE & Cloud Cost Optimization:** Auto-scale cloud server instances based on predicted usage curves.

---

## 🔮 5 Future Roadmap Features

- **Automated LLM Incident Reports:** Pairing numerical forecasts with Mistral Large for written executive summaries.
- **Streaming Sliding-Window Mode:** Real-time socket ingestion for sub-second live sensor feeds.
- **ONNX & OpenVINO Quantization:** Running Timestral on microcontrollers and embedded NPU chips.
- **Cross-Correlated Market Pairs:** Joint multivariate spread forecasting for algorithmic market making.
- **Zero-Code Dashboard UI:** Web-based interactive drag-and-drop CSV forecaster with interactive charts.

---

## 📜 License & Acknowledgments

Built upon Mistral AI's official open-weight release. Released under Modified MIT license. Official repository: [github.com/mistralai/timestral](https://github.com/mistralai/timestral).

---

## 🏷️ Keywords & Search Tags

`mistral timestral`, `timestral 209m`, `timestral 5m`, `timestral 28m`, `timestral 123m`, `mistral ai`, `timesfm 3.0`, `amazon chronos`, `gift eval`, `time benchmark`, `fev bench`, `zero shot time series`, `forecasting foundation model`, `cpu ai forecasting`, `no gpu ai`, `stock prediction ai`, `demand forecasting`, `multivariate forecasting`, `local ai`, `python time series`, `yfinance python`
