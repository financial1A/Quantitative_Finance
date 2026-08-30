# Quantitative Finance

A collection of tools, models, and research notebooks for quantitative analysis of financial markets — covering trading signal generation, risk management, portfolio optimization, and market data pipelines.

## Overview

This repository serves as a working lab for quantitative finance experiments, including:

- **Signal research** — indicators and statistical models for identifying trading opportunities
- **Risk management** — position sizing, volatility modeling, and drawdown analysis
- **Portfolio optimization** — allocation strategies and backtesting frameworks
- **Data pipelines** — ingestion and processing of market data (equities, crypto, FX)
- **Dashboards** — Streamlit-based tools for visualizing signals and performance

## Tech Stack

- **Language:** Python
- **Core libraries:** pandas, numpy, scipy
- **Visualization:** matplotlib / plotly / Streamlit
- **Backtesting:** (add framework, e.g. backtrader, vectorbt, or custom)
- **Data sources:** (add exchanges / APIs, e.g. IBKR, Binance, yfinance)

## Project Structure

```
quant-finance/
├── data/                # Raw and processed market data
├── notebooks/           # Exploratory research (Jupyter)
├── src/
│   ├── signals/         # Signal generation logic
│   ├── risk/            # Risk models and position sizing
│   ├── backtest/        # Backtesting engine and utilities
│   └── pipelines/       # Data ingestion and processing
├── dashboards/          # Streamlit apps
├── tests/               # Unit tests
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/financial1A/Quantitative_Finance.git
cd quant-finance
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Run a Streamlit dashboard
streamlit run dashboards/main.py

# Run backtests
python -m src.backtest.run --strategy <strategy_name>
```

## Roadmap

- [ ] Add data ingestion pipeline for chosen broker/exchange
- [ ] Implement core signal library
- [ ] Build backtesting engine
- [ ] Add risk management module
- [ ] Deploy live dashboard

## Disclaimer

This project is for research and educational purposes only. Nothing in this repository constitutes financial advice. Trading involves risk, and past performance is not indicative of future results.

## License

(Add your preferred license, e.g. MIT)
