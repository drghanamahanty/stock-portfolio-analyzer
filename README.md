# Stock Portfolio Analyzer

A Streamlit dashboard that analyzes a stock portfolio from CSV files: what it's worth, how it has performed against a benchmark, and how risky it is. Built for **The Gen Academy, Week 1 (Path A: Data Analysis with Vibe Coding)**.

## Features

| Tab | What it shows |
|---|---|
| **Overview** | Market value, cost basis, unrealized P&L, day change, period return vs. benchmark; portfolio value vs. invested capital over time; allocation by sector and by position |
| **Holdings** | Searchable position table (price, value, P&L, weight); CSV download; P&L-by-position chart |
| **Performance** | Cumulative return vs. benchmark; each stock rebased to 100; monthly returns |
| **Risk** | Annualized volatility, Sharpe ratio, max drawdown, beta; drawdown chart; correlation heatmap; risk-vs-return scatter |
| **Forecast** | 3-month Monte Carlo scenarios for the portfolio and each stock: median, 50% and 90% bands, chance of a loss |
| **Alerts** | Trend breaks (50/200-day crosses, unusually large daily moves, drawdowns from the 1-year high) with a "Send test alert email" button |
| **Assistant** | Chatbot that answers plain-English questions ("Am I beating the market?", "Tell me about NVDA") from the dashboard's numbers. It is rule-based, with no AI model |

Sidebar filters (date range, sector, ticker, benchmark, risk-free rate, alert thresholds) apply to every tab. If a filter leaves nothing to show, the app says so instead of erroring.

## Email alerts setup (optional)

Alerts are sent through [Resend](https://resend.com), whose free tier is 3,000 emails a month.

1. Sign up at resend.com and create an API key under **API Keys**.
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in `RESEND_API_KEY` and `ALERT_EMAIL_TO`.
3. Restart the app. On the **Alerts** tab, click **Send test alert email**.

Without a verified domain, Resend only delivers to the address you signed up with. `secrets.toml` is git-ignored, so your key stays off GitHub.

## Run it

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
uv run streamlit run app.py
```

Without uv: `pip install -r requirements.txt && streamlit run app.py`.

## Data

The app loads the bundled sample data in `data/` by default. You can upload your own CSVs from the sidebar.

**Prices CSV:** one row per ticker per day. Any ticker that isn't in your holdings (e.g. `SPY`) can be picked as the benchmark.

```csv
date,ticker,close
2024-10-01,AAPL,226.00
```

**Holdings CSV**

```csv
ticker,company,sector,shares,avg_cost,purchase_date
AAPL,Apple Inc.,Technology,60,224.08,2024-10-01
```

> The sample data is **synthetic**. `scripts/generate_sample_data.py` simulates 2 years of correlated daily prices for 10 large-cap stocks plus SPY, using a fixed seed. Regenerate it with `uv run python scripts/generate_sample_data.py`. It is not real market data and is not investment advice.

## How the numbers are calculated

- **Portfolio return** is time-weighted: each day's return comes only from the positions held the day before. Buying a new stock therefore doesn't show up as a gain.
- **Annualized** figures assume 252 trading days.
- **Sharpe** = (annualized return − risk-free rate) / annualized volatility.
- **Beta** = cov(portfolio, benchmark) / var(benchmark), using daily returns.
- **Forecast**: 2,000 simulated paths of 63 trading days, each built by resampling whole historical days. Resampling whole days keeps the stocks' real co-movement. It assumes buy-and-hold with today's share counts. These are scenarios, not predictions.

## Project layout

```
app.py                         Streamlit UI
src/w1_project/analytics.py    Loading, validation and portfolio math (no UI code)
src/w1_project/forecast.py     Monte Carlo scenarios
src/w1_project/alerts.py       Trend-break detection + Resend email
src/w1_project/assistant.py    Rule-based chatbot
.streamlit/secrets.toml.example
scripts/generate_sample_data.py
data/sample_prices.csv, data/sample_holdings.csv
```
