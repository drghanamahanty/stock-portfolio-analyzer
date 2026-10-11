"""Generate a synthetic (but realistic-looking) stock portfolio dataset.

Writes two CSVs into data/:
  - sample_prices.csv   : daily closing prices (date, ticker, close), incl. SPY benchmark
  - sample_holdings.csv : the portfolio (ticker, company, sector, shares, avg_cost, purchase_date)

Prices are simulated with a one-factor model (market + stock-specific noise),
so stocks are correlated the way real ones are. Seeded for reproducibility.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START, END = "2024-10-01", "2026-10-09"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ticker: (company, sector, start_price, annual_drift, annual_idio_vol, market_beta)
STOCKS = {
    "AAPL": ("Apple Inc.", "Technology", 226.0, 0.14, 0.18, 1.1),
    "MSFT": ("Microsoft Corp.", "Technology", 420.0, 0.13, 0.16, 1.0),
    "NVDA": ("NVIDIA Corp.", "Technology", 118.0, 0.35, 0.38, 1.7),
    "GOOGL": ("Alphabet Inc.", "Communication Services", 165.0, 0.12, 0.20, 1.1),
    "AMZN": ("Amazon.com Inc.", "Consumer Discretionary", 185.0, 0.15, 0.22, 1.2),
    "JPM": ("JPMorgan Chase & Co.", "Financials", 210.0, 0.10, 0.15, 1.0),
    "V": ("Visa Inc.", "Financials", 275.0, 0.09, 0.13, 0.9),
    "JNJ": ("Johnson & Johnson", "Healthcare", 162.0, 0.03, 0.12, 0.5),
    "XOM": ("Exxon Mobil Corp.", "Energy", 117.0, -0.02, 0.20, 0.7),
    "PG": ("Procter & Gamble Co.", "Consumer Staples", 172.0, 0.05, 0.11, 0.4),
}
BENCHMARK = ("SPY", 570.0)

# ticker: (shares, purchase_date)
HOLDINGS = {
    "AAPL": (60, "2024-10-01"),
    "MSFT": (25, "2024-10-01"),
    "NVDA": (90, "2024-11-15"),
    "GOOGL": (50, "2025-01-10"),
    "AMZN": (45, "2024-10-01"),
    "JPM": (40, "2025-03-03"),
    "V": (30, "2025-02-14"),
    "JNJ": (35, "2024-12-02"),
    "XOM": (55, "2025-04-07"),
    "PG": (40, "2025-06-02"),
}


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(START, END)
    n = len(dates)
    dt = 1 / 252

    market = rng.normal(0.09 * dt, 0.15 * np.sqrt(dt), n)
    market[0] = 0.0

    rows = []
    bench_ticker, bench_start = BENCHMARK
    bench = bench_start * np.exp(np.cumsum(market))
    rows += [{"date": d, "ticker": bench_ticker, "close": p} for d, p in zip(dates, bench)]

    prices = {}
    for ticker, (_, _, start, drift, vol, beta) in STOCKS.items():
        idio = rng.normal((drift - beta * 0.09) * dt, vol * np.sqrt(dt), n)
        idio[0] = 0.0
        path = start * np.exp(np.cumsum(beta * market + idio))
        prices[ticker] = pd.Series(path, index=dates)
        rows += [{"date": d, "ticker": ticker, "close": p} for d, p in zip(dates, path)]

    price_df = pd.DataFrame(rows)
    price_df["date"] = price_df["date"].dt.strftime("%Y-%m-%d")
    price_df["close"] = price_df["close"].round(2)

    holdings = []
    for ticker, (shares, bought) in HOLDINGS.items():
        company, sector, *_ = STOCKS[ticker]
        fill = prices[ticker].loc[:bought].iloc[-1] * rng.uniform(0.99, 1.01)  # small slippage
        holdings.append(
            {
                "ticker": ticker,
                "company": company,
                "sector": sector,
                "shares": shares,
                "avg_cost": round(fill, 2),
                "purchase_date": bought,
            }
        )

    DATA_DIR.mkdir(exist_ok=True)
    price_df.to_csv(DATA_DIR / "sample_prices.csv", index=False)
    pd.DataFrame(holdings).to_csv(DATA_DIR / "sample_holdings.csv", index=False)
    print(f"Wrote {len(price_df):,} price rows and {len(holdings)} holdings to {DATA_DIR}")


if __name__ == "__main__":
    main()
