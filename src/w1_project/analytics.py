"""Portfolio analytics used by the Streamlit app (pure pandas, no UI code)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
PRICE_COLUMNS = {"date", "ticker", "close"}
HOLDING_COLUMNS = {"ticker", "company", "sector", "shares", "avg_cost", "purchase_date"}


def _normalise(df: pd.DataFrame, required: set[str], name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} CSV is missing column(s): {', '.join(sorted(missing))}")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df


def load_prices(source) -> pd.DataFrame:
    """Read a long-format price CSV: date, ticker, close."""
    df = _normalise(pd.read_csv(source), PRICE_COLUMNS, "Prices")
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def load_holdings(source) -> pd.DataFrame:
    """Read a holdings CSV: ticker, company, sector, shares, avg_cost, purchase_date."""
    df = _normalise(pd.read_csv(source), HOLDING_COLUMNS, "Holdings")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce")
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    return df.dropna(subset=["shares", "avg_cost", "purchase_date"]).reset_index(drop=True)


def price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide table of closes (dates x tickers), forward-filled over gaps."""
    return prices.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()


def position_values(holdings: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Daily market value of each holding; zero before its purchase date."""
    values: dict[str, pd.Series] = {}
    for h in holdings.itertuples():
        if h.ticker not in wide:
            continue
        held = wide.index >= h.purchase_date
        series = wide[h.ticker].where(held, 0.0).fillna(0.0) * h.shares
        values[h.ticker] = values.get(h.ticker, 0.0) + series
    return pd.DataFrame(values, index=wide.index)


def invested_capital(holdings: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Cumulative cost basis deployed over time."""
    total = pd.Series(0.0, index=index)
    for h in holdings.itertuples():
        total += np.where(index >= h.purchase_date, h.shares * h.avg_cost, 0.0)
    return total


def portfolio_returns(values: pd.DataFrame, wide: pd.DataFrame) -> pd.Series:
    """Daily portfolio return, excluding the effect of new money being added.

    Each day's return is the value-weighted return of the positions already held
    the day before, so buying a new stock doesn't look like a price jump.
    """
    asset_returns = wide[values.columns].pct_change()
    prev = values.shift(1)
    prev_total = prev.sum(axis=1).replace(0.0, np.nan)
    return ((prev * asset_returns).sum(axis=1) / prev_total).dropna()


def snapshot(holdings: pd.DataFrame, wide: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Per-holding position summary as of a given date."""
    history = wide.loc[:as_of]
    last = history.iloc[-1]
    prev = history.iloc[-2] if len(history) > 1 else last

    df = holdings[(holdings["purchase_date"] <= as_of) & holdings["ticker"].isin(wide.columns)].copy()
    df["price"] = df["ticker"].map(last)
    df["market_value"] = df["shares"] * df["price"]
    df["cost_basis"] = df["shares"] * df["avg_cost"]
    df["unrealized_pnl"] = df["market_value"] - df["cost_basis"]
    df["pnl_pct"] = df["unrealized_pnl"] / df["cost_basis"]
    df["day_change"] = df["shares"] * (df["price"] - df["ticker"].map(prev))
    df["day_change_pct"] = df["price"] / df["ticker"].map(prev) - 1
    total = df["market_value"].sum()
    df["weight"] = df["market_value"] / total if total else 0.0
    return df.sort_values("market_value", ascending=False).reset_index(drop=True)


def cumulative_return(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod() - 1


def drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def risk_metrics(
    returns: pd.Series, benchmark: pd.Series | None = None, risk_free: float = 0.0
) -> dict[str, float]:
    """Annualised return/volatility, Sharpe, max drawdown and (optionally) beta."""
    returns = returns.dropna()
    if len(returns) < 2:
        return {}
    total = (1 + returns).prod() - 1
    ann_return = (1 + total) ** (TRADING_DAYS / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    metrics = {
        "total_return": total,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": (ann_return - risk_free) / ann_vol if ann_vol else np.nan,
        "max_drawdown": drawdown(returns).min(),
    }
    if benchmark is not None:
        aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
        if len(aligned) > 1 and aligned.iloc[:, 1].var():
            metrics["beta"] = aligned.cov().iloc[0, 1] / aligned.iloc[:, 1].var()
    return metrics
