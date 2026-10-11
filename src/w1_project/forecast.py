"""Monte Carlo price scenarios built by resampling historical daily returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

QUANTILES = {"p5": 0.05, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p95": 0.95}


@dataclass
class Forecast:
    portfolio: pd.DataFrame  # future dates x quantile columns, in $
    stocks: dict[str, pd.DataFrame]  # ticker -> future dates x quantile columns, in $ per share
    summary: pd.DataFrame  # one row per ticker + a PORTFOLIO row


def _bands(paths: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """paths: (sims, horizon) -> quantile bands per future date."""
    return pd.DataFrame(
        {name: np.quantile(paths, q, axis=0) for name, q in QUANTILES.items()}, index=dates
    )


def _summary_row(name: str, start: float, final: np.ndarray) -> dict:
    return {
        "ticker": name,
        "today": start,
        "median": np.median(final),
        "low_p5": np.quantile(final, 0.05),
        "high_p95": np.quantile(final, 0.95),
        "expected_change": np.median(final) / start - 1,
        "prob_loss": (final < start).mean(),
    }


def simulate(
    window: pd.DataFrame,
    shares: pd.Series,
    horizon: int = 63,
    n_sims: int = 2000,
    seed: int = 7,
) -> Forecast:
    """Simulate `horizon` trading days ahead for each ticker in `shares`.

    Whole historical days are resampled (bootstrap), so the stocks keep their
    real co-movement: a bad day for the market is a bad day for most holdings.
    The portfolio is buy-and-hold: today's share counts, no rebalancing.
    """
    tickers = list(shares.index)
    returns = window[tickers].pct_change().dropna()
    if len(returns) < 20:
        raise ValueError("Need at least 20 trading days of history to simulate.")

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(returns), size=(n_sims, horizon))
    growth = np.cumprod(1 + returns.to_numpy()[picks], axis=1)  # (sims, horizon, stocks)

    last = window[tickers].iloc[-1].to_numpy()
    dates = pd.bdate_range(window.index[-1] + pd.offsets.BDay(1), periods=horizon)

    stock_paths = growth * last  # price per share
    port_paths = stock_paths @ shares.to_numpy(dtype=float)  # $ value

    rows = [_summary_row(t, last[i], stock_paths[:, -1, i]) for i, t in enumerate(tickers)]
    rows.append(_summary_row("PORTFOLIO", float(last @ shares.to_numpy(dtype=float)), port_paths[:, -1]))
    return Forecast(
        portfolio=_bands(port_paths, dates),
        stocks={t: _bands(stock_paths[:, :, i], dates) for i, t in enumerate(tickers)},
        summary=pd.DataFrame(rows),
    )
