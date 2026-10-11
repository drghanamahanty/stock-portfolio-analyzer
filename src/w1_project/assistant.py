"""Rule-based portfolio assistant: answers questions from the dashboard's own numbers.

No AI model is involved. The question is matched to a topic by keywords, and
the answer is filled in from the same figures the dashboard shows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

EXAMPLES = [
    "How is my portfolio doing?",
    "Which stock is my best performer?",
    "What is dragging my returns down?",
    "Am I too concentrated?",
    "How risky is my portfolio?",
    "Am I beating the market?",
    "What's the 3-month outlook?",
    "Any trend alerts?",
    "Tell me about NVDA",
]


@dataclass
class Context:
    snap: pd.DataFrame  # analytics.snapshot output
    metrics: dict  # portfolio risk metrics
    bench_metrics: dict
    benchmark: str | None
    forecast: pd.DataFrame | None  # forecast summary (incl. PORTFOLIO row)
    alerts: pd.DataFrame
    as_of: pd.Timestamp


def _money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def _find_ticker(question: str, snap: pd.DataFrame) -> str | None:
    tokens = re.findall(r"[A-Za-z&.]+", question)
    for tok in tokens:
        # Short tickers like "V" or "PG" only count when typed in capitals.
        if tok.upper() in set(snap["ticker"]) and (tok.isupper() or len(tok) >= 3):
            return tok.upper()
    lowered = question.lower()
    for row in snap.itertuples():
        first_word = re.sub(r"[^a-z]", "", row.company.lower().split()[0])
        if len(first_word) >= 4 and first_word in lowered:
            return row.ticker
    return None


def _overview(c: Context) -> str:
    mv, cb = c.snap["market_value"].sum(), c.snap["cost_basis"].sum()
    day = c.snap["day_change"].sum()
    text = (
        f"As of **{c.as_of:%b %d, %Y}** your {len(c.snap)} positions are worth **{_money(mv)}** "
        f"against **{_money(cb)}** invested — an unrealized {'gain' if mv >= cb else 'loss'} of "
        f"**{_money(mv - cb)} ({(mv - cb) / cb:+.1%})**. The last trading day moved the portfolio {_money(day)}."
    )
    if c.metrics:
        text += f" Over the selected period the portfolio returned **{c.metrics['total_return']:+.1%}**."
    return text


def _ranked(c: Context, best: bool) -> str:
    s = c.snap.sort_values("pnl_pct", ascending=not best)
    top = s.head(3)
    lines = [
        f"- **{r.ticker}** ({r.company}): {r.pnl_pct:+.1%} ({_money(r.unrealized_pnl)}), {r.weight:.1%} of the portfolio"
        for r in top.itertuples()
    ]
    head = "Your best performers" if best else "The positions dragging returns down the most"
    biggest = s.sort_values("unrealized_pnl", ascending=not best).iloc[0]
    tail = (
        f"\n\nIn dollar terms the biggest {'gain' if best else 'loss'} is **{biggest.ticker}** "
        f"at {_money(biggest.unrealized_pnl)}."
    )
    if not best and (s["unrealized_pnl"] >= 0).all():
        return "Good news: every position is currently above its purchase price. " + head.lower() + " are the smallest gainers:\n\n" + "\n".join(lines)
    return f"{head} (by % gain since purchase):\n\n" + "\n".join(lines) + tail


def _concentration(c: Context) -> str:
    sectors = c.snap.groupby("sector")["weight"].sum().sort_values(ascending=False)
    top_stock = c.snap.sort_values("weight", ascending=False).iloc[0]
    lines = [f"- {name}: {w:.1%}" for name, w in sectors.items()]
    flags = []
    if sectors.iloc[0] > 0.40:
        flags.append(f"**{sectors.index[0]}** is over 40% of the portfolio — that's a heavy sector bet.")
    if top_stock.weight > 0.20:
        flags.append(f"**{top_stock.ticker}** alone is {top_stock.weight:.1%} — a single-stock concentration risk.")
    verdict = " ".join(flags) or (
        f"No sector is above 40% and no single stock is above 20% (largest: {top_stock.ticker} "
        f"at {top_stock.weight:.1%}), so the portfolio is reasonably diversified."
    )
    return f"{verdict}\n\nAllocation by sector:\n\n" + "\n".join(lines)


def _risk(c: Context) -> str:
    m = c.metrics
    if not m:
        return "There isn't enough data in the selected date range to measure risk. Try widening it."
    vol_word = "low" if m["ann_vol"] < 0.12 else "moderate" if m["ann_vol"] < 0.22 else "high"
    text = (
        f"Annualized volatility is **{m['ann_vol']:.1%}** ({vol_word}). The worst peak-to-trough fall in "
        f"this period was **{m['max_drawdown']:.1%}**, and the Sharpe ratio is **{m['sharpe']:.2f}** "
        f"({'return beats the risk-free rate' if m['sharpe'] > 0 else 'return did not beat the risk-free rate'})."
    )
    if "beta" in m:
        text += (
            f" Beta vs {c.benchmark} is **{m['beta']:.2f}**, so the portfolio tends to move "
            f"{'more than' if m['beta'] > 1.05 else 'less than' if m['beta'] < 0.95 else 'about as much as'} the market."
        )
    return text


def _benchmark(c: Context) -> str:
    if not c.bench_metrics or not c.metrics:
        return "No benchmark is available in the data, so I can't compare against the market."
    p, b = c.metrics["total_return"], c.bench_metrics["total_return"]
    verdict = "beating" if p > b else "trailing"
    return (
        f"Over the selected period the portfolio returned **{p:+.1%}** vs **{b:+.1%}** for {c.benchmark} — "
        f"**{verdict}** the market by {abs(p - b):.1%}. "
        f"Risk-adjusted: portfolio Sharpe {c.metrics['sharpe']:.2f} vs {c.bench_metrics['sharpe']:.2f} for {c.benchmark}."
    )


def _outlook(c: Context) -> str:
    if c.forecast is None:
        return "The forecast needs at least 20 trading days in the selected range. Try widening the dates."
    f = c.forecast.set_index("ticker")
    p = f.loc["PORTFOLIO"]
    stocks = f.drop(index="PORTFOLIO").sort_values("expected_change")
    return (
        f"Based on 2,000 simulated scenarios built from past daily returns, the portfolio's median value in "
        f"3 months is **{_money(p['median'])}** ({p['expected_change']:+.1%}), with a likely range of "
        f"{_money(p['low_p5'])} to {_money(p['high_p95'])}. The chance of being below today's value is "
        f"**{p['prob_loss']:.0%}**.\n\n"
        f"Strongest median outlook: **{stocks.index[-1]}** ({stocks['expected_change'].iloc[-1]:+.1%}). "
        f"Weakest: **{stocks.index[0]}** ({stocks['expected_change'].iloc[0]:+.1%}).\n\n"
        "_These are scenarios from history, not predictions._"
    )


def _alerts(c: Context, ticker: str | None = None) -> str:
    a = c.alerts if ticker is None else c.alerts[c.alerts["ticker"] == ticker]
    if a.empty:
        return f"No trend breaks{' for ' + ticker if ticker else ''} in the alert window."
    lines = [f"- **{r.ticker}** · {r.signal} ({r.severity}, {r.date:%b %d}): {r.detail}" for r in a.head(6).itertuples()]
    more = f"\n\n…and {len(a) - 6} more on the Alerts tab." if len(a) > 6 else ""
    return f"{len(a)} trend alert(s){' for ' + ticker if ticker else ''}:\n\n" + "\n".join(lines) + more


def _stock(c: Context, ticker: str) -> str:
    r = c.snap.set_index("ticker").loc[ticker]
    text = (
        f"**{ticker} — {r.company}** ({r.sector})\n\n"
        f"- {r.shares:,.0f} shares bought at ${r.avg_cost:,.2f}, now ${r.price:,.2f}\n"
        f"- Value {_money(r.market_value)}, {r.weight:.1%} of the portfolio\n"
        f"- Unrealized P&L {_money(r.unrealized_pnl)} ({r.pnl_pct:+.1%}); last day {r.day_change_pct:+.2%}"
    )
    if c.forecast is not None and ticker in set(c.forecast["ticker"]):
        f = c.forecast.set_index("ticker").loc[ticker]
        text += (
            f"\n- 3-month scenarios: median ${f['median']:,.2f} ({f['expected_change']:+.1%}), "
            f"range ${f['low_p5']:,.2f}–${f['high_p95']:,.2f}, {f['prob_loss']:.0%} chance of a loss"
        )
    alerts = c.alerts[c.alerts["ticker"] == ticker]
    if not alerts.empty:
        text += "\n- ⚠ " + "; ".join(f"{a.signal}: {a.detail}" for a in alerts.itertuples())
    return text


TOPICS = [
    (("alert", "trend", "warning", "signal", "cross", "break"), "alerts"),
    (("forecast", "outlook", "predict", "future", "next", "month", "expect", "will"), "outlook"),
    (("best", "top", "winner", "gainer", "strongest"), "best"),
    (("worst", "loser", "drag", "hurt", "losing", "weakest", "bottom"), "worst"),
    (("concentrat", "diversif", "sector", "allocation", "exposure", "spread"), "concentration"),
    (("market", "benchmark", "spy", "beat", "compare", "index"), "benchmark"),
    (("risk", "volatil", "sharpe", "drawdown", "beta", "safe"), "risk"),
    (("doing", "worth", "value", "total", "overall", "summary", "p&l", "profit", "today"), "overview"),
]


def answer(question: str, c: Context) -> str:
    q = question.lower()
    if c.snap.empty:
        return "There are no holdings in the current filters, so there's nothing to analyze."
    # Match keywords at word starts so "top" doesn't fire on "stop".
    topic = next(
        (name for words, name in TOPICS if any(re.search(rf"\b{re.escape(w)}", q) for w in words)), None
    )
    ticker = _find_ticker(question, c.snap)

    if ticker and topic == "alerts":
        return _alerts(c, ticker)
    if ticker:
        return _stock(c, ticker)
    handlers = {
        "alerts": _alerts, "outlook": _outlook, "best": lambda c: _ranked(c, True),
        "worst": lambda c: _ranked(c, False), "concentration": _concentration,
        "benchmark": _benchmark, "risk": _risk, "overview": _overview,
    }
    if topic:
        return handlers[topic](c)
    return (
        "I answer questions about this portfolio using the dashboard's numbers. Try one of these:\n\n"
        + "\n".join(f"- {e}" for e in EXAMPLES)
    )
