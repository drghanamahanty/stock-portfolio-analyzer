"""Trend-break detection and email delivery (via the Resend API)."""

from __future__ import annotations

import html
import json
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from w1_project import analytics as an

SEVERITY_ORDER = {"High": 0, "Medium": 1, "Info": 2}


def detect(
    wide: pd.DataFrame,
    tickers: list[str],
    as_of: pd.Timestamp,
    portfolio_returns: pd.Series | None = None,
    lookback: int = 20,
    move_sigma: float = 3.0,
    drawdown_limit: float = 0.15,
) -> pd.DataFrame:
    """Find trend breaks in the last `lookback` trading days up to `as_of`.

    - Golden / death cross: 50-day average crosses the 200-day average
    - Large move: a daily return beyond `move_sigma` x the prior 60-day volatility
    - Drawdown: price `drawdown_limit` or more below its 1-year high (also for the portfolio)
    """
    hist = wide.loc[:as_of]
    recent = hist.index[-lookback:]
    found: list[dict] = []

    for t in tickers:
        price = hist[t].dropna()
        if len(price) < 2:
            continue

        fast, slow = price.rolling(50).mean(), price.rolling(200).mean()
        above = (fast > slow).where(slow.notna())
        crossed = above.ne(above.shift()) & above.notna() & above.shift().notna()
        for day in crossed[crossed].index.intersection(recent):
            golden = bool(above[day])
            found.append({
                "date": day, "ticker": t,
                "signal": "Golden cross" if golden else "Death cross",
                "severity": "Info" if golden else "High",
                "detail": f"50-day average moved {'above' if golden else 'below'} the 200-day average "
                          f"({fast[day]:.2f} vs {slow[day]:.2f}).",
            })

        ret = price.pct_change()
        z = ret / ret.rolling(60).std().shift(1)
        for day in z[z.abs() >= move_sigma].index.intersection(recent):
            found.append({
                "date": day, "ticker": t, "signal": "Large move",
                "severity": "High" if ret[day] < 0 else "Medium",
                "detail": f"{'Fell' if ret[day] < 0 else 'Rose'} {abs(ret[day]):.1%} in one day "
                          f"({abs(z[day]):.1f}x its usual daily swing).",
            })

        year = price.iloc[-252:]
        dd = year.iloc[-1] / year.max() - 1
        if dd <= -drawdown_limit:
            found.append({
                "date": price.index[-1], "ticker": t, "signal": "Drawdown", "severity": "High",
                "detail": f"{abs(dd):.1%} below its 1-year high of ${year.max():,.2f} "
                          f"(on {year.idxmax():%b %d, %Y}).",
            })

    if portfolio_returns is not None and len(portfolio_returns) > 1:
        dd = an.drawdown(portfolio_returns.iloc[-252:])
        if dd.iloc[-1] <= -drawdown_limit:
            found.append({
                "date": dd.index[-1], "ticker": "PORTFOLIO", "signal": "Drawdown", "severity": "High",
                "detail": f"Portfolio is {abs(dd.iloc[-1]):.1%} below its 1-year peak.",
            })

    df = pd.DataFrame(found, columns=["date", "ticker", "signal", "severity", "detail"])
    if df.empty:
        return df
    df["rank"] = df["severity"].map(SEVERITY_ORDER)
    return df.sort_values(["rank", "date"], ascending=[True, False]).drop(columns="rank").reset_index(drop=True)


def email_body(alerts: pd.DataFrame, as_of: pd.Timestamp) -> tuple[str, str]:
    """Subject line and HTML body summarising the alerts."""
    high = int((alerts["severity"] == "High").sum())
    subject = f"Portfolio alert: {len(alerts)} trend break(s), {high} high severity ({as_of:%b %d, %Y})"
    rows = "".join(
        f"<tr><td>{r.date:%Y-%m-%d}</td><td><b>{html.escape(r.ticker)}</b></td>"
        f"<td>{html.escape(r.signal)}</td><td>{r.severity}</td><td>{html.escape(r.detail)}</td></tr>"
        for r in alerts.itertuples()
    )
    body = (
        f"<h2>Stock Portfolio Analyzer &mdash; trend alerts</h2>"
        f"<p>As of {as_of:%A, %B %d, %Y}: {len(alerts)} alert(s).</p>"
        "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;font-family:sans-serif'>"
        "<tr><th>Date</th><th>Ticker</th><th>Signal</th><th>Severity</th><th>Detail</th></tr>"
        f"{rows}</table><p style='color:#666'>Sent from the Stock Portfolio Analyzer. Not investment advice.</p>"
    )
    return subject, body


def send_email(api_key: str, sender: str, to: str, subject: str, html_body: str) -> str:
    """Send through Resend (https://resend.com). Returns the message id."""
    request = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps({"from": sender, "to": [to], "subject": subject, "html": html_body}).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "stock-portfolio-analyzer",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as resp:
            return json.loads(resp.read()).get("id", "")
    except urllib.error.HTTPError as err:
        detail = err.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("message", detail)
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"Resend rejected the email ({err.code}): {detail}") from None
    except urllib.error.URLError as err:
        raise RuntimeError(f"Couldn't reach Resend: {err.reason}") from None
