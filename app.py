"""Stock Portfolio Analyzer — Streamlit app.

Run with:  uv run streamlit run app.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make src/ importable on hosts that install only requirements.txt (e.g. Streamlit Community Cloud).
sys.path.insert(0, str(Path(__file__).parent / "src"))

from w1_project import alerts as al  # noqa: E402
from w1_project import analytics as an
from w1_project import assistant as bot
from w1_project import forecast as fc

DATA_DIR = Path(__file__).parent / "data"
UP, DOWN, NEUTRAL = "#1a9850", "#d73027", "#4c78a8"
HORIZON = 63  # trading days ≈ 3 months

st.set_page_config(page_title="Stock Portfolio Analyzer", page_icon="📈", layout="wide")


@st.cache_data
def load(prices_src, holdings_src):
    return an.load_prices(prices_src), an.load_holdings(holdings_src)


@st.cache_data
def run_forecast(window: pd.DataFrame, shares: pd.Series) -> fc.Forecast:
    return fc.simulate(window, shares, horizon=HORIZON)


def secret(name: str) -> str | None:
    """Read from .streamlit/secrets.toml, falling back to an environment variable."""
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file at all
        value = None
    return value or os.environ.get(name)


def fan_chart(history: pd.Series, bands: pd.DataFrame, title: str, tick_prefix: str = "$") -> go.Figure:
    """History line + forecast median with 50% and 90% scenario bands."""
    fig = go.Figure()
    for lo, hi, label, opacity in (("p5", "p95", "90% of scenarios", 0.15), ("p25", "p75", "50% of scenarios", 0.30)):
        fig.add_trace(go.Scatter(x=bands.index, y=bands[hi], line_width=0, showlegend=False, hoverinfo="skip"))
        fig.add_trace(
            go.Scatter(
                x=bands.index, y=bands[lo], fill="tonexty", line_width=0, name=label,
                fillcolor=f"rgba(76,120,168,{opacity})", hoverinfo="skip",
            )
        )
    fig.add_trace(go.Scatter(x=history.index, y=history, name="History", line_color="#555"))
    fig.add_trace(go.Scatter(x=bands.index, y=bands["p50"], name="Median scenario", line=dict(color=NEUTRAL, dash="dash")))
    fig.update_layout(title=title, yaxis_tickprefix=tick_prefix, hovermode="x unified")
    return fig


def money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def pct(x: float) -> str:
    return f"{x:+.2%}"


# ---------------------------------------------------------------- data + sidebar
st.sidebar.title("📈 Portfolio Analyzer")
st.sidebar.caption("Uses the bundled sample data unless you upload your own CSVs.")
prices_file = st.sidebar.file_uploader("Prices CSV (date, ticker, close)", type="csv")
holdings_file = st.sidebar.file_uploader(
    "Holdings CSV (ticker, company, sector, shares, avg_cost, purchase_date)", type="csv"
)

try:
    prices, holdings = load(
        prices_file or DATA_DIR / "sample_prices.csv",
        holdings_file or DATA_DIR / "sample_holdings.csv",
    )
except ValueError as err:
    st.error(f"Couldn't read the CSV: {err}")
    st.stop()

wide = an.price_matrix(prices)
if wide.empty or holdings.empty:
    st.error("The uploaded data has no usable rows.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("Filters")
min_d, max_d = wide.index.min().date(), wide.index.max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
sectors = sorted(holdings["sector"].unique())
picked_sectors = st.sidebar.multiselect("Sectors", sectors, default=sectors)
ticker_options = sorted(holdings.loc[holdings["sector"].isin(picked_sectors), "ticker"].unique())
picked_tickers = st.sidebar.multiselect("Tickers", ticker_options, default=ticker_options)

benchmarks = [t for t in wide.columns if t not in set(holdings["ticker"])]
benchmark = st.sidebar.selectbox("Benchmark", benchmarks or ["(none)"], disabled=not benchmarks)
risk_free = st.sidebar.number_input("Risk-free rate (annual %)", 0.0, 15.0, 4.0, 0.25) / 100

with st.sidebar.expander("Alert settings"):
    alert_lookback = st.slider("Look back (trading days)", 5, 120, 60)
    alert_sigma = st.slider("Large move = this many × usual daily swing", 2.0, 5.0, 3.0, 0.5)
    alert_drawdown = st.slider("Drawdown alert from 1-year high (%)", 5, 50, 15) / 100

if len(date_range) != 2:
    st.info("Pick an end date to complete the date range.")
    st.stop()

start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
window = wide.loc[start:end]
selected = holdings[holdings["ticker"].isin(picked_tickers)]
if selected.empty:
    st.warning("No holdings match the current filters — select at least one sector and ticker.")
    st.stop()
if len(window) < 2:
    st.warning("The date range contains fewer than two trading days — widen it to see results.")
    st.stop()

as_of = window.index[-1]
snap = an.snapshot(selected, window, as_of)
values = an.position_values(selected, window)
port_ret = an.portfolio_returns(values, window)
bench_ret = window[benchmark].pct_change().dropna() if benchmarks else None
metrics = an.risk_metrics(port_ret, bench_ret, risk_free)
bench_metrics = an.risk_metrics(bench_ret, None, risk_free) if benchmarks else {}

st.title("Stock Portfolio Analyzer")
st.caption(
    f"{len(snap)} positions · {start:%b %d, %Y} → {as_of:%b %d, %Y}"
    + (f" · benchmark {benchmark}" if benchmarks else "")
)
if snap.empty:
    st.warning("None of the selected holdings had been purchased by the end of this date range.")
    st.stop()

try:
    outlook = run_forecast(window, snap.set_index("ticker")["shares"])
except ValueError:
    outlook = None
trend_alerts = al.detect(
    wide, list(snap["ticker"]), as_of, port_ret,
    lookback=alert_lookback, move_sigma=alert_sigma, drawdown_limit=alert_drawdown,
)

overview, holdings_tab, performance, risk, forecast_tab, alerts_tab, assistant_tab = st.tabs(
    ["Overview", "Holdings", "Performance", "Risk", "Forecast", "Alerts", "Assistant"]
)

# ---------------------------------------------------------------- overview
with overview:
    mv, cb = snap["market_value"].sum(), snap["cost_basis"].sum()
    pnl, day = mv - cb, snap["day_change"].sum()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Market value", money(mv))
    c2.metric("Cost basis", money(cb))
    c3.metric("Unrealized P&L", money(pnl), pct(pnl / cb) if cb else None)
    c4.metric("Day change", money(day), pct(day / (mv - day)) if mv - day else None)
    if metrics:
        delta = (
            f"{metrics['total_return'] - bench_metrics['total_return']:+.2%} vs {benchmark}"
            if bench_metrics
            else None
        )
        c5.metric("Period return", pct(metrics["total_return"]), delta)

    value_df = pd.DataFrame(
        {
            "Market value": values.sum(axis=1),
            "Invested capital": an.invested_capital(selected, window.index),
        }
    )
    fig = px.line(value_df, labels={"value": "USD", "date": "", "variable": ""})
    fig.update_layout(title="Portfolio value vs. money invested", yaxis_tickprefix="$", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    by_sector = snap.groupby("sector", as_index=False)["market_value"].sum()
    fig = px.pie(by_sector, names="sector", values="market_value", hole=0.5, title="Allocation by sector")
    fig.update_traces(hovertemplate="%{label}<br>$%{value:,.0f} (%{percent})<extra></extra>")
    left.plotly_chart(fig, width="stretch")

    fig = px.bar(
        snap.sort_values("weight"), x="weight", y="ticker", orientation="h",
        title="Position weights", labels={"weight": "Weight", "ticker": ""},
        color_discrete_sequence=[NEUTRAL],
    )
    fig.update_layout(xaxis_tickformat=".0%")
    right.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- holdings
with holdings_tab:
    query = st.text_input("Search by ticker or company", placeholder="e.g. NVDA or Apple")
    table = snap
    if query:
        q = query.strip().lower()
        table = snap[
            snap["ticker"].str.lower().str.contains(q, regex=False)
            | snap["company"].str.lower().str.contains(q, regex=False)
        ]
    if table.empty:
        st.info(f"No holdings match “{query}”.")
    else:
        cols = ["ticker", "company", "sector", "shares", "avg_cost", "price", "market_value",
                "cost_basis", "unrealized_pnl", "pnl_pct", "day_change_pct", "weight", "purchase_date"]
        view = table[cols].copy()
        for c in ("pnl_pct", "day_change_pct", "weight"):
            view[c] *= 100
        st.dataframe(
            view,
            hide_index=True,
            width="stretch",
            column_config={
                "ticker": "Ticker", "company": "Company", "sector": "Sector",
                "shares": st.column_config.NumberColumn("Shares", format="%d"),
                "avg_cost": st.column_config.NumberColumn("Avg cost ($)", format="$%.2f"),
                "price": st.column_config.NumberColumn("Price ($)", format="$%.2f"),
                "market_value": st.column_config.NumberColumn("Market value ($)", format="$%,.0f"),
                "cost_basis": st.column_config.NumberColumn("Cost basis ($)", format="$%,.0f"),
                "unrealized_pnl": st.column_config.NumberColumn("Unrealized P&L ($)", format="$%,.0f"),
                "pnl_pct": st.column_config.NumberColumn("P&L (%)", format="%.1f%%"),
                "day_change_pct": st.column_config.NumberColumn("Day (%)", format="%.2f%%"),
                "weight": st.column_config.ProgressColumn("Weight (%)", format="%.1f%%", min_value=0, max_value=100),
                "purchase_date": st.column_config.DateColumn("Bought"),
            },
        )
        st.download_button(
            "Download table as CSV", table[cols].to_csv(index=False), "holdings_snapshot.csv", "text/csv"
        )

        ranked = table.sort_values("unrealized_pnl")
        fig = go.Figure(
            go.Bar(
                x=ranked["unrealized_pnl"], y=ranked["ticker"], orientation="h",
                marker_color=[UP if v >= 0 else DOWN for v in ranked["unrealized_pnl"]],
                customdata=ranked["pnl_pct"],
                hovertemplate="%{y}: $%{x:,.0f} (%{customdata:+.1%})<extra></extra>",
            )
        )
        fig.update_layout(title="Unrealized P&L by position", xaxis_tickprefix="$")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- performance
with performance:
    cum = pd.DataFrame({"Portfolio": an.cumulative_return(port_ret)})
    if bench_ret is not None:
        cum[benchmark] = an.cumulative_return(bench_ret)
    fig = px.line(cum, labels={"value": "Cumulative return", "date": "", "variable": ""})
    fig.update_layout(title="Cumulative return", yaxis_tickformat=".0%", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    rebased = window[snap["ticker"]].dropna(how="all")
    rebased = rebased / rebased.bfill().iloc[0] * 100
    fig = px.line(rebased, labels={"value": "Price (start = 100)", "date": "", "ticker": ""})
    fig.update_layout(title="Individual stocks, rebased to 100 at start of range", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    monthly = (1 + port_ret).resample("ME").prod() - 1
    fig = go.Figure(
        go.Bar(
            x=monthly.index, y=monthly.values,
            marker_color=[UP if v >= 0 else DOWN for v in monthly.values],
            hovertemplate="%{x|%b %Y}: %{y:+.2%}<extra></extra>",
        )
    )
    fig.update_layout(title="Monthly portfolio returns", yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- risk
with risk:
    if not metrics:
        st.info("Not enough data in this range to compute risk metrics.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annualized volatility", f"{metrics['ann_vol']:.1%}")
        c2.metric("Sharpe ratio", f"{metrics['sharpe']:.2f}", help=f"Uses a {risk_free:.2%} risk-free rate")
        c3.metric("Max drawdown", f"{metrics['max_drawdown']:.1%}")
        c4.metric(f"Beta vs {benchmark}" if "beta" in metrics else "Beta", f"{metrics.get('beta', float('nan')):.2f}")

        dd = pd.DataFrame({"Portfolio": an.drawdown(port_ret)})
        if bench_ret is not None:
            dd[benchmark] = an.drawdown(bench_ret)
        fig = px.area(dd, labels={"value": "Drawdown", "date": "", "variable": ""})
        fig.update_traces(stackgroup=None, fill="tozeroy")
        fig.update_layout(title="Drawdown from previous peak", yaxis_tickformat=".0%", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

        left, right = st.columns(2)
        stock_ret = window[snap["ticker"]].pct_change().dropna(how="all")
        if len(snap) > 1:
            corr = stock_ret.corr()
            fig = px.imshow(
                corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                title="Correlation of daily returns",
            )
            left.plotly_chart(fig, width="stretch")
        else:
            left.info("Select at least two tickers to see correlations.")

        per_stock = pd.DataFrame(
            {t: an.risk_metrics(stock_ret[t], None, risk_free) for t in snap["ticker"]}
        ).T.reset_index(names="ticker")
        fig = px.scatter(
            per_stock, x="ann_vol", y="ann_return", text="ticker",
            title="Risk vs. return (annualized)",
            labels={"ann_vol": "Volatility", "ann_return": "Return"},
        )
        fig.update_traces(textposition="top center", marker=dict(size=12, color=NEUTRAL))
        fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        right.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- forecast
with forecast_tab:
    st.caption(
        "Scenarios, not predictions: 2,000 simulated 3-month paths built by replaying random days "
        "from the selected history. Assumes you keep today's shares and don't trade."
    )
    if outlook is None:
        st.info("The forecast needs at least 20 trading days in the selected date range — widen it.")
    else:
        p = outlook.summary.set_index("ticker").loc["PORTFOLIO"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Value today", money(p["today"]))
        c2.metric("Median in 3 months", money(p["median"]), pct(p["expected_change"]))
        c3.metric("Likely range (90%)", f"{money(p['low_p5'])} – {money(p['high_p95'])}")
        c4.metric("Chance of a loss", f"{p['prob_loss']:.0%}")

        st.plotly_chart(
            fan_chart(values.sum(axis=1).iloc[-126:], outlook.portfolio, "Portfolio value: last 6 months and next 3 months"),
            width="stretch",
        )

        pick = st.selectbox("Stock", list(snap["ticker"]), format_func=lambda t: f"{t} — {snap.set_index('ticker').loc[t, 'company']}")
        st.plotly_chart(
            fan_chart(window[pick].iloc[-126:], outlook.stocks[pick], f"{pick} share price: last 6 months and next 3 months"),
            width="stretch",
        )

        table = outlook.summary[outlook.summary["ticker"] != "PORTFOLIO"].sort_values("expected_change", ascending=False)
        view = table.copy()
        view[["expected_change", "prob_loss"]] *= 100
        st.dataframe(
            view,
            hide_index=True,
            width="stretch",
            column_config={
                "ticker": "Ticker",
                "today": st.column_config.NumberColumn("Price today ($)", format="$%.2f"),
                "median": st.column_config.NumberColumn("Median in 3 mo ($)", format="$%.2f"),
                "low_p5": st.column_config.NumberColumn("Low, 5th pct ($)", format="$%.2f"),
                "high_p95": st.column_config.NumberColumn("High, 95th pct ($)", format="$%.2f"),
                "expected_change": st.column_config.NumberColumn("Median change (%)", format="%+.1f%%"),
                "prob_loss": st.column_config.NumberColumn("Chance of loss (%)", format="%.0f%%"),
            },
        )

# ---------------------------------------------------------------- alerts
with alerts_tab:
    st.caption(
        f"Trend breaks in the last {alert_lookback} trading days to {as_of:%b %d, %Y}: "
        "50/200-day average crosses, unusually large daily moves, and drawdowns from the 1-year high. "
        "Adjust thresholds under **Alert settings** in the sidebar."
    )
    counts = trend_alerts["severity"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("High", int(counts.get("High", 0)))
    c2.metric("Medium", int(counts.get("Medium", 0)))
    c3.metric("Info", int(counts.get("Info", 0)))

    if trend_alerts.empty:
        st.success("No trend breaks with the current settings.")
    else:
        icons = {"High": "🔴 High", "Medium": "🟠 Medium", "Info": "🔵 Info"}
        st.dataframe(
            trend_alerts.assign(severity=trend_alerts["severity"].map(icons)),
            hide_index=True,
            width="stretch",
            column_config={
                "date": st.column_config.DateColumn("Date"), "ticker": "Ticker", "signal": "Signal",
                "severity": "Severity", "detail": st.column_config.TextColumn("Detail", width="large"),
            },
        )

    st.subheader("Email alert")
    api_key = secret("RESEND_API_KEY")
    sender = secret("ALERT_EMAIL_FROM") or "Portfolio Analyzer <onboarding@resend.dev>"
    recipient = st.text_input("Send to", value=secret("ALERT_EMAIL_TO") or "")
    if not api_key:
        st.info(
            "Email isn't set up yet. Add `RESEND_API_KEY` (and optionally `ALERT_EMAIL_TO`) to "
            "`.streamlit/secrets.toml` — see `.streamlit/secrets.toml.example` — then restart the app."
        )
    if st.button("Send test alert email", type="primary", disabled=not (api_key and recipient and not trend_alerts.empty)):
        subject, body = al.email_body(trend_alerts, as_of)
        try:
            with st.spinner("Sending…"):
                message_id = al.send_email(api_key, sender, recipient, subject, body)
            st.success(f"Sent “{subject}” to {recipient} (id {message_id}).")
        except RuntimeError as err:
            st.error(str(err))

# ---------------------------------------------------------------- assistant
with assistant_tab:
    st.caption(
        "Ask about this portfolio in plain English. Answers use the dashboard's own numbers "
        "and respect the sidebar filters. (Rule-based — no AI model, no data leaves your computer.)"
    )
    st.session_state.setdefault("chat", [])
    context = bot.Context(
        snap=snap, metrics=metrics, bench_metrics=bench_metrics,
        benchmark=benchmark if benchmarks else None,
        forecast=outlook.summary if outlook else None, alerts=trend_alerts, as_of=as_of,
    )

    def ask_suggestion():
        st.session_state.pending = st.session_state.suggestion
        st.session_state.suggestion = None  # unselect, so the same pill can be clicked again

    st.pills("Try asking", bot.EXAMPLES, key="suggestion", on_change=ask_suggestion)
    typed = st.chat_input("Ask a question about your portfolio")
    question = typed or st.session_state.pop("pending", None)
    if question:
        st.session_state.chat.append(("user", question))
        st.session_state.chat.append(("assistant", bot.answer(question, context)))

    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(text)
    if st.session_state.chat and st.button("Clear chat"):
        st.session_state.chat = []
        st.rerun()

st.caption("Sample data is synthetic and for demonstration only — not investment advice.")
