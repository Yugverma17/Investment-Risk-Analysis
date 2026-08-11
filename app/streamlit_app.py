"""RiskLens — portfolio risk & allocation dashboard.

Run with:
    streamlit run app/streamlit_app.py

The app is a thin presentation layer. All the actual analytics (metrics, VaR,
allocators, backtest) live in `riskengine` and are unit-tested there — this
file does formatting and layout only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riskengine.config import DATA_PROCESSED, PROFILES, RESULTS, TABLES  # noqa: E402
from riskengine.data import sector_map  # noqa: E402
from riskengine.optimize.allocators import ALLOCATORS  # noqa: E402
from riskengine.optimize.constraints import Constraints, select_candidates  # noqa: E402
from riskengine.optimize.scoring import composite_score, compute_metrics  # noqa: E402
from riskengine.risk.covariance import ledoit_wolf_cov  # noqa: E402
from riskengine.risk.var import conditional_var, historical_var, var_in_rupees  # noqa: E402

st.set_page_config(page_title="RiskLens", page_icon="📊", layout="wide")

PRIMARY = "#0072B2"
GOOD = "#009E73"
BAD = "#D55E00"
NEUTRAL = "#7F7F7F"


# --------------------------------------------------------------------- data --
@st.cache_data(show_spinner=False)
def load_data():
    missing = []
    paths = {
        "close": DATA_PROCESSED / "close.parquet",
        "bench": DATA_PROCESSED / "benchmark.parquet",
    }
    for _k, p in paths.items():
        if not p.exists():
            missing.append(str(p))
    if missing:
        return None, None, None, missing

    close = pd.read_parquet(paths["close"])
    bench = pd.read_parquet(paths["bench"])["close"]
    mkt = bench.reindex(close.index).pct_change()
    smap = sector_map()
    return close, mkt, smap, []


@st.cache_data(show_spinner=False)
def load_results_table(name: str) -> pd.DataFrame | None:
    path = TABLES / name
    return pd.read_csv(path, index_col=0) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_strategy_returns() -> pd.DataFrame | None:
    path = RESULTS / "strategy_returns.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def build_portfolio(close: pd.DataFrame, mkt: pd.Series, smap: dict, profile_key: str, n_select: int):
    profile = PROFILES[profile_key]
    train_end = close.index[-1]
    train_start = train_end - pd.DateOffset(months=36)
    train = close.pct_change().loc[(close.index > train_start) & (close.index <= train_end)]
    train = train.dropna(axis=1, thresh=int(len(train) * 0.9))
    mkt_train = mkt.reindex(train.index)

    metrics = compute_metrics(train, mkt_train)
    scores = composite_score(metrics, profile.score_weights)
    chosen = select_candidates(
        scores, train, n_select=n_select, corr_threshold=0.85, sector_map=smap, max_per_sector=5
    )
    if len(chosen) < profile.min_positions:
        chosen = list(scores.sort_values(ascending=False).head(n_select).index)

    sub_train = train[chosen]
    cov = ledoit_wolf_cov(sub_train)
    cons = Constraints(max_weight=profile.max_weight, max_sector_weight=0.35, sector_map=smap)
    weights = ALLOCATORS["score_based"](
        sub_train, cons, cov=cov, scores=scores.reindex(chosen), expected_returns=metrics["mean_return"]
    )
    weights = weights[weights > 0].sort_values(ascending=False)
    return weights, metrics.reindex(weights.index), sub_train


# --------------------------------------------------------------------- UI ---
st.title("📊 RiskLens")
st.caption(
    "A walk-forward-backtested portfolio risk & allocation engine for NSE equities. "
    "Not investment advice — see **Methodology** for every assumption."
)

close, mkt, smap, missing = load_data()

if close is None:
    st.error(
        "No processed market data found. Run the data pipeline first:\n\n"
        "```\npython scripts/fetch_data.py\n```"
    )
    st.caption("Missing: " + ", ".join(missing))
    st.stop()

tab_build, tab_backtest, tab_risk, tab_about = st.tabs(
    ["🎯 Build my portfolio", "📈 Backtest vs Nifty", "⚠️ Risk report card", "ℹ️ Methodology"]
)

# ============================================================ Build tab ====
with tab_build:
    col_ctrl, col_out = st.columns([1, 2.4])

    with col_ctrl:
        st.subheader("Your profile")
        profile_key = st.select_slider(
            "Risk appetite", options=["conservative", "balanced", "aggressive"], value="balanced"
        )
        capital = st.number_input("Capital to invest (₹)", min_value=10_000, value=500_000, step=10_000)
        n_select = st.slider("Max number of holdings", 8, 30, 20)
        horizon = st.selectbox("Investment horizon", ["1 year", "3 years", "5+ years"], index=1)

        p = PROFILES[profile_key]
        st.markdown(
            f"""
**{p.name} profile constraints**
- Max single-stock weight: **{p.max_weight:.0%}**
- Minimum holdings: **{p.min_positions}**
- Target volatility: **{f'{p.target_vol:.0%}' if p.target_vol else 'unconstrained'}**
"""
        )
        st.caption(
            "Scoring uses only the trailing 36 months of data as of the latest "
            "available price — exactly the same rule the walk-forward backtest uses "
            "at every historical rebalance."
        )

    with st.spinner("Scoring the universe on the trailing 36-month window..."):
        weights, wmetrics, sub_train = build_portfolio(close, mkt, smap, profile_key, n_select)

    with col_out:
        st.subheader(f"Suggested allocation — {p.name}")

        c1, c2, c3, c4 = st.columns(4)
        port_ret = float((weights * wmetrics["mean_return"]).sum())
        # weights can be a strict subset of sub_train's columns (anything an
        # allocator floors to exactly zero is dropped) — align explicitly
        # rather than assuming the two are the same shape.
        cov_full = ledoit_wolf_cov(sub_train)
        cov_aligned = cov_full.loc[weights.index, weights.index].to_numpy()
        port_vol_est = float(np.sqrt(weights.to_numpy() @ cov_aligned @ weights.to_numpy()))
        c1.metric("Holdings", len(weights))
        c2.metric("Est. annual return", f"{port_ret:.1%}")
        c3.metric("Est. annual volatility", f"{port_vol_est:.1%}")
        c4.metric("Est. Sharpe", f"{(port_ret - 0.065) / port_vol_est:.2f}" if port_vol_est else "—")
        st.caption(
            "⚠️ These are trailing 36-month figures for stocks the model just selected "
            "*because* they scored well over that window — not a forward-looking return "
            "guarantee. It will read as optimistic almost by construction. The realistic "
            "expectation is the walk-forward, out-of-sample result in the **Backtest vs "
            "Nifty** tab, which is deliberately lower."
        )

        fig = go.Figure(
            go.Pie(
                labels=[t.replace(".NS", "") for t in weights.index],
                values=weights.values,
                hole=0.45,
                marker=dict(line=dict(color="white", width=1)),
                textinfo="label+percent",
                textfont=dict(size=10),
            )
        )
        fig.update_layout(height=430, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        table = pd.DataFrame(
            {
                "Weight": weights,
                "₹ Allocated": weights * capital,
                "Sector": [smap.get(t, "—") for t in weights.index],
                "Sharpe (36m)": wmetrics["sharpe"],
                "Vol (36m, ann.)": wmetrics["volatility"],
                "Beta": wmetrics["beta"],
            }
        ).sort_values("Weight", ascending=False)
        st.dataframe(
            table.style.format(
                {
                    "Weight": "{:.1%}",
                    "₹ Allocated": "₹{:,.0f}",
                    "Sharpe (36m)": "{:.2f}",
                    "Vol (36m, ann.)": "{:.1%}",
                    "Beta": "{:.2f}",
                }
            ),
            use_container_width=True,
            height=380,
        )

        port_ret_series = (sub_train * weights).sum(axis=1)
        var95 = historical_var(port_ret_series, 0.95)
        var99 = historical_var(port_ret_series, 0.99)
        cv95 = conditional_var(port_ret_series, 0.95)

        st.subheader("What this means in rupees")
        r1, r2, r3 = st.columns(3)
        r1.metric("1-day VaR (95%)", f"₹{var_in_rupees(var95, capital):,.0f}", help="On 1 in 20 trading days, expect to lose at least this much.")
        r2.metric("1-day VaR (99%)", f"₹{var_in_rupees(var99, capital):,.0f}", help="On 1 in 100 trading days, expect to lose at least this much.")
        r3.metric("1-day CVaR (95%)", f"₹{var_in_rupees(cv95, capital):,.0f}", help="Average loss on the worst 5% of days.")
        st.caption(
            "VaR is estimated from the same 36-month window used for scoring, using the historical-quantile "
            "method. It is validated out-of-sample in the **Risk report card** tab via Kupiec/Christoffersen "
            "backtests — read that before trusting a single point estimate."
        )

# ========================================================== Backtest tab ===
with tab_backtest:
    streams = load_strategy_returns()
    comparison = load_results_table("strategy_comparison.csv")
    sig = load_results_table("significance_tests.csv")

    if streams is None or comparison is None:
        st.warning(
            "No backtest results found yet. Run:\n\n```\npython scripts/run_backtests.py\n```"
        )
    else:
        st.subheader("Six strategies, one walk-forward test, net of costs")
        st.caption(
            "Every strategy is trained on a rolling 36-month window, holds for 3 months, pays 15bps "
            "transaction cost on every rupee traded, and is evaluated on the SAME window as every other "
            "strategy and the Nifty 50 benchmark."
        )

        default_pick = [c for c in ["equal_weight", "score_based", "max_sharpe", "min_variance"] if c in streams.columns]
        picks = st.multiselect("Strategies to plot", list(streams.columns), default=default_pick)

        if picks:
            curves = (1 + streams[picks].fillna(0)).cumprod()
            fig = go.Figure()
            for col in curves.columns:
                fig.add_trace(go.Scatter(x=curves.index, y=curves[col], name=col, mode="lines"))
            fig.update_layout(
                height=460,
                yaxis_type="log",
                yaxis_title="Growth of ₹1 (log scale)",
                margin=dict(t=20, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Results table")
        st.dataframe(comparison.style.format(precision=3), use_container_width=True)

        if sig is not None:
            st.subheader("Is any of this distinguishable from luck?")
            st.caption(
                "Bootstrap 95% CI on (strategy Sharpe − Nifty Sharpe), using a stationary block bootstrap "
                "to respect autocorrelation. A CI that excludes zero is the bar for 'significant'."
            )
            st.dataframe(sig.style.format(precision=3), use_container_width=True)

        stress = load_results_table("stress_tests.csv")
        if stress is not None:
            st.subheader("Stress windows")
            st.dataframe(stress.style.format("{:.1%}"), use_container_width=True)

# ============================================================== Risk tab ===
with tab_risk:
    var_bt = load_results_table("var_backtest.csv")
    var_sum = load_results_table("var_summary.csv")
    vol_cmp = load_results_table("vol_model_comparison.csv")

    st.subheader("VaR backtest — did the risk model actually work?")
    if var_bt is not None:
        st.caption(
            "Kupiec tests whether the NUMBER of breaches matches expectation. Christoffersen tests whether "
            "breaches were independent (not clustered). A model can pass one and fail the other."
        )
        st.dataframe(var_bt, use_container_width=True)
    else:
        st.info("Run `python scripts/run_backtests.py` to generate this table.")

    if var_sum is not None:
        st.subheader("VaR / CVaR by method")
        st.dataframe(var_sum.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Volatility forecasting — LightGBM vs. EWMA vs. GARCH")
    if vol_cmp is not None:
        st.caption(
            "Lower RMSE/QLIKE is better. QLIKE is the metric the volatility-forecasting literature actually "
            "uses because it's robust to noise in the realised-vol proxy."
        )
        st.dataframe(vol_cmp.style.format(precision=4), use_container_width=True)
        dm = load_results_table("vol_dm_tests.csv")
        if dm is not None:
            st.caption("Diebold-Mariano test for whether the improvement over each baseline is significant:")
            st.dataframe(dm, use_container_width=True)
    else:
        st.info("Run `python scripts/run_vol_model.py` to generate this table.")

# ============================================================= About tab ===
with tab_about:
    st.subheader("What this is")
    st.markdown(
        """
RiskLens is a walk-forward-backtested portfolio construction engine over ~120 liquid NSE stocks
(2015–2025). It is a resume / learning project, **not financial advice**, and should not be used
to make real investment decisions.

**Known limitations — read before trusting a number:**

1. **Survivorship bias.** The universe is a current-membership snapshot, not point-in-time. See
   [`docs/methodology.md`](../docs/methodology.md) for the mitigations applied and why they're
   partial.
2. **A constant 6.5% risk-free rate** is used throughout instead of the actual time-varying G-Sec
   yield.
3. **Transaction costs (15bps)** are a reasonable estimate for discount-broker delivery trades, not
   a live quote.
4. **VaR assumes the recent past represents the near future.** The backtest tab shows exactly how
   often that assumption failed.

Full methodology, every formula, and the ADRs behind each design choice: see the `docs/` folder in
the repository.
"""
    )
    st.caption("Built with pandas, scikit-learn, LightGBM, arch, scipy, and Streamlit.")
