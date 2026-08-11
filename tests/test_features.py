"""Feature engineering: volatility estimators and market-relative measures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskengine.features.market import momentum, rolling_beta, turnover_liquidity
from riskengine.features.volatility import (
    downside_deviation,
    ewma_vol,
    garman_klass,
    parkinson,
    realised_vol,
    rogers_satchell,
)


def test_realised_vol_recovers_known_std():
    """21-day constant-volatility synthetic series: realised vol should recover
    the generating volatility to within sampling noise."""
    rng = np.random.default_rng(1)
    n = 3000
    true_daily_vol = 0.015
    r = pd.DataFrame({"A": rng.normal(0, true_daily_vol, n)})
    rv = realised_vol(r, window=252, annualise=False)
    assert rv["A"].iloc[-1] == pytest.approx(true_daily_vol, rel=0.08)


def test_ewma_reacts_faster_than_long_window_realised_vol():
    """After a vol regime shift, EWMA(0.94) should move closer to the new level
    than a 252-day trailing window still dominated by the old regime."""
    rng = np.random.default_rng(2)
    calm = rng.normal(0, 0.005, 800)
    wild = rng.normal(0, 0.030, 60)
    r = pd.DataFrame({"A": np.concatenate([calm, wild])})

    e = ewma_vol(r, lam=0.94, annualise=False)["A"].iloc[-1]
    rv_long = realised_vol(r, window=252, annualise=False)["A"].iloc[-1]
    assert abs(e - 0.030) < abs(rv_long - 0.030)


def test_parkinson_is_more_efficient_than_close_to_close():
    """On simulated GBM (no drift, no jumps) Parkinson's estimate of a KNOWN
    constant vol should have lower variance across repeated short windows than
    close-to-close — that lower variance is the entire point of range estimators."""
    rng = np.random.default_rng(3)
    true_vol = 0.02
    n_days, n_trials, window = 21, 400, 21

    cc_estimates, pk_estimates = [], []
    for _ in range(n_trials):
        # simulate one intraday path per day to get a genuine high/low
        opens = np.zeros(n_days)
        closes = np.zeros(n_days)
        highs = np.zeros(n_days)
        lows = np.zeros(n_days)
        price = 100.0
        for d in range(n_days):
            intraday = price * np.exp(np.cumsum(rng.normal(0, true_vol / np.sqrt(20), 20)))
            opens[d] = price
            highs[d] = intraday.max()
            lows[d] = intraday.min()
            closes[d] = intraday[-1]
            price = closes[d]

        close_s = pd.DataFrame({"A": closes})
        high_s = pd.DataFrame({"A": highs})
        low_s = pd.DataFrame({"A": lows})
        rets = close_s.pct_change()

        cc = realised_vol(rets, window=window, annualise=False)["A"].iloc[-1]
        pk = parkinson(high_s, low_s, window=window, annualise=False)["A"].iloc[-1]
        cc_estimates.append(cc)
        pk_estimates.append(pk)

    cc_var = np.var(cc_estimates)
    pk_var = np.var(pk_estimates)
    assert pk_var < cc_var


def test_garman_klass_nonnegative_and_finite(ohlcv, prices):
    op = ohlcv.pivot(index="date", columns="ticker", values="open").reindex(prices.index)
    hi = ohlcv.pivot(index="date", columns="ticker", values="high").reindex(prices.index)
    lo = ohlcv.pivot(index="date", columns="ticker", values="low").reindex(prices.index)
    gk = garman_klass(op, hi, lo, prices, window=21)
    finite = gk.to_numpy()[np.isfinite(gk.to_numpy())]
    assert (finite >= 0).all()


def test_rogers_satchell_nonnegative(ohlcv, prices):
    op = ohlcv.pivot(index="date", columns="ticker", values="open").reindex(prices.index)
    hi = ohlcv.pivot(index="date", columns="ticker", values="high").reindex(prices.index)
    lo = ohlcv.pivot(index="date", columns="ticker", values="low").reindex(prices.index)
    rs = rogers_satchell(op, hi, lo, prices, window=21)
    finite = rs.to_numpy()[np.isfinite(rs.to_numpy())]
    assert (finite >= 0).all()


def test_downside_deviation_ignores_upside_moves():
    """A series of alternating +5% / -1% has small downside deviation despite
    large total volatility — semi-deviation must reflect only the downside."""
    r = pd.Series([0.05, -0.01] * 200)
    dd = downside_deviation(r, window=len(r), mar=0.0, annualise=False).iloc[-1]
    total_vol = r.std(ddof=1)
    assert dd < total_vol


def test_momentum_positive_for_a_steadily_rising_stock():
    idx = pd.bdate_range("2015-01-01", periods=400)
    p = pd.DataFrame({"A": 100 * 1.0005 ** np.arange(400)}, index=idx)
    mom = momentum(p, lookback=252, skip=21)
    assert mom["A"].iloc[-1] > 0


def test_rolling_beta_recovers_known_loading(market_returns):
    rng = np.random.default_rng(5)
    asset = 0.8 * market_returns + rng.normal(0, 1e-5, len(market_returns))
    b = rolling_beta(asset.to_frame("A"), market_returns, window=252)
    assert b["A"].dropna().iloc[-1] == pytest.approx(0.8, abs=0.02)


def test_turnover_liquidity_is_traded_value():
    idx = pd.bdate_range("2020-01-01", periods=30)
    vol = pd.DataFrame({"A": 1000.0}, index=idx)
    close = pd.DataFrame({"A": 10.0}, index=idx)
    liq = turnover_liquidity(vol, close, window=10)
    assert liq["A"].iloc[-1] == pytest.approx(10_000.0)
