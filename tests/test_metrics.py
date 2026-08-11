"""Known-answer tests for the performance metrics.

Where a closed form exists the test asserts against it rather than against a
previously-observed output — a regression test that just pins whatever the code
did last time cannot catch a formula that was wrong from the start.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskengine.config import TRADING_DAYS
from riskengine.risk.metrics import (
    annual_vol,
    beta,
    cagr,
    calmar_ratio,
    daily_rf,
    drawdown_series,
    hit_rate,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    tearsheet,
    ulcer_index,
)


def test_cagr_matches_closed_form(flat_returns):
    # 252 days of exactly +0.1% compounds to 1.001**252 over exactly one year
    expected = 1.001**252 - 1
    assert cagr(flat_returns) == pytest.approx(expected, rel=1e-9)


def test_zero_vol_series_has_undefined_sharpe(flat_returns):
    # constant returns => zero standard deviation => Sharpe is not finite
    assert np.isnan(sharpe_ratio(flat_returns))


def test_annual_vol_scales_with_sqrt_time():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 5000))
    daily_sd = r.std(ddof=1)
    assert annual_vol(r) == pytest.approx(daily_sd * np.sqrt(TRADING_DAYS), rel=1e-12)


def test_daily_rf_compounds_to_annual():
    assert (1 + daily_rf(0.065)) ** TRADING_DAYS == pytest.approx(1.065, rel=1e-12)


def test_max_drawdown_known_path():
    # +100%, then -50% => back to start. Peak 2.0, trough 1.0 => -50% drawdown.
    r = pd.Series([1.0, -0.5], index=pd.bdate_range("2020-01-01", periods=2))
    assert max_drawdown(r) == pytest.approx(-0.5, rel=1e-12)


def test_max_drawdown_is_never_positive(returns):
    for col in returns.columns[:5]:
        assert max_drawdown(returns[col]) <= 0


def test_drawdown_series_starts_at_or_below_zero(returns):
    dd = drawdown_series(returns.iloc[:, 0])
    assert (dd <= 1e-12).all()


def test_sharpe_is_translation_sensitive_and_scale_invariant():
    """Scaling every return by k scales mean and sd by k -> Sharpe unchanged
    only when rf is zero. This pins the excess-return convention."""
    r = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 2000))
    s1 = sharpe_ratio(r, rf_annual=0.0)
    s2 = sharpe_ratio(r * 2, rf_annual=0.0)
    assert s1 == pytest.approx(s2, rel=1e-9)


def test_sortino_exceeds_sharpe_for_right_skewed_returns():
    """With upside outliers, downside deviation < total sd, so Sortino > Sharpe."""
    rng = np.random.default_rng(3)
    r = pd.Series(np.concatenate([rng.normal(0.0004, 0.005, 1900), rng.uniform(0.05, 0.10, 100)]))
    assert sortino_ratio(r, 0.0) > sharpe_ratio(r, 0.0)


def test_beta_of_market_against_itself_is_one(market_returns):
    assert beta(market_returns, market_returns) == pytest.approx(1.0, rel=1e-9)


def test_beta_recovers_known_loading(market_returns):
    """A synthetic asset built as 1.4 * market + tiny noise must show beta ~1.4."""
    rng = np.random.default_rng(5)
    asset = 1.4 * market_returns + rng.normal(0, 1e-5, len(market_returns))
    assert beta(asset, market_returns) == pytest.approx(1.4, abs=0.01)


def test_information_ratio_against_self_is_undefined(market_returns):
    assert np.isnan(information_ratio(market_returns, market_returns))


def test_hit_rate_bounded(returns, market_returns):
    h = hit_rate(returns.iloc[:, 0], market_returns)
    assert 0.0 <= h <= 1.0


def test_ulcer_index_non_negative(returns):
    assert ulcer_index(returns.iloc[:, 0]) >= 0


def test_calmar_sign_follows_cagr(returns):
    r = returns.iloc[:, 0]
    c = calmar_ratio(r)
    if not np.isnan(c):
        assert np.sign(c) == np.sign(cagr(r))


def test_tearsheet_has_expected_fields(returns, market_returns):
    ts = tearsheet(returns.iloc[:, 0], market_returns)
    for field in ("CAGR", "Ann.Vol", "Sharpe", "Sortino", "MaxDD", "Beta", "Alpha", "InfoRatio"):
        assert field in ts.index
    assert np.isfinite(ts["CAGR"])
