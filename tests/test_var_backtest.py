"""Kupiec and Christoffersen tests, validated against constructed cases.

Each test builds a breach pattern whose correct verdict is known a priori, so
these check the statistics themselves rather than pinning previous output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskengine.risk.var_backtest import (
    _christoffersen_independence,
    _kupiec,
    evaluate_var,
    rolling_var_forecasts,
    var_backtest_table,
)


def test_kupiec_accepts_correct_breach_count():
    """50 breaches in 1000 days at 95% is exactly the expected rate."""
    stat, p = _kupiec(n=1000, x=50, p=0.05)
    assert stat == pytest.approx(0.0, abs=1e-9)
    assert p > 0.99


def test_kupiec_rejects_far_too_many_breaches():
    """150 breaches where 50 were expected must be rejected decisively."""
    _, p = _kupiec(n=1000, x=150, p=0.05)
    assert p < 0.001


def test_kupiec_rejects_far_too_few_breaches():
    """A VaR that is never breached is not conservative, it is miscalibrated."""
    _, p = _kupiec(n=1000, x=0, p=0.05)
    assert p < 0.001


def test_christoffersen_accepts_independent_breaches():
    """Randomly scattered breaches must not be flagged as clustered."""
    rng = np.random.default_rng(11)
    breaches = rng.random(3000) < 0.05
    _, p = _christoffersen_independence(breaches)
    assert p > 0.05


def test_christoffersen_rejects_clustered_breaches():
    """The failure mode that plain Kupiec cannot see: right number of breaches,
    all landing consecutively."""
    breaches = np.zeros(1000, dtype=bool)
    breaches[200:250] = True  # 50 breaches (the correct count) in one block
    _, p = _christoffersen_independence(breaches)
    assert p < 0.01


def test_evaluate_var_on_a_perfectly_calibrated_forecast():
    """Gaussian returns with the true Gaussian VaR should pass every test."""
    rng = np.random.default_rng(23)
    n = 4000
    r = pd.Series(rng.normal(0, 0.01, n), index=pd.bdate_range("2010-01-01", periods=n))
    true_var = pd.Series(1.6448536 * 0.01, index=r.index)

    res = evaluate_var(r, true_var, 0.95)
    assert res.breach_rate == pytest.approx(0.05, abs=0.012)
    assert res.kupiec_pvalue > 0.05
    assert res.passes


def test_evaluate_var_flags_a_too_optimistic_model():
    """Deliberately halving the VaR must be caught."""
    rng = np.random.default_rng(24)
    n = 3000
    r = pd.Series(rng.normal(0, 0.01, n), index=pd.bdate_range("2010-01-01", periods=n))
    bad_var = pd.Series(0.5 * 1.6448536 * 0.01, index=r.index)

    res = evaluate_var(r, bad_var, 0.95)
    assert res.breach_rate > 0.05
    assert res.kupiec_pvalue < 0.01
    assert not res.passes


def test_rolling_var_forecasts_are_strictly_out_of_sample():
    """The forecast on row t must not depend on the return on row t.

    Constructed check: replace one day's return with an enormous outlier and
    confirm the VaR forecast FOR that same day is unchanged.
    """
    rng = np.random.default_rng(31)
    n = 800
    idx = pd.bdate_range("2015-01-01", periods=n)
    base = pd.Series(rng.normal(0, 0.01, n), index=idx)

    spiked = base.copy()
    target_pos = 600
    spiked.iloc[target_pos] = -0.45

    f_base = rolling_var_forecasts(base, window=252, confidence=0.95)
    f_spiked = rolling_var_forecasts(spiked, window=252, confidence=0.95)

    day = idx[target_pos]
    assert f_base.loc[day] == pytest.approx(f_spiked.loc[day], rel=1e-12)
    # and the very next day's forecast SHOULD change, since the spike is now history
    nxt = idx[target_pos + 1]
    assert f_base.loc[nxt] != pytest.approx(f_spiked.loc[nxt], rel=1e-9)


def test_var_backtest_table_shape(market_returns):
    tbl = var_backtest_table(market_returns.iloc[:1500], window=252)
    assert len(tbl) == 4  # 2 methods x 2 confidence levels
    assert "verdict" in tbl.columns
    assert set(tbl["verdict"]).issubset({"PASS", "REJECT"})
