"""VaR estimator tests, including the sign convention that everything depends on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from riskengine.risk.var import (
    conditional_var,
    cornish_fisher_var,
    historical_var,
    monte_carlo_var,
    parametric_var,
    var_in_rupees,
    var_summary,
)


@pytest.fixture
def normal_returns():
    """Mean 0, sd 1% — the parametric answer is known in closed form."""
    return pd.Series(stats.norm.rvs(0, 0.01, 200_000, random_state=42))


def test_var_is_reported_as_a_positive_loss(normal_returns):
    """The single convention every other module relies on."""
    assert parametric_var(normal_returns, 0.95) > 0
    assert historical_var(normal_returns, 0.95) > 0
    assert conditional_var(normal_returns, 0.95) > 0


def test_parametric_var_matches_gaussian_quantile(normal_returns):
    # VaR_95 for N(0, 0.01) is 1.645 * 0.01
    expected = -stats.norm.ppf(0.05) * 0.01
    assert parametric_var(normal_returns, 0.95) == pytest.approx(expected, rel=0.01)


def test_higher_confidence_means_larger_var(normal_returns):
    assert parametric_var(normal_returns, 0.99) > parametric_var(normal_returns, 0.95)
    assert historical_var(normal_returns, 0.99) > historical_var(normal_returns, 0.95)


def test_cvar_always_at_least_var(normal_returns):
    """Expected shortfall is the mean of the tail beyond VaR, so it cannot be smaller."""
    for c in (0.90, 0.95, 0.99):
        assert conditional_var(normal_returns, c) >= historical_var(normal_returns, c)


def test_historical_and_parametric_agree_under_normality(normal_returns):
    """With genuinely Gaussian data the two methods must nearly coincide.
    Any large gap here would indicate a quantile/sign bug."""
    p = parametric_var(normal_returns, 0.95)
    h = historical_var(normal_returns, 0.95)
    assert abs(p - h) / p < 0.02


def test_parametric_understates_risk_for_fat_tails():
    """The motivating failure: Gaussian VaR under-reports tail risk on fat-tailed
    data, which is why the project backtests VaR instead of trusting it."""
    t_returns = pd.Series(stats.t.rvs(df=3, size=100_000, random_state=1) * 0.004)
    assert historical_var(t_returns, 0.99) > parametric_var(t_returns, 0.99)


def test_monte_carlo_bootstrap_tracks_historical(normal_returns):
    mc = monte_carlo_var(normal_returns, 0.95, n_sims=40_000)
    h = historical_var(normal_returns, 0.95)
    assert abs(mc - h) / h < 0.05


def test_cornish_fisher_equals_parametric_for_normal_data(normal_returns):
    """Zero skew and zero excess kurtosis => the correction terms vanish."""
    cf = cornish_fisher_var(normal_returns, 0.95)
    p = parametric_var(normal_returns, 0.95)
    assert cf == pytest.approx(p, rel=0.02)


def test_var_horizon_scaling(normal_returns):
    v1 = parametric_var(normal_returns, 0.95, horizon=1)
    v10 = parametric_var(normal_returns, 0.95, horizon=10)
    assert v10 == pytest.approx(v1 * np.sqrt(10), rel=1e-9)


def test_var_in_rupees():
    assert var_in_rupees(0.023, 500_000) == pytest.approx(11_500.0)


def test_var_summary_shape(normal_returns):
    s = var_summary(normal_returns.iloc[:5000])
    assert list(s.index) == ["95%", "99%"]
    assert set(s.columns) == {"parametric", "historical", "cornish_fisher", "monte_carlo", "cvar"}


def test_empty_series_raises():
    with pytest.raises(ValueError):
        parametric_var(pd.Series(dtype=float))
