"""Cross-sectional scoring: z-scores, composite scoring, metric computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskengine.optimize.scoring import composite_score, compute_metrics, zscore


def test_zscore_has_zero_mean_and_unit_std_before_winsorising():
    s = pd.Series(np.random.default_rng(1).normal(5, 2, 500))
    z = zscore(s, winsor=100)  # effectively no clipping
    assert z.mean() == pytest.approx(0.0, abs=1e-9)
    assert z.std(ddof=1) == pytest.approx(1.0, rel=1e-9)


def test_zscore_winsorises_outliers():
    s = pd.Series([0.0, 1.0, 2.0, 1000.0])
    z = zscore(s, winsor=2.0)
    assert z.max() <= 2.0 + 1e-9
    assert z.min() >= -2.0 - 1e-9


def test_zscore_constant_series_is_all_zero():
    s = pd.Series([5.0] * 10)
    z = zscore(s)
    assert (z == 0).all()


def test_composite_score_orders_by_positive_weight():
    metrics = pd.DataFrame({"sharpe": [0.5, 1.0, 1.5], "volatility": [0.2, 0.2, 0.2]}, index=list("ABC"))
    score = composite_score(metrics, {"sharpe": 1.0})
    assert score["C"] > score["B"] > score["A"]


def test_composite_score_flips_sign_for_negative_weight():
    metrics = pd.DataFrame({"volatility": [0.1, 0.2, 0.3]}, index=list("ABC"))
    score = composite_score(metrics, {"volatility": -1.0})
    # lower volatility should score higher when the weight is negative
    assert score["A"] > score["B"] > score["C"]


def test_composite_score_ignores_unknown_columns():
    metrics = pd.DataFrame({"sharpe": [1.0, 2.0]}, index=list("AB"))
    score = composite_score(metrics, {"sharpe": 1.0, "nonexistent_col": 5.0})
    assert score["B"] > score["A"]


def test_composite_score_empty_metrics_returns_empty():
    assert composite_score(pd.DataFrame(), {"sharpe": 1.0}).empty


def test_compute_metrics_excludes_short_history_stocks(returns, market_returns):
    trimmed = returns.copy()
    trimmed.iloc[:-50, trimmed.columns.get_loc("STK00.NS")] = np.nan  # only 50 obs
    m = compute_metrics(trimmed, market_returns, min_obs=200)
    assert "STK00.NS" not in m.index
    assert "STK01.NS" in m.index


def test_compute_metrics_beta_against_self_composed_market(market_returns, returns):
    """A stock built as exactly 1x the market (zero noise) must show beta ~1."""
    df = returns.copy()
    df["PURE_MKT.NS"] = market_returns
    m = compute_metrics(df, market_returns, min_obs=100)
    assert m.loc["PURE_MKT.NS", "beta"] == pytest.approx(1.0, abs=0.02)


def test_compute_metrics_var_is_positive(returns, market_returns):
    m = compute_metrics(returns, market_returns)
    valid = m["var_95"].dropna()
    assert (valid > 0).all()
