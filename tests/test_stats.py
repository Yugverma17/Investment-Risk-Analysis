"""Statistical significance tooling: bootstrap CI, PSR/DSR, HAC t-stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskengine.backtest.stats import (
    bootstrap_sharpe_difference,
    deflated_sharpe_ratio,
    newey_west_tstat,
    probabilistic_sharpe_ratio,
    stationary_bootstrap_indices,
)


def test_stationary_bootstrap_indices_are_in_range():
    rng = np.random.default_rng(1)
    idx = stationary_bootstrap_indices(500, mean_block=20, rng=rng)
    assert idx.min() >= 0
    assert idx.max() < 500
    assert len(idx) == 500


def test_bootstrap_ci_is_centred_near_zero_for_identical_series():
    """A strategy compared against itself must show a Sharpe difference of
    (near) zero, with a CI that comfortably straddles zero."""
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2018-01-01", periods=1500)
    r = pd.Series(rng.normal(0.0005, 0.012, len(idx)), index=idx)
    out = bootstrap_sharpe_difference(r, r, n_boot=500)
    assert out["diff"] == pytest.approx(0.0, abs=1e-9)
    assert out["ci_low"] <= 0 <= out["ci_high"]


def test_bootstrap_ci_detects_a_real_difference():
    """A strategy with a genuinely and substantially higher mean return should
    show a positive Sharpe difference whose 95% CI excludes zero."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", periods=2000)
    base = rng.normal(0.0002, 0.011, len(idx))
    strategy = pd.Series(base + 0.0009, index=idx)  # large, persistent outperformance
    benchmark = pd.Series(base, index=idx)

    out = bootstrap_sharpe_difference(strategy, benchmark, n_boot=800)
    assert out["diff"] > 0
    assert out["ci_low"] > 0


def test_bootstrap_handles_short_series_gracefully():
    r = pd.Series([0.01, -0.01, 0.02])
    out = bootstrap_sharpe_difference(r, r)
    assert np.isnan(out["diff"])


def test_psr_is_high_for_a_strong_consistent_strategy():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2015-01-01", periods=2500)
    r = pd.Series(rng.normal(0.0012, 0.010, len(idx)), index=idx)
    psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    assert psr > 0.9


def test_psr_is_inconclusive_for_a_flat_strategy():
    """True Sharpe is 0, so PSR should land somewhere in the middle of (0, 1)
    rather than saturating near either extreme — the single-draw sample Sharpe
    is noisy, so this checks the qualitative claim, not a tight point estimate."""
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-01-01", periods=1000)
    r = pd.Series(rng.normal(0.0, 0.01, len(idx)), index=idx)
    psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    assert 0.05 < psr < 0.95


def test_deflated_sharpe_falls_as_trial_count_rises():
    """The more strategies you tried, the harder it should be to call the
    winner's Sharpe genuinely significant."""
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    r = pd.Series(rng.normal(0.0006, 0.011, len(idx)), index=idx)

    dsr_1 = deflated_sharpe_ratio(r, n_trials=1)
    dsr_20 = deflated_sharpe_ratio(r, n_trials=20)
    assert dsr_20 <= dsr_1 + 1e-9


def test_newey_west_tstat_matches_naive_tstat_under_no_autocorrelation():
    """With i.i.d. data, the HAC correction should barely move the t-stat."""
    rng = np.random.default_rng(7)
    x = pd.Series(rng.normal(0.001, 0.01, 3000))
    t_hac, _ = newey_west_tstat(x, lags=5)

    naive_t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert t_hac == pytest.approx(naive_t, rel=0.15)


def test_newey_west_shrinks_tstat_under_positive_autocorrelation():
    """Positively autocorrelated noise (like overlapping-window artefacts)
    must produce a SMALLER |t-stat| under HAC than under the naive formula —
    that shrinkage is the entire reason to use it."""
    rng = np.random.default_rng(8)
    n = 3000
    eps = rng.normal(0, 0.01, n)
    x = np.zeros(n)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = 0.6 * x[i - 1] + eps[i]
    x = pd.Series(x + 0.0005)

    t_hac, _ = newey_west_tstat(x, lags=10)
    naive_t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(t_hac) < abs(naive_t)
