"""Covariance estimator tests — including the ill-conditioning that motivates shrinkage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskengine.config import TRADING_DAYS
from riskengine.risk.covariance import (
    ESTIMATORS,
    condition_number,
    constant_correlation_cov,
    cov_from_vols,
    ewma_cov,
    ledoit_wolf_cov,
    nearest_psd,
    sample_cov,
)


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_every_estimator_is_symmetric_and_psd(name, returns):
    cov = ESTIMATORS[name](returns.iloc[-500:])
    A = cov.to_numpy()
    assert np.allclose(A, A.T, atol=1e-12)
    eigs = np.linalg.eigvalsh(A)
    assert eigs.min() > -1e-10


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_estimator_shape_and_labels(name, returns):
    cov = ESTIMATORS[name](returns.iloc[-500:])
    assert cov.shape == (returns.shape[1], returns.shape[1])
    assert list(cov.index) == list(returns.columns)


def test_shrinkage_improves_conditioning_when_assets_outnumber_observations(returns):
    """The motivating case: 20 assets, only 25 days of data. The sample matrix is
    near-singular; shrinkage must fix it by orders of magnitude."""
    tiny = returns.iloc[-25:]
    cond_sample = condition_number(sample_cov(tiny))
    cond_shrunk = condition_number(ledoit_wolf_cov(tiny))
    assert cond_shrunk < cond_sample / 10


def test_sample_covariance_is_singular_with_too_few_observations(returns):
    """T < N means the sample covariance has zero eigenvalues and cannot be
    trusted by any optimiser that inverts it."""
    tiny = returns.iloc[-10:]  # 10 obs, 20 assets
    eigs = np.linalg.eigvalsh(sample_cov(tiny).to_numpy())
    assert (eigs < 1e-12).sum() >= tiny.shape[1] - tiny.shape[0]


def test_annualisation_scales_by_trading_days(returns):
    daily = sample_cov(returns, annualise=False)
    annual = sample_cov(returns, annualise=True)
    assert np.allclose(annual.to_numpy(), daily.to_numpy() * TRADING_DAYS)


def test_constant_correlation_preserves_individual_variances(returns):
    sub = returns.iloc[-500:]
    cov = constant_correlation_cov(sub, annualise=False)
    np.testing.assert_allclose(np.diag(cov.to_numpy()), sub.var(ddof=1).to_numpy(), rtol=1e-9)


def test_constant_correlation_has_one_off_diagonal_correlation(returns):
    cov = constant_correlation_cov(returns.iloc[-500:], annualise=False).to_numpy()
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    off = corr[~np.eye(len(corr), dtype=bool)]
    assert off.std() == pytest.approx(0.0, abs=1e-10)


def test_ewma_weights_recent_data_more_heavily():
    """A vol regime shift late in the sample should move EWMA more than sample cov."""
    n = 1000
    rng = np.random.default_rng(17)
    calm = rng.normal(0, 0.005, (n // 2, 2))
    wild = rng.normal(0, 0.030, (n // 2, 2))
    df = pd.DataFrame(np.vstack([calm, wild]), columns=["A", "B"], index=pd.bdate_range("2015-01-01", periods=n))

    s = sample_cov(df, annualise=False).iloc[0, 0]
    e = ewma_cov(df, lam=0.94, annualise=False).iloc[0, 0]
    assert e > s


def test_cov_from_vols_installs_the_forecast_on_the_diagonal(returns):
    """The vol-forecast injection must replace variances exactly while leaving
    correlations alone."""
    sub = returns.iloc[-500:]
    target = pd.Series(0.25, index=sub.columns)  # 25% annualised for everything

    cov = cov_from_vols(sub, target)
    diag_vol = np.sqrt(np.diag(cov.to_numpy()))
    np.testing.assert_allclose(diag_vol, 0.25, rtol=1e-8)

    # correlations should be unchanged from the shrunk historical estimate
    base = ledoit_wolf_cov(sub)
    sd_b = np.sqrt(np.diag(base.to_numpy()))
    corr_b = base.to_numpy() / np.outer(sd_b, sd_b)
    corr_new = cov.to_numpy() / np.outer(diag_vol, diag_vol)
    np.testing.assert_allclose(corr_new, corr_b, atol=1e-8)


def test_cov_from_vols_falls_back_when_a_forecast_is_missing(returns):
    sub = returns.iloc[-500:]
    partial = pd.Series(0.30, index=sub.columns[:5])
    cov = cov_from_vols(sub, partial.reindex(sub.columns))
    assert np.isfinite(cov.to_numpy()).all()
    assert np.linalg.eigvalsh(cov.to_numpy()).min() > -1e-10


def test_nearest_psd_repairs_a_non_psd_matrix():
    bad = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=["A", "B"], columns=["A", "B"])
    assert np.linalg.eigvalsh(bad.to_numpy()).min() < 0
    fixed = nearest_psd(bad)
    assert np.linalg.eigvalsh(fixed.to_numpy()).min() >= -1e-12


def test_condition_number_of_identity_is_one():
    eye = pd.DataFrame(np.eye(5))
    assert condition_number(eye) == pytest.approx(1.0)
