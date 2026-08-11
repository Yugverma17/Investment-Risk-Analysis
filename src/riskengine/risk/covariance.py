"""Covariance estimation for portfolio optimisation.

With around 25 selected stocks per rebalance and 756 daily observations (3
years), the sample covariance matrix has a few hundred free parameters
estimated from around 19,000 numbers. It's technically invertible but still
noticeably ill-conditioned — the smallest eigenvalues are mostly noise, and a
mean-variance optimizer will pile weight into exactly those directions
because they look like free risk reduction.

This is basically THE reason naive Markowitz portfolios blow up out of
sample. Swapping sample covariance for Ledoit-Wolf shrinkage was probably the
single highest-leverage change I made in the whole optimization stack.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import OAS, LedoitWolf

from ..config import TRADING_DAYS


def sample_cov(returns: pd.DataFrame, annualise: bool = True) -> pd.DataFrame:
    cov = returns.dropna(how="all").cov()
    return cov * TRADING_DAYS if annualise else cov


def ledoit_wolf_cov(returns: pd.DataFrame, annualise: bool = True) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage toward a scaled identity matrix.

    The shrinkage intensity is chosen analytically to minimise expected squared
    Frobenius error — there is no hyperparameter to tune, which is a large part
    of why it is the default here.
    """
    X = returns.dropna()
    if X.shape[0] < 2 or X.shape[1] < 2:
        return sample_cov(returns, annualise)
    lw = LedoitWolf().fit(X.to_numpy())
    cov = pd.DataFrame(lw.covariance_, index=X.columns, columns=X.columns)
    return cov * TRADING_DAYS if annualise else cov


def oas_cov(returns: pd.DataFrame, annualise: bool = True) -> pd.DataFrame:
    """Oracle Approximating Shrinkage — usually shrinks harder than Ledoit-Wolf."""
    X = returns.dropna()
    if X.shape[0] < 2 or X.shape[1] < 2:
        return sample_cov(returns, annualise)
    oas = OAS().fit(X.to_numpy())
    cov = pd.DataFrame(oas.covariance_, index=X.columns, columns=X.columns)
    return cov * TRADING_DAYS if annualise else cov


def constant_correlation_cov(returns: pd.DataFrame, annualise: bool = True) -> pd.DataFrame:
    """Elton-Gruber: keep each stock's own vol, replace all pairwise correlations
    with their average.

    Crude, but remarkably hard to beat out of sample, and it makes an honest
    baseline for "does the fancy estimator actually help?".
    """
    X = returns.dropna()
    corr = X.corr()
    n = len(corr)
    if n < 2:
        return sample_cov(returns, annualise)
    off = corr.to_numpy()[~np.eye(n, dtype=bool)]
    rho = float(np.mean(off))
    target = np.full((n, n), rho)
    np.fill_diagonal(target, 1.0)
    sd = X.std(ddof=1).to_numpy()
    cov = pd.DataFrame(target * np.outer(sd, sd), index=corr.index, columns=corr.columns)
    return cov * TRADING_DAYS if annualise else cov


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94, annualise: bool = True) -> pd.DataFrame:
    """Exponentially weighted covariance — recent days matter more.

    Useful when the correlation regime has just shifted (e.g. post-COVID), at
    the cost of a much smaller effective sample size.
    """
    X = returns.dropna().to_numpy()
    n_obs, n_assets = X.shape
    if n_obs < 2:
        return sample_cov(returns, annualise)
    weights = (1 - lam) * lam ** np.arange(n_obs - 1, -1, -1)
    weights /= weights.sum()
    mu = weights @ X
    Xc = X - mu
    cov = (Xc * weights[:, None]).T @ Xc
    cov = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    return cov * TRADING_DAYS if annualise else cov


def condition_number(cov: pd.DataFrame) -> float:
    """Ratio of largest to smallest eigenvalue — the ill-conditioning diagnostic.

    Above ~10^4 the matrix inverse is numerically meaningless for optimisation.
    """
    eig = np.linalg.eigvalsh(cov.to_numpy())
    eig = eig[eig > 0]
    if eig.size == 0:
        return np.inf
    return float(eig.max() / eig.min())


def nearest_psd(cov: pd.DataFrame, epsilon: float = 1e-10) -> pd.DataFrame:
    """Clip negative eigenvalues so the matrix is positive semi-definite.

    Needed after ffill/alignment can introduce tiny numerical asymmetries.
    """
    A = cov.to_numpy()
    A = (A + A.T) / 2
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, epsilon, None)
    return pd.DataFrame(vecs @ np.diag(vals) @ vecs.T, index=cov.index, columns=cov.columns)


def cov_from_vols(
    returns: pd.DataFrame, vols_annual: pd.Series, shrink: bool = True
) -> pd.DataFrame:
    """Rebuild a covariance matrix from FORECAST volatilities + historical correlations.

    Sigma = D * C * D, where D = diag(forecast vol) and C is the (shrunk)
    historical correlation matrix.

    This is how the volatility model actually feeds into the portfolio:
    correlations are relatively stable and can be estimated from history,
    while volatility is the part that actually moves and is worth
    forecasting. Only touching the diagonal isolates exactly what the model
    contributes — any change in performance is attributable to the vol
    forecast and nothing else.
    """
    base = ledoit_wolf_cov(returns, annualise=True) if shrink else sample_cov(returns)
    sd = np.sqrt(np.diag(base.to_numpy()))
    sd[sd <= 0] = np.nan
    corr = base.to_numpy() / np.outer(sd, sd)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    v = vols_annual.reindex(base.index).astype(float)
    v = v.fillna(pd.Series(sd, index=base.index)).clip(lower=1e-4).to_numpy()
    cov = corr * np.outer(v, v)
    return nearest_psd(pd.DataFrame(cov, index=base.index, columns=base.columns))


ESTIMATORS = {
    "sample": sample_cov,
    "ledoit_wolf": ledoit_wolf_cov,
    "oas": oas_cov,
    "constant_corr": constant_correlation_cov,
    "ewma": ewma_cov,
}
