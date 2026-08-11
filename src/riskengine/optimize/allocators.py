"""Portfolio allocators.

Every allocator has the same signature and returns a weight Series summing to 1
over the assets it was given. That uniformity is what makes the horse race in
`backtest.walk_forward` a fair comparison rather than an apples-to-oranges one.

Optimisation uses scipy SLSQP rather than a dedicated convex solver. For the
problem sizes here (20-40 assets, linear constraints, quadratic objective) SLSQP
converges reliably and removes a heavyweight dependency — see
docs/decisions/ADR-002-slsqp-over-cvxpy.md.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..config import RISK_FREE_ANNUAL, TRADING_DAYS
from ..risk.covariance import ledoit_wolf_cov, nearest_psd
from .constraints import Constraints, apply_caps

__all__ = [
    "equal_weight",
    "inverse_volatility",
    "min_variance",
    "risk_parity",
    "max_sharpe",
    "score_based",
    "ALLOCATORS",
]


def _clean(weights: np.ndarray, assets: list[str]) -> pd.Series:
    w = pd.Series(weights, index=assets).clip(lower=0)
    w[w < 1e-6] = 0.0  # scrub optimiser dust so turnover isn't inflated by noise
    total = w.sum()
    if total <= 0:
        return pd.Series(1.0 / len(assets), index=assets)
    return w / total


def _sum_to_one():
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _sector_constraints(cons: Constraints, assets: list[str]) -> list[dict]:
    if not cons.sector_map or cons.max_sector_weight >= 1.0:
        return []
    S, _ = cons.sector_matrix(assets)
    return [{"type": "ineq", "fun": lambda w, S=S: cons.max_sector_weight - S @ w}]


# --------------------------------------------------------------------------
# Closed-form allocators
# --------------------------------------------------------------------------


def equal_weight(returns: pd.DataFrame, cons: Constraints, **_) -> pd.Series:
    """1/N.

    The baseline that matters. DeMiguel, Garlappi & Uppal (2009) showed 1/N beats
    most optimised portfolios out of sample once estimation error is accounted
    for. If a strategy in this project cannot beat 1/N, that is the finding.
    """
    assets = list(returns.columns)
    return apply_caps(pd.Series(1.0, index=assets), cons.effective_max_weight(len(assets)))


def inverse_volatility(returns: pd.DataFrame, cons: Constraints, **_) -> pd.Series:
    """Weight proportional to 1/sigma — risk parity's diagonal approximation.

    Ignores correlations entirely, which is a real limitation, but it needs no
    matrix inversion and is therefore immune to the ill-conditioning that
    wrecks min-variance.
    """
    assets = list(returns.columns)
    vol = returns.std(ddof=1).replace(0, np.nan)
    inv = (1.0 / vol).fillna(0.0)
    if inv.sum() <= 0:
        return equal_weight(returns, cons)
    return apply_caps(inv, cons.effective_max_weight(len(assets)))


# --------------------------------------------------------------------------
# Optimised allocators
# --------------------------------------------------------------------------


def min_variance(
    returns: pd.DataFrame, cons: Constraints, cov: pd.DataFrame | None = None, **_
) -> pd.Series:
    """Global minimum-variance portfolio.

    Uses shrunk covariance by default. With the raw sample covariance this
    allocator produces extreme, unstable weights — that failure is demonstrated
    deliberately in notebook 03 rather than hidden.
    """
    assets = list(returns.columns)
    n = len(assets)
    if n == 1:
        return pd.Series([1.0], index=assets)

    Sigma = nearest_psd(cov if cov is not None else ledoit_wolf_cov(returns)).to_numpy()
    mw = cons.effective_max_weight(n)

    def obj(w):
        return float(w @ Sigma @ w)

    def grad(w):
        return 2 * Sigma @ w

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            obj,
            x0=np.full(n, 1.0 / n),
            jac=grad,
            method="SLSQP",
            bounds=[(cons.min_weight, mw)] * n,
            constraints=[_sum_to_one(), *_sector_constraints(cons, assets)],
            options={"maxiter": 300, "ftol": 1e-12},
        )
    return _clean(res.x if res.success else np.full(n, 1.0 / n), assets)


def risk_parity(
    returns: pd.DataFrame, cons: Constraints, cov: pd.DataFrame | None = None, **_
) -> pd.Series:
    """Equal risk contribution.

    Each asset contributes the same share of total portfolio variance:
        RC_i = w_i * (Sigma w)_i / (w' Sigma w)  ->  1/n for all i

    Unlike inverse-vol, this accounts for correlations, so two highly correlated
    banks share one bank's worth of risk budget instead of two.
    """
    assets = list(returns.columns)
    n = len(assets)
    if n == 1:
        return pd.Series([1.0], index=assets)

    Sigma = nearest_psd(cov if cov is not None else ledoit_wolf_cov(returns)).to_numpy()
    target = 1.0 / n
    mw = cons.effective_max_weight(n)

    def obj(w):
        port_var = float(w @ Sigma @ w)
        if port_var <= 0:
            return 1e6
        rc = w * (Sigma @ w) / port_var
        return float(np.sum((rc - target) ** 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            obj,
            x0=np.full(n, 1.0 / n),
            method="SLSQP",
            bounds=[(max(cons.min_weight, 1e-4), mw)] * n,
            constraints=[_sum_to_one(), *_sector_constraints(cons, assets)],
            options={"maxiter": 500, "ftol": 1e-12},
        )
    return _clean(res.x if res.success else np.full(n, 1.0 / n), assets)


def max_sharpe(
    returns: pd.DataFrame,
    cons: Constraints,
    cov: pd.DataFrame | None = None,
    expected_returns: pd.Series | None = None,
    rf_annual: float = RISK_FREE_ANNUAL,
    **_,
) -> pd.Series:
    """Tangency portfolio — classic Markowitz.

    Included precisely because it is the textbook answer and it tends to
    disappoint out of sample: it is maximally sensitive to expected-return
    estimates, which are the noisiest inputs in finance. Showing that
    empirically is more valuable than omitting it.
    """
    assets = list(returns.columns)
    n = len(assets)
    if n == 1:
        return pd.Series([1.0], index=assets)

    Sigma = nearest_psd(cov if cov is not None else ledoit_wolf_cov(returns)).to_numpy()
    mu = (
        expected_returns.reindex(assets).to_numpy()
        if expected_returns is not None
        else (returns.mean() * TRADING_DAYS).to_numpy()
    )
    mu = np.nan_to_num(mu, nan=0.0)
    mw = cons.effective_max_weight(n)

    def neg_sharpe(w):
        port_ret = float(w @ mu) - rf_annual
        port_vol = float(np.sqrt(max(w @ Sigma @ w, 1e-12)))
        return -port_ret / port_vol

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            neg_sharpe,
            x0=np.full(n, 1.0 / n),
            method="SLSQP",
            bounds=[(cons.min_weight, mw)] * n,
            constraints=[_sum_to_one(), *_sector_constraints(cons, assets)],
            options={"maxiter": 500, "ftol": 1e-10},
        )
    return _clean(res.x if res.success else np.full(n, 1.0 / n), assets)


# --------------------------------------------------------------------------
# The original notebook's approach, rebuilt
# --------------------------------------------------------------------------


def score_based(
    returns: pd.DataFrame,
    cons: Constraints,
    scores: pd.Series | None = None,
    softmax_temp: float = 1.0,
    **_,
) -> pd.Series:
    """Weight proportional to a cross-sectional score.

    This is the original project's method, with three corrections:

    1. Min-max normalisation is replaced by cross-sectional z-scores. Min-max is
       destroyed by a single outlier — one stock with a freak Sharpe compresses
       every other stock into a narrow band near zero.
    2. Weights come from a softmax over z-scores rather than from normalised
       scores directly, so a stock with a below-average score gets a small
       weight rather than a *negative-then-clipped-to-zero* weight.
    3. Concentration caps are enforced afterwards.
    """
    assets = list(returns.columns)
    if scores is None:
        return equal_weight(returns, cons)

    s = scores.reindex(assets).astype(float)
    if s.notna().sum() == 0:
        return equal_weight(returns, cons)
    s = s.fillna(s.mean())

    sd = s.std(ddof=1)
    z = (s - s.mean()) / sd if sd and sd > 0 else s * 0.0
    w = np.exp(np.clip(z / max(softmax_temp, 1e-6), -10, 10))
    return apply_caps(pd.Series(w, index=assets), cons.effective_max_weight(len(assets)))


ALLOCATORS = {
    "equal_weight": equal_weight,
    "inverse_vol": inverse_volatility,
    "min_variance": min_variance,
    "risk_parity": risk_parity,
    "max_sharpe": max_sharpe,
    "score_based": score_based,
}
