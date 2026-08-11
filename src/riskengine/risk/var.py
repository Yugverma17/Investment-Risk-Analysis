"""Value-at-Risk and Conditional VaR.

One convention to remember: every function here returns VaR as a POSITIVE
number representing a loss. VaR of 0.023 at 95% means "on 5% of days I'd
expect to lose more than 2.3%." Mixing sign conventions is a classic way VaR
code quietly produces nonsense, so I nailed it down once here and check it in
tests instead of trusting myself to remember.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import SEED, TRADING_DAYS


def _as_array(returns: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if arr.size == 0:
        raise ValueError("empty return series")
    return arr


def parametric_var(returns: pd.Series, confidence: float = 0.95, horizon: int = 1) -> float:
    """Gaussian (variance-covariance) VaR.

    Fast and closed-form, but it assumes normality. Indian equity returns have
    fat tails and negative skew, so this systematically understates tail risk.
    I show that directly rather than just claiming it — see the Kupiec
    backtest in `var_backtest.py`.
    """
    r = _as_array(returns)
    z = stats.norm.ppf(1 - confidence)
    var_1d = -(r.mean() + z * r.std(ddof=1))
    return float(var_1d * np.sqrt(horizon))


def historical_var(returns: pd.Series, confidence: float = 0.95, horizon: int = 1) -> float:
    """Empirical quantile VaR — no distributional assumption.

    Scaling by sqrt(horizon) assumes i.i.d. returns; for horizon=1 (the only
    case we backtest) that assumption is not invoked.
    """
    r = _as_array(returns)
    q = np.quantile(r, 1 - confidence)
    return float(-q * np.sqrt(horizon))


def cornish_fisher_var(returns: pd.Series, confidence: float = 0.95, horizon: int = 1) -> float:
    """Gaussian VaR with a skew/kurtosis correction to the quantile.

    A cheap middle ground: keeps the closed form but adjusts the z-score for the
    third and fourth moments.
    """
    r = _as_array(returns)
    z = stats.norm.ppf(1 - confidence)
    s = stats.skew(r)
    k = stats.kurtosis(r, fisher=True)
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )
    return float(-(r.mean() + z_cf * r.std(ddof=1)) * np.sqrt(horizon))


def monte_carlo_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    n_sims: int = 50_000,
    bootstrap: bool = True,
    seed: int = SEED,
) -> float:
    """Simulated VaR.

    `bootstrap=True` resamples actual historical days, so the simulation
    inherits the real return distribution's fat tails. `bootstrap=False` falls
    back to a Gaussian draw, which is only useful as a sanity check that the
    simulation machinery agrees with `parametric_var`.
    """
    r = _as_array(returns)
    rng = np.random.default_rng(seed)
    if bootstrap:
        draws = rng.choice(r, size=(n_sims, horizon), replace=True)
    else:
        draws = rng.normal(r.mean(), r.std(ddof=1), size=(n_sims, horizon))
    paths = (1 + draws).prod(axis=1) - 1
    return float(-np.quantile(paths, 1 - confidence))


def conditional_var(returns: pd.Series, confidence: float = 0.95, horizon: int = 1) -> float:
    """CVaR / Expected Shortfall: the mean loss given that VaR was breached.

    CVaR is the number a regulator would actually ask for; VaR is the one
    every textbook teaches first. CVaR is sub-additive (diversifying can
    never make it worse) and VaR technically isn't — a real shortcoming of
    VaR, not just a technicality.
    """
    r = _as_array(returns)
    threshold = np.quantile(r, 1 - confidence)
    tail = r[r <= threshold]
    if tail.size == 0:
        return float(-threshold * np.sqrt(horizon))
    return float(-tail.mean() * np.sqrt(horizon))


def var_summary(
    returns: pd.Series, confidences: tuple[float, ...] = (0.95, 0.99), horizon: int = 1
) -> pd.DataFrame:
    """All four VaR estimators plus CVaR, side by side."""
    rows = {}
    for c in confidences:
        rows[f"{c:.0%}"] = {
            "parametric": parametric_var(returns, c, horizon),
            "historical": historical_var(returns, c, horizon),
            "cornish_fisher": cornish_fisher_var(returns, c, horizon),
            "monte_carlo": monte_carlo_var(returns, c, horizon),
            "cvar": conditional_var(returns, c, horizon),
        }
    return pd.DataFrame(rows).T


def var_in_rupees(var_fraction: float, capital: float) -> float:
    """Translate a VaR fraction into money — the only form a real user reads."""
    return float(var_fraction * capital)


def annualise_var(var_1d: float) -> float:
    """Scale a 1-day VaR to 1 year under the i.i.d. square-root rule.

    Stated explicitly because the square-root-of-time rule is *wrong* under
    volatility clustering; it is included for comparability with industry
    reporting, not because it is accurate.
    """
    return float(var_1d * np.sqrt(TRADING_DAYS))
