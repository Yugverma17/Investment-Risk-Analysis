"""Performance and risk-adjusted return metrics.

Everything here takes a daily simple-return Series unless I say otherwise,
and returns annualised numbers. Spelling out these conventions because half
the time two backtests disagree it's just because one used 365 days and the
other used 252.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RISK_FREE_ANNUAL, TRADING_DAYS


def daily_rf(annual: float = RISK_FREE_ANNUAL) -> float:
    """Geometric daily equivalent of an annual risk-free rate."""
    return (1 + annual) ** (1 / TRADING_DAYS) - 1


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate from a daily return stream."""
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    total = float((1 + returns).prod())
    years = len(returns) / TRADING_DAYS
    if total <= 0 or years <= 0:
        return np.nan
    return total ** (1 / years) - 1


def annual_return(returns: pd.Series) -> float:
    """Arithmetic annualised mean — used for Sharpe's numerator."""
    return float(returns.dropna().mean() * TRADING_DAYS)


def annual_vol(returns: pd.Series) -> float:
    return float(returns.dropna().std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    """Annualised Sharpe using excess daily returns.

    One thing my original notebook got sloppy about: subtracting an annual rf
    from an annualised mean is fine, but the denominator needs to be the vol
    of the *excess* returns, not the raw returns. With a constant rf those
    happen to coincide, but I wanted it right in general.
    """
    ex = returns.dropna() - daily_rf(rf_annual)
    sd = ex.std(ddof=1)
    # A near-constant return stream can leave sd a few ULPs above zero rather
    # than exactly 0.0 (mean/sum rounding), which would otherwise blow the
    # ratio up to a meaningless O(1e17) number instead of returning NaN.
    if not np.isfinite(sd) or sd < 1e-10:
        return np.nan
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino_ratio(
    returns: pd.Series, rf_annual: float = RISK_FREE_ANNUAL, mar: float = 0.0
) -> float:
    """Sharpe's downside-only cousin.

    Denominator divides by ALL observations, not just the downside ones —
    that's the standard definition. Dividing only by downside days would make
    a portfolio look artificially better just for losing money less often.
    """
    ex = returns.dropna() - daily_rf(rf_annual)
    downside = np.minimum(ex - mar, 0.0)
    dd = np.sqrt((downside**2).mean())
    if not np.isfinite(dd) or dd < 1e-10:
        return np.nan
    return float(ex.mean() / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the cumulative curve (negative)."""
    curve = (1 + returns.dropna()).cumprod()
    if curve.empty:
        return np.nan
    return float((curve / curve.cummax() - 1.0).min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    curve = (1 + returns.dropna()).cumprod()
    return curve / curve.cummax() - 1.0


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR divided by the absolute max drawdown."""
    mdd = max_drawdown(returns)
    if mdd is None or np.isnan(mdd) or mdd == 0:
        return np.nan
    return float(cagr(returns) / abs(mdd))


def beta(returns: pd.Series, market: pd.Series) -> float:
    df = pd.concat([returns, market], axis=1).dropna()
    if len(df) < 20:
        return np.nan
    var_m = df.iloc[:, 1].var(ddof=1)
    if var_m == 0:
        return np.nan
    return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / var_m)


def alpha(returns: pd.Series, market: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    """Annualised Jensen's alpha."""
    b = beta(returns, market)
    if np.isnan(b):
        return np.nan
    rf = daily_rf(rf_annual)
    df = pd.concat([returns, market], axis=1).dropna()
    return float((df.iloc[:, 0].mean() - rf - b * (df.iloc[:, 1].mean() - rf)) * TRADING_DAYS)


def treynor_ratio(
    returns: pd.Series, market: pd.Series, rf_annual: float = RISK_FREE_ANNUAL
) -> float:
    """Excess return per unit of *systematic* risk."""
    b = beta(returns, market)
    if np.isnan(b) or b == 0:
        return np.nan
    return float((annual_return(returns) - rf_annual) / b)


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    """Active return divided by tracking error."""
    df = pd.concat([returns, benchmark], axis=1).dropna()
    active = df.iloc[:, 0] - df.iloc[:, 1]
    te = active.std(ddof=1)
    if te == 0 or np.isnan(te):
        return np.nan
    return float(active.mean() / te * np.sqrt(TRADING_DAYS))


def hit_rate(returns: pd.Series, benchmark: pd.Series, freq: str = "ME") -> float:
    """Fraction of periods the portfolio beat the benchmark."""
    p = (1 + returns.dropna()).resample(freq).prod() - 1
    b = (1 + benchmark.dropna()).resample(freq).prod() - 1
    df = pd.concat([p, b], axis=1).dropna()
    if df.empty:
        return np.nan
    return float((df.iloc[:, 0] > df.iloc[:, 1]).mean())


def ulcer_index(returns: pd.Series) -> float:
    """RMS drawdown — penalises deep *and* long drawdowns, unlike max DD."""
    dd = drawdown_series(returns)
    if dd.empty:
        return np.nan
    return float(np.sqrt((dd**2).mean()))


def tearsheet(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    rf_annual: float = RISK_FREE_ANNUAL,
    name: str = "portfolio",
) -> pd.Series:
    """The full summary row used in every results table in this project."""
    out = {
        "CAGR": cagr(returns),
        "Ann.Return": annual_return(returns),
        "Ann.Vol": annual_vol(returns),
        "Sharpe": sharpe_ratio(returns, rf_annual),
        "Sortino": sortino_ratio(returns, rf_annual),
        "MaxDD": max_drawdown(returns),
        "Calmar": calmar_ratio(returns),
        "Ulcer": ulcer_index(returns),
        "Skew": float(returns.dropna().skew()),
        "Kurtosis": float(returns.dropna().kurtosis()),
    }
    if benchmark is not None:
        out.update(
            {
                "Beta": beta(returns, benchmark),
                "Alpha": alpha(returns, benchmark, rf_annual),
                "Treynor": treynor_ratio(returns, benchmark, rf_annual),
                "InfoRatio": information_ratio(returns, benchmark),
                "HitRate": hit_rate(returns, benchmark),
            }
        )
    return pd.Series(out, name=name)
