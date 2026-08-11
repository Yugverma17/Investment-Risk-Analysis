"""Volatility estimators.

Plain close-to-close realised vol throws away the intraday range. The
range-based ones here (Parkinson, Garman-Klass, Rogers-Satchell) use the same
number of days but come out a lot less noisy — Parkinson is roughly 5x more
efficient than close-to-close, Garman-Klass around 7x. That's exactly what a
21-day forecasting window needs, which is why these end up as the core
features for the volatility model in `models.vol_forecast`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS


def realised_vol(returns: pd.DataFrame | pd.Series, window: int = 21, annualise: bool = True) -> pd.DataFrame | pd.Series:
    """Rolling close-to-close standard deviation."""
    vol = returns.rolling(window, min_periods=max(5, window // 2)).std()
    return vol * np.sqrt(TRADING_DAYS) if annualise else vol


def ewma_vol(
    returns: pd.DataFrame | pd.Series, lam: float = 0.94, annualise: bool = True
) -> pd.DataFrame | pd.Series:
    """RiskMetrics EWMA volatility.

    lam=0.94 is the RiskMetrics daily standard. This is the benchmark my ML
    model has to beat, and it's a genuinely strong baseline, so beating it
    actually means something.
    """
    var = returns.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    vol = np.sqrt(var)
    return vol * np.sqrt(TRADING_DAYS) if annualise else vol


def parkinson(
    high: pd.DataFrame, low: pd.DataFrame, window: int = 21, annualise: bool = True
) -> pd.DataFrame:
    """Parkinson (1980) high-low range estimator.

    sigma^2 = (1 / (4 ln2)) * mean( ln(H/L)^2 )

    Assumes zero drift and continuous monitoring; it therefore *underestimates*
    when there are large overnight gaps, which is why we also carry
    Garman-Klass (which uses the open) as a feature.
    """
    hl = np.log(high / low) ** 2
    var = hl.rolling(window, min_periods=max(5, window // 2)).mean() / (4 * np.log(2))
    vol = np.sqrt(var)
    return vol * np.sqrt(TRADING_DAYS) if annualise else vol


def garman_klass(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 21,
    annualise: bool = True,
) -> pd.DataFrame:
    """Garman-Klass (1980) OHLC estimator."""
    hl = 0.5 * np.log(high / low) ** 2
    co = (2 * np.log(2) - 1) * np.log(close / open_) ** 2
    var = (hl - co).rolling(window, min_periods=max(5, window // 2)).mean()
    var = var.clip(lower=0)  # the estimator can go slightly negative on quiet days
    vol = np.sqrt(var)
    return vol * np.sqrt(TRADING_DAYS) if annualise else vol


def rogers_satchell(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 21,
    annualise: bool = True,
) -> pd.DataFrame:
    """Rogers-Satchell (1991) — unlike Parkinson/GK, it is drift-robust."""
    term = np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_)
    var = term.rolling(window, min_periods=max(5, window // 2)).mean().clip(lower=0)
    vol = np.sqrt(var)
    return vol * np.sqrt(TRADING_DAYS) if annualise else vol


def downside_deviation(
    returns: pd.DataFrame | pd.Series, window: int = 252, mar: float = 0.0, annualise: bool = True
) -> pd.DataFrame | pd.Series:
    """Semi-deviation below a minimum acceptable return — the Sortino denominator."""
    downside = returns.where(returns < mar, 0.0)
    dd = np.sqrt(downside.pow(2).rolling(window, min_periods=window // 2).mean())
    return dd * np.sqrt(TRADING_DAYS) if annualise else dd


def realised_vol_forward(returns: pd.DataFrame, horizon: int = 21, annualise: bool = True) -> pd.DataFrame:
    """FORWARD realised vol over the next `horizon` days — the ML target.

    Row t holds the vol realised over returns[t+1 .. t+horizon], i.e. strictly
    the future relative to t. Any model consuming this must be trained only on
    features known at t — see `tests/test_leakage.py`, which asserts exactly
    this alignment.
    """
    # rolling(h).std() at row t covers [t-h+1 .. t]; shifting by -h moves the
    # value at row t+h onto row t, which covers [t+1 .. t+h].
    fwd = returns.rolling(horizon, min_periods=horizon).std().shift(-horizon)
    return fwd * np.sqrt(TRADING_DAYS) if annualise else fwd
