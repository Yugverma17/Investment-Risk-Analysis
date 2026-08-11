"""Market-relative features: beta, alpha, downside beta, momentum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS


def rolling_beta(
    returns: pd.DataFrame, market: pd.Series, window: int = 252, min_periods: int | None = None
) -> pd.DataFrame:
    """Rolling OLS beta of each stock against the market.

    beta = cov(r_i, r_m) / var(r_m)

    A 252-day window is used rather than the notebook's full-sample beta because
    beta is demonstrably non-stationary in Indian equities — bank betas in
    2020-21 look nothing like 2017.
    """
    min_periods = min_periods or window // 2
    mkt = market.reindex(returns.index)
    cov = returns.rolling(window, min_periods=min_periods).cov(mkt)
    var = mkt.rolling(window, min_periods=min_periods).var()
    return cov.div(var, axis=0)


def rolling_alpha(
    returns: pd.DataFrame,
    market: pd.Series,
    beta: pd.DataFrame,
    window: int = 252,
    rf_daily: float = 0.0,
) -> pd.DataFrame:
    """Annualised Jensen's alpha: mean(r_i - rf) - beta * mean(r_m - rf)."""
    min_periods = window // 2
    mkt = market.reindex(returns.index)
    mean_i = returns.rolling(window, min_periods=min_periods).mean() - rf_daily
    mean_m = mkt.rolling(window, min_periods=min_periods).mean() - rf_daily
    alpha = mean_i.sub(beta.mul(mean_m, axis=0))
    return alpha * TRADING_DAYS


def downside_beta(
    returns: pd.DataFrame, market: pd.Series, window: int = 252, threshold: float = 0.0
) -> pd.DataFrame:
    """Beta estimated only on days the market fell below `threshold`.

    This is the number a risk-averse investor actually cares about: how much do
    I lose when the market drops? A stock can have beta 1.0 overall but
    downside beta 1.4, and the ordinary beta hides that entirely.
    """
    mkt = market.reindex(returns.index)
    mask = mkt < threshold
    out = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    idx = returns.index
    for end in range(window, len(idx) + 1):
        sl = slice(end - window, end)
        m = mask.iloc[sl]
        if m.sum() < 20:  # too few down days to estimate anything
            continue
        rm = mkt.iloc[sl][m.values]
        ri = returns.iloc[sl][m.values]
        var_m = rm.var()
        if var_m == 0 or np.isnan(var_m):
            continue
        out.iloc[end - 1] = ri.apply(lambda c, rm=rm: c.cov(rm)).values / var_m
    return out


def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Classic 12-1 momentum: return over `lookback` days, skipping the last `skip`.

    The skip removes the well-documented short-term reversal effect that
    contaminates raw 12-month momentum.
    """
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def max_drawdown_rolling(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Rolling maximum drawdown over a trailing window (negative number)."""
    roll_max = prices.rolling(window, min_periods=window // 2).max()
    return (prices / roll_max - 1.0).rolling(window, min_periods=window // 2).min()


def turnover_liquidity(volume: pd.DataFrame, close: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Average daily traded value (rupees) — a liquidity screen."""
    return (volume * close).rolling(window, min_periods=window // 2).mean()
