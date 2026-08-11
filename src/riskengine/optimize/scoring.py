"""Cross-sectional stock scoring from a training window.

This is the module that turns "which stocks look good right now?" into numbers.
Everything here consumes ONLY the training slice it is handed — the walk-forward
engine is responsible for never handing it future data, and
`tests/test_leakage.py` verifies that contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RISK_FREE_ANNUAL, TRADING_DAYS
from ..risk.var import historical_var


def compute_metrics(
    train_returns: pd.DataFrame,
    market_returns: pd.Series,
    rf_annual: float = RISK_FREE_ANNUAL,
    min_obs: int = 200,
) -> pd.DataFrame:
    """Per-stock risk/return metrics over the training window.

    Stocks with fewer than `min_obs` observations in the window are excluded —
    a stock that listed two months ago has no estimable beta, and letting it
    through with a noisy estimate is how newly-listed names end up dominating
    a score-ranked portfolio.
    """
    valid = train_returns.columns[train_returns.notna().sum() >= min_obs]
    R = train_returns[valid]
    if R.empty:
        return pd.DataFrame()

    mkt = market_returns.reindex(R.index)
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1

    mean_ret = R.mean() * TRADING_DAYS
    vol = R.std(ddof=1) * np.sqrt(TRADING_DAYS)

    excess = R.sub(rf_daily)
    sharpe = (excess.mean() / R.std(ddof=1)) * np.sqrt(TRADING_DAYS)

    var_m = mkt.var(ddof=1)
    betas = R.apply(lambda c: c.cov(mkt) / var_m if var_m > 0 else np.nan)

    downside = R.where(R < 0, 0.0)
    dd = np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = (excess.mean() * TRADING_DAYS) / dd.replace(0, np.nan)

    curve = (1 + R.fillna(0)).cumprod()
    mdd = (curve / curve.cummax() - 1.0).min()

    var95 = R.apply(lambda c: historical_var(c.dropna(), 0.95) if c.notna().sum() > 30 else np.nan)
    var99 = R.apply(lambda c: historical_var(c.dropna(), 0.99) if c.notna().sum() > 30 else np.nan)

    # 12-1 momentum computed inside the window only
    n = len(R)
    mom_window = min(252, n)
    skip = min(21, mom_window // 4)
    cum = (1 + R.fillna(0)).cumprod()
    momentum = cum.iloc[-1 - skip] / cum.iloc[max(0, n - mom_window)] - 1.0

    treynor = (mean_ret - rf_annual) / betas.replace(0, np.nan)

    return pd.DataFrame(
        {
            "mean_return": mean_ret,
            "volatility": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "beta": betas,
            "treynor": treynor,
            "max_dd": mdd,
            "var_95": var95,
            "var_99": var99,
            "momentum": momentum,
            "n_obs": R.notna().sum(),
        }
    )


def zscore(s: pd.Series, winsor: float = 3.0) -> pd.Series:
    """Cross-sectional z-score, winsorised at +/- `winsor` sigma.

    Winsorising matters: without it a single stock that 5x'd sets the scale for
    the entire ranking and every other stock's score collapses toward zero.
    """
    s = s.astype(float)
    sd = s.std(ddof=1)
    if not sd or np.isnan(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / sd).clip(-winsor, winsor)


def composite_score(metrics: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted sum of z-scored metrics.

    Negative weights mean "less is better" (volatility, beta, VaR). Because
    every component is z-scored first, the weights are directly comparable —
    which was not true of the original min-max approach, where a weight of 0.3
    on volatility and 0.3 on Sharpe did not mean equal influence.
    """
    if metrics.empty:
        return pd.Series(dtype=float)

    total = pd.Series(0.0, index=metrics.index)
    used = 0.0
    for col, w in weights.items():
        if col not in metrics.columns:
            continue
        total += w * zscore(metrics[col])
        used += abs(w)
    return total / used if used > 0 else total


def rank_stocks(
    train_returns: pd.DataFrame,
    market_returns: pd.Series,
    weights: dict[str, float],
    rf_annual: float = RISK_FREE_ANNUAL,
) -> tuple[pd.Series, pd.DataFrame]:
    """Convenience wrapper: metrics -> composite score, returned together."""
    metrics = compute_metrics(train_returns, market_returns, rf_annual)
    return composite_score(metrics, weights), metrics
