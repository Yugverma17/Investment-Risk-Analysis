"""Feature panel construction for the volatility model.

Every feature here is computable from information available at time `t`. The
target is realised volatility over `(t, t+21]`. The separation is enforced
structurally — features come from `.rolling()` (backward-looking) and the target
from a negative `.shift()` — and asserted in `tests/test_leakage.py`.

Sampling
--------
Observations are taken at MONTH-ENDS, not daily. Daily sampling of a 21-day
forward target produces massively overlapping windows: consecutive observations
share 20 of 21 days. That does not add information, it just inflates the apparent
sample size and makes every error bar look 5x tighter than it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.volatility import (
    ewma_vol,
    garman_klass,
    parkinson,
    realised_vol,
    realised_vol_forward,
    rogers_satchell,
)

TARGET = "fwd_vol_21"
HORIZON = 21


def _panel(ohlcv: pd.DataFrame, field: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    return (
        ohlcv.pivot(index="date", columns="ticker", values=field)
        .reindex(calendar)
        .ffill(limit=5)
    )


def build_feature_panel(
    ohlcv: pd.DataFrame,
    close: pd.DataFrame,
    market_close: pd.Series,
    vix: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a long-format (date, ticker) feature + target table.

    Returns a DataFrame with columns: date, ticker, <features...>, fwd_vol_21.
    """
    calendar = pd.DatetimeIndex(close.index)
    tickers = list(close.columns)

    op = _panel(ohlcv, "open", calendar)[tickers]
    hi = _panel(ohlcv, "high", calendar)[tickers]
    lo = _panel(ohlcv, "low", calendar)[tickers]
    cl = close[tickers]
    vol = _panel(ohlcv, "volume", calendar)[tickers]

    rets = cl.pct_change()
    mkt_ret = market_close.reindex(calendar).ffill().pct_change()

    feats: dict[str, pd.DataFrame] = {}

    # --- close-to-close realised vol at several horizons -------------------
    for w in (5, 21, 63, 126, 252):
        feats[f"rv_{w}"] = realised_vol(rets, w)

    # --- range-based estimators (the efficiency win) -----------------------
    feats["parkinson_21"] = parkinson(hi, lo, 21)
    feats["parkinson_63"] = parkinson(hi, lo, 63)
    feats["gk_21"] = garman_klass(op, hi, lo, cl, 21)
    feats["rs_21"] = rogers_satchell(op, hi, lo, cl, 21)

    # --- EWMA (the baseline the model must beat) ---------------------------
    feats["ewma_94"] = ewma_vol(rets, 0.94)
    feats["ewma_97"] = ewma_vol(rets, 0.97)

    # --- vol dynamics -------------------------------------------------------
    rv21 = feats["rv_21"]
    feats["vol_of_vol"] = rv21.rolling(63, min_periods=30).std()
    feats["vol_ratio_5_63"] = feats["rv_5"] / feats["rv_63"].replace(0, np.nan)
    feats["vol_ratio_21_252"] = rv21 / feats["rv_252"].replace(0, np.nan)
    feats["vol_trend_21"] = rv21 - rv21.shift(21)

    # --- leverage effect: falling prices predict rising vol ----------------
    feats["ret_21"] = cl / cl.shift(21) - 1
    feats["ret_63"] = cl / cl.shift(63) - 1
    feats["neg_ret_21"] = feats["ret_21"].clip(upper=0)
    feats["abs_ret_5"] = rets.abs().rolling(5, min_periods=3).mean()
    feats["downside_frac_21"] = (rets < 0).rolling(21, min_periods=10).mean()

    # --- drawdown state -----------------------------------------------------
    roll_max = cl.rolling(252, min_periods=60).max()
    feats["drawdown"] = cl / roll_max - 1.0

    # --- liquidity / activity ----------------------------------------------
    turnover = (vol * cl).replace(0, np.nan)
    log_to = np.log(turnover)
    feats["log_turnover"] = log_to.rolling(21, min_periods=10).mean()
    feats["turnover_z"] = (
        log_to.rolling(21, min_periods=10).mean() - log_to.rolling(252, min_periods=60).mean()
    ) / log_to.rolling(252, min_periods=60).std()

    # --- market-wide state (same value for every stock on a date) ----------
    mkt_rv21 = mkt_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    mkt_rv63 = mkt_ret.rolling(63, min_periods=30).std() * np.sqrt(252)
    broadcast = pd.DataFrame(index=calendar, columns=tickers, dtype=float)
    feats["mkt_rv_21"] = broadcast.apply(lambda c: mkt_rv21, axis=0)
    feats["mkt_rv_63"] = broadcast.apply(lambda c: mkt_rv63, axis=0)
    feats["mkt_ret_21"] = broadcast.apply(
        lambda c: market_close.reindex(calendar).ffill().pct_change(21), axis=0
    )

    if vix is not None and not vix.empty:
        v = vix.reindex(calendar).ffill()
        feats["vix"] = broadcast.apply(lambda c: v, axis=0)
        feats["vix_chg_21"] = broadcast.apply(lambda c: v / v.shift(21) - 1, axis=0)

    # --- beta (systematic exposure scales vol with market vol) -------------
    cov = rets.rolling(252, min_periods=120).cov(mkt_ret)
    var = mkt_ret.rolling(252, min_periods=120).var()
    feats["beta_252"] = cov.div(var, axis=0)

    # --- target -------------------------------------------------------------
    target = realised_vol_forward(rets, HORIZON)

    # --- stack to long format at month-end sample points -------------------
    sample_dates = pd.Series(index=calendar, data=calendar).resample("ME").last().dropna()
    sample_dates = pd.DatetimeIndex(sample_dates.values)

    long_frames = []
    for name, frame in feats.items():
        s = frame.reindex(sample_dates).stack(future_stack=True).rename(name)
        long_frames.append(s)
    y = target.reindex(sample_dates).stack(future_stack=True).rename(TARGET)
    long_frames.append(y)

    panel = pd.concat(long_frames, axis=1)
    panel.index.names = ["date", "ticker"]
    panel = panel.reset_index()

    # A row is usable only if it has a target and the core vol features.
    panel = panel.dropna(subset=[TARGET, "rv_21", "rv_63", "ewma_94"])
    panel = panel[np.isfinite(panel[TARGET]) & (panel[TARGET] > 0)]
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in ("date", "ticker", TARGET)]
