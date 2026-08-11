"""Data quality auditing and panel construction.

The point of this module is to make the data's flaws *visible and quantified*
rather than silently absorbed into the backtest. Every filter applied here is
counted and reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import MAX_MISSING_FRAC, MIN_HISTORY_DAYS

# A single-day move beyond this is almost always a data error (unadjusted
# corporate action) rather than a real price move for a large-cap.
EXTREME_RETURN = 0.50
# More than this fraction of exactly-zero returns means a stale / illiquid feed.
MAX_STALE_FRAC = 0.20


def data_quality_report(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    requested: list[str] | None = None,
) -> pd.DataFrame:
    """Per-ticker audit against a reference trading calendar.

    Parameters
    ----------
    prices : wide date x ticker close prices
    calendar : reference trading days (use the benchmark's index)
    requested : the tickers we *asked* for, so we can report outright failures
    """
    rows = []
    aligned = prices.reindex(calendar)

    for tk in aligned.columns:
        s = aligned[tk]
        obs = s.notna()
        n_obs = int(obs.sum())
        if n_obs == 0:
            rows.append(
                {
                    "ticker": tk,
                    "n_obs": 0,
                    "first_date": pd.NaT,
                    "last_date": pd.NaT,
                    "missing_frac": 1.0,
                    "stale_frac": np.nan,
                    "n_extreme": 0,
                    "status": "no_data",
                }
            )
            continue

        first, last = s.first_valid_index(), s.last_valid_index()
        window = calendar[(calendar >= first) & (calendar <= last)]
        missing_frac = 1.0 - n_obs / max(len(window), 1)

        rets = s.pct_change()
        stale_frac = float((rets.dropna() == 0).mean()) if rets.notna().sum() else np.nan
        n_extreme = int((rets.abs() > EXTREME_RETURN).sum())

        if n_obs < MIN_HISTORY_DAYS:
            status = "short_history"
        elif missing_frac > MAX_MISSING_FRAC:
            status = "gappy"
        elif stale_frac > MAX_STALE_FRAC:
            status = "stale"
        else:
            status = "ok"

        rows.append(
            {
                "ticker": tk,
                "n_obs": n_obs,
                "first_date": first,
                "last_date": last,
                "missing_frac": round(missing_frac, 4),
                "stale_frac": round(stale_frac, 4) if stale_frac == stale_frac else np.nan,
                "n_extreme": n_extreme,
                "status": status,
            }
        )

    report = pd.DataFrame(rows).set_index("ticker")

    if requested:
        missing = sorted(set(requested) - set(report.index))
        if missing:
            extra = pd.DataFrame(
                {
                    "n_obs": 0,
                    "first_date": pd.NaT,
                    "last_date": pd.NaT,
                    "missing_frac": 1.0,
                    "stale_frac": np.nan,
                    "n_extreme": 0,
                    "status": "download_failed",
                },
                index=pd.Index(missing, name="ticker"),
            )
            report = pd.concat([report, extra])

    return report.sort_values(["status", "ticker"])


def clean_prices(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    report: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the audit: keep 'ok' tickers, align to calendar, forward-fill gaps.

    Forward-fill is limited to 5 days. A longer gap means the stock genuinely
    was not trading (suspension), and carrying a stale price across it would
    manufacture a fake zero-volatility period.
    """
    keep = report.index[report["status"] == "ok"]
    keep = [t for t in keep if t in prices.columns]
    out = prices[keep].reindex(calendar).ffill(limit=5)
    return out


def to_returns(prices: pd.DataFrame, log_returns: bool = False) -> pd.DataFrame:
    """Simple (default) or log daily returns.

    Simple returns are the default because portfolio return is a linear
    combination of simple returns — log returns do not aggregate across assets.
    """
    if log_returns:
        return np.log(prices / prices.shift(1))
    return prices.pct_change()


def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end prices (used for monthly rebalancing arithmetic)."""
    return prices.resample("ME").last()


def winsorise_returns(returns: pd.DataFrame, limit: float = EXTREME_RETURN) -> pd.DataFrame:
    """Clip implausible single-day moves.

    This is a blunt instrument and it *does* remove some genuine crash days.
    It is applied only to per-stock feature computation (vol, beta), never to
    the return stream used to compute portfolio P&L — clipping the P&L stream
    would flatter every drawdown statistic.
    """
    return returns.clip(lower=-limit, upper=limit)
