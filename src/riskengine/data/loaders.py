"""Downloads market data and caches it to disk so I only pay for it once.

A few things worth knowing:
* I keep the full OHLCV panel, not just close, because the range-based vol
  estimators (Parkinson, Garman-Klass) need high/low and are noticeably
  better than close-to-close ones.
* `auto_adjust=True` gives split/dividend-adjusted prices. Skipping this is
  probably the most common bug in DIY backtests — every stock split shows up
  as a fake -50% crash otherwise.
* Everything gets cached to parquet. A cold fetch of the whole universe takes
  a few minutes and I didn't want to pay that cost more than once.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from ..config import DATA_RAW, END_DATE, START_DATE

log = logging.getLogger(__name__)

_BATCH = 20
_RETRIES = 3


def _normalise(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Flatten yfinance output into long format: date, ticker, o/h/l/c/volume."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    # yfinance returns MultiIndex (field, ticker) for multi-ticker requests and
    # flat columns for a single ticker. Normalise both to the same shape.
    if not isinstance(raw.columns, pd.MultiIndex):
        raw = pd.concat({tickers[0]: raw}, axis=1)
        raw.columns = raw.columns.swaplevel(0, 1)

    frames = []
    for tk in tickers:
        try:
            sub = raw.xs(tk, axis=1, level=1)
        except KeyError:
            continue
        sub = sub.rename(columns=str.lower)
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in sub.columns]
        sub = sub[keep].dropna(how="all")
        if sub.empty:
            continue
        sub = sub.assign(ticker=tk)
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames).reset_index()
    out = out.rename(columns={out.columns[0]: "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def _download_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    for attempt in range(1, _RETRIES + 1):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
            )
            return _normalise(raw, tickers)
        except Exception as exc:  # noqa: BLE001 - network layer, retry anything
            log.warning("batch failed (attempt %d/%d): %s", attempt, _RETRIES, exc)
            time.sleep(2 * attempt)
    return pd.DataFrame()


def fetch_ohlcv(
    tickers: list[str],
    start: str = START_DATE,
    end: str = END_DATE,
    cache_name: str = "ohlcv",
    force: bool = False,
) -> pd.DataFrame:
    """Fetch daily adjusted OHLCV in long format, caching to parquet.

    Returns columns: date, ticker, open, high, low, close, volume.
    Tickers that return no data are silently dropped here and reported by
    `quality.data_quality_report`.
    """
    path: Path = DATA_RAW / f"{cache_name}_{start}_{end}.parquet"
    if path.exists() and not force:
        log.info("loading cached OHLCV from %s", path.name)
        return pd.read_parquet(path)

    frames = []
    for i in range(0, len(tickers), _BATCH):
        batch = tickers[i : i + _BATCH]
        log.info("downloading %d-%d of %d", i + 1, i + len(batch), len(tickers))
        df = _download_batch(batch, start, end)
        if not df.empty:
            frames.append(df)
        time.sleep(0.5)  # be polite to the endpoint

    if not frames:
        raise RuntimeError("no data downloaded — check network access to Yahoo Finance")

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(path, index=False)
    log.info("cached %d rows / %d tickers to %s", len(out), out.ticker.nunique(), path.name)
    return out


def to_wide(long_df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Pivot long OHLCV to a wide date x ticker frame for one field."""
    return long_df.pivot(index="date", columns="ticker", values=field).sort_index()


def fetch_benchmark(
    ticker: str, start: str = START_DATE, end: str = END_DATE, force: bool = False
) -> pd.Series:
    """Fetch a single index series (Nifty, India VIX) as a named Series."""
    safe = ticker.replace("^", "").lower()
    df = fetch_ohlcv([ticker], start, end, cache_name=f"index_{safe}", force=force)
    if df.empty:
        raise RuntimeError(f"no data for index {ticker}")
    s = df.set_index("date")["close"].sort_index()
    s.name = ticker
    return s
