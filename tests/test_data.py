"""Data layer: universe integrity, quality auditing, and return construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskengine.data.loaders import to_wide
from riskengine.data.quality import (
    clean_prices,
    data_quality_report,
    to_monthly,
    to_returns,
    winsorise_returns,
)
from riskengine.data.universe import get_tickers, get_universe, sector_map


# ------------------------------------------------------------- universe ----
def test_universe_is_non_trivial():
    assert len(get_tickers()) >= 100


def test_every_ticker_has_the_nse_suffix():
    assert all(t.endswith(".NS") for t in get_tickers())


def test_no_duplicate_tickers():
    tickers = get_tickers()
    assert len(tickers) == len(set(tickers))


def test_sector_map_covers_the_whole_universe():
    smap = sector_map()
    assert set(smap) == set(get_tickers())
    assert all(isinstance(v, str) and v for v in smap.values())


def test_universe_is_reasonably_diversified():
    """No single sector should dominate, or sector caps become meaningless."""
    counts = get_universe()["sector"].value_counts(normalize=True)
    assert counts.max() < 0.30


def test_universe_retains_known_underperformers():
    """Survivorship mitigation: the list must not be only winners."""
    tickers = set(get_tickers())
    laggards = {"IDEA.NS", "BHEL.NS", "SAIL.NS", "ZEEL.NS", "PAYTM.NS", "RBLBANK.NS"}
    assert laggards.issubset(tickers)


# -------------------------------------------------------------- quality ----
@pytest.fixture
def wide_prices(calendar):
    rng = np.random.default_rng(2)
    n = len(calendar)
    df = pd.DataFrame(
        {
            "GOOD.NS": 100 * (1 + rng.normal(0.0005, 0.01, n)).cumprod(),
            "SHORT.NS": np.nan,
            "GAPPY.NS": 100 * (1 + rng.normal(0.0005, 0.01, n)).cumprod(),
            "STALE.NS": 100.0,
        },
        index=calendar,
    )
    df.loc[calendar[-300:], "SHORT.NS"] = 50.0  # only 300 obs -> short history
    gappy_idx = calendar[rng.random(n) < 0.25]
    df.loc[gappy_idx, "GAPPY.NS"] = np.nan  # ~25% missing -> gappy
    return df


def test_quality_report_classifies_each_failure_mode(wide_prices, calendar):
    rep = data_quality_report(wide_prices, calendar)
    assert rep.loc["GOOD.NS", "status"] == "ok"
    assert rep.loc["SHORT.NS", "status"] == "short_history"
    assert rep.loc["GAPPY.NS", "status"] == "gappy"
    assert rep.loc["STALE.NS", "status"] == "stale"


def test_quality_report_flags_tickers_that_never_downloaded(wide_prices, calendar):
    rep = data_quality_report(wide_prices, calendar, requested=list(wide_prices.columns) + ["MISSING.NS"])
    assert rep.loc["MISSING.NS", "status"] == "download_failed"
    assert rep.loc["MISSING.NS", "n_obs"] == 0


def test_quality_report_counts_extreme_moves(calendar):
    rng = np.random.default_rng(3)
    s = pd.Series(100 * (1 + rng.normal(0, 0.01, len(calendar))).cumprod(), index=calendar)
    s.iloc[500] *= 2.5  # an unadjusted-split-shaped artefact
    rep = data_quality_report(pd.DataFrame({"X.NS": s}), calendar)
    assert rep.loc["X.NS", "n_extreme"] >= 1


def test_clean_prices_keeps_only_ok_tickers(wide_prices, calendar):
    rep = data_quality_report(wide_prices, calendar)
    clean = clean_prices(wide_prices, calendar, rep)
    assert list(clean.columns) == ["GOOD.NS"]
    assert clean.index.equals(calendar)


def test_clean_prices_does_not_bridge_long_suspensions(calendar):
    """A gap longer than 5 days must stay NaN — carrying a stale price across a
    trading halt manufactures a fake zero-volatility stretch."""
    s = pd.Series(100.0, index=calendar)
    s.iloc[100:130] = np.nan
    df = pd.DataFrame({"A.NS": s})
    rep = data_quality_report(df, calendar)
    rep.loc["A.NS", "status"] = "ok"  # force through the filter for this check
    clean = clean_prices(df, calendar, rep)
    assert clean["A.NS"].iloc[100:130].isna().any()


# -------------------------------------------------------------- returns ----
def test_simple_returns_are_correct():
    p = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    r = to_returns(p)
    assert r["A"].iloc[1] == pytest.approx(0.10)
    assert r["A"].iloc[2] == pytest.approx(-0.10)


def test_log_returns_are_additive():
    p = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
    lr = to_returns(p, log_returns=True)["A"].dropna()
    assert lr.sum() == pytest.approx(np.log(1.21), rel=1e-12)


def test_first_return_is_nan():
    p = pd.DataFrame({"A": [100.0, 110.0]})
    assert np.isnan(to_returns(p)["A"].iloc[0])


def test_winsorise_clips_both_tails():
    r = pd.DataFrame({"A": [-0.9, 0.0, 0.9]})
    w = winsorise_returns(r, limit=0.5)
    assert w["A"].min() == pytest.approx(-0.5)
    assert w["A"].max() == pytest.approx(0.5)


def test_to_monthly_takes_period_end_prices(calendar):
    p = pd.DataFrame({"A": range(len(calendar))}, index=calendar, dtype=float)
    m = to_monthly(p)
    assert len(m) < len(p)
    assert m["A"].is_monotonic_increasing


def test_to_wide_pivots_long_ohlcv(ohlcv):
    wide = to_wide(ohlcv, "close")
    assert wide.shape[1] == ohlcv["ticker"].nunique()
    assert wide.index.is_monotonic_increasing
