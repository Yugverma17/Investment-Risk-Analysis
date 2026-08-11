"""Look-ahead bias tests.

These are the most important tests in the repository. Every other number the
project reports is worthless if any of them fail.

The technique throughout: take a dataset, corrupt only the FUTURE portion of it,
recompute, and assert that everything decided in the past is bit-for-bit
identical. A quantity that changes when the future changes was reading the
future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskengine.backtest.engine import BacktestConfig, rebalance_dates, run_backtest
from riskengine.features.volatility import realised_vol, realised_vol_forward
from riskengine.models.features import build_feature_panel
from riskengine.optimize.scoring import compute_metrics


def test_forward_vol_target_is_aligned_to_the_future():
    """Row t of the ML target must equal the std of returns[t+1 .. t+h]."""
    h = 5
    n = 60
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(4)
    r = pd.DataFrame({"A": rng.normal(0, 0.01, n)}, index=idx)

    fwd = realised_vol_forward(r, horizon=h, annualise=False)

    for t in (10, 25, 40):
        expected = r["A"].iloc[t + 1 : t + 1 + h].std(ddof=1)
        assert fwd["A"].iloc[t] == pytest.approx(expected, rel=1e-12), f"misaligned at t={t}"


def test_forward_vol_is_nan_at_the_end():
    """The last `horizon` rows have no future to measure, so they must be NaN —
    not silently filled, which would create fake training rows."""
    h = 5
    n = 40
    idx = pd.bdate_range("2020-01-01", periods=n)
    r = pd.DataFrame({"A": np.random.default_rng(6).normal(0, 0.01, n)}, index=idx)
    fwd = realised_vol_forward(r, horizon=h)
    assert fwd["A"].iloc[-h:].isna().all()


def test_trailing_vol_never_reads_the_future():
    """The mirror image: a backward-looking feature must be immune to a shock
    placed after the observation date."""
    n = 300
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(8)
    r = pd.DataFrame({"A": rng.normal(0, 0.01, n)}, index=idx)

    shocked = r.copy()
    shocked.iloc[200:] *= 8.0  # enormous future volatility

    v_base = realised_vol(r, 21)
    v_shocked = realised_vol(shocked, 21)

    pd.testing.assert_series_equal(v_base["A"].iloc[:200], v_shocked["A"].iloc[:200])


def test_feature_panel_features_do_not_change_when_the_future_changes(ohlcv, prices, market_returns):
    """Corrupt the last 18 months of prices; every feature dated before the
    corruption must be identical."""
    market_close = 100 * (1 + market_returns).cumprod()
    cutoff = prices.index[-400]

    panel_a = build_feature_panel(ohlcv, prices, market_close)

    shocked_prices = prices.copy()
    shocked_prices.loc[shocked_prices.index > cutoff] *= 1.9
    shocked_ohlcv = ohlcv.copy()
    mask = shocked_ohlcv["date"] > cutoff
    for col in ("open", "high", "low", "close"):
        shocked_ohlcv.loc[mask, col] *= 1.9

    panel_b = build_feature_panel(shocked_ohlcv, shocked_prices, market_close)

    # compare only rows whose *entire* 252-day feature lookback predates the cutoff
    safe_cut = prices.index[-400 - 260]
    a = panel_a[panel_a.date <= safe_cut].set_index(["date", "ticker"]).sort_index()
    b = panel_b[panel_b.date <= safe_cut].set_index(["date", "ticker"]).sort_index()

    common = a.index.intersection(b.index)
    feature_cols = [c for c in a.columns if c != "fwd_vol_21"]
    pd.testing.assert_frame_equal(
        a.loc[common, feature_cols], b.loc[common, feature_cols], check_exact=False, rtol=1e-9
    )


def test_compute_metrics_only_sees_its_window(returns, market_returns):
    """Metrics computed on a slice must not change when data outside the slice does."""
    window = returns.iloc[500:1250]
    mkt = market_returns.iloc[500:1250]

    m_a = compute_metrics(window, mkt)

    shocked = returns.copy()
    shocked.iloc[1250:] *= 5.0
    m_b = compute_metrics(shocked.iloc[500:1250], mkt)

    pd.testing.assert_frame_equal(m_a, m_b)


def test_backtest_weights_are_immune_to_future_prices(prices, market_returns, sector_map):
    """The end-to-end guarantee.

    Multiply every price after a cutoff by 1.5. Any weight chosen at a rebalance
    date on or before that cutoff must be exactly unchanged — the allocator
    cannot have seen the shock.
    """
    mkt = market_returns.reindex(prices.index)
    cfg = BacktestConfig(allocator="score_based", profile="balanced", n_select=12)

    res_a = run_backtest(prices, mkt, sector_map, cfg)

    cutoff = prices.index[int(len(prices) * 0.8)]
    shocked = prices.copy()
    shocked.loc[shocked.index > cutoff] *= 1.5

    res_b = run_backtest(shocked, mkt, sector_map, cfg)

    past_a = res_a.weights[res_a.weights.index <= cutoff]
    past_b = res_b.weights[res_b.weights.index <= cutoff]

    assert len(past_a) > 2, "test needs at least a few pre-cutoff rebalances to be meaningful"

    # res_a.weights and res_b.weights each carry the UNION of tickers selected
    # across every rebalance in their respective run, including ones after the
    # cutoff — which legitimately differ between A and B since that's where the
    # shock lives. Restrict the comparison to tickers that actually appear with
    # nonzero weight in the pre-cutoff rows, which is the only thing the
    # look-ahead guarantee makes a claim about.
    used_a = past_a.columns[(past_a != 0).any()]
    used_b = past_b.columns[(past_b != 0).any()]
    assert set(used_a) == set(used_b), "pre-cutoff stock selection changed when only future prices did"

    cols = sorted(used_a)
    pd.testing.assert_frame_equal(past_a[cols], past_b[cols], check_exact=False, rtol=1e-10)


def test_returns_start_the_day_after_the_rebalance(prices, market_returns, sector_map):
    """Weights set at the close of t must earn their first return on t+1, not t.

    Trading on t's close using t's own data and booking t's return is the
    classic one-day look-ahead; this pins the +1 shift.
    """
    mkt = market_returns.reindex(prices.index)
    cfg = BacktestConfig(allocator="equal_weight", profile="balanced", n_select=12)
    res = run_backtest(prices, mkt, sector_map, cfg)

    first_rebalance = res.weights.index[0]
    assert res.returns.index[0] > first_rebalance


def test_rebalance_dates_leave_room_for_training(prices):
    """No rebalance may occur before a full training window exists."""
    dates = rebalance_dates(pd.DatetimeIndex(prices.index), train_months=36, hold_months=3)
    earliest_allowed = prices.index[0] + pd.DateOffset(months=36)
    assert dates
    assert min(dates) >= earliest_allowed
