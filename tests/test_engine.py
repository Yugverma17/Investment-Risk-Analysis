"""Backtest engine mechanics: cost accounting, weight drift, rebalancing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskengine.backtest.engine import (
    BacktestConfig,
    _period_returns,
    buy_and_hold_benchmark,
    rebalance_dates,
    run_backtest,
)
from riskengine.risk.metrics import cagr


@pytest.fixture
def mkt(market_returns, prices):
    return market_returns.reindex(prices.index)


def test_run_produces_a_coherent_result(prices, mkt, sector_map):
    res = run_backtest(prices, mkt, sector_map, BacktestConfig(allocator="equal_weight", n_select=12))
    assert len(res.returns) > 200
    assert res.returns.index.is_monotonic_increasing
    assert not res.returns.index.has_duplicates
    assert len(res.weights) == len(res.turnover) == len(res.costs)
    # every rebalance's weights sum to 1
    assert np.allclose(res.weights.sum(axis=1), 1.0, atol=1e-8)


def test_zero_cost_makes_gross_equal_net(prices, mkt, sector_map):
    res = run_backtest(
        prices, mkt, sector_map, BacktestConfig(allocator="equal_weight", cost_bps=0.0, n_select=12)
    )
    pd.testing.assert_series_equal(res.returns, res.gross_returns)
    assert res.costs.sum() == pytest.approx(0.0)


def test_costs_reduce_returns_monotonically(prices, mkt, sector_map):
    """Charging more per trade must never produce a higher net return."""
    cagrs = []
    for bps in (0.0, 15.0, 100.0):
        res = run_backtest(
            prices, mkt, sector_map, BacktestConfig(allocator="score_based", cost_bps=bps, n_select=12)
        )
        cagrs.append(cagr(res.returns))
    assert cagrs[0] > cagrs[1] > cagrs[2]


def test_cost_drag_is_positive_when_costs_are_charged(prices, mkt, sector_map):
    res = run_backtest(
        prices, mkt, sector_map, BacktestConfig(allocator="score_based", cost_bps=50.0, n_select=12)
    )
    assert res.total_cost_drag > 0


def test_first_rebalance_is_charged_a_full_buy(prices, mkt, sector_map):
    """Building the initial portfolio from cash trades 100% of capital."""
    bps = 20.0
    res = run_backtest(
        prices, mkt, sector_map, BacktestConfig(allocator="equal_weight", cost_bps=bps, n_select=12)
    )
    first_cost = res.costs.iloc[0]
    assert first_cost == pytest.approx(bps / 10_000.0, rel=1e-9)


def test_turnover_is_bounded(prices, mkt, sector_map):
    """One-way turnover of a long-only fully-invested book cannot exceed 100%."""
    res = run_backtest(prices, mkt, sector_map, BacktestConfig(allocator="score_based", n_select=12))
    assert (res.turnover >= 0).all()
    assert (res.turnover <= 1.0 + 1e-9).all()


def test_weights_drift_between_rebalances():
    """Two assets, one doubling and one flat: the winner's weight must grow."""
    n = 30
    idx = pd.bdate_range("2020-01-01", periods=n)
    rets = pd.DataFrame({"UP.NS": 0.02, "FLAT.NS": 0.0}, index=idx)
    w0 = pd.Series({"UP.NS": 0.5, "FLAT.NS": 0.5})

    daily, drifted = _period_returns(rets, w0, start_pos=0, end_pos=n - 1)

    assert drifted["UP.NS"] > 0.5
    assert drifted["FLAT.NS"] < 0.5
    assert drifted.sum() == pytest.approx(1.0)
    # portfolio return on the first day is the weighted average of asset returns
    assert daily.iloc[0] == pytest.approx(0.5 * 0.02 + 0.5 * 0.0)


def test_period_returns_match_manual_compounding():
    n = 10
    idx = pd.bdate_range("2020-01-01", periods=n)
    rets = pd.DataFrame({"A.NS": 0.01, "B.NS": -0.005}, index=idx)
    w0 = pd.Series({"A.NS": 0.6, "B.NS": 0.4})

    daily, _ = _period_returns(rets, w0, 0, n - 1)
    realised = float((1 + daily).prod())
    expected = 0.6 * (1.01 ** len(daily)) + 0.4 * (0.995 ** len(daily))
    assert realised == pytest.approx(expected, rel=1e-12)


def test_rebalance_cadence_matches_hold_months(prices):
    idx = pd.DatetimeIndex(prices.index)
    quarterly = rebalance_dates(idx, train_months=36, hold_months=3)
    annual = rebalance_dates(idx, train_months=36, hold_months=12)
    assert len(quarterly) > len(annual)
    gaps = pd.Series(quarterly).diff().dropna().dt.days
    assert gaps.between(80, 100).all()


def test_longer_training_window_delays_the_first_trade(prices, mkt, sector_map):
    short = run_backtest(prices, mkt, sector_map, BacktestConfig(train_months=24, n_select=12))
    long = run_backtest(prices, mkt, sector_map, BacktestConfig(train_months=48, n_select=12))
    assert short.weights.index[0] < long.weights.index[0]


def test_benchmark_is_restricted_to_the_evaluation_window(prices, mkt, sector_map):
    res = run_backtest(prices, mkt, sector_map, BacktestConfig(n_select=12))
    bench = buy_and_hold_benchmark(mkt, res.returns)
    assert bench.index.equals(res.returns.index)


def test_insufficient_history_raises(prices, mkt, sector_map):
    short_prices = prices.iloc[:100]
    with pytest.raises(ValueError):
        run_backtest(short_prices, mkt.iloc[:100], sector_map, BacktestConfig(train_months=36))
