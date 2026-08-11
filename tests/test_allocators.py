"""Allocator contract tests.

Every allocator must satisfy the same invariants — long-only, fully invested,
concentration-capped. Where an optimality property is provable (min-variance
must not be beaten in-sample by a feasible alternative), the test asserts it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskengine.optimize.allocators import (
    ALLOCATORS,
    equal_weight,
    inverse_volatility,
    max_sharpe,
    min_variance,
    risk_parity,
    score_based,
)
from riskengine.optimize.constraints import (
    Constraints,
    apply_caps,
    correlation_filter,
    select_candidates,
)
from riskengine.risk.covariance import ledoit_wolf_cov

ALL_NAMES = list(ALLOCATORS)


@pytest.fixture
def cons(sector_map):
    return Constraints(max_weight=0.15, max_sector_weight=0.40, sector_map=sector_map)


@pytest.fixture
def train(returns):
    return returns.iloc[-756:]


@pytest.mark.parametrize("name", ALL_NAMES)
def test_weights_sum_to_one(name, train, cons):
    w = ALLOCATORS[name](train, cons, cov=ledoit_wolf_cov(train), scores=pd.Series(1.0, index=train.columns))
    assert w.sum() == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_weights_are_long_only(name, train, cons):
    w = ALLOCATORS[name](train, cons, cov=ledoit_wolf_cov(train), scores=pd.Series(1.0, index=train.columns))
    assert (w >= -1e-12).all()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_weights_respect_the_concentration_cap(name, train, cons):
    w = ALLOCATORS[name](train, cons, cov=ledoit_wolf_cov(train), scores=pd.Series(1.0, index=train.columns))
    # tolerance covers SLSQP's convergence slack
    assert w.max() <= cons.max_weight + 1e-4


@pytest.mark.parametrize("name", ALL_NAMES)
def test_weights_cover_exactly_the_given_assets(name, train, cons):
    w = ALLOCATORS[name](train, cons, cov=ledoit_wolf_cov(train), scores=pd.Series(1.0, index=train.columns))
    assert set(w.index) == set(train.columns)


def test_equal_weight_is_actually_equal(train):
    cons = Constraints(max_weight=1.0)
    w = equal_weight(train, cons)
    assert w.std() == pytest.approx(0.0, abs=1e-12)
    assert w.iloc[0] == pytest.approx(1 / len(train.columns))


def test_inverse_vol_gives_more_weight_to_calmer_assets(train):
    cons = Constraints(max_weight=1.0)
    w = inverse_volatility(train, cons)
    vol = train.std()
    # the lowest-vol asset must receive more weight than the highest-vol asset
    assert w[vol.idxmin()] > w[vol.idxmax()]


def test_min_variance_beats_equal_weight_in_sample(train):
    """Equal weight is inside the feasible set, so the optimiser's solution can
    never have higher in-sample variance. If this fails, the optimiser is not
    converging."""
    cons = Constraints(max_weight=1.0)
    cov = ledoit_wolf_cov(train)
    w_mv = min_variance(train, cons, cov=cov)
    w_ew = equal_weight(train, cons)

    var_mv = float(w_mv @ cov @ w_mv)
    var_ew = float(w_ew @ cov @ w_ew)
    assert var_mv <= var_ew + 1e-10


def test_risk_parity_equalises_risk_contributions(train):
    cons = Constraints(max_weight=1.0)
    cov = ledoit_wolf_cov(train)
    w = risk_parity(train, cons, cov=cov)

    Sigma = cov.loc[w.index, w.index].to_numpy()
    wv = w.to_numpy()
    port_var = float(wv @ Sigma @ wv)
    rc = wv * (Sigma @ wv) / port_var

    # every asset should carry roughly 1/n of total risk
    assert rc.std() < 0.02
    assert rc.sum() == pytest.approx(1.0, rel=1e-9)


def test_max_sharpe_prefers_the_better_asset():
    """Two uncorrelated assets, identical vol, one with double the mean return.
    The tangency portfolio must overweight the higher-return asset."""
    rng = np.random.default_rng(9)
    n = 1500
    idx = pd.bdate_range("2015-01-01", periods=n)
    df = pd.DataFrame(
        {
            "GOOD.NS": rng.normal(0.0010, 0.01, n),
            "BAD.NS": rng.normal(0.0002, 0.01, n),
        },
        index=idx,
    )
    w = max_sharpe(df, Constraints(max_weight=1.0), cov=ledoit_wolf_cov(df))
    assert w["GOOD.NS"] > w["BAD.NS"]


def test_score_based_ranks_monotonically(train):
    """Higher score must never receive lower weight (before caps bind)."""
    cons = Constraints(max_weight=1.0)
    scores = pd.Series(np.linspace(-2, 2, len(train.columns)), index=train.columns)
    w = score_based(train, cons, scores=scores)
    ordered = w.reindex(scores.sort_values().index).to_numpy()
    assert np.all(np.diff(ordered) >= -1e-12)


def test_score_based_falls_back_to_equal_weight_without_scores(train):
    cons = Constraints(max_weight=1.0)
    w = score_based(train, cons, scores=None)
    assert w.std() == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------- constraints ---


def test_apply_caps_enforces_the_cap_and_still_sums_to_one():
    w = pd.Series([0.7, 0.2, 0.05, 0.05], index=list("ABCD"))
    capped = apply_caps(w, 0.30)
    assert capped.max() <= 0.30 + 1e-9
    assert capped.sum() == pytest.approx(1.0)


def test_apply_caps_is_a_no_op_when_nothing_binds():
    w = pd.Series([0.25, 0.25, 0.25, 0.25], index=list("ABCD"))
    assert apply_caps(w, 0.30).equals(w / w.sum())


def test_infeasible_cap_is_relaxed_not_crashed():
    """max_weight=0.10 with 5 assets cannot sum to 1; the code must widen it."""
    cons = Constraints(max_weight=0.10)
    assert not cons.feasible(5)
    assert cons.effective_max_weight(5) == pytest.approx(0.20)


def test_correlation_filter_drops_a_near_duplicate():
    rng = np.random.default_rng(13)
    n = 1000
    idx = pd.bdate_range("2015-01-01", periods=n)
    base = rng.normal(0, 0.01, n)
    df = pd.DataFrame(
        {
            "A.NS": base,
            "A_CLONE.NS": base + rng.normal(0, 0.0005, n),  # ~0.99 correlated
            "B.NS": rng.normal(0, 0.01, n),
        },
        index=idx,
    )
    ranking = pd.Series({"A.NS": 3.0, "A_CLONE.NS": 2.0, "B.NS": 1.0})
    kept = correlation_filter(df, ranking, threshold=0.85)
    assert "A.NS" in kept
    assert "A_CLONE.NS" not in kept  # lower-ranked twin is dropped
    assert "B.NS" in kept


def test_select_candidates_respects_sector_cap(train, sector_map):
    scores = pd.Series(np.linspace(2, -2, len(train.columns)), index=train.columns)
    chosen = select_candidates(
        scores, train, n_select=20, corr_threshold=0.99, sector_map=sector_map, max_per_sector=2
    )
    counts: dict[str, int] = {}
    for t in chosen:
        counts[sector_map[t]] = counts.get(sector_map[t], 0) + 1
    assert max(counts.values()) <= 2
