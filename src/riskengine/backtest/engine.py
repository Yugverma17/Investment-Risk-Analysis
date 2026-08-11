"""Walk-forward backtest engine.

The rules, stated once so they can be checked
---------------------------------------------
1. At each rebalance date `t`, the allocator sees returns strictly up to and
   including `t`. Nothing after `t` is available to it, ever.
2. Weights decided at `t` are applied at the CLOSE of `t` and earn returns from
   `t+1` onwards. Trading on the same day's close using that day's data is the
   most common look-ahead bug in retail backtests; the +1 shift removes it.
3. Between rebalances, weights DRIFT with prices. They are not silently reset to
   target every day — that would be a free daily rebalance nobody pays for.
4. Transaction costs are charged on the drifted-to-target difference at each
   rebalance, so a strategy that churns is penalised for churning.
5. The benchmark and every strategy are evaluated on exactly the same date
   range, which is the walk-forward period only — never the training burn-in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import COST_BPS, HOLD_MONTHS, PROFILES, TRAIN_MONTHS
from ..optimize.allocators import ALLOCATORS
from ..optimize.constraints import Constraints, select_candidates
from ..optimize.scoring import composite_score, compute_metrics
from ..risk.covariance import cov_from_vols, ledoit_wolf_cov, sample_cov

log = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    allocator: str = "equal_weight"
    profile: str = "balanced"
    train_months: int = TRAIN_MONTHS
    hold_months: int = HOLD_MONTHS
    n_select: int = 25
    corr_threshold: float = 0.85
    max_per_sector: int = 5
    cost_bps: float = COST_BPS
    cov_estimator: str = "ledoit_wolf"  # "ledoit_wolf" | "sample"
    label: str | None = None

    def name(self) -> str:
        return self.label or f"{self.allocator}/{self.profile}"


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series  # daily, net of costs
    gross_returns: pd.Series  # daily, before costs
    weights: pd.DataFrame  # rebalance date x ticker
    turnover: pd.Series  # one-way, per rebalance
    costs: pd.Series  # cost drag charged, per rebalance
    n_holdings: pd.Series
    config: BacktestConfig = field(default_factory=BacktestConfig)

    @property
    def equity_curve(self) -> pd.Series:
        return (1 + self.returns.fillna(0)).cumprod()

    @property
    def total_cost_drag(self) -> float:
        """Annualised return given up to transaction costs."""
        n_years = len(self.returns) / 252
        if n_years <= 0:
            return np.nan
        gross = float((1 + self.gross_returns.fillna(0)).prod()) ** (1 / n_years) - 1
        net = float((1 + self.returns.fillna(0)).prod()) ** (1 / n_years) - 1
        return gross - net


def rebalance_dates(
    index: pd.DatetimeIndex, train_months: int, hold_months: int
) -> list[pd.Timestamp]:
    """Quarter-end (or hold_months-end) trading dates that have enough history behind them."""
    freq = f"{hold_months}ME"
    period_ends = pd.Series(index=index, data=index).resample(freq).last().dropna()
    first_valid = index[0] + pd.DateOffset(months=train_months)
    return [d for d in period_ends if d >= first_valid and d < index[-1]]


def _period_returns(
    returns: pd.DataFrame, weights: pd.Series, start_pos: int, end_pos: int
) -> tuple[pd.Series, pd.Series]:
    """Daily portfolio returns over (start_pos, end_pos] with drifting weights.

    Returns the daily series and the drifted end-of-period weights.
    """
    assets = list(weights.index)
    window = returns.iloc[start_pos + 1 : end_pos + 1][assets].fillna(0.0)
    if window.empty:
        return pd.Series(dtype=float), weights

    # Track holding values so weights drift exactly as a real portfolio would.
    values = weights.to_numpy(dtype=float).copy()
    daily = np.empty(len(window))
    mat = window.to_numpy()

    for i in range(len(window)):
        prev_total = values.sum()
        values = values * (1.0 + mat[i])
        new_total = values.sum()
        daily[i] = new_total / prev_total - 1.0 if prev_total > 0 else 0.0

    drifted = pd.Series(values / values.sum() if values.sum() > 0 else values, index=assets)
    return pd.Series(daily, index=window.index), drifted


def run_backtest(
    prices: pd.DataFrame,
    market_returns: pd.Series,
    sector_map: dict[str, str],
    config: BacktestConfig,
    vol_forecasts: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run one strategy walk-forward across the full sample.

    Parameters
    ----------
    prices : wide date x ticker adjusted close
    market_returns : benchmark daily returns, aligned to `prices.index`
    sector_map : ticker -> sector, for sector caps
    vol_forecasts : optional date x ticker ANNUALISED vol forecasts. When given,
        optimiser covariance is rebuilt as D(forecast) * Corr(historical) * D.
    """
    profile = PROFILES[config.profile]
    allocator = ALLOCATORS[config.allocator]
    returns = prices.pct_change()
    index = pd.DatetimeIndex(prices.index)
    positions = {d: i for i, d in enumerate(index)}

    cons = Constraints(
        max_weight=profile.max_weight,
        max_sector_weight=0.35,
        sector_map=sector_map,
    )

    dates = rebalance_dates(index, config.train_months, config.hold_months)
    if not dates:
        raise ValueError("no rebalance dates — sample is shorter than the training window")

    all_daily: list[pd.Series] = []
    weight_rows: dict[pd.Timestamp, pd.Series] = {}
    turnover_rows: dict[pd.Timestamp, float] = {}
    cost_rows: dict[pd.Timestamp, float] = {}
    holdings_rows: dict[pd.Timestamp, int] = {}

    prev_drifted = pd.Series(dtype=float)

    for k, t in enumerate(dates):
        pos = positions[t]
        train_start = t - pd.DateOffset(months=config.train_months)
        train = returns.loc[(returns.index > train_start) & (returns.index <= t)]
        train = train.dropna(axis=1, thresh=int(len(train) * 0.9))
        if train.shape[1] < profile.min_positions:
            continue

        mkt_train = market_returns.reindex(train.index)

        # ---- rank, filter, select -------------------------------------
        metrics = compute_metrics(train, mkt_train)
        if metrics.empty:
            continue
        scores = composite_score(metrics, profile.score_weights)
        chosen = select_candidates(
            scores,
            train,
            n_select=config.n_select,
            corr_threshold=config.corr_threshold,
            sector_map=sector_map,
            max_per_sector=config.max_per_sector,
        )
        if len(chosen) < profile.min_positions:
            chosen = list(scores.sort_values(ascending=False).head(config.n_select).index)
        if not chosen:
            continue

        sub_train = train[chosen]

        # ---- covariance ------------------------------------------------
        if vol_forecasts is not None:
            vf = vol_forecasts.reindex(index).ffill()
            row = vf.loc[t, [c for c in chosen if c in vf.columns]] if t in vf.index else pd.Series(dtype=float)
            cov = cov_from_vols(sub_train, row.reindex(chosen))
        elif config.cov_estimator == "sample":
            cov = sample_cov(sub_train)
        else:
            cov = ledoit_wolf_cov(sub_train)

        # ---- allocate --------------------------------------------------
        w = allocator(
            sub_train,
            cons,
            cov=cov,
            scores=scores.reindex(chosen),
            expected_returns=metrics["mean_return"].reindex(chosen),
        )
        w = w[w > 0]
        if w.empty:
            continue

        # ---- transaction costs ----------------------------------------
        # Traded value as a fraction of portfolio = sum |w_new - w_drifted|
        # (buys + sells). Cost is charged on every rupee traded.
        union = w.index.union(prev_drifted.index)
        delta = w.reindex(union).fillna(0.0) - prev_drifted.reindex(union).fillna(0.0)
        traded = float(delta.abs().sum())
        if prev_drifted.empty:
            traded = float(w.sum())  # initial build-out is a full buy
        cost = traded * config.cost_bps / 10_000.0

        turnover_rows[t] = traded / 2.0
        cost_rows[t] = cost
        weight_rows[t] = w
        holdings_rows[t] = int(len(w))

        # ---- hold ------------------------------------------------------
        end_pos = positions[dates[k + 1]] if k + 1 < len(dates) else len(index) - 1
        period, prev_drifted = _period_returns(returns, w, pos, end_pos)
        if period.empty:
            continue

        # Charge the cost on the first day of the holding period. Spreading it
        # would flatter the drawdown; charging it up front is conservative.
        period = period.copy()
        period.iloc[0] -= cost
        all_daily.append(period)

    if not all_daily:
        raise ValueError(f"backtest produced no periods for {config.name()}")

    net = pd.concat(all_daily).sort_index()
    net = net[~net.index.duplicated(keep="first")]

    gross = net.copy()
    for t, c in cost_rows.items():
        future = gross.index[gross.index > t]
        if len(future):
            gross.loc[future[0]] += c

    return BacktestResult(
        name=config.name(),
        returns=net,
        gross_returns=gross,
        weights=pd.DataFrame(weight_rows).T.fillna(0.0),
        turnover=pd.Series(turnover_rows).sort_index(),
        costs=pd.Series(cost_rows).sort_index(),
        n_holdings=pd.Series(holdings_rows).sort_index(),
        config=config,
    )


def buy_and_hold_benchmark(market_returns: pd.Series, like: pd.Series) -> pd.Series:
    """Benchmark returns restricted to exactly the strategy's evaluation window."""
    return market_returns.reindex(like.index).fillna(0.0)
