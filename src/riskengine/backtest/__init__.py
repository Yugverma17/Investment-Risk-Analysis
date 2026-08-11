"""Walk-forward backtesting and statistical validation."""

from .engine import (
    BacktestConfig,
    BacktestResult,
    buy_and_hold_benchmark,
    rebalance_dates,
    run_backtest,
)
from .stats import (
    bootstrap_sharpe_difference,
    deflated_sharpe_ratio,
    newey_west_tstat,
    probabilistic_sharpe_ratio,
    significance_table,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "rebalance_dates",
    "buy_and_hold_benchmark",
    "bootstrap_sharpe_difference",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "newey_west_tstat",
    "significance_table",
]
