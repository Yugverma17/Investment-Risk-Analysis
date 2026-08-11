"""Figures and result tables."""

from .plots import (
    correlation_heatmap,
    drawdown_chart,
    equity_curves,
    feature_importance,
    rolling_sharpe,
    sector_exposure,
    stress_chart,
    turnover_chart,
    var_breach_chart,
    vol_model_by_year,
    vol_model_scatter,
)
from .tables import (
    comparison_table,
    format_table,
    stress_test,
    stress_windows,
    to_markdown,
)

__all__ = [
    "equity_curves",
    "drawdown_chart",
    "rolling_sharpe",
    "var_breach_chart",
    "vol_model_scatter",
    "feature_importance",
    "vol_model_by_year",
    "sector_exposure",
    "turnover_chart",
    "correlation_heatmap",
    "stress_chart",
    "comparison_table",
    "format_table",
    "to_markdown",
    "stress_test",
    "stress_windows",
]
