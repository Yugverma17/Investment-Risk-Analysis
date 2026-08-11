"""All the tunables live here so I can trace any number in the README back to
the exact settings that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

for _p in (DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- universe ----
BENCHMARK = "^NSEI"  # Nifty 50 total-return proxy (price index; see docs/methodology.md)
VIX_TICKER = "^INDIAVIX"

START_DATE = "2015-01-01"
END_DATE = "2025-06-30"

# ------------------------------------------------------------- calendar ----
TRADING_DAYS = 252
MONTHS_PER_YEAR = 12

# ---------------------------------------------------------------- rates ----
# 10y GoI yield averaged ~6.9% over 2015-2025; 6.5% is a conservative constant.
# A constant risk-free rate is a simplification, documented in docs/methodology.md.
RISK_FREE_ANNUAL = 0.065

# ----------------------------------------------------------------- costs ----
# Round-trip cost assumption for NSE delivery trades, in basis points of turnover.
#   brokerage (discount broker, delivery) ~0 bps
#   STT on sell                            10.0 bps
#   exchange txn charge + SEBI + stamp      ~1.5 bps
#   GST on charges                          ~0.3 bps
#   impact/slippage for large caps          ~3.0 bps
COST_BPS = 15.0

# ------------------------------------------------------------- backtest ----
TRAIN_MONTHS = 36
HOLD_MONTHS = 3
MIN_HISTORY_DAYS = 500  # a stock needs this much history to enter the universe
MAX_MISSING_FRAC = 0.05  # drop a stock if >5% of trading days are missing in-window


@dataclass(frozen=True)
class RiskProfile:
    """A risk profile = hard constraints + scoring preference.

    My original notebook only had scoring weights, which meant a
    "Conservative" portfolio could theoretically still dump 60% into one
    stock. Added actual constraints here so that can't happen.
    """

    name: str
    max_weight: float  # cap on any single position
    min_positions: int  # minimum names held after filtering
    target_vol: float | None  # annualised; None = unconstrained
    # scoring preference used by the score-based allocator
    score_weights: dict[str, float] = field(default_factory=dict)


PROFILES: dict[str, RiskProfile] = {
    "conservative": RiskProfile(
        name="Conservative",
        max_weight=0.10,
        min_positions=15,
        target_vol=0.12,
        score_weights={"sharpe": 0.25, "volatility": -0.35, "beta": -0.20, "var_95": -0.20},
    ),
    "balanced": RiskProfile(
        name="Balanced",
        max_weight=0.15,
        min_positions=12,
        target_vol=0.18,
        score_weights={"sharpe": 0.40, "volatility": -0.20, "beta": -0.10, "mean_return": 0.30},
    ),
    "aggressive": RiskProfile(
        name="Aggressive",
        max_weight=0.20,
        min_positions=8,
        target_vol=None,
        score_weights={"sharpe": 0.30, "mean_return": 0.50, "volatility": -0.05, "beta": 0.15},
    ),
}

# ------------------------------------------------------------ reporting ----
SEED = 42
