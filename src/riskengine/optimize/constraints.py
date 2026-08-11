"""Portfolio constraints and candidate selection.

I kept constraints separate from the optimizers so every allocator — even
the trivial ones like equal-weight — plays by the same rules. Otherwise a
"comparison between strategies" quietly turns into a comparison between
constraint sets instead, which is a really easy way for a backtest to
mislead you without anyone noticing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Constraints:
    """Long-only, fully-invested, with concentration limits."""

    max_weight: float = 0.15
    min_weight: float = 0.0
    max_sector_weight: float = 0.35
    sector_map: dict[str, str] = field(default_factory=dict)

    def bounds(self, assets: list[str]) -> list[tuple[float, float]]:
        return [(self.min_weight, self.max_weight)] * len(assets)

    def feasible(self, n_assets: int) -> bool:
        """A max_weight of 0.10 with only 5 assets cannot sum to 1."""
        return n_assets * self.max_weight >= 1.0 - 1e-9

    def effective_max_weight(self, n_assets: int) -> float:
        """Relax max_weight just enough to keep the problem feasible."""
        if self.feasible(n_assets):
            return self.max_weight
        return max(self.max_weight, 1.0 / n_assets)

    def sector_matrix(self, assets: list[str]) -> tuple[np.ndarray, list[str]]:
        """Indicator matrix S (n_sectors x n_assets) for sector-cap constraints."""
        sectors = sorted({self.sector_map.get(a, "UNKNOWN") for a in assets})
        S = np.zeros((len(sectors), len(assets)))
        for j, a in enumerate(assets):
            i = sectors.index(self.sector_map.get(a, "UNKNOWN"))
            S[i, j] = 1.0
        return S, sectors


def apply_caps(weights: pd.Series, max_weight: float, iters: int = 100) -> pd.Series:
    """Cap weights and redistribute the excess proportionally among uncapped names.

    Used by the closed-form allocators (equal-weight, inverse-vol, score-based)
    which have no optimiser to enforce bounds for them.
    """
    w = weights.clip(lower=0).astype(float)
    if w.sum() <= 0:
        return pd.Series(1.0 / len(w), index=w.index)
    w = w / w.sum()
    max_weight = max(max_weight, 1.0 / len(w))

    for _ in range(iters):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        free = ~over
        if not free.any() or w[free].sum() <= 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def correlation_filter(
    returns: pd.DataFrame, ranking: pd.Series, threshold: float = 0.85
) -> list[str]:
    """Drop the lower-ranked member of any pair correlated above `threshold`.

    This was my original notebook's idea, kept because it's a sound one, but
    with two fixes: it walks the ranking in order now (so the outcome is
    deterministic instead of depending on column order), and it checks
    against the names already kept rather than the whole universe.
    """
    corr = returns.corr().abs()
    ordered = [t for t in ranking.sort_values(ascending=False).index if t in corr.columns]
    kept: list[str] = []
    for cand in ordered:
        if all(corr.loc[cand, k] < threshold for k in kept):
            kept.append(cand)
    return kept


def select_candidates(
    scores: pd.Series,
    returns: pd.DataFrame,
    n_select: int = 25,
    corr_threshold: float = 0.85,
    sector_map: dict[str, str] | None = None,
    max_per_sector: int = 5,
) -> list[str]:
    """Rank -> de-correlate -> sector-cap -> take top n.

    Order matters here: de-correlating before truncating means the full
    ranked list gets considered when hunting for diversifiers, not just the
    top 25 — which would defeat the point, since the top 25 by Sharpe in
    Indian equities is frequently just 12 banks.
    """
    scores = scores.dropna()
    if scores.empty:
        return []

    kept = correlation_filter(returns[scores.index.intersection(returns.columns)], scores, corr_threshold)

    if sector_map:
        counts: dict[str, int] = {}
        capped = []
        for t in kept:
            sec = sector_map.get(t, "UNKNOWN")
            if counts.get(sec, 0) < max_per_sector:
                capped.append(t)
                counts[sec] = counts.get(sec, 0) + 1
        kept = capped

    return kept[:n_select]
