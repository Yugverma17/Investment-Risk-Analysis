"""VaR validation.

A VaR number that has never been backtested is decoration. These tests answer
two separate questions:

  Kupiec (unconditional coverage) — did we breach the right NUMBER of times?
  Christoffersen (independence)   — were the breaches spread out, or clustered?

A model can pass Kupiec and fail Christoffersen: it breaches 5% of the time,
but all the breaches land in one week of March 2020. That model is useless for
risk management, and only the joint test catches it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .var import historical_var, parametric_var


@dataclass
class VaRTestResult:
    n_obs: int
    n_breaches: int
    expected_breaches: float
    breach_rate: float
    kupiec_stat: float
    kupiec_pvalue: float
    christoffersen_stat: float
    christoffersen_pvalue: float
    cc_stat: float
    cc_pvalue: float

    @property
    def passes(self) -> bool:
        """Joint conditional-coverage test at the 5% level."""
        return bool(self.cc_pvalue > 0.05)

    def to_series(self, name: str = "var_backtest") -> pd.Series:
        return pd.Series(
            {
                "n_obs": self.n_obs,
                "n_breaches": self.n_breaches,
                "expected": round(self.expected_breaches, 1),
                "breach_rate": round(self.breach_rate, 4),
                "kupiec_p": round(self.kupiec_pvalue, 4),
                "christoffersen_p": round(self.christoffersen_pvalue, 4),
                "cc_p": round(self.cc_pvalue, 4),
                "verdict": "PASS" if self.passes else "REJECT",
            },
            name=name,
        )


def _kupiec(n: int, x: int, p: float) -> tuple[float, float]:
    """Proportion-of-failures likelihood ratio, chi-square(1)."""
    if n == 0:
        return np.nan, np.nan
    if x == 0:
        lr = -2 * (n * np.log(1 - p))
    elif x == n:
        lr = -2 * (n * np.log(p))
    else:
        pi = x / n
        lr = -2 * (
            (n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - pi) - x * np.log(pi)
        )
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def _christoffersen_independence(breaches: np.ndarray) -> tuple[float, float]:
    """Markov-chain test that a breach today is independent of a breach yesterday."""
    b = breaches.astype(int)
    if b.size < 2:
        return np.nan, np.nan

    prev, curr = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    # If no breach ever follows a breach, the transition model is degenerate and
    # the test carries no information — report it as non-rejecting.
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return 0.0, 1.0

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _ll(p: float, zeros: int, ones: int) -> float:
        if p <= 0 or p >= 1:
            return 0.0
        return zeros * np.log(1 - p) + ones * np.log(p)

    ll_null = _ll(pi, n00 + n10, n01 + n11)
    ll_alt = _ll(pi01, n00, n01) + _ll(pi11, n10, n11)
    lr = -2 * (ll_null - ll_alt)
    lr = max(lr, 0.0)
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def evaluate_var(
    realised_returns: pd.Series, var_forecasts: pd.Series, confidence: float = 0.95
) -> VaRTestResult:
    """Score a stream of one-day-ahead VaR forecasts against what happened.

    Parameters
    ----------
    realised_returns : actual return on day t
    var_forecasts    : VaR (positive loss) forecast made at t-1 for day t
    """
    df = pd.concat(
        [realised_returns.rename("r"), var_forecasts.rename("var")], axis=1
    ).dropna()
    if df.empty:
        raise ValueError("no overlapping observations between returns and VaR forecasts")

    breaches = (df["r"] < -df["var"]).to_numpy()
    n, x = len(breaches), int(breaches.sum())
    p = 1 - confidence

    k_stat, k_p = _kupiec(n, x, p)
    c_stat, c_p = _christoffersen_independence(breaches)
    cc_stat = (k_stat if not np.isnan(k_stat) else 0.0) + (c_stat if not np.isnan(c_stat) else 0.0)
    cc_p = float(1 - stats.chi2.cdf(cc_stat, df=2))

    return VaRTestResult(
        n_obs=n,
        n_breaches=x,
        expected_breaches=n * p,
        breach_rate=x / n,
        kupiec_stat=k_stat,
        kupiec_pvalue=k_p,
        christoffersen_stat=c_stat,
        christoffersen_pvalue=c_p,
        cc_stat=cc_stat,
        cc_pvalue=cc_p,
    )


def rolling_var_forecasts(
    returns: pd.Series, window: int = 252, confidence: float = 0.95, method: str = "historical"
) -> pd.Series:
    """Walk-forward VaR: at each t, estimate VaR from the trailing `window` days only.

    Critically, the forecast on row t uses data up to and including t-1, so it
    is genuinely out-of-sample against the return on row t.
    """
    fn = {"historical": historical_var, "parametric": parametric_var}[method]
    r = returns.dropna()
    out = pd.Series(index=r.index, dtype=float)
    values = r.to_numpy()

    for i in range(window, len(r)):
        hist = pd.Series(values[i - window : i])
        out.iloc[i] = fn(hist, confidence)
    return out.dropna()


def var_backtest_table(
    returns: pd.Series, window: int = 252, confidences: tuple[float, ...] = (0.95, 0.99)
) -> pd.DataFrame:
    """Backtest both estimators at both confidence levels — the headline table."""
    rows = []
    for method in ("historical", "parametric"):
        for c in confidences:
            fc = rolling_var_forecasts(returns, window, c, method)
            res = evaluate_var(returns.reindex(fc.index), fc, c)
            rows.append(res.to_series(name=f"{method}@{c:.0%}"))
    return pd.DataFrame(rows)
