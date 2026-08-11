"""Statistical significance testing for backtest results.

Why bother
----------
A backtest that reports "Sharpe 0.94 vs benchmark 0.71" invites the question
nobody usually asks: is that difference distinguishable from luck over ~7 years
of data? Usually it is not. These functions produce the confidence interval that
answers it, and the project reports that interval rather than the point estimate
alone.

Two effects are handled:

* Autocorrelation — daily returns are not i.i.d., so a naive bootstrap
  understates the standard error. The stationary bootstrap (Politis & Romano)
  resamples blocks of random length, preserving short-range dependence.
* Multiple testing — trying six strategies and reporting the best one inflates
  the winner's Sharpe. `deflated_sharpe_ratio` adjusts for exactly that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import SEED, TRADING_DAYS
from ..risk.metrics import sharpe_ratio


def stationary_bootstrap_indices(
    n: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap index draw."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < p:
            idx[i] = rng.integers(0, n)
        else:
            idx[i] = (idx[i - 1] + 1) % n
    return idx


def bootstrap_sharpe_difference(
    strategy: pd.Series,
    benchmark: pd.Series,
    n_boot: int = 2000,
    mean_block: int = 21,
    seed: int = SEED,
    rf_annual: float = 0.065,
) -> dict[str, float]:
    """Bootstrap CI for (Sharpe_strategy - Sharpe_benchmark).

    Both series are resampled with the SAME indices each iteration, which
    preserves their contemporaneous correlation — resampling them independently
    would inflate the variance of the difference enormously.
    """
    df = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    if len(df) < 60:
        return {"diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}

    s = df["s"].to_numpy()
    b = df["b"].to_numpy()
    observed = sharpe_ratio(df["s"], rf_annual) - sharpe_ratio(df["b"], rf_annual)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    n = len(df)
    for i in range(n_boot):
        idx = stationary_bootstrap_indices(n, mean_block, rng)
        diffs[i] = sharpe_ratio(pd.Series(s[idx]), rf_annual) - sharpe_ratio(
            pd.Series(b[idx]), rf_annual
        )

    diffs = diffs[np.isfinite(diffs)]
    centred = diffs - diffs.mean()
    # two-sided p-value: how often does a centred bootstrap draw exceed |observed|?
    p = float(np.mean(np.abs(centred) >= abs(observed))) if diffs.size else np.nan

    return {
        "diff": float(observed),
        "ci_low": float(np.percentile(diffs, 2.5)) if diffs.size else np.nan,
        "ci_high": float(np.percentile(diffs, 97.5)) if diffs.size else np.nan,
        "p_value": p,
        "n_obs": int(n),
    }


def probabilistic_sharpe_ratio(
    returns: pd.Series, benchmark_sr: float = 0.0, rf_annual: float = 0.065
) -> float:
    """Bailey & Lopez de Prado's PSR: P(true Sharpe > benchmark_sr).

    Corrects the Sharpe standard error for skewness and kurtosis. Negatively
    skewed, fat-tailed return streams — which is every equity strategy — have a
    LARGER Sharpe standard error than the Gaussian formula suggests, so the
    naive t-stat overstates significance.
    """
    r = returns.dropna()
    n = len(r)
    if n < 30:
        return np.nan

    sr = sharpe_ratio(r, rf_annual) / np.sqrt(TRADING_DAYS)  # per-period
    sr_b = benchmark_sr / np.sqrt(TRADING_DAYS)
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))

    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    z = (sr - sr_b) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int, rf_annual: float = 0.065
) -> float:
    """PSR with the benchmark set to the Sharpe you'd expect from the BEST of
    `n_trials` random strategies.

    If you test 6 strategies and pick the winner, the winner's Sharpe is biased
    upward even if all 6 are worthless. This is the correction for that, and it
    is why the README reports how many strategy variants were actually tried.
    """
    r = returns.dropna()
    if len(r) < 30 or n_trials < 1:
        return np.nan

    e = np.euler_gamma
    # Expected maximum of n_trials draws from a standard normal
    if n_trials == 1:
        expected_max = 0.0
    else:
        expected_max = (1 - e) * stats.norm.ppf(1 - 1 / n_trials) + e * stats.norm.ppf(
            1 - 1 / (n_trials * np.e)
        )

    sr_std = r.std(ddof=1)  # variance of the per-period Sharpe estimate ~ 1/sqrt(n)
    if sr_std == 0:
        return np.nan
    threshold_daily = expected_max / np.sqrt(len(r) - 1)
    return probabilistic_sharpe_ratio(r, threshold_daily * np.sqrt(TRADING_DAYS), rf_annual)


def newey_west_tstat(active_returns: pd.Series, lags: int = 5) -> tuple[float, float]:
    """t-statistic on mean active return with a HAC (Newey-West) standard error."""
    x = active_returns.dropna().to_numpy()
    n = len(x)
    if n < 30:
        return np.nan, np.nan
    mu = x.mean()
    e = x - mu
    gamma0 = float(e @ e / n)
    var = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1 - lag / (lags + 1)
        gamma = float(e[lag:] @ e[:-lag] / n)
        var += 2 * w * gamma
    se = np.sqrt(max(var / n, 1e-18))
    t = mu / se
    return float(t), float(2 * (1 - stats.norm.cdf(abs(t))))


def significance_table(
    results: dict[str, pd.Series], benchmark: pd.Series, n_boot: int = 1000
) -> pd.DataFrame:
    """One row per strategy: Sharpe difference, bootstrap CI, HAC t-stat, DSR."""
    rows = []
    n_trials = len(results)
    for name, r in results.items():
        boot = bootstrap_sharpe_difference(r, benchmark, n_boot=n_boot)
        bench_aligned = benchmark.reindex(r.index)
        t, p = newey_west_tstat(r - bench_aligned)
        rows.append(
            {
                "strategy": name,
                "sharpe_diff": round(boot["diff"], 3) if boot["diff"] == boot["diff"] else np.nan,
                "ci_95_low": round(boot["ci_low"], 3),
                "ci_95_high": round(boot["ci_high"], 3),
                "bootstrap_p": round(boot["p_value"], 3),
                "active_t_stat": round(t, 2) if t == t else np.nan,
                "active_p": round(p, 3) if p == p else np.nan,
                "PSR": round(probabilistic_sharpe_ratio(r), 3),
                "DSR": round(deflated_sharpe_ratio(r, n_trials), 3),
                "significant_5pct": bool(
                    boot["ci_low"] == boot["ci_low"] and boot["ci_low"] > 0
                ),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")
