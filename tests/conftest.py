"""Synthetic fixtures.

Every test in this suite runs offline on generated data. No test touches
yfinance. That is deliberate: a test suite that needs a market data API is a
test suite that fails in CI on a Sunday, and people stop trusting it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

N_ASSETS = 20
N_DAYS = 2200  # ~8.7 years of business days
SEED = 7


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def calendar() -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=N_DAYS)


@pytest.fixture(scope="session")
def market_returns(calendar) -> pd.Series:
    """A market factor with realistic volatility clustering."""
    r = np.random.default_rng(SEED)
    vol = np.zeros(len(calendar))
    vol[0] = 0.011
    shocks = r.standard_normal(len(calendar))
    # GARCH-ish recursion so the series has genuine vol persistence
    for t in range(1, len(vol)):
        vol[t] = np.sqrt(0.000002 + 0.08 * (vol[t - 1] * shocks[t - 1]) ** 2 + 0.90 * vol[t - 1] ** 2)
    return pd.Series(0.0004 + vol * shocks, index=calendar, name="market")


@pytest.fixture(scope="session")
def returns(calendar, market_returns) -> pd.DataFrame:
    """20 assets with heterogeneous betas, idiosyncratic vol, and alphas."""
    r = np.random.default_rng(SEED + 1)
    betas = r.uniform(0.5, 1.6, N_ASSETS)
    idio_vol = r.uniform(0.008, 0.022, N_ASSETS)
    alphas = r.normal(0.0001, 0.0002, N_ASSETS)

    m = market_returns.to_numpy()[:, None]
    eps = r.standard_normal((len(calendar), N_ASSETS)) * idio_vol
    data = alphas + betas * m + eps
    cols = [f"STK{i:02d}.NS" for i in range(N_ASSETS)]
    return pd.DataFrame(data, index=calendar, columns=cols)


@pytest.fixture(scope="session")
def prices(returns) -> pd.DataFrame:
    return 100 * (1 + returns).cumprod()


@pytest.fixture(scope="session")
def sector_map(returns) -> dict[str, str]:
    sectors = ["Financials", "IT", "Pharma", "Energy", "FMCG"]
    return {c: sectors[i % len(sectors)] for i, c in enumerate(returns.columns)}


@pytest.fixture(scope="session")
def ohlcv(prices, returns) -> pd.DataFrame:
    """Long-format OHLCV consistent with the close series."""
    r = np.random.default_rng(SEED + 2)
    frames = []
    for tk in prices.columns:
        close = prices[tk]
        noise = np.abs(r.normal(0, 0.006, len(close)))
        high = close * (1 + noise)
        low = close * (1 - np.abs(r.normal(0, 0.006, len(close))))
        open_ = close.shift(1).fillna(close.iloc[0])
        frames.append(
            pd.DataFrame(
                {
                    "date": close.index,
                    "ticker": tk,
                    "open": open_.to_numpy(),
                    "high": np.maximum.reduce([high.to_numpy(), close.to_numpy(), open_.to_numpy()]),
                    "low": np.minimum.reduce([low.to_numpy(), close.to_numpy(), open_.to_numpy()]),
                    "close": close.to_numpy(),
                    "volume": r.integers(1e5, 1e7, len(close)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def flat_returns(calendar) -> pd.Series:
    """Deterministic +0.1%/day — exact answers are computable by hand."""
    return pd.Series(0.001, index=calendar[:252])
