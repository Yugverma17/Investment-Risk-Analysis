"""Cold-start data fetch. Run once; everything else reads the parquet cache.

    python scripts/fetch_data.py [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from riskengine.config import (  # noqa: E402
    BENCHMARK,
    DATA_PROCESSED,
    END_DATE,
    START_DATE,
    TABLES,
    VIX_TICKER,
)
from riskengine.data import (  # noqa: E402
    clean_prices,
    data_quality_report,
    fetch_benchmark,
    fetch_ohlcv,
    get_tickers,
    to_wide,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("fetch")


def main(force: bool = False) -> None:
    tickers = get_tickers()
    log.info("universe: %d tickers, %s to %s", len(tickers), START_DATE, END_DATE)

    log.info("=== benchmark ===")
    bench = fetch_benchmark(BENCHMARK, force=force)
    calendar = pd.DatetimeIndex(bench.index)
    log.info("benchmark: %d trading days (%s to %s)", len(calendar), calendar[0].date(), calendar[-1].date())

    log.info("=== India VIX ===")
    try:
        vix = fetch_benchmark(VIX_TICKER, force=force)
        log.info("vix: %d obs", len(vix))
    except Exception as exc:  # noqa: BLE001
        log.warning("India VIX unavailable (%s) — vol model will run without it", exc)
        vix = pd.Series(dtype=float, name=VIX_TICKER)

    log.info("=== universe OHLCV ===")
    ohlcv = fetch_ohlcv(tickers, force=force)
    log.info("downloaded %d rows across %d tickers", len(ohlcv), ohlcv.ticker.nunique())

    close = to_wide(ohlcv, "close")
    report = data_quality_report(close, calendar, requested=tickers)
    report.to_csv(TABLES / "data_quality_report.csv")

    counts = report["status"].value_counts()
    log.info("data quality: %s", counts.to_dict())

    clean = clean_prices(close, calendar, report)
    log.info("clean panel: %d days x %d tickers", *clean.shape)

    # Persist the processed panel plus the OHLC fields needed by range-based
    # volatility estimators, restricted to the surviving tickers.
    keep = list(clean.columns)
    ohlcv_clean = ohlcv[ohlcv.ticker.isin(keep)]

    clean.to_parquet(DATA_PROCESSED / "close.parquet")
    ohlcv_clean.to_parquet(DATA_PROCESSED / "ohlcv.parquet", index=False)
    bench.to_frame("close").to_parquet(DATA_PROCESSED / "benchmark.parquet")
    if not vix.empty:
        vix.to_frame("close").to_parquet(DATA_PROCESSED / "vix.parquet")

    log.info("wrote processed panels to %s", DATA_PROCESSED)
    log.info("wrote quality report to %s", TABLES / "data_quality_report.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore cache and re-download")
    main(**vars(ap.parse_args()))
