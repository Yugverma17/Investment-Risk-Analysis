"""Phase 4 + 5 experiment: the strategy horse race, end to end.

    python scripts/run_backtests.py [--no-vol-model] [--quick]

Produces every number and figure the README cites.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from riskengine.backtest import (  # noqa: E402
    BacktestConfig,
    run_backtest,
    significance_table,
)
from riskengine.config import DATA_PROCESSED, RESULTS, TABLES  # noqa: E402
from riskengine.data import sector_map  # noqa: E402
from riskengine.report import (  # noqa: E402
    comparison_table,
    drawdown_chart,
    equity_curves,
    rolling_sharpe,
    sector_exposure,
    stress_chart,
    stress_test,
    to_markdown,
    turnover_chart,
    var_breach_chart,
)
from riskengine.risk import (  # noqa: E402
    condition_number,
    ledoit_wolf_cov,
    rolling_var_forecasts,
    sample_cov,
    tearsheet,
    var_backtest_table,
    var_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("backtest")

ALLOCATORS = ["equal_weight", "inverse_vol", "min_variance", "risk_parity", "max_sharpe", "score_based"]


def load_data():
    close = pd.read_parquet(DATA_PROCESSED / "close.parquet")
    bench = pd.read_parquet(DATA_PROCESSED / "benchmark.parquet")["close"]
    mkt = bench.reindex(close.index).pct_change()
    return close, mkt


def main(no_vol_model: bool = False, quick: bool = False) -> None:
    close, mkt = load_data()
    smap = sector_map()
    log.info("panel %s | %s to %s", close.shape, close.index[0].date(), close.index[-1].date())

    results = {}
    n_boot = 300 if quick else 2000

    # ------------------------------------------------- 1. allocator horse race
    log.info("=== allocator comparison (balanced profile) ===")
    for alloc in ALLOCATORS:
        t0 = time.time()
        cfg = BacktestConfig(allocator=alloc, profile="balanced", label=alloc)
        results[alloc] = run_backtest(close, mkt, smap, cfg)
        log.info("  %-14s %.1fs", alloc, time.time() - t0)

    # ------------------------------------------ 2. shrinkage vs sample covariance
    log.info("=== covariance estimator ablation ===")
    cfg = BacktestConfig(
        allocator="min_variance", profile="balanced", cov_estimator="sample", label="min_variance (sample cov)"
    )
    results["min_variance_sample_cov"] = run_backtest(close, mkt, smap, cfg)

    # --------------------------------------------------------- 3. risk profiles
    log.info("=== risk profiles (score_based allocator) ===")
    profile_results = {}
    for prof in ("conservative", "balanced", "aggressive"):
        cfg = BacktestConfig(allocator="score_based", profile=prof, label=f"score_based/{prof}")
        profile_results[prof] = run_backtest(close, mkt, smap, cfg)

    # ------------------------------------- 4. does the vol forecast add anything?
    vol_results = {}
    fc_path = DATA_PROCESSED / "vol_forecasts.parquet"
    if not no_vol_model and fc_path.exists():
        log.info("=== volatility-forecast integration ===")
        vf = pd.read_parquet(fc_path)
        for alloc in ("min_variance", "risk_parity"):
            cfg = BacktestConfig(allocator=alloc, profile="balanced", label=f"{alloc} + vol forecast")
            vol_results[f"{alloc}_volfc"] = run_backtest(close, mkt, smap, cfg, vol_forecasts=vf)
            log.info("  %s + vol forecast done", alloc)
    elif not no_vol_model:
        log.warning("no vol forecasts found — run scripts/run_vol_model.py first")

    # ============================================================== reporting ==
    # Evaluate everything on the common window every strategy actually covers.
    common = None
    for r in list(results.values()) + list(profile_results.values()) + list(vol_results.values()):
        idx = r.returns.index
        common = idx if common is None else common.intersection(idx)
    bench_r = mkt.reindex(common).fillna(0.0)
    log.info("common evaluation window: %s to %s (%d days)", common[0].date(), common[-1].date(), len(common))

    def ts(res):
        return tearsheet(res.returns.reindex(common), bench_r, name=res.name)

    main_sheets = {r.name: ts(r) for r in results.values()}
    main_sheets["Nifty 50 (buy & hold)"] = tearsheet(bench_r, bench_r, name="Nifty 50 (buy & hold)")

    table = comparison_table(
        main_sheets,
        turnover={r.name: float(r.turnover.mean()) for r in results.values()},
        cost_drag={r.name: float(r.total_cost_drag) for r in results.values()},
    )
    table.to_csv(TABLES / "strategy_comparison.csv")
    log.info("\n%s", table.round(3).to_string())

    prof_table = comparison_table(
        {r.name: ts(r) for r in profile_results.values()},
        turnover={r.name: float(r.turnover.mean()) for r in profile_results.values()},
    )
    prof_table.to_csv(TABLES / "profile_comparison.csv")

    if vol_results:
        vol_cmp = {}
        for r in vol_results.values():
            vol_cmp[r.name] = ts(r)
        for base in ("min_variance", "risk_parity"):
            vol_cmp[f"{base} (historical vol)"] = ts(results[base])
        vol_table = comparison_table(vol_cmp)
        vol_table.to_csv(TABLES / "vol_forecast_impact.csv")
        log.info("\n=== vol forecast impact ===\n%s", vol_table.round(3).to_string())

    # ------------------------------------------------------ significance tests
    log.info("=== statistical significance vs Nifty (this is the slow part) ===")
    sig_input = {r.name: r.returns.reindex(common) for r in results.values()}
    sig = significance_table(sig_input, bench_r, n_boot=n_boot)
    sig.to_csv(TABLES / "significance_tests.csv")
    log.info("\n%s", sig.to_string())

    # ------------------------------------- best strategy vs equal weight, head to head
    from riskengine.backtest import bootstrap_sharpe_difference

    best_name = table.drop(index="Nifty 50 (buy & hold)")["Sharpe"].idxmax()
    best = next(r for r in results.values() if r.name == best_name)
    head_to_head = bootstrap_sharpe_difference(
        best.returns.reindex(common), results["equal_weight"].returns.reindex(common), n_boot=n_boot
    )
    log.info("BEST (%s) vs equal_weight: %s", best_name, head_to_head)
    (TABLES / "best_vs_equalweight.json").write_text(
        json.dumps({"best": best_name, **head_to_head}, indent=2, default=float)
    )

    # ---------------------------------------------------------- VaR validation
    log.info("=== VaR backtesting on the portfolio return stream ===")
    var_tbl = var_backtest_table(best.returns.reindex(common).dropna())
    var_tbl.to_csv(TABLES / "var_backtest.csv")
    log.info("\n%s", var_tbl.to_string())

    var_sum = var_summary(best.returns.reindex(common).dropna())
    var_sum.to_csv(TABLES / "var_summary.csv")

    # ------------------------------------------------------------ stress tests
    stress_input = {r.name: r.returns.reindex(common) for r in results.values() if r.name in ("equal_weight", "score_based", "min_variance")}
    stress = stress_test(stress_input, bench_r)
    stress.to_csv(TABLES / "stress_tests.csv")
    log.info("\n=== stress windows ===\n%s", stress.round(3).to_string())

    # --------------------------------------------- covariance conditioning demo
    train = close.pct_change().loc["2019-01-01":"2021-12-31"].dropna(axis=1, thresh=600)
    cond = {
        "sample": condition_number(sample_cov(train)),
        "ledoit_wolf": condition_number(ledoit_wolf_cov(train)),
        "n_assets": int(train.shape[1]),
        "n_obs": int(train.shape[0]),
    }
    (TABLES / "covariance_conditioning.json").write_text(json.dumps(cond, indent=2, default=float))
    log.info("condition numbers: %s", cond)

    # ================================================================ figures ==
    log.info("=== figures ===")
    curve_input = {r.name: r.returns.reindex(common) for r in results.values()}
    equity_curves(curve_input, bench_r)
    drawdown_chart(
        {k: v for k, v in curve_input.items() if k in ("equal_weight", "score_based", "min_variance")},
        bench_r,
    )
    rolling_sharpe(
        {k: v for k, v in curve_input.items() if k in ("equal_weight", "score_based", "min_variance")},
        bench_r,
    )
    turnover_chart({r.name: r.turnover for r in results.values()})
    sector_exposure(best.weights, smap)
    stress_chart(stress)

    vfc = rolling_var_forecasts(best.returns.reindex(common).dropna(), window=252, confidence=0.95)
    var_breach_chart(best.returns.reindex(common), vfc, 0.95)

    # ------------------------------------------------------- markdown snippets
    md = [
        "## Strategy comparison (walk-forward, net of costs)\n",
        f"_Evaluation window: {common[0].date()} to {common[-1].date()} "
        f"({len(common)/252:.1f} years, {len(results['equal_weight'].weights)} quarterly rebalances)_\n",
        to_markdown(table),
        "\n\n## Statistical significance vs Nifty 50\n",
        sig.to_markdown(),
        "\n\n## Risk profiles\n",
        to_markdown(prof_table),
        "\n\n## Stress windows (total return)\n",
        stress.map(lambda v: f"{v:.1%}" if pd.notna(v) else "—").to_markdown(),
        "\n\n## VaR backtest (best strategy)\n",
        var_tbl.to_markdown(),
    ]
    if vol_results:
        md += ["\n\n## Volatility-forecast integration\n", to_markdown(vol_table)]

    (RESULTS / "results_snippets.md").write_text("\n".join(md), encoding="utf-8")

    # persist raw return streams so the app and notebooks need not re-run anything
    streams = pd.DataFrame({r.name: r.returns for r in results.values()})
    streams["Nifty 50"] = mkt
    for r in vol_results.values():
        streams[r.name] = r.returns
    for r in profile_results.values():
        streams[r.name] = r.returns
    streams.to_parquet(RESULTS / "strategy_returns.parquet")
    best.weights.to_parquet(RESULTS / "best_weights.parquet")

    log.info("done — tables in %s, figures in %s", TABLES, RESULTS / "figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vol-model", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fewer bootstrap iterations")
    main(**vars(ap.parse_args()))
