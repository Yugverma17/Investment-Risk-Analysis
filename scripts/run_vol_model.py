"""Phase 5 experiment: volatility forecasting horse race.

    python scripts/run_vol_model.py [--no-garch]

Outputs
-------
results/tables/vol_model_comparison.csv   model scores (RMSE / MAE / QLIKE / R2)
results/tables/vol_dm_tests.csv           Diebold-Mariano significance tests
results/tables/vol_feature_importance.csv LightGBM gain by feature
results/tables/vol_by_year.csv            per-fold breakdown
data/processed/vol_forecasts.parquet      OOS forecast panel for the backtester
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from riskengine.config import DATA_PROCESSED, TABLES  # noqa: E402
from riskengine.models import (  # noqa: E402
    build_feature_panel,
    diebold_mariano,
    evaluate_predictions,
    fit_garch_forecasts,
    forecasts_to_panel,
    walk_forward_predict,
)
from riskengine.models.vol_forecast import score_all  # noqa: E402
from riskengine.report.plots import (  # noqa: E402
    feature_importance,
    vol_model_by_year,
    vol_model_scatter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("volmodel")

START_YEAR = 2018


def main(no_garch: bool = False, rebuild: bool = False) -> None:
    panel_path = DATA_PROCESSED / "vol_panel.parquet"

    if panel_path.exists() and not rebuild:
        panel = pd.read_parquet(panel_path)
        log.info("loaded cached feature panel %s", panel.shape)
    else:
        close = pd.read_parquet(DATA_PROCESSED / "close.parquet")
        ohlcv = pd.read_parquet(DATA_PROCESSED / "ohlcv.parquet")
        bench = pd.read_parquet(DATA_PROCESSED / "benchmark.parquet")["close"]
        vix_path = DATA_PROCESSED / "vix.parquet"
        vix = pd.read_parquet(vix_path)["close"] if vix_path.exists() else None
        panel = build_feature_panel(ohlcv, close, bench, vix)
        panel.to_parquet(panel_path, index=False)
        log.info("built feature panel %s", panel.shape)

    # ---------------------------------------------------------- LightGBM ---
    log.info("=== walk-forward LightGBM (expanding window, one fold per year) ===")
    t0 = time.time()
    preds, importance = walk_forward_predict(panel, start_year=START_YEAR)
    log.info("%d OOS predictions across %d folds in %.1fs", len(preds), preds.fold_year.nunique(), time.time() - t0)

    importance[["mean_gain"]].to_csv(TABLES / "vol_feature_importance.csv")

    # -------------------------------------------------------------- GARCH ---
    garch = pd.DataFrame()
    garch_path = DATA_PROCESSED / "garch_forecasts.parquet"
    if not no_garch:
        pts = pd.DatetimeIndex(sorted(preds.date.unique()))
        if garch_path.exists() and not rebuild:
            garch = pd.read_parquet(garch_path)
            # cache is only valid if it covers every sample point this run needs
            if not pd.DatetimeIndex(pts).difference(garch.index).empty:
                log.info("cached GARCH forecasts are missing sample points — refitting")
                garch = pd.DataFrame()
            else:
                log.info("loaded cached GARCH forecasts %s", garch.shape)
        if garch.empty:
            log.info("=== GARCH(1,1)-t baseline (this takes ~15-20 min) ===")
            close = pd.read_parquet(DATA_PROCESSED / "close.parquet")
            rets = close.pct_change()
            t0 = time.time()
            garch = fit_garch_forecasts(rets, pts)
            log.info("GARCH done in %.1f min", (time.time() - t0) / 60)
            garch.to_parquet(garch_path)

    # --------------------------------------------------------- evaluation ---
    table = evaluate_predictions(preds, garch if not garch.empty else None)
    table.round(5).to_csv(TABLES / "vol_model_comparison.csv")
    log.info("\n%s", table.round(4).to_string())

    # ------------------------------------------------ Diebold-Mariano tests --
    df = preds.copy()
    if not garch.empty:
        g = garch.stack().rename("garch").reset_index()
        g.columns = ["date", "ticker", "garch"]
        df = df.merge(g, on=["date", "ticker"], how="left")

    dm_rows = []
    contenders = {"Random walk (RV21)": "rv_21", "EWMA(0.94)": "ewma_94"}
    if "garch" in df.columns:
        contenders["GARCH(1,1)-t"] = "garch"

    for label, col in contenders.items():
        sub = df.dropna(subset=["actual", "lgbm", col])
        for loss in ("qlike", "mse"):
            stat, p = diebold_mariano(
                sub["actual"].to_numpy(), sub["lgbm"].to_numpy(), sub[col].to_numpy(), loss=loss
            )
            dm_rows.append(
                {
                    "comparison": f"LightGBM vs {label}",
                    "loss": loss.upper(),
                    "DM_stat": round(stat, 3) if stat == stat else None,
                    "p_value": round(p, 4) if p == p else None,
                    "n": len(sub),
                    "verdict": (
                        "LightGBM better (5%)"
                        if (p == p and p < 0.05 and stat < 0)
                        else ("baseline better (5%)" if (p == p and p < 0.05 and stat > 0) else "no significant difference")
                    ),
                }
            )
    dm = pd.DataFrame(dm_rows)
    dm.to_csv(TABLES / "vol_dm_tests.csv", index=False)
    log.info("\n%s", dm.to_string(index=False))

    # -------------------------------------------------------- per-year ------
    year_rows = []
    for y, grp in preds.groupby("fold_year"):
        for name, col in [("LightGBM", "lgbm"), ("EWMA(0.94)", "ewma_94"), ("RandomWalk", "rv_21")]:
            s = score_all(grp["actual"].to_numpy(), grp[col].to_numpy())
            year_rows.append({"year": y, "model": name, **{k: round(v, 5) for k, v in s.items()}, "n": len(grp)})
    by_year = pd.DataFrame(year_rows)
    by_year.to_csv(TABLES / "vol_by_year.csv", index=False)

    # ------------------------------------------ forecasts for the backtest --
    fc = forecasts_to_panel(preds, "lgbm")
    fc.to_parquet(DATA_PROCESSED / "vol_forecasts.parquet")
    preds.to_parquet(DATA_PROCESSED / "vol_predictions.parquet", index=False)
    log.info("wrote forecast panel %s", fc.shape)
    log.info("top features:\n%s", importance["mean_gain"].head(12).round(0).to_string())

    # -------------------------------------------------------------- figures --
    log.info("=== figures ===")
    vol_model_scatter(preds)
    feature_importance(importance["mean_gain"])
    vol_model_by_year(by_year, metric="QLIKE")
    log.info("wrote vol_pred_vs_actual.png, vol_feature_importance.png, vol_by_year.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-garch", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the feature panel")
    main(**vars(ap.parse_args()))
