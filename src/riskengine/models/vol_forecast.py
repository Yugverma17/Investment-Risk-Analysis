"""Volatility forecasting: LightGBM vs. EWMA vs. GARCH(1,1).

What I'm actually testing: can a gradient-boosted model, given range-based
estimators and market-state features, forecast 21-day-ahead realised
volatility better than the two standard baselines?

Why volatility and not returns: volatility is autocorrelated and clusters —
that's the empirical fact basically every GARCH paper since 1986 is built
on — so it's genuinely something you can forecast. Returns, at daily and
monthly horizons for liquid large-caps, essentially aren't. Forecasting the
thing that's actually forecastable is the whole methodological point of this
part of the project, which is also why I made sure the baselines here are
strong and not strawmen.

On evaluation:
* Walk-forward: train on everything before year Y, predict year Y, roll
  forward. No fold ever sees its own future.
* QLIKE alongside RMSE. RMSE on volatility over-weights the high-vol names —
  a 10% error on a 60%-vol stock dwarfs a 10% error on a 15%-vol stock. QLIKE
  is what the volatility literature actually uses, because it's robust to
  that and to noise in the realised-vol proxy itself.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from ..config import SEED
from .features import TARGET, feature_columns

log = logging.getLogger(__name__)

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
    "num_threads": 4,
}
N_ROUNDS = 400


# --------------------------------------------------------------------------
# Loss functions
# --------------------------------------------------------------------------


def qlike(actual: np.ndarray, pred: np.ndarray) -> float:
    """QLIKE loss on variances: mean( s2/f2 - log(s2/f2) - 1 ).

    Minimised at f2 == s2, asymmetric (penalises under-forecasting more than
    over-forecasting), and robust to the fact that realised vol is itself a
    noisy proxy for true vol.
    """
    a = np.asarray(actual, dtype=float) ** 2
    p = np.clip(np.asarray(pred, dtype=float) ** 2, 1e-10, None)
    r = a / p
    return float(np.mean(r - np.log(r) - 1))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))))


def r2(actual: np.ndarray, pred: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def score_all(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"RMSE": rmse(actual, pred), "MAE": mae(actual, pred), "QLIKE": qlike(actual, pred), "R2": r2(actual, pred)}


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def baseline_random_walk(panel: pd.DataFrame) -> np.ndarray:
    """Tomorrow's 21-day vol = today's trailing 21-day vol. The naive benchmark."""
    return panel["rv_21"].to_numpy()


def baseline_ewma(panel: pd.DataFrame) -> np.ndarray:
    """RiskMetrics EWMA(0.94). A genuinely strong baseline, not a strawman."""
    return panel["ewma_94"].to_numpy()


def baseline_hybrid(panel: pd.DataFrame) -> np.ndarray:
    """Simple average of short and long realised vol — a crude mean-reversion proxy."""
    return (0.5 * panel["rv_21"] + 0.5 * panel["rv_252"].fillna(panel["rv_63"])).to_numpy()


def fit_garch_forecasts(
    returns: pd.DataFrame,
    sample_points: pd.DatetimeIndex,
    horizon: int = 21,
    window: int = 750,
    refit_every: int = 3,
    max_tickers: int | None = None,
) -> pd.DataFrame:
    """GARCH(1,1) 21-day-ahead volatility forecasts at each sample point.

    Refit every `refit_every` sample points and carry parameters forward, which
    is standard practice (GARCH parameters are stable) and cuts runtime ~3x.
    Failures (non-convergence) are returned as NaN rather than silently filled,
    so the comparison table can report how often GARCH simply did not fit.
    """
    try:
        from arch import arch_model
    except ImportError:  # pragma: no cover
        log.warning("arch not installed — skipping GARCH baseline")
        return pd.DataFrame()

    tickers = list(returns.columns)
    if max_tickers:
        rng = np.random.default_rng(SEED)
        tickers = sorted(rng.choice(tickers, size=min(max_tickers, len(tickers)), replace=False))

    out = pd.DataFrame(index=sample_points, columns=tickers, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tk in tickers:
            series = returns[tk].dropna() * 100  # arch prefers percent returns
            params = None
            for i, dt in enumerate(sample_points):
                hist = series[series.index <= dt].tail(window)
                if len(hist) < 250:
                    continue
                try:
                    am = arch_model(hist, vol="Garch", p=1, q=1, mean="Constant", dist="t")
                    if params is None or i % refit_every == 0:
                        fitted = am.fit(disp="off", show_warning=False)
                        params = fitted.params
                    else:
                        fitted = am.fix(params)
                    fc = fitted.forecast(horizon=horizon, reindex=False)
                    var_path = fc.variance.to_numpy()[-1]
                    # average daily variance over the horizon -> annualised vol
                    ann = np.sqrt(np.mean(var_path)) / 100 * np.sqrt(252)
                    out.loc[dt, tk] = float(ann)
                except Exception:  # noqa: BLE001 - a failed fit is data, not a crash
                    params = None
                    continue
    return out


# --------------------------------------------------------------------------
# LightGBM walk-forward
# --------------------------------------------------------------------------


def walk_forward_predict(
    panel: pd.DataFrame,
    start_year: int = 2018,
    min_train_rows: int = 3000,
    log_target: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window walk-forward LightGBM.

    For each year Y >= start_year: train on everything dated before Y, predict Y.

    `log_target=True` regresses log(vol) instead of vol directly. Volatility
    is right-skewed and strictly positive, so modelling the log makes the
    residuals more symmetric, guarantees positive predictions, and — since
    RMSE in log space works out to roughly a percentage error — keeps a
    handful of crisis observations from dominating the whole loss.
    """
    import lightgbm as lgb

    feats = feature_columns(panel)
    panel = panel.copy()
    panel["year"] = panel["date"].dt.year
    years = sorted(y for y in panel["year"].unique() if y >= start_year)

    preds: list[pd.DataFrame] = []
    importances: list[pd.Series] = []

    for y in years:
        train = panel[panel["year"] < y]
        test = panel[panel["year"] == y]
        if len(train) < min_train_rows or test.empty:
            continue

        X_tr = train[feats]
        y_tr = np.log(train[TARGET]) if log_target else train[TARGET]

        model = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(X_tr, label=y_tr, feature_name=feats),
            num_boost_round=N_ROUNDS,
        )

        raw = model.predict(test[feats])
        pred = np.exp(raw) if log_target else raw

        preds.append(
            pd.DataFrame(
                {
                    "date": test["date"].to_numpy(),
                    "ticker": test["ticker"].to_numpy(),
                    "actual": test[TARGET].to_numpy(),
                    "lgbm": pred,
                    "rv_21": test["rv_21"].to_numpy(),
                    "ewma_94": test["ewma_94"].to_numpy(),
                    "rv_252": test["rv_252"].to_numpy(),
                    "fold_year": y,
                }
            )
        )
        importances.append(
            pd.Series(model.feature_importance("gain"), index=feats, name=str(y))
        )
        log.info("fold %d: train=%d test=%d", y, len(train), len(test))

    if not preds:
        raise ValueError("no walk-forward folds produced — check start_year and panel size")

    out = pd.concat(preds, ignore_index=True)
    imp = pd.concat(importances, axis=1)
    imp["mean_gain"] = imp.mean(axis=1)
    return out, imp.sort_values("mean_gain", ascending=False)


def evaluate_predictions(
    preds: pd.DataFrame, garch: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Model comparison table across every forecaster, on identical rows."""
    df = preds.copy()
    models = {
        "Random walk (RV21)": df["rv_21"],
        "EWMA(0.94)": df["ewma_94"],
        "Hybrid (RV21+RV252)/2": 0.5 * df["rv_21"] + 0.5 * df["rv_252"].fillna(df["rv_21"]),
        "LightGBM": df["lgbm"],
    }

    if garch is not None and not garch.empty:
        g = garch.stack().rename("garch").reset_index()
        g.columns = ["date", "ticker", "garch"]
        df = df.merge(g, on=["date", "ticker"], how="left")
        models["GARCH(1,1)-t"] = df["garch"]
        # Restrict every model to rows where GARCH also produced a forecast, so
        # the comparison is on identical observations.
        mask = df["garch"].notna()
    else:
        mask = pd.Series(True, index=df.index)

    rows = []
    for name, pred in models.items():
        sub_a = df.loc[mask, "actual"].to_numpy()
        sub_p = pd.Series(pred).loc[mask].to_numpy()
        ok = np.isfinite(sub_a) & np.isfinite(sub_p) & (sub_p > 0)
        s = score_all(sub_a[ok], sub_p[ok])
        s["n"] = int(ok.sum())
        rows.append(pd.Series(s, name=name))

    table = pd.DataFrame(rows)
    base = table.loc["EWMA(0.94)"]
    table["RMSE_vs_EWMA_%"] = (table["RMSE"] / base["RMSE"] - 1) * 100
    table["QLIKE_vs_EWMA_%"] = (table["QLIKE"] / base["QLIKE"] - 1) * 100
    return table


def diebold_mariano(
    actual: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, loss: str = "qlike"
) -> tuple[float, float]:
    """Diebold-Mariano test: is model A's forecast loss significantly below B's?

    Without this, "LightGBM beat EWMA by 4%" is just a claim about a
    difference that could easily be noise. This is what actually backs it up.
    """
    from scipy import stats

    a = np.asarray(actual, dtype=float)
    pa = np.clip(np.asarray(pred_a, dtype=float), 1e-8, None)
    pb = np.clip(np.asarray(pred_b, dtype=float), 1e-8, None)

    if loss == "qlike":
        la = (a**2) / (pa**2) - np.log((a**2) / (pa**2)) - 1
        lb = (a**2) / (pb**2) - np.log((a**2) / (pb**2)) - 1
    else:
        la = (a - pa) ** 2
        lb = (a - pb) ** 2

    d = la - lb
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        return np.nan, np.nan
    # Newey-West standard error (h=21 overlapping horizon)
    lags = 21
    e = d - d.mean()
    var = float(e @ e / n)
    for lag in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - lag / (lags + 1)) * float(e[lag:] @ e[:-lag] / n)
    se = np.sqrt(max(var / n, 1e-18))
    stat = d.mean() / se
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def forecasts_to_panel(preds: pd.DataFrame, column: str = "lgbm") -> pd.DataFrame:
    """Pivot walk-forward predictions into a date x ticker panel for the backtester."""
    return preds.pivot(index="date", columns="ticker", values=column).sort_index()
