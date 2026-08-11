"""Figures for the README, notebooks, and app.

One consistent visual language across every chart: same palette, same grid
weight, same annotation style. Charts are written to results/figures/ at 150 dpi.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from ..config import FIGURES  # noqa: E402
from ..risk.metrics import drawdown_series, sharpe_ratio  # noqa: E402

# Okabe-Ito: colourblind-safe, prints legibly in greyscale.
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky
    "#D55E00",  # vermillion
    "#7F7F7F",  # grey
]
BENCH_COLOR = "#333333"
GRID = {"color": "#DDDDDD", "linewidth": 0.7}


def _style(ax, title: str = "", ylabel: str = "", xlabel: str = ""):
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(True, alpha=0.6, **GRID)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")
    ax.tick_params(labelsize=9, colors="#444444")
    return ax


def _save(fig, name: str, outdir: Path | None = None) -> Path:
    outdir = outdir or FIGURES
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def equity_curves(
    strategies: dict[str, pd.Series],
    benchmark: pd.Series,
    name: str = "equity_curves",
    title: str = "Growth of ₹1 — walk-forward, net of costs",
    logy: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, (label, r) in enumerate(strategies.items()):
        curve = (1 + r.fillna(0)).cumprod()
        ax.plot(curve.index, curve.values, label=label, color=PALETTE[i % len(PALETTE)], lw=1.6)

    bcurve = (1 + benchmark.fillna(0)).cumprod()
    ax.plot(bcurve.index, bcurve.values, label="Nifty 50", color=BENCH_COLOR, lw=2.0, ls="--")

    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"₹{v:.1f}"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"₹{v:.1f}"))

    _style(ax, title, "Cumulative value of ₹1 (log scale)" if logy else "Cumulative value of ₹1")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, name)


def drawdown_chart(
    strategies: dict[str, pd.Series], benchmark: pd.Series, name: str = "drawdowns"
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (label, r) in enumerate(strategies.items()):
        dd = drawdown_series(r)
        ax.plot(dd.index, dd.values, label=label, color=PALETTE[i % len(PALETTE)], lw=1.4)
    bdd = drawdown_series(benchmark)
    ax.fill_between(bdd.index, bdd.values, 0, color=BENCH_COLOR, alpha=0.15, label="Nifty 50")

    _style(ax, "Drawdown — peak-to-trough decline", "Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower left")
    return _save(fig, name)


def rolling_sharpe(
    strategies: dict[str, pd.Series],
    benchmark: pd.Series,
    window: int = 252,
    name: str = "rolling_sharpe",
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (label, r) in enumerate(strategies.items()):
        rs = r.rolling(window).apply(lambda x: sharpe_ratio(pd.Series(x)), raw=False)
        ax.plot(rs.index, rs.values, label=label, color=PALETTE[i % len(PALETTE)], lw=1.4)
    brs = benchmark.rolling(window).apply(lambda x: sharpe_ratio(pd.Series(x)), raw=False)
    ax.plot(brs.index, brs.values, label="Nifty 50", color=BENCH_COLOR, lw=1.8, ls="--")
    ax.axhline(0, color="#888888", lw=0.8)

    _style(ax, f"Rolling {window//21}-month Sharpe ratio", "Sharpe (annualised)")
    ax.legend(frameon=False, fontsize=9, ncol=3)
    return _save(fig, name)


def var_breach_chart(
    returns: pd.Series, var_forecasts: pd.Series, confidence: float = 0.95, name: str = "var_breaches"
) -> Path:
    df = pd.concat([returns.rename("r"), var_forecasts.rename("v")], axis=1).dropna()
    breach = df["r"] < -df["v"]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df.index, df["r"], color="#BBBBBB", lw=0.7, label="Daily return")
    ax.plot(df.index, -df["v"], color=PALETTE[0], lw=1.4, label=f"{confidence:.0%} VaR threshold")
    ax.scatter(
        df.index[breach],
        df["r"][breach],
        color=PALETTE[5],
        s=18,
        zorder=5,
        label=f"Breaches ({int(breach.sum())})",
    )

    expected = len(df) * (1 - confidence)
    _style(
        ax,
        f"VaR backtest — {int(breach.sum())} breaches observed vs {expected:.0f} expected",
        "Daily return",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower left")
    return _save(fig, name)


def vol_model_scatter(preds: pd.DataFrame, name: str = "vol_pred_vs_actual") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (label, col) in zip(axes, [("LightGBM", "lgbm"), ("EWMA(0.94)", "ewma_94")], strict=True):
        sub = preds.dropna(subset=["actual", col])
        ax.scatter(sub[col], sub["actual"], s=4, alpha=0.15, color=PALETTE[0], edgecolors="none")
        lim = [0, float(np.nanpercentile(sub["actual"], 99))]
        ax.plot(lim, lim, color="#CC0000", lw=1.2, ls="--", label="perfect forecast")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        corr = float(np.corrcoef(sub[col], sub["actual"])[0, 1])
        _style(ax, f"{label}  (corr = {corr:.3f})", "Realised vol (next 21d)", "Forecast vol")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Forecast vs realised 21-day volatility (out-of-sample)",
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    return _save(fig, name)


def feature_importance(importance: pd.Series, top_n: int = 15, name: str = "vol_feature_importance") -> Path:
    top = importance.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.index, top.values / top.values.max(), color=PALETTE[0], height=0.7)
    _style(ax, f"LightGBM feature importance (top {top_n})", "", "Relative gain")
    return _save(fig, name)


def vol_model_by_year(by_year: pd.DataFrame, metric: str = "QLIKE", name: str = "vol_by_year") -> Path:
    piv = by_year.pivot(index="year", columns="model", values=metric)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(piv))
    width = 0.8 / len(piv.columns)
    for i, col in enumerate(piv.columns):
        ax.bar(x + i * width, piv[col].values, width, label=col, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x + width * (len(piv.columns) - 1) / 2)
    ax.set_xticklabels(piv.index)
    _style(ax, f"{metric} by walk-forward fold (lower is better)", metric, "Out-of-sample year")
    ax.legend(frameon=False, fontsize=9, ncol=3)
    return _save(fig, name)


def sector_exposure(weights: pd.DataFrame, sector_map: dict[str, str], name: str = "sector_exposure") -> Path:
    sectors = pd.Series({c: sector_map.get(c, "UNKNOWN") for c in weights.columns})
    by_sector = weights.T.groupby(sectors).sum().T.sort_index()
    keep = by_sector.mean().sort_values(ascending=False).head(8).index
    other = by_sector.drop(columns=keep).sum(axis=1)
    plot_df = by_sector[keep].copy()
    plot_df["Other"] = other

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.stackplot(
        plot_df.index,
        plot_df.T.values,
        labels=plot_df.columns,
        colors=[PALETTE[i % len(PALETTE)] for i in range(len(plot_df.columns))],
        alpha=0.9,
    )
    _style(ax, "Portfolio sector exposure through time", "Weight")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(fig, name)


def turnover_chart(results: dict[str, pd.Series], name: str = "turnover") -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.0))
    labels = list(results.keys())
    means = [float(v.mean()) for v in results.values()]
    ax.bar(labels, means, color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))], width=0.6)
    for i, v in enumerate(means):
        ax.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    _style(ax, "Average one-way turnover per quarterly rebalance", "Turnover")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, name)


def correlation_heatmap(returns: pd.DataFrame, name: str = "correlation_heatmap", max_n: int = 40) -> Path:
    corr = returns.iloc[:, :max_n].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels([c.replace(".NS", "") for c in corr.columns], rotation=90, fontsize=6)
    ax.set_yticklabels([c.replace(".NS", "") for c in corr.index], fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Correlation")
    ax.set_title("Pairwise daily return correlation", fontsize=12, fontweight="bold", loc="left")
    return _save(fig, name)


def stress_chart(stress: pd.DataFrame, name: str = "stress_tests") -> Path:
    # Labels are single-line in the DataFrame (they're written straight into
    # markdown tables elsewhere) — wrap only for the x-axis here, at render
    # time, rather than baking a newline into the data itself.
    wrapped = [lbl.replace(" (", "\n(", 1) for lbl in stress.index]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(stress.index))
    width = 0.8 / len(stress.columns)
    for i, col in enumerate(stress.columns):
        ax.bar(x + i * width, stress[col].values, width, label=col, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x + width * (len(stress.columns) - 1) / 2)
    ax.set_xticklabels(wrapped, fontsize=9)
    ax.axhline(0, color="#444444", lw=0.8)
    _style(ax, "Stress test — total return through historical crisis windows", "Return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=9, ncol=3)
    return _save(fig, name)
