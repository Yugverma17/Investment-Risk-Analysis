"""Result tables, formatted for both CSV and README markdown."""

from __future__ import annotations

import pandas as pd

PCT_COLS = {
    "CAGR",
    "Ann.Return",
    "Ann.Vol",
    "MaxDD",
    "Alpha",
    "Ulcer",
    "HitRate",
    "AvgTurnover",
    "CostDrag",
}
RATIO_COLS = {"Sharpe", "Sortino", "Calmar", "Beta", "Treynor", "InfoRatio", "Skew", "Kurtosis"}


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Percent columns as percentages, ratios to 2dp — for display only."""
    out = df.copy()
    for col in out.columns:
        if col in PCT_COLS:
            out[col] = out[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
        elif col in RATIO_COLS:
            out[col] = out[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    return out


def to_markdown(df: pd.DataFrame, index_name: str = "Strategy") -> str:
    d = format_table(df)
    d.index.name = index_name
    return d.to_markdown()


def comparison_table(
    tearsheets: dict[str, pd.Series],
    turnover: dict[str, float] | None = None,
    cost_drag: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Assemble per-strategy tearsheets into the headline results table."""
    df = pd.DataFrame(tearsheets).T
    if turnover:
        df["AvgTurnover"] = pd.Series(turnover)
    if cost_drag:
        df["CostDrag"] = pd.Series(cost_drag)
    order = [
        c
        for c in [
            "CAGR",
            "Ann.Vol",
            "Sharpe",
            "Sortino",
            "MaxDD",
            "Calmar",
            "Beta",
            "Alpha",
            "InfoRatio",
            "HitRate",
            "AvgTurnover",
            "CostDrag",
        ]
        if c in df.columns
    ]
    return df[order + [c for c in df.columns if c not in order]]


def stress_windows() -> dict[str, tuple[str, str]]:
    """Historical stress windows for Indian equities.

    Labels are kept as single lines — they end up as DataFrame index values
    that get written straight into markdown tables (README, notebooks). An
    embedded newline there corrupts the table (each row would span two lines
    of markdown). `report.plots.stress_chart` wraps them for the x-axis
    itself instead of relying on a pre-baked line break.
    """
    return {
        "COVID crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
        "COVID recovery (Apr-Dec 2020)": ("2020-03-24", "2020-12-31"),
        "2022 rate shock (Jan-Jun 2022)": ("2022-01-01", "2022-06-17"),
        "Oct 2021 - Mar 2023 (flat market)": ("2021-10-19", "2023-03-31"),
        "2024-25 correction (Sep 24-Feb 25)": ("2024-09-27", "2025-02-28"),
    }


def stress_test(
    strategies: dict[str, pd.Series], benchmark: pd.Series
) -> pd.DataFrame:
    """Total return of each strategy through each stress window."""
    rows = {}
    for label, (start, end) in stress_windows().items():
        row = {}
        for name, r in strategies.items():
            sl = r.loc[start:end]
            row[name] = float((1 + sl.fillna(0)).prod() - 1) if len(sl) else float("nan")
        bsl = benchmark.loc[start:end]
        row["Nifty 50"] = float((1 + bsl.fillna(0)).prod() - 1) if len(bsl) else float("nan")
        rows[label] = row
    return pd.DataFrame(rows).T
