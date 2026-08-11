"""Fill README.md result placeholders directly from results/tables/*.csv.

Keeps the README's numbers mechanically tied to the actual output of
scripts/run_backtests.py and scripts/run_vol_model.py — no hand-copied figures
that can drift from a rerun.

    python scripts/render_readme.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from riskengine.config import ROOT, TABLES  # noqa: E402
from riskengine.report.tables import format_table  # noqa: E402

README = ROOT / "README.md"


def _md(df: pd.DataFrame, formatted: bool = True) -> str:
    d = format_table(df) if formatted else df
    return d.to_markdown()


def build_replacements() -> dict[str, str]:
    out = {}

    strat = pd.read_csv(TABLES / "strategy_comparison.csv", index_col=0)
    out["STRATEGY_TABLE"] = _md(strat.round(3))

    sig = pd.read_csv(TABLES / "significance_tests.csv", index_col=0)
    out["SIGNIFICANCE_TABLE"] = _md(sig, formatted=False)

    stress = pd.read_csv(TABLES / "stress_tests.csv", index_col=0)
    out["STRESS_TABLE"] = _md(stress.map(lambda v: f"{v:.1%}" if pd.notna(v) else "—"), formatted=False)

    var_bt = pd.read_csv(TABLES / "var_backtest.csv", index_col=0)
    out["VAR_TABLE"] = _md(var_bt, formatted=False)

    vol = pd.read_csv(TABLES / "vol_model_comparison.csv", index_col=0)
    out["VOL_TABLE"] = _md(vol.round(4), formatted=False)

    dm = pd.read_csv(TABLES / "vol_dm_tests.csv")
    out["DM_TABLE"] = "\n\n" + dm.to_markdown(index=False) + "\n"

    vol_impact = pd.read_csv(TABLES / "vol_forecast_impact.csv", index_col=0)
    out["VOL_IMPACT_TABLE"] = _md(vol_impact.round(3))

    return out


def main() -> None:
    text = README.read_text(encoding="utf-8")
    replacements = build_replacements()

    for key, table_md in replacements.items():
        pattern = re.compile(rf"<!-- {key} -->")
        if not pattern.search(text):
            print(f"WARNING: placeholder {key} not found in README.md")
            continue
        # Passing a callable (not a string) as the replacement means re.sub
        # treats its return value literally — no backreference processing —
        # which matters because markdown table cells can legitimately contain
        # backslashes (e.g. a stray "—" render) that must not be doubled.
        text = pattern.sub(lambda _, t=table_md: t, text)

    README.write_text(text, encoding="utf-8")
    print(f"README.md updated with {len(replacements)} tables")


if __name__ == "__main__":
    main()
