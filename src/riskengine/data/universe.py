"""Investable universe: large- and mid-cap NSE names with sector labels.

SURVIVORSHIP BIAS — read this before trusting any backtest number
-----------------------------------------------------------------
This list is a *current-membership snapshot* of the Nifty 100 plus a set of
liquid mid-caps. It is NOT point-in-time. A stock that was in the index in 2016
and was later delisted or demoted does not appear here, so the backtest never
gets to lose money on it.

Free point-in-time NSE constituent history does not exist. Rather than pretend
otherwise, the project does three things:

1. Deliberately retains names that performed *badly* over the sample
   (IDEA, BHEL, SAIL, ZEEL, PAYTM, RBLBANK, IDFCFIRSTB, INDUSTOWER...) so the
   universe is not purely a winners' list.
2. Compares every strategy against an equal-weight portfolio *of the same
   universe*. Survivorship inflates both sides, so the relative comparison —
   which is what the project actually claims — is far less contaminated than
   the absolute CAGR.
3. Reports absolute returns with an explicit caveat and never as the headline.

See docs/methodology.md#survivorship for the full treatment.
"""

from __future__ import annotations

import pandas as pd

# ticker (without .NS) -> sector
_UNIVERSE: dict[str, str] = {
    # ---------------------------------------------------------- Financials
    "HDFCBANK": "Financials",
    "ICICIBANK": "Financials",
    "KOTAKBANK": "Financials",
    "AXISBANK": "Financials",
    "SBIN": "Financials",
    "INDUSINDBK": "Financials",
    "BANKBARODA": "Financials",
    "PNB": "Financials",
    "CANBK": "Financials",
    "IDFCFIRSTB": "Financials",
    "AUBANK": "Financials",
    "FEDERALBNK": "Financials",
    "RBLBANK": "Financials",
    "BAJFINANCE": "Financials",
    "BAJAJFINSV": "Financials",
    "CHOLAFIN": "Financials",
    "MUTHOOTFIN": "Financials",
    "PFC": "Financials",
    "RECLTD": "Financials",
    "HDFCLIFE": "Financials",
    "SBILIFE": "Financials",
    "ICICIPRULI": "Financials",
    "ICICIGI": "Financials",
    "HDFCAMC": "Financials",
    # ------------------------------------------------------------------ IT
    "TCS": "IT",
    "INFY": "IT",
    "HCLTECH": "IT",
    "WIPRO": "IT",
    "TECHM": "IT",
    "LTIM": "IT",
    "PERSISTENT": "IT",
    "MPHASIS": "IT",
    "COFORGE": "IT",
    "OFSS": "IT",
    # -------------------------------------------------------- Energy/Power
    "RELIANCE": "Energy",
    "ONGC": "Energy",
    "IOC": "Energy",
    "BPCL": "Energy",
    "HINDPETRO": "Energy",
    "GAIL": "Energy",
    "PETRONET": "Energy",
    "COALINDIA": "Energy",
    "NTPC": "Utilities",
    "POWERGRID": "Utilities",
    "TATAPOWER": "Utilities",
    "NHPC": "Utilities",
    "TORNTPOWER": "Utilities",
    # ---------------------------------------------------------------- FMCG
    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG",
    "DABUR": "FMCG",
    "GODREJCP": "FMCG",
    "MARICO": "FMCG",
    "COLPAL": "FMCG",
    "TATACONSUM": "FMCG",
    "UNITDSPR": "FMCG",
    # ---------------------------------------------------------------- Auto
    "MARUTI": "Auto",
    "M&M": "Auto",
    "TATAMOTORS": "Auto",
    "BAJAJ-AUTO": "Auto",
    "EICHERMOT": "Auto",
    "HEROMOTOCO": "Auto",
    "TVSMOTOR": "Auto",
    "ASHOKLEY": "Auto",
    "BOSCHLTD": "Auto",
    "MOTHERSON": "Auto",
    "BALKRISIND": "Auto",
    # -------------------------------------------------------------- Pharma
    "SUNPHARMA": "Pharma",
    "DRREDDY": "Pharma",
    "CIPLA": "Pharma",
    "DIVISLAB": "Pharma",
    "LUPIN": "Pharma",
    "AUROPHARMA": "Pharma",
    "TORNTPHARM": "Pharma",
    "ALKEM": "Pharma",
    "BIOCON": "Pharma",
    "APOLLOHOSP": "Healthcare",
    # ------------------------------------------------------------- Metals
    "TATASTEEL": "Metals",
    "JSWSTEEL": "Metals",
    "HINDALCO": "Metals",
    "VEDL": "Metals",
    "JINDALSTEL": "Metals",
    "SAIL": "Metals",
    "NMDC": "Metals",
    "HINDZINC": "Metals",
    # ------------------------------------------------------ Cement/Chem
    "ULTRACEMCO": "Materials",
    "GRASIM": "Materials",
    "SHREECEM": "Materials",
    "AMBUJACEM": "Materials",
    "ACC": "Materials",
    "PIDILITIND": "Materials",
    "ASIANPAINT": "Materials",
    "BERGEPAINT": "Materials",
    "SRF": "Materials",
    "UPL": "Materials",
    "TATACHEM": "Materials",
    "DEEPAKNTR": "Materials",
    # ------------------------------------------------------- Telecom/Media
    "BHARTIARTL": "Telecom",
    "IDEA": "Telecom",
    "INDUSTOWER": "Telecom",
    "ZEEL": "Media",
    # --------------------------------------------------- Industrials/Infra
    "LT": "Industrials",
    "SIEMENS": "Industrials",
    "ABB": "Industrials",
    "HAVELLS": "Industrials",
    "CUMMINSIND": "Industrials",
    "BEL": "Industrials",
    "BHEL": "Industrials",
    "ADANIPORTS": "Industrials",
    "ADANIENT": "Industrials",
    "CONCOR": "Industrials",
    "DLF": "Realty",
    "GODREJPROP": "Realty",
    "OBEROIRLTY": "Realty",
    # --------------------------------------------------- Consumer/Internet
    "TITAN": "Consumer",
    "TRENT": "Consumer",
    "DMART": "Consumer",
    "PAGEIND": "Consumer",
    "VOLTAS": "Consumer",
    "NAUKRI": "Internet",
    "PAYTM": "Internet",
    "IRCTC": "Consumer",
    "INDIGO": "Transport",
}


def get_universe() -> pd.DataFrame:
    """Return the universe as a DataFrame indexed by yfinance ticker."""
    df = pd.DataFrame(
        {
            "symbol": list(_UNIVERSE.keys()),
            "sector": list(_UNIVERSE.values()),
        }
    )
    df["ticker"] = df["symbol"] + ".NS"
    return df.set_index("ticker").sort_index()


def get_tickers() -> list[str]:
    """yfinance tickers for the full universe."""
    return sorted(f"{s}.NS" for s in _UNIVERSE)


def sector_map() -> dict[str, str]:
    """ticker -> sector, for sector-cap constraints."""
    return {f"{s}.NS": sec for s, sec in _UNIVERSE.items()}
