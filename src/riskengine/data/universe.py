"""The investable universe: large/mid-cap NSE names with sector labels.

Quick note on survivorship bias before anyone trusts a backtest number: this
is a snapshot of what the Nifty 100 + liquid mid-caps look like today, not a
point-in-time reconstruction. A stock that was in the index in 2016 and got
delisted or demoted since then just isn't here, so the backtest never had a
chance to lose money on it. I couldn't find a free source for real
point-in-time NSE constituent history, so instead I: kept known
underperformers in the list on purpose (IDEA, BHEL, SAIL, ZEEL, PAYTM,
RBLBANK, IDFCFIRSTB, INDUSTOWER...) instead of only listing winners, compare
everything against equal-weight *of this same universe* so the relative
comparison stays meaningful even though survivorship inflates both sides, and
never lead with a bare absolute return number.

Full writeup in docs/methodology.md#survivorship.
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
