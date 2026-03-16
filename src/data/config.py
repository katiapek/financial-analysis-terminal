"""
Market configuration — single source of truth for all market definitions.

Every module imports from here. To add a new market, add one entry to MARKETS.
"""

# ---------------------------------------------------------------------------
# Market definitions
# ---------------------------------------------------------------------------
MARKETS: dict[str, dict] = {
    "NQ": {
        "name": "E-mini Nasdaq 100",
        "yfinance": "NQ=F",
        "cftc_code": "209742",
        "cot_report": "tff",
        "complex": "equity_index",
    },
    "GC": {
        "name": "Gold",
        "yfinance": "GC=F",
        "cftc_code": "088691",
        "cot_report": "disaggregated",
        "complex": "metals",
    },
    "CL": {
        "name": "WTI Crude Oil",
        "yfinance": "CL=F",
        "cftc_code": "067651",
        "cot_report": "disaggregated",
        "complex": "energy",
    },
    "ZC": {
        "name": "Corn",
        "yfinance": "ZC=F",
        "cftc_code": "002602",
        "cot_report": "disaggregated",
        "complex": "agriculture",
    },
}

# ---------------------------------------------------------------------------
# COT column mappings by report type
# ---------------------------------------------------------------------------
# These map to actual column names in the cot_reports library output.
# "speculator" = the category we track for positioning signals.
#   - TFF: leveraged money (hedge funds)
#   - Disaggregated: managed money (hedge funds / CTAs)

COT_COLUMNS: dict[str, dict[str, str]] = {
    "tff": {
        "spec_long": "Lev_Money_Positions_Long_All",
        "spec_short": "Lev_Money_Positions_Short_All",
        "asset_mgr_long": "Asset_Mgr_Positions_Long_All",
        "asset_mgr_short": "Asset_Mgr_Positions_Short_All",
        "comm_long": "Dealer_Positions_Long_All",
        "comm_short": "Dealer_Positions_Short_All",
        "open_interest": "Open_Interest_All",
    },
    "disaggregated": {
        "spec_long": "M_Money_Positions_Long_All",
        "spec_short": "M_Money_Positions_Short_All",
        "swap_long": "Swap_Positions_Long_All",
        "swap_short": "Swap__Positions_Short_All",  # NOTE: double underscore — cot_reports quirk
        "comm_long": "Prod_Merc_Positions_Long_All",
        "comm_short": "Prod_Merc_Positions_Short_All",
        "open_interest": "Open_Interest_All",
    },
}

# ---------------------------------------------------------------------------
# Convenience lookups
# ---------------------------------------------------------------------------
MARKET_SYMBOLS: list[str] = list(MARKETS.keys())
