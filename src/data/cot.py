"""
COT data ingestion - fetch and filter CFTC Commitment of Traders data.
"""
import pandas as pd
import cot_reports as cot
from pathlib import Path

from src.data.config import MARKETS, COT_COLUMNS

from src.models.positioning import (compute_zscore, compute_26w_index,
                                    compute_percentile_rank, compute_rate_of_change)

_REPORT_TYPE_MAP = {
    "tff": "traders_in_financial_futures_futopt",
    "disaggregated": "disaggregated_futopt"
}

DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")


def fetch_cot(
        symbol: str,
        start_year: int = 2026,
        end_year: int = 2026,
        force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch COT positioning data for a single market.

      Returns a DataFrame indexed by report date with columns:
      spec_long, spec_short, comm_long, comm_short, open_interest
      (plus asset_mgr or swap columns depending on report type).

    Args:
      symbol: Market symbol from config (e.g. "NQ", "GC")
      start_year: First year to pull (Disaggregated available from 2006)
      end_year: Last year to pull (inclusive)
      force_refresh: If True, re-download from CFTC even if cache exists
    """
    market = MARKETS[symbol]
    report_type = market['cot_report']
    cftc_code = market['cftc_code']
    lib_report_type = _REPORT_TYPE_MAP[report_type]

    # _-- Check cache ---
    cache_path = DATA_RAW_DIR / f"cot_{report_type}_{start_year}_{end_year}.parquet"
    if cache_path.exists() and not force_refresh:
        raw_df = pd.read_parquet(cache_path)
    else:
        raw_df = _download_cot_range(lib_report_type, start_year, end_year)
        # Coerce mixed-type columns so Parquet can handle them
        for col in raw_df.select_dtypes(include="object").columns:
            raw_df[col] = raw_df[col].astype(str)
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_df.to_parquet(cache_path)

    # --- Filter out our market ---
    mask = raw_df['CFTC_Contract_Market_Code'].astype(str).str.strip() == cftc_code
    df = raw_df.loc[mask].copy()

    # --- Parse date index ---
    df['date'] = pd.to_datetime(df['Report_Date_as_YYYY-MM-DD'])
    df = df.set_index("date").sort_index()

    # --- Select and rename columns ---
    col_map = COT_COLUMNS[report_type]
    result = df[list(col_map.values())].copy()
    result.columns = list(col_map.keys())

    # Drop duplicate dates (if exists)
    result = result[~result.index.duplicated(keep='last')]

    return result

def _download_cot_range(
        lib_report_type: str,
        start_year: int,
        end_year: int,
) -> pd.DataFrame:
    """Download COT data year by year and concatenate"""
    frames = []
    for year in range(start_year, end_year +1):
        try:
            df = cot.cot_year(year, cot_report_type=lib_report_type)
            frames.append(df)
        except Exception as e:
            print(f"Skipping {year}: {e}")
    return pd.concat(frames, ignore_index=True)


def compute_net_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Add net position columns for all trader categories."""
    df = df.copy()
    df['spec_net'] = df['spec_long'] - df['spec_short']
    df['comm_net'] = df['comm_long'] - df['comm_short']

    # Asset managers (TFF Only) or Swap dealers (Disaggregated only)
    if "asset_mgr_long" in df.columns:
        df['asset_mgr_net'] = df['asset_mgr_long'] - df['asset_mgr_short']
    if "swap_long" in df.columns:
        df['swap_net'] = df['swap_long'] - df['swap_short']

    return df


def build_cot_dataset(
        symbol: str,
        start_year: int = 2006,
        end_year: int = 2026,
        force_refresh : bool = False,
) -> pd.DataFrame:
    """
    Full COT pipeline: fetch, compute all positioning metrics, save to Parquet.

      This is the main entry point for COT data. Returns a fully enriched
      DataFrame with net positions, 26w index, z-scores, percentile ranks,
      and rate of change.

      Saves processed output to data/processed/cot_{symbol}.parquet.
      On subsequent calls, loads from processed cache unless force_refresh=True.
    """

    processed_path = DATA_PROCESSED_DIR / f"cot_{symbol}.parquet"

    if processed_path.exists() and not force_refresh:
        return pd.read_parquet(processed_path)

    df = fetch_cot(symbol, start_year, end_year, force_refresh)
    df = compute_net_positions(df)
    df = compute_26w_index(df)
    df = compute_zscore(df)
    df = compute_percentile_rank(df)
    df = compute_rate_of_change(df)

    # Save
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path)

    return df
