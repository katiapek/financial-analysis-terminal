"""
 Price data ingestion — daily OHLCV via yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import date
from pathlib import Path

from src.data.config import MARKETS

DATA_RAW_DIR = Path('data/raw')
DATA_PROCESSED_DIR = Path('data/processed')


def fetch_prices(
        symbol: str,
        start: str = "2006-01-01",
        end: str | None = None,
        force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a futures market.

      Args:
          symbol: Market symbol from config (e.g. "NQ", "GC")
          start: Start date as "YYYY-MM-DD"
          end: End date as "YYYY-MM-DD" (defaults to today)
          force_refresh: If True, re-download even if cache exists

      Returns:
          DataFrame with DatetimeIndex and columns:
          open, high, low, close, volume
    """
    if end is None:
        end = str(date.today())

    ticker = MARKETS[symbol]['yfinance']
    cache_path = DATA_RAW_DIR / f"prices_{symbol}.parquet"

    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    df = yf.download(ticker, start=start, end=end, progress=False)

    # Flatten MultiIndex columns (yfinance v1.2+ returns ('Close', 'GC=F') tuples)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Standardize column names to lowercase
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Drop adj_close if present (same as close for futures)
    if 'adj_close' in df.columns:
        df = df.drop(columns=['adj_close'])

    # Ensure DatetimeIndex
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'

    # Cache
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)

    return df

def compute_moving_averages(
        df: pd.DataFrame,
        windows: list[int] = [20, 50, 100, 200],
) -> pd.DataFrame:
    """
     Add simple moving averages of the close price.

      Args:
          df: Price DataFrame with 'close' column
          windows: List of MA periods in days

      Adds columns: sma_20, sma_50, sma_100, sma_200
    """
    df = df.copy()
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(window=w).mean()

    return df

def compute_atr(
        df: pd.DataFrame,
        windows: list[int] = [14,20],
) -> pd.DataFrame:
    """
    Add Average True Range columns.

      True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
      ATR = rolling mean of True Range.

      Args:
          df: Price DataFrame with 'high', 'low', 'close' columns
          windows: ATR periods (default 14 and 20 day)

      Adds columns: atr_14, atr_20
    """
    df = df.copy()
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)

    for w in windows:
        df[f'atr_{w}'] = tr.rolling(window=w).mean()

    return df


def compute_donchian(
        df: pd.DataFrame,
        windows: list[int] = [20, 50],
) -> pd.DataFrame:
    """
    Add Donchian channel columns (rolling high/low).

      Upper = highest high over window.
      Lower = lowest low over window.

      Args:
          df: Price DataFrame with 'high', 'low' columns
          windows: Channel periods (default 20 and 50 day)

      Adds columns: donchian_20_upper, donchian_20_lower,
                    donchian_50_upper, donchian_50_lower

    """
    df = df.copy()
    for w in windows:
        df[f'donchian_{w}_upper'] = df['high'].rolling(window=w).max()
        df[f'donchian_{w}_lower'] = df['low'].rolling(window=w).min()

    return df


def compute_weekly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily prices to weekly and compute returns.

      Returns a weekly DataFrame with columns:
          close: last close of the week
          weekly_return: week-over-week percentage change
          week_of_year: ISO week number (1-53)
          year: year of the observation
    """
    weekly = df[['close']].resample('W').last().copy()
    weekly['weekly_return'] = weekly['close'].pct_change()
    weekly['week_of_year'] = weekly.index.isocalendar().week.astype(int)
    weekly['year'] = weekly.index.year

    # Drop first row (NaN return, no prior week)
    weekly = weekly.dropna(subset=['weekly_return'])

    return weekly


def build_price_dataset(
        symbol: str,
        start: str = "2006-01-01",
        end: str | None = None,
        force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Full price pipeline: fetch OHLCV, compute all indicators, save to Parquet.
    """
    processed_path = DATA_PROCESSED_DIR / f"prices_{symbol}.parquet"

    if processed_path.exists() and not force_refresh:
        return pd.read_parquet(processed_path)

    df = fetch_prices(symbol, start, end, force_refresh)
    df = compute_moving_averages(df)
    df = compute_atr(df)
    df = compute_donchian(df)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path)

    return df


def build_weekly_dataset(
        symbol: str,
        start: str = "2006-01-01",
        end: str | None = None,
        force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Build weekly returns dataset for seasonal analysis.
    """
    processed_path = DATA_PROCESSED_DIR / f"weekly_{symbol}.parquet"

    if processed_path.exists() and not force_refresh:
        return pd.read_parquet(processed_path)

    df = fetch_prices(symbol, start, end, force_refresh)
    weekly = compute_weekly_returns(df)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(processed_path)

    return weekly

