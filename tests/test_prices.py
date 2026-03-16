import pytest
import pandas as pd
from src.data.prices import build_price_dataset, build_weekly_dataset

SYMBOLS = ['NQ', 'GC', 'CL', 'ZC']

DAILY_COLUMNS = [
  'open', 'high', 'low', 'close', 'volume',
  'sma_20', 'sma_50', 'sma_100', 'sma_200',
  'atr_14', 'atr_20',
  'donchian_20_upper', 'donchian_20_lower',
  'donchian_50_upper', 'donchian_50_lower',
]

WEEKLY_COLUMNS = ['close', 'weekly_return', 'week_of_year', 'year']


@pytest.fixture(params=SYMBOLS)
def price_data(request):
  """Load processed price data for a market."""
  return request.param, build_price_dataset(request.param)


@pytest.fixture(params=SYMBOLS)
def weekly_data(request):
  """Load weekly returns data for a market."""
  return request.param, build_weekly_dataset(request.param)


# --- Daily price tests ---

def test_daily_not_empty(price_data):
  symbol, df = price_data
  assert len(df) > 0, f"{symbol}: DataFrame is empty"


def test_daily_has_all_columns(price_data):
  symbol, df = price_data
  missing = [c for c in DAILY_COLUMNS if c not in df.columns]
  assert not missing, f"{symbol}: missing columns {missing}"


def test_daily_datetime_index(price_data):
  symbol, df = price_data
  assert isinstance(df.index, pd.DatetimeIndex), f"{symbol}: index is not DatetimeIndex"


def test_daily_numeric_values(price_data):
  symbol, df = price_data
  for col in DAILY_COLUMNS:
      assert pd.api.types.is_numeric_dtype(df[col]), f"{symbol}: {col} is not numeric"


def test_daily_no_all_nan_columns(price_data):
  symbol, df = price_data
  all_nan = [c for c in DAILY_COLUMNS if df[c].isna().all()]
  assert not all_nan, f"{symbol}: all-NaN columns: {all_nan}"


def test_sma_ordering(price_data):
  """Shorter SMAs should have fewer NaN rows than longer ones."""
  symbol, df = price_data
  nan_20 = df['sma_20'].isna().sum()
  nan_200 = df['sma_200'].isna().sum()
  assert nan_20 < nan_200, f"{symbol}: sma_20 has more NaNs than sma_200"


# --- Weekly tests ---

def test_weekly_not_empty(weekly_data):
  symbol, df = weekly_data
  assert len(df) > 0, f"{symbol}: weekly DataFrame is empty"


def test_weekly_has_all_columns(weekly_data):
  symbol, df = weekly_data
  missing = [c for c in WEEKLY_COLUMNS if c not in df.columns]
  assert not missing, f"{symbol}: missing weekly columns {missing}"


def test_weekly_no_nan_returns(weekly_data):
  symbol, df = weekly_data
  assert df['weekly_return'].isna().sum() == 0, f"{symbol}: weekly_return has NaN values"


def test_weekly_week_of_year_range(weekly_data):
  symbol, df = weekly_data
  assert df['week_of_year'].min() >= 1, f"{symbol}: week_of_year below 1"
  assert df['week_of_year'].max() <= 53, f"{symbol}: week_of_year above 53"
