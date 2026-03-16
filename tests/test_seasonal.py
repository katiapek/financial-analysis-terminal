import pytest
import pandas as pd
from src.models.seasonal import compute_seasonal_matrix

SYMBOLS = ['NQ', 'GC', 'CL', 'ZC']

EXPECTED_COLUMNS = [
    'mean_return_5yr', 'win_rate_5yr', 'std_5yr',
    'mean_return_10yr', 'win_rate_10yr', 'std_10yr',
    'mean_return_20yr', 'win_rate_20yr', 'std_20yr',
]


@pytest.fixture(params=SYMBOLS)
def seasonal_data(request):
    return request.param, compute_seasonal_matrix(request.param)


def test_not_empty(seasonal_data):
    symbol, df = seasonal_data
    assert len(df) > 0, f"{symbol}: seasonal matrix is empty"


def test_has_all_columns(seasonal_data):
    symbol, df = seasonal_data
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    assert not missing, f"{symbol}: missing columns {missing}"


def test_week_range(seasonal_data):
    symbol, df = seasonal_data
    assert df.index.min() >= 1, f"{symbol}: week below 1"
    assert df.index.max() <= 53, f"{symbol}: week above 53"


def test_win_rate_bounds(seasonal_data):
    symbol, df = seasonal_data
    for col in [c for c in df.columns if 'win_rate' in c]:
        assert df[col].min() >= 0, f"{symbol}: {col} below 0"
        assert df[col].max() <= 100, f"{symbol}: {col} above 100"


def test_std_non_negative(seasonal_data):
    symbol, df = seasonal_data
    for col in [c for c in df.columns if 'std' in c]:
        assert (df[col].dropna() >= 0).all(), f"{symbol}: {col} has negative values"