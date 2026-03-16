import pytest
import pandas as pd
from src.data.cot import build_cot_dataset

SYMBOLS = ['NQ', 'GC', 'CL', 'ZC']

# Columns that must exist for every market
BASE_COLUMNS = [
    "spec_long", "spec_short", "comm_long", "comm_short", "open_interest",
    "spec_net", "comm_net",
    "spec_net_26w", "comm_net_26w",
    "spec_net_z1yr", "spec_net_z3yr", "spec_net_z5yr",
    "spec_net_pctl", "comm_net_pctl",
    "spec_net_roc", "comm_net_roc",
]

@pytest.fixture(params=SYMBOLS)
def cot_data(request):
    """Load processed COT data for a market (uses cache)."""
    return request.param, build_cot_dataset(request.param, start_year=2024, end_year=2024)


def test_not_empty(cot_data):
    symbol, df = cot_data
    assert len(df) > 0, f"{symbol}: DataFrame is empty"


def test_has_base_columns(cot_data):
    symbol, df = cot_data
    missing =[c for c in BASE_COLUMNS if c not in df.columns]
    assert not missing, f"{symbol}: missing columns {missing}"


def test_datetime_index(cot_data):
    symbol, df = cot_data
    assert isinstance(df.index, pd.DatetimeIndex), f"{symbol}: index is not DatetimeIndex"


def test_numeric_values(cot_data):
    symbol, df = cot_data
    for col in ['spec_net', 'comm_net', 'open_interest']:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{symbol}: {col} is not numeric"


def test_no_all_nan_columns(cot_data):
    symbol, df = cot_data
    all_nan = [c for c in df.columns if df[c].isna().all()]
    assert not all_nan, f"{symbol}: all_NaN columns: {all_nan}"


def test_tff_has_asset_mgr():
    df = build_cot_dataset("NQ", start_year=2024, end_year=2024)
    assert "asset_mgr_net" in df.columns, "NQ (TFF) should have asset_mgr_net"
    assert "swap_net" not in df.columns, "NQ (TFF) should not have swap_net"


def test_disaggregated_has_swap():
    df = build_cot_dataset("GC", start_year=2024, end_year=2024)
    assert "swap_net" in df.columns, "GC (Disaggregated) should have swap_net"
    assert "asset_mgr_net" not in df.columns, "GC (Disaggregated) should not have asset_mgr_net"

