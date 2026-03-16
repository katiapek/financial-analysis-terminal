"""
COT positioning analytics — derived metrics from raw positioning data.

Takes the output of fetch_cot() + compute_net_positions() and adds
analytical layers: 26-week index, z-scores, percentile ranks, and
rate-of-change signals. These feed into the regime classification model
and power the weekly positioning charts.
"""
import pandas as pd


def compute_26w_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 26-week index (0-100 scale) for all net position columns.

    Rolling min-max normalization over a 26-week window.
    100 = highest positioning in last 26 weeks.
    0 = lowest positioning in last 26 weeks.
    """
    df = df.copy()
    net_cols = [c for c in df.columns if c.endswith('_net')]

    for col in net_cols:
        rolling_min = df[col].rolling(window=26, min_periods=1).min()
        rolling_max = df[col].rolling(window=26, min_periods=1).max()
        spread = rolling_max - rolling_min
        index = ((df[col] - rolling_min) / spread) * 100
        # Handle zero spread (no change over window) -> midpoint
        index = index.fillna(50.0)
        df[f"{col}_26w"] = index

    return df


def compute_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add z-score columns for all net position columns.

      Computes rolling z-scores over 1-year (52w), 3-year (156w),
      and 5-year (260w) windows. Measures how many standard deviations
      current positioning is from the rolling mean.

      |z| > 2 is generally considered an extreme.
    """
    df = df.copy()
    windows = {'1yr': 52, '3yr': 156, '5yr': 260}
    net_cols = [c for c in df.columns if c.endswith("_net")]

    for col in net_cols:
        for label, window in windows.items():
            rolling_mean = df[col].rolling(window=window, min_periods=26).mean()
            rolling_std = df[col].rolling(window=window, min_periods=26).std()
            df[f"{col}_z{label}"] = ((df[col] - rolling_mean) / rolling_std)

    return df


def compute_percentile_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add percentile rank (0-100) for all net position columns.

      Uses expanding window (all available history up to current row).
      90+ = historically high positioning, 10- = historically low.
    """
    df = df.copy()
    net_cols = [c for c in df.columns if c.endswith('_net')]

    for col in net_cols:
        df[f"{col}_pctl"] = df[col].expanding(min_periods=26).apply(
            lambda x: x.rank(pct=True).iloc[-1] * 100,
            raw=False
        )

    return df


def compute_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add week-over-week change for all net position columns.

      Positive = speculators adding to position.
      Negative = speculators reducing/reversing position.
    """

    df = df.copy()
    net_cols = [c for c in df.columns if c.endswith('_net')]

    for col in net_cols:
        df[f"{col}_roc"] = df[col].diff()

    return df
