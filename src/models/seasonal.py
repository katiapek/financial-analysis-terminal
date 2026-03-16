"""
Seasonal analysis engine — week-of-year return patterns.
"""
import pandas as pd
from src.data.prices import build_weekly_dataset


def compute_seasonal_matrix(
        symbol: str,
        lookbacks: list[int] = [5,10,20]
) -> pd.DataFrame:
    """
    Build week-of-year seasonal matrix with mean return, win rate,
    and std dev for each lookback period.

    Returns DataFrame indexed by week_of_year (1-53) with columns:
          mean_return_5yr, win_rate_5yr, std_5yr,
          mean_return_10yr, win_rate_10yr, std_10yr,
          mean_return_20yr, win_rate_20yr, std_20yr
    """
    weekly = build_weekly_dataset(symbol)
    current_year = weekly['year'].max()

    results = pd.DataFrame(index=range(1,54))
    results.index.name = 'week_of_year'

    for years in lookbacks:
        cutoff_year = current_year - years
        subset = weekly[weekly['year'] > cutoff_year]
        grouped = subset.groupby('week_of_year')['weekly_return']

        label = f"{years}yr"
        results[f'mean_return_{label}'] = grouped.mean()
        results[f'win_rate_{label}'] = grouped.apply(
            lambda x: (x>0).sum() / len(x) * 100
        )
        results[f'std_{label}'] = grouped.std()

    # Drop weeks with no data (week 53 in most years)
    results = results.dropna(how='all')

    return results
