"""
Historical analog engine — find past instances of similar positioning
conditions and measure what happened to price afterwards.

Supports 5 trigger types:
  zscore     — z-score crosses a threshold (most common)
  flip       — net position crosses zero (direction change)
  roc        — week-over-week change is extreme (panic/conviction)
  divergence — spec vs commercial z-scores diverge (disagreement)
  percentile — all-time percentile hits extreme (historically rare)
"""
import numpy as np
import pandas as pd

from src.data.cot import build_cot_dataset
from src.data.prices import build_price_dataset
from src.data.config import MARKETS


# Group mappings per report type
GROUP_MAP = {
    'tff':          ['spec', 'comm', 'asset_mgr'],
    'disaggregated': ['spec', 'comm', 'swap'],
}

GROUP_LABELS = {
    'spec': 'Managed Money', 'comm': 'Commercials',
    'asset_mgr': 'Asset Managers', 'swap': 'Swap Dealers',
}

# Human-readable labels for trigger descriptions
TRIGGER_LABELS = {
    'zscore': 'Z-Score Extreme',
    'flip': 'Positioning Flip',
    'roc': 'Rate of Change Extreme',
    'divergence': 'Spec vs Comm Divergence',
    'percentile': 'Percentile Extreme',
}


def scan_triggers(
    symbols: list[str] | None = None,
    z_window: str = '3yr',
    zscore_threshold: float = 2.0,
    divergence_threshold: float = 3.0,
    roc_threshold: float = 2.0,
    percentile_high: float = 95,
    percentile_low: float = 5,
) -> pd.DataFrame:
    """
    Scan all markets and groups for currently active positioning triggers.

    Checks the latest week's data against all 5 trigger types and returns
    a DataFrame of everything that's firing right now.

    Returns:
        DataFrame with columns: market, group, trigger, direction,
        value, threshold, n_analogs (how many historical matches exist)
    """
    if symbols is None:
        symbols = list(MARKETS.keys())

    active = []

    for sym in symbols:
        df = build_cot_dataset(sym)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        report = MARKETS[sym]['cot_report']

        for group in GROUP_MAP[report]:
            net_col = f'{group}_net'

            # --- 1. Z-score extreme ---
            for zw in [z_window]:
                z_col = f'{net_col}_z{zw}'
                z_val = latest[z_col]
                if z_val >= zscore_threshold:
                    active.append({
                        'market': sym, 'group': GROUP_LABELS[group],
                        'trigger': 'zscore', 'direction': 'LONG',
                        'value': z_val,
                        'threshold': zscore_threshold,
                        'params': {'trigger': 'zscore', 'threshold': zscore_threshold,
                                   'group': group, 'z_window': zw},
                    })
                elif z_val <= -zscore_threshold:
                    active.append({
                        'market': sym, 'group': GROUP_LABELS[group],
                        'trigger': 'zscore', 'direction': 'SHORT',
                        'value': z_val,
                        'threshold': -zscore_threshold,
                        'params': {'trigger': 'zscore', 'threshold': -zscore_threshold,
                                   'group': group, 'z_window': zw},
                    })

            # --- 2. Positioning flip ---
            curr_sign = np.sign(latest[net_col])
            prev_sign = np.sign(prev[net_col])
            if curr_sign != prev_sign and curr_sign != 0 and prev_sign != 0:
                direction = 'LONG' if curr_sign > 0 else 'SHORT'
                flip_th = 1 if curr_sign > 0 else -1
                active.append({
                    'market': sym, 'group': GROUP_LABELS[group],
                    'trigger': 'flip', 'direction': f'flipped {direction}',
                    'value': latest[net_col],
                    'threshold': flip_th,
                    'params': {'trigger': 'flip', 'threshold': flip_th,
                               'group': group},
                })

            # --- 3. RoC extreme ---
            roc_col = f'{net_col}_roc'
            roc_series = df[roc_col].dropna()
            if len(roc_series) >= 26:
                roc_mean = roc_series.expanding(min_periods=26).mean().iloc[-1]
                roc_std = roc_series.expanding(min_periods=26).std().iloc[-1]
                if roc_std > 0:
                    roc_z = (latest[roc_col] - roc_mean) / roc_std
                    if roc_z >= roc_threshold:
                        active.append({
                            'market': sym, 'group': GROUP_LABELS[group],
                            'trigger': 'roc', 'direction': 'large ADD',
                            'value': roc_z,
                            'threshold': roc_threshold,
                            'params': {'trigger': 'roc', 'threshold': roc_threshold,
                                       'group': group},
                        })
                    elif roc_z <= -roc_threshold:
                        active.append({
                            'market': sym, 'group': GROUP_LABELS[group],
                            'trigger': 'roc', 'direction': 'large CUT',
                            'value': roc_z,
                            'threshold': -roc_threshold,
                            'params': {'trigger': 'roc', 'threshold': -roc_threshold,
                                       'group': group},
                        })

            # --- 4. Spec vs Comm divergence ---
            if group == 'spec':
                spec_z = latest[f'spec_net_z{z_window}']
                comm_z = latest[f'comm_net_z{z_window}']
                spread = spec_z - comm_z
                if spread >= divergence_threshold:
                    active.append({
                        'market': sym, 'group': 'Spec vs Comm',
                        'trigger': 'divergence', 'direction': 'spec MORE LONG',
                        'value': spread,
                        'threshold': divergence_threshold,
                        'params': {'trigger': 'divergence',
                                   'threshold': divergence_threshold,
                                   'group': 'spec', 'z_window': z_window},
                    })
                elif spread <= -divergence_threshold:
                    active.append({
                        'market': sym, 'group': 'Spec vs Comm',
                        'trigger': 'divergence', 'direction': 'spec MORE SHORT',
                        'value': spread,
                        'threshold': -divergence_threshold,
                        'params': {'trigger': 'divergence',
                                   'threshold': -divergence_threshold,
                                   'group': 'spec', 'z_window': z_window},
                    })

            # --- 5. Percentile extreme ---
            pctl_col = f'{net_col}_pctl'
            pctl_val = latest[pctl_col]
            if pctl_val >= percentile_high:
                active.append({
                    'market': sym, 'group': GROUP_LABELS[group],
                    'trigger': 'percentile', 'direction': 'LONG',
                    'value': pctl_val,
                    'threshold': percentile_high,
                    'params': {'trigger': 'percentile',
                               'threshold': percentile_high,
                               'group': group},
                })
            elif pctl_val <= percentile_low:
                active.append({
                    'market': sym, 'group': GROUP_LABELS[group],
                    'trigger': 'percentile', 'direction': 'SHORT',
                    'value': pctl_val,
                    'threshold': percentile_low,
                    'params': {'trigger': 'percentile',
                               'threshold': percentile_low,
                               'group': group},
                })

    if not active:
        return pd.DataFrame()

    result = pd.DataFrame(active)

    # Count historical analogs for each trigger
    n_analogs = []
    for _, row in result.iterrows():
        p = row['params']
        analogs = find_analogs(
            row['market'], group=p['group'], trigger=p['trigger'],
            threshold=p['threshold'], z_window=p.get('z_window', '3yr'),
        )
        n_analogs.append(len(analogs))
    result['n_analogs'] = n_analogs

    return result[['market', 'group', 'trigger', 'direction',
                    'value', 'threshold', 'n_analogs', 'params']]


def find_analogs(
    symbol: str,
    group: str = 'spec',
    trigger: str = 'zscore',
    threshold: float = 2.0,
    z_window: str = '3yr',
    cooldown_weeks: int = 8,
    forward_weeks: list[int] | None = None,
) -> pd.DataFrame:
    """
    Find historical dates where a positioning trigger fired,
    then compute forward price returns from each date.

    Args:
        symbol:     Market symbol (NQ, GC, CL, ZC)
        group:      Trader group (spec, comm, swap, asset_mgr)
        trigger:    One of: zscore, flip, roc, divergence, percentile
        threshold:  Signed value — positive = above, negative = below.
                      zscore:     z-score level (e.g. +2.0 or -2.0)
                      flip:       +1 = short→long, -1 = long→short
                      roc:        z-score of the weekly change
                      divergence: spec_z minus comm_z spread
                      percentile: level (e.g. 95 for high, 5 for low)
        z_window:   Which z-score — 1yr, 3yr, or 5yr (for zscore/divergence)
        cooldown_weeks: Min gap between independent signals (avoids clustering)
        forward_weeks:  Return horizons in weeks (default: [1, 2, 4, 8])

    Returns:
        DataFrame indexed by signal_date with columns:
          trigger_value, ret_1w, ret_2w, ret_4w, ret_8w
        Summary stats attached as result.attrs['summary']
    """
    if forward_weeks is None:
        forward_weeks = [1, 2, 4, 8, 12]

    cot = build_cot_dataset(symbol)
    prices = build_price_dataset(symbol)['close']

    # Detect trigger dates
    signal_dates, trigger_values = _detect_signals(
        cot, group, trigger, threshold, z_window, cooldown_weeks
    )

    # Compute forward returns for each signal
    rows = []
    for sig_date, trig_val in zip(signal_dates, trigger_values):
        base_price = prices.asof(sig_date)
        if pd.isna(base_price):
            continue

        row = {'signal_date': sig_date, 'trigger_value': trig_val}

        for weeks in forward_weeks:
            future_date = sig_date + pd.DateOffset(weeks=weeks)
            if future_date > prices.index[-1]:
                row[f'ret_{weeks}w'] = np.nan
            else:
                future_price = prices.asof(future_date)
                row[f'ret_{weeks}w'] = (future_price / base_price - 1) * 100

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index('signal_date')

    # Attach summary stats and metadata
    summary = {}
    for weeks in forward_weeks:
        col = f'ret_{weeks}w'
        valid = result[col].dropna()
        if len(valid) > 0:
            summary[f'{weeks}w'] = {
                'median': valid.median(),
                'mean': valid.mean(),
                'win_rate': (valid > 0).mean() * 100,
                'best': valid.max(),
                'worst': valid.min(),
                'n': int(len(valid)),
            }

    result.attrs['summary'] = summary
    result.attrs['trigger'] = trigger
    result.attrs['threshold'] = threshold
    result.attrs['z_window'] = z_window
    result.attrs['symbol'] = symbol
    result.attrs['group'] = group
    result.attrs['forward_weeks'] = forward_weeks

    return result


def get_analog_paths(
    symbol: str,
    signal_dates: list[pd.Timestamp],
    forward_weeks: int = 8,
) -> pd.DataFrame:
    """
    Get normalized daily price paths from each signal date forward.

    Returns DataFrame where each column is a signal date, index is
    trading days from signal (0 to ~forward_weeks*5), values are
    cumulative % return from signal date.
    """
    prices = build_price_dataset(symbol)['close']
    max_days = forward_weeks * 7 + 5  # calendar days with buffer

    paths = {}
    for sig_date in signal_dates:
        base_price = prices.asof(sig_date)
        if pd.isna(base_price):
            continue

        end_date = sig_date + pd.DateOffset(days=max_days)
        window = prices[(prices.index >= sig_date) & (prices.index <= end_date)]

        if len(window) < 2:
            continue

        # Normalize to % return from signal, reindex to trading day number
        pct = (window / base_price - 1) * 100
        pct.index = range(len(pct))
        paths[sig_date] = pct

    return pd.DataFrame(paths)


# ---------------------------------------------------------------------------
# Trigger detection (internal)
# ---------------------------------------------------------------------------

def _detect_signals(df, group, trigger, threshold, z_window, cooldown_weeks):
    """Find dates where trigger fires, with cooldown dedup."""
    net_col = f'{group}_net'

    if trigger == 'zscore':
        col = f'{net_col}_z{z_window}'
        if threshold >= 0:
            mask = df[col] >= threshold
        else:
            mask = df[col] <= threshold
        values = df[col]

    elif trigger == 'flip':
        signs = np.sign(df[net_col])
        prev_signs = signs.shift(1)
        if threshold >= 0:      # short → long
            mask = (signs > 0) & (prev_signs < 0)
        else:                   # long → short
            mask = (signs < 0) & (prev_signs > 0)
        values = df[net_col]

    elif trigger == 'roc':
        roc_col = f'{net_col}_roc'
        # Z-score the rate of change itself (expanding window)
        roc_mean = df[roc_col].expanding(min_periods=26).mean()
        roc_std = df[roc_col].expanding(min_periods=26).std()
        roc_z = (df[roc_col] - roc_mean) / roc_std
        if threshold >= 0:
            mask = roc_z >= threshold
        else:
            mask = roc_z <= threshold
        values = roc_z

    elif trigger == 'divergence':
        spec_z = df[f'spec_net_z{z_window}']
        comm_z = df[f'comm_net_z{z_window}']
        spread = spec_z - comm_z
        if threshold >= 0:
            mask = spread >= threshold
        else:
            mask = spread <= threshold
        values = spread

    elif trigger == 'percentile':
        pctl_col = f'{net_col}_pctl'
        if threshold >= 50:
            mask = df[pctl_col] >= threshold
        else:
            mask = df[pctl_col] <= threshold
        values = df[pctl_col]

    else:
        raise ValueError(
            f"Unknown trigger '{trigger}'. "
            f"Valid: zscore, flip, roc, divergence, percentile"
        )

    # Apply cooldown — keep only the first signal per cluster
    candidate_dates = df.index[mask].tolist()
    signal_dates = []
    signal_values = []
    last_signal = None

    for d in candidate_dates:
        if last_signal is None or (d - last_signal).days > cooldown_weeks * 7:
            signal_dates.append(d)
            signal_values.append(values.loc[d])
            last_signal = d

    return signal_dates, signal_values
