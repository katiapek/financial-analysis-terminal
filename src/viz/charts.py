"""
Chart generators — branded chart functions for X/Substack content.
"""
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

from src.viz.styles import (
    apply_style, COLORS, FONT,
    add_title, add_watermark, add_signature_stripe, add_source, save_chart
)
from src.data.cot import build_cot_dataset
from src.data.prices import build_price_dataset, build_weekly_dataset
from src.models.seasonal import compute_seasonal_matrix

from src.data.config import MARKETS

GROUP_LABELS = {
      "spec": "Managed Money",
      "comm": "Commercials",
      "asset_mgr": "Asset Managers",
      "swap": "Swap Dealers",
  }


def chart_cot_positioning(
        symbol: str,
        lookback_years: int = 3,
        save: bool = True,
) -> plt.Figure:
    """
    COT positioning chart — managed money net + commercials net
      with z-score context bands.
    """
    apply_style()
    df = build_cot_dataset(symbol)

    # Filter to lookback window
    cutoff = pd.Timestamp(date.today()) - pd.DateOffset(years=lookback_years)
    df = df[df.index > cutoff]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 4.5),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.3},
    )

    # --- Top panel: Net positions ---
    ax1.fill_between(df.index, df['spec_net'], 0,
                     where=df['spec_net'] >= 0,
                     color=COLORS['bull'], alpha=0.2, interpolate=True)
    ax1.fill_between(df.index, df['spec_net'], 0,
                     where=df['spec_net'] < 0,
                     color=COLORS['bear'], alpha=0.2, interpolate=True)
    ax1.plot(df.index, df['spec_net'], color=COLORS['spec'],
             linewidth=1.5, label="Managed Money")
    ax1.plot(df.index, df['comm_net'], color=COLORS['comm'],
             linewidth=1.5, label="Commercials")

    # Inline labels at right edge
    for col, color, label in [
        ('spec_net', COLORS['spec'], "Managed Money"),
        ('comm_net', COLORS['comm'], "Commercials"),
    ]:
        last_val = df[col].iloc[-1]
        ax1.annotate(
            f" {label}",
            xy=(df.index[-1], last_val),
            fontsize=FONT['annotation'],
            color=color,
            va='center'
        )

    ax1.set_ylabel("Net Contracts", fontsize=FONT['label'])
    ax1.tick_params(labelbottom=False)

    # --- Bottom panel: Z-score (1yr) ---
    z_col = 'spec_net_z1yr'
    ax2.fill_between(df.index, df[z_col], 0,
                     where=df[z_col] >= 0,
                     color=COLORS['bull'], alpha=0.3, interpolate=True)
    ax2.fill_between(df.index, df[z_col], 0,
                     where=df[z_col] < 0,
                     color=COLORS['bear'], alpha=0.3, interpolate=True)
    ax2.plot(df.index, df[z_col], color=COLORS['spec'], linewidth=1.0)

    # Z-score bands
    for level in [1, 2, -1, -2]:
        ax2.axhline(level, color=COLORS['text_muted'],
                    linewidth=0.5, linestyle='--', alpha=0.4)

    ax2.set_ylabel("Z-Score (1yr)", fontsize=FONT['label'])
    ax2.set_ylim(-3.5, 3.5)

    # --- Branding ---
    add_title(ax1,  f"{symbol} | Managed Money Net Positioning",
              f"Week of {date.today().strftime('%b %d, %Y')} | Source: CFTC COT")
    add_watermark(fig)
    add_signature_stripe(fig)
    add_source(fig)

    if save:
        save_chart(fig, f"cot_{symbol}.png")

    return fig


def chart_cot_overview(
        symbol: str,
        group: str = 'spec',
        lookback_years: int = 3,
        save: bool = True,
) -> plt.Figure:
    """
    COT positioning overview — net positions, 26-week index, and percentile rank.

    Three-panel chart for a single trader group showing absolute positioning
    (top), short-term context via 26w index (middle), and historical context
    via all-time percentile rank (bottom).
    """
    apply_style()
    df = build_cot_dataset(symbol)

    net_col = f'{group}_net'
    if group not in GROUP_LABELS:
        raise ValueError(f"Unknown group '{group}'. Valid: {list(GROUP_LABELS.keys())}")
    if net_col not in df.columns:
        raise ValueError(f"Group '{group}' not available for {symbol}. "
                         f"Available: {[c.replace('_net','') for c in df.columns if c.endswith('_net')]}")

    cutoff = pd.Timestamp(date.today()) - pd.DateOffset(years=lookback_years)
    df = df[df.index > cutoff]

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, figsize=[8, 4.5],
        gridspec_kw={'height_ratios': [3, 1, 1], 'hspace': 0.3}
    )

    # Top panel
    ax1.fill_between(df.index, df[net_col], 0,
                     where=df[net_col] > 0,
                     color=COLORS['bull'], alpha=0.3, interpolate=True)
    ax1.fill_between(df.index, df[net_col], 0,
                     where=df[net_col] < 0,
                     color=COLORS['bear'], alpha=0.3, interpolate=True)
    ax1.plot(df.index, df[net_col], color=COLORS[f'{group}'],
             linewidth=1.5, label=f'{GROUP_LABELS[group]}')
    ax1.set_ylabel("Net Contracts", fontsize=FONT['label'])
    ax1.tick_params(labelbottom=False)

    # Middle panel
    w26_col = f'{net_col}_26w'
    ax2.fill_between(df.index, df[w26_col], 80,
                     where=df[w26_col]>80,
                     color=COLORS['bull'], alpha=0.3, interpolate=True)
    ax2.fill_between(df.index, df[w26_col], 20,
                     where=df[w26_col] < 20,
                     color=COLORS['bear'], alpha=0.3, interpolate=True)
    ax2.plot(df.index, df[w26_col], color=COLORS[f'{group}'], linewidth=0.5)
    ax2.set_ylabel('26W Index', fontsize=FONT['label'])
    ax2.axhline(80, color=COLORS['text_muted'], linewidth=0.5,
                linestyle='--', alpha=0.6)
    ax2.axhline(20, color=COLORS['text_muted'], linewidth=0.5,
                linestyle='--', alpha=0.6)
    ax2.set_ylim(0,100)
    ax2.tick_params(labelbottom=False)

    # Bottom label
    pctl_col = f'{net_col}_pctl'
    ax3.fill_between(df.index, df[pctl_col], 90,
                     where=df[pctl_col] > 90,
                     color=COLORS['bull'], alpha=0.3, interpolate=True)
    ax3.fill_between(df.index, df[pctl_col], 10,
                     where=df[pctl_col] < 10,
                     color=COLORS['bear'], alpha=0.3, interpolate=True)
    ax3.plot(df.index, df[pctl_col], color=COLORS[f'{group}'], linewidth=0.5)
    ax3.set_ylabel('Percentile', fontsize=FONT['label'])
    ax3.axhline(90, color=COLORS['text_muted'], linewidth=0.5,
                linestyle='--', alpha=0.6)
    ax3.axhline(10, color=COLORS['text_muted'], linewidth=0.5,
                linestyle='--', alpha=0.6)
    ax3.set_ylim(0,100)

    # --- Branding ---
    add_title(ax1, f"{symbol} | {GROUP_LABELS[group]} — Positioning Overview",
              f"Week of {date.today().strftime('%b %d, %Y')} | Source: CFTC COT")
    add_watermark(fig)
    add_signature_stripe(fig)
    add_source(fig)

    if save:
        save_chart(fig, f"cot_{symbol}_{group}.png")

    return fig


def chart_cot_momentum(
        symbol: str,
        group: str = 'spec',
        lookback_years: int = 3,
        save: bool = True
) -> plt.Figure:
    """
    COT positioning momentum — weekly change, 1-year z-score, and 3-year z-score.

    Three-panel chart for a single trader group showing rate of change as bars
    (top), short-term statistical extremes via 1yr z-score (middle), and
    medium-term extremes via 3yr z-score (bottom).
    """
    apply_style()
    df = build_cot_dataset(symbol)

    net_col = f'{group}_net'
    if group not in GROUP_LABELS:
        raise ValueError(f"Unknown group '{group}'. Valid: {list(GROUP_LABELS.keys())}")
    if net_col not in df.columns:
        raise ValueError(f"Group '{group}' not available for {symbol}. "
                         f"Available: {[c.replace('_net', '') for c in df.columns if c.endswith('_net')]}")

    cutoff = pd.Timestamp(date.today()) - pd.DateOffset(years=lookback_years)
    df = df[df.index > cutoff]

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, figsize=[8, 4.5],
        gridspec_kw={'height_ratios': [2, 1, 1], 'hspace': 0.3}
    )

    # Top panel
    roc_col = f'{net_col}_roc'

    colors = [COLORS['bull'] if v >= 0
              else COLORS['bear']
              for v in df[roc_col]]

    ax1.bar(df.index, df[roc_col], color=colors, width=5, alpha=0.3)
    ax1.set_ylabel("Weekly Change", fontsize=FONT['label'])
    ax1.tick_params(labelbottom=False)


    # Mid panel
    z1yr_col = f'{net_col}_z1yr'
    ax2.fill_between(df.index, df[z1yr_col], 0,
                     where=df[z1yr_col] >= 0,
                     color=COLORS['bull'], alpha=0.3, interpolate=True)
    ax2.fill_between(df.index, df[z1yr_col], 0,
                     where=df[z1yr_col] < 0,
                     color=COLORS['bear'], alpha=0.3, interpolate=True)
    ax2.plot(df.index, df[z1yr_col], color=COLORS[f'{group}'],
             linewidth=1.5, label=GROUP_LABELS[group])
    ax2.set_ylabel("Z-Score (1yr)", fontsize=FONT['label'])
    ax2.axhline(2, color=COLORS['text_muted'], linewidth=0.5,
                linestyle='--', alpha=0.6)
    ax2.axhline(1, color=COLORS['text_muted'], linewidth=0.5,
                linestyle='--', alpha=0.6)
    ax2.axhline(-1, color=COLORS['text_muted'], linewidth=0.5,
                linestyle='--', alpha=0.6)
    ax2.axhline(-2, color=COLORS['text_muted'], linewidth=0.5,
                linestyle='--', alpha=0.6)
    ax2.tick_params(labelbottom=False)
    ax2.set_ylim(-3.5, 3.5)

    # Bottom panel
    z3yr_col = f'{net_col}_z3yr'
    ax3.fill_between(df.index, df[z3yr_col], 0,
                     where=df[z3yr_col] >= 0,
                     color=COLORS['bull'], alpha=0.3, interpolate=True)
    ax3.fill_between(df.index, df[z3yr_col], 0,
                     where=df[z3yr_col] < 0,
                     color=COLORS['bear'], alpha=0.3, interpolate=True)
    ax3.plot(df.index, df[z3yr_col], color=COLORS[f'{group}'],
             linewidth=1.5, label=GROUP_LABELS[group])
    ax3.set_ylabel("Z-Score (3yr)", fontsize=FONT['label'])
    ax3.axhline(2, color=COLORS['text_muted'], linewidth=0.5,
                linestyle='--', alpha=0.6)
    ax3.axhline(1, color=COLORS['text_muted'], linewidth=0.5,
                linestyle='--', alpha=0.6)
    ax3.axhline(-1, color=COLORS['text_muted'], linewidth=0.5,
                linestyle='--', alpha=0.6)
    ax3.axhline(-2, color=COLORS['text_muted'], linewidth=0.5,
                linestyle='--', alpha=0.6)

    ax3.set_ylim(-3.5, 3.5)


    # --- Branding ---
    add_title(ax1, f"{symbol} | {GROUP_LABELS[group]} — Positioning Momentum",
              f"Week of {date.today().strftime('%b %d, %Y')} | Source: CFTC COT")
    add_watermark(fig)
    add_signature_stripe(fig)
    add_source(fig)

    if save:
        save_chart(fig, f"cot_momentum_{symbol}_{group}.png")

    return fig


def chart_cot_all(
        symbol: str,
        lookback_years: int = 3,
        save: bool = True,
) -> list[plt.Figure]:
    """
    Generate all COT charts for a symbol — overview + momentum for each group.

    Determines available groups from the market's report type (TFF vs
    Disaggregated) and generates both chart types per group.
    Returns a list of all generated figures.
    """
    groups = {
        'tff': ["spec", "comm", "asset_mgr"],
        'disaggregated': ["spec", "comm", "swap"]
    }

    current_group = MARKETS[symbol]['cot_report']

    group = groups[current_group]

    fig_list = []
    for g in group:
        fig_list.append(chart_cot_overview(symbol, g, lookback_years, save))
        fig_list.append(chart_cot_momentum(symbol, g, lookback_years, save))

    return fig_list


def chart_seasonal(
        symbol: str,
        lookback: int = 10,
        save: bool = True,
) -> plt.Figure:
    """
    Seasonal bar chart — week-of-year average returns
      with current week highlighted.
    """
    apply_style()
    sm = compute_seasonal_matrix(symbol, lookbacks=[lookback])
    label = f"{lookback}yr"

    current_week = date.today().isocalendar()[1]

    fig, ax = plt.subplots()

    colors = [
        COLORS['current_week'] if w == current_week
        else COLORS['seasonal_positive'] if v >= 0
        else COLORS['seasonal_negative']
        for w, v in zip(sm.index, sm[f'mean_return_{label}'])
    ]

    ax.bar(sm.index, sm[f'mean_return_{label}'] * 100,
           width=0.6, color=colors, alpha=0.8)

    # Current week glow effect
    if current_week in sm.index:
        val = sm.loc[current_week, f'mean_return_{label}'] * 100
        ax.bar(current_week, val, width=0.9, color=COLORS['current_week'], alpha=0.2)

        win_rate = sm.loc[current_week, f'win_rate_{label}']
        ax.annotate(
            f'Week {current_week}\n{val:+.2f}\nWR: {win_rate:.0f}%',
            xy=(current_week, val),
            xytext=(current_week + 4, val),
            fontsize=FONT['annotation'],
            color=COLORS['current_week'],
            arrowprops=dict(arrowstyle="-", color=COLORS["current_week"], alpha=0.5),
            va='center',
        )

    # Average line
    avg = sm[f"mean_return_{label}"].mean() * 100
    ax.axhline(avg, color=COLORS["text_muted"], linewidth=0.8,
               linestyle="--", alpha=0.6)
    ax.annotate(f" avg: {avg:+.2f}%", xy=(53, avg),
                fontsize=FONT["annotation"], color=COLORS["text_muted"],
                va="center")

    ax.axhline(0, color=COLORS["text_muted"], linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Week of Year", fontsize=FONT["label"])
    ax.set_ylabel("Avg Weekly Return (%)", fontsize=FONT["label"])
    ax.set_xlim(0.5, 53.5)

    add_title(ax, f"{symbol} | Seasonal Returns ({lookback}yr)",
              f"Current: Week {current_week} | Source: yfinance")
    add_watermark(fig)
    add_signature_stripe(fig)
    add_source(fig, f"Source: yfinance | {lookback}-year average | @marketsmanners")

    if save:
        save_chart(fig, f"seasonal_{symbol}.png")

    return fig


def chart_price_ma(
      symbol: str,
      lookback_days: int = 252,
      save: bool = True,
) -> plt.Figure:
    """
    Price + moving average overlay with Donchian channel.
    """
    apply_style()
    df = build_price_dataset(symbol)

    # Filter to lookback
    df = df.iloc[-lookback_days:]

    fig, ax = plt.subplots()

    # Donchian channel fill
    ax.fill_between(df.index, df["donchian_50_upper"], df["donchian_50_lower"],
                    color=COLORS["donchian"], alpha=0.08)

    # Price line
    ax.plot(df.index, df["close"], color=COLORS["price"],
            linewidth=1.2, label="Price")

    # MAs with descending thickness
    for ma, color, lw in [
      ("sma_200", COLORS["sma_200"], 1.8),
      ("sma_100", COLORS["sma_100"], 1.4),
      ("sma_50", COLORS["sma_50"], 1.0),
      ("sma_20", COLORS["sma_20"], 0.8),
    ]:
        ax.plot(df.index, df[ma], color=color, linewidth=lw)
        # Inline label at right edge
        last_val = df[ma].dropna().iloc[-1]
        ax.annotate(
          f" {ma.upper().replace('_', ' ')}",
          xy=(df.index[-1], last_val),
          fontsize=FONT["annotation"],
          color=color,
          va="center",
        )

    ax.set_ylabel("Price", fontsize=FONT["label"])

    add_title(ax, f"{symbol} | Price & Moving Averages",
              f"As of {date.today().strftime('%b %d, %Y')} | 1-year view")
    add_watermark(fig)
    add_signature_stripe(fig)
    add_source(fig, f"Source: yfinance | @marketsmanners")

    if save:
        save_chart(fig, f"price_ma_{symbol}.png")

    return fig
