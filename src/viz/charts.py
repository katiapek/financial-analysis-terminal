"""
Chart generators — branded chart functions for X/Substack content.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
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

GROUP_MAP = {
    'tff':          ['spec', 'comm', 'asset_mgr'],
    'disaggregated': ['spec', 'comm', 'swap'],
}

# ---------------------------------------------------------------------------
# Heatmap color helpers
# ---------------------------------------------------------------------------

def _zscore_color(z: float) -> str:
    """Map z-score to a red/green color. |z|>=2 = fully saturated."""
    if np.isnan(z):
        return COLORS['bg_panel']
    intensity = min(abs(z) / 2.5, 1.0)
    if z > 0:
        r, g, b = 34, 197, 94      # COLORS['bull'] #22c55e
    else:
        r, g, b = 239, 68, 68      # COLORS['bear'] #ef4444
    # Blend with panel background (#111827 = 17, 24, 39)
    br, bg_, bb = 17, 24, 39
    r = int(br + (r - br) * intensity)
    g = int(bg_ + (g - bg_) * intensity)
    b = int(bb + (b - bb) * intensity)
    return f"#{r:02x}{g:02x}{b:02x}"


def _index_color(val: float) -> str:
    """Map 0-100 index to red (0) → gray (50) → green (100)."""
    if np.isnan(val):
        return COLORS['bg_panel']
    if val >= 50:
        intensity = min((val - 50) / 40, 1.0)   # 90+ = full green
        r, g, b = 34, 197, 94
    else:
        intensity = min((50 - val) / 40, 1.0)    # 10- = full red
        r, g, b = 239, 68, 68
    br, bg_, bb = 17, 24, 39
    r = int(br + (r - br) * intensity)
    g = int(bg_ + (g - bg_) * intensity)
    b = int(bb + (b - bb) * intensity)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Cross-market positioning heatmap
# ---------------------------------------------------------------------------

def chart_positioning_heatmap(
        symbols: list[str] | None = None,
        save: bool = True,
) -> plt.Figure:
    """
    Cross-market positioning heatmap — all markets and groups in one chart.

    Shows Net, WoW Change, Z-1yr, Z-3yr, 26W Index, and Percentile for
    every trader group across all markets. Color-coded z-scores and index
    values. Market blocks visually grouped with alternating backgrounds.
    """
    apply_style()

    if symbols is None:
        symbols = list(MARKETS.keys())

    # --- Build data rows ---
    rows = []
    for sym in symbols:
        df = build_cot_dataset(sym)
        latest = df.iloc[-1]
        report = MARKETS[sym]['cot_report']

        for group in GROUP_MAP[report]:
            net = f'{group}_net'
            rows.append({
                'market': sym,
                'group': GROUP_LABELS[group],
                'net': latest[net],
                'wow': latest[f'{net}_roc'],
                'z1yr': latest[f'{net}_z1yr'],
                'z3yr': latest[f'{net}_z3yr'],
                'idx26w': latest[f'{net}_26w'],
                'pctl': latest[f'{net}_pctl'],
            })

    # --- Layout constants ---
    n_rows = len(rows)
    row_h = 1.0                     # height per data row
    header_h = 1.2                  # header row height
    total_h = header_h + n_rows * row_h

    # Column x-positions and widths: (x_start, width, label, align)
    columns = [
        (0.0,  1.8, 'Market',   'left'),
        (1.8,  2.8, 'Group',    'left'),
        (4.6,  2.0, 'Net',      'right'),
        (6.6,  1.8, 'WoW Chg',  'right'),
        (8.4,  1.3, 'Z-1yr',    'center'),
        (9.7,  1.3, 'Z-3yr',    'center'),
        (11.0, 1.3, '26W Idx',  'center'),
        (12.3, 1.2, 'Pctl',     'center'),
    ]
    total_w = 13.5

    fig_w = 8.0
    fig_h = 4.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis('off')

    # --- Draw header row ---
    header_y = total_h - header_h
    ax.add_patch(mpatches.Rectangle(
        (0, header_y), total_w, header_h,
        facecolor=COLORS['accent'], alpha=0.25, edgecolor='none',
    ))
    for x, w, label, align in columns:
        tx = x + 0.15 if align == 'left' else x + w - 0.15 if align == 'right' else x + w / 2
        ha = align if align != 'center' else 'center'
        ax.text(tx, header_y + header_h / 2, label,
                fontsize=FONT['annotation'], fontweight='bold',
                color=COLORS['text'], ha=ha, va='center')

    # --- Draw data rows ---
    current_market = None
    market_idx = -1

    for i, row in enumerate(rows):
        y = header_y - (i + 1) * row_h

        # Track market blocks for alternating backgrounds
        if row['market'] != current_market:
            current_market = row['market']
            market_idx += 1

        # Alternating market block background
        bg_alpha = 0.08 if market_idx % 2 == 0 else 0.0
        ax.add_patch(mpatches.Rectangle(
            (0, y), total_w, row_h,
            facecolor=COLORS['text'], alpha=bg_alpha, edgecolor='none',
        ))

        # Thin separator line between market blocks
        if i > 0 and rows[i - 1]['market'] != row['market']:
            ax.plot([0, total_w], [y + row_h, y + row_h],
                    color=COLORS['grid'], linewidth=0.8, alpha=0.6)

        cell_y = y + row_h / 2   # vertical center of row

        # Col 0: Market — show only on first row of each block
        if i == 0 or rows[i - 1]['market'] != row['market']:
            ax.text(0.15, cell_y, row['market'],
                    fontsize=FONT['label'], fontweight='bold',
                    color=COLORS['accent'], ha='left', va='center')

        # Col 1: Group
        ax.text(1.95, cell_y, row['group'],
                fontsize=FONT['tick'], color=COLORS['text'],
                ha='left', va='center')

        # Col 2: Net
        ax.text(6.45, cell_y, f"{row['net']:+,.0f}",
                fontsize=FONT['tick'], color=COLORS['text'],
                ha='right', va='center')

        # Col 3: WoW Change
        wow_color = COLORS['bull'] if row['wow'] >= 0 else COLORS['bear']
        ax.text(8.25, cell_y, f"{row['wow']:+,.0f}",
                fontsize=FONT['tick'], color=wow_color,
                ha='right', va='center')

        # Col 4: Z-1yr (color-coded background)
        z1_x, z1_w = 8.4, 1.3
        ax.add_patch(mpatches.FancyBboxPatch(
            (z1_x + 0.05, y + 0.1), z1_w - 0.1, row_h - 0.2,
            boxstyle="round,pad=0.05",
            facecolor=_zscore_color(row['z1yr']), edgecolor='none',
        ))
        ax.text(z1_x + z1_w / 2, cell_y, f"{row['z1yr']:+.1f}",
                fontsize=FONT['tick'], fontweight='bold',
                color=COLORS['text'], ha='center', va='center')

        # Col 5: Z-3yr (color-coded background)
        z3_x, z3_w = 9.7, 1.3
        ax.add_patch(mpatches.FancyBboxPatch(
            (z3_x + 0.05, y + 0.1), z3_w - 0.1, row_h - 0.2,
            boxstyle="round,pad=0.05",
            facecolor=_zscore_color(row['z3yr']), edgecolor='none',
        ))
        ax.text(z3_x + z3_w / 2, cell_y, f"{row['z3yr']:+.1f}",
                fontsize=FONT['tick'], fontweight='bold',
                color=COLORS['text'], ha='center', va='center')

        # Col 6: 26W Index (color-coded background)
        idx_x, idx_w = 11.0, 1.3
        ax.add_patch(mpatches.FancyBboxPatch(
            (idx_x + 0.05, y + 0.1), idx_w - 0.1, row_h - 0.2,
            boxstyle="round,pad=0.05",
            facecolor=_index_color(row['idx26w']), edgecolor='none',
        ))
        ax.text(idx_x + idx_w / 2, cell_y, f"{row['idx26w']:.0f}",
                fontsize=FONT['tick'], fontweight='bold',
                color=COLORS['text'], ha='center', va='center')

        # Col 7: Percentile (color-coded background)
        pct_x, pct_w = 12.3, 1.2
        ax.add_patch(mpatches.FancyBboxPatch(
            (pct_x + 0.05, y + 0.1), pct_w - 0.1, row_h - 0.2,
            boxstyle="round,pad=0.05",
            facecolor=_index_color(row['pctl']), edgecolor='none',
        ))
        ax.text(pct_x + pct_w / 2, cell_y, f"{row['pctl']:.0f}",
                fontsize=FONT['tick'], fontweight='bold',
                color=COLORS['text'], ha='center', va='center')

    # --- Branding ---
    fig.text(0.5, 0.95,
             "COT Positioning Heatmap",
             fontsize=FONT['title'], fontweight='bold',
             color=COLORS['text'], ha='center', va='top')
    fig.text(0.5, 0.91,
             f"Week of {date.today().strftime('%b %d, %Y')} | Source: CFTC COT",
             fontsize=FONT['subtitle'], color=COLORS['text_muted'],
             ha='center', va='top')

    add_watermark(fig)
    add_signature_stripe(fig)
    add_source(fig)

    if save:
        save_chart(fig, "cot_heatmap.png")

    return fig


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
        std = sm.loc[current_week, f'std_{label}'] * 100

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

    subtitle = f"Week {current_week}"
    if current_week in sm.index:
        subtitle += f" | {val:+.2f}% | WR: {win_rate:.0f}% | σ: {std:.2f}%"
    subtitle += " | Source: yfinance"
    add_title(ax, f"{symbol} | Seasonal Returns ({lookback}yr)", subtitle)
    add_watermark(fig)
    add_signature_stripe(fig)
    add_source(fig, f"Source: yfinance | {lookback}-year average | @marketsmanners")

    if save:
        save_chart(fig, f"seasonal_{symbol}.png")

    return fig


def chart_price_daily(
      symbol: str,
      lookback_days: int = 252,
      save: bool = True,
) -> plt.Figure:
    """
    Daily price + moving average overlay with Donchian channel.
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


def chart_price_weekly(
        symbol: str,
        lookback_years: int = 3,
        save: bool = True,
) -> plt.Figure:
    """
    Weekly price with moving averages and COT open interest.

    Top panel shows weekly close with 20w, 50w, 100w, 200w SMAs.
    Bottom panel shows total open interest from COT data as bars.
    """
    apply_style()
    df = build_weekly_dataset(symbol)
    cot = build_cot_dataset(symbol)

    # Filter to lookback
    cutoff = pd.Timestamp(date.today()) - pd.DateOffset(years=lookback_years)
    df = df[df.index > cutoff]
    cot = cot[cot.index > cutoff]

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=[8, 4.5],
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3}
    )

    # Top panel — weekly price + MAs
    ax1.plot(df.index, df['close'], color=COLORS['price'],
             linewidth=1.2, label='Price')

    for ma, color, lw in [
        ('sma_200', COLORS['sma_200'], 1.8),
        ('sma_100', COLORS['sma_100'], 1.4),
        ('sma_50', COLORS['sma_50'], 1.0),
        ('sma_20', COLORS['sma_20'], 0.8),
    ]:
        ax1.plot(df.index, df[ma], color=color, linewidth=lw)
        last_val = df[ma].dropna().iloc[-1]
        ax1.annotate(
            f" {ma.upper().replace('_', ' ')}",
            xy=(df.index[-1], last_val),
            fontsize=FONT['annotation'],
            color=color,
            va='center',
        )

    ax1.set_ylabel('Price', fontsize=FONT['label'])
    ax1.tick_params(labelbottom=False)

    # Bottom panel — open interest
    ax2.bar(cot.index, cot['open_interest'], color=COLORS['accent'],
            width=5, alpha=0.4)
    ax2.set_ylabel('Open Interest', fontsize=FONT['label'])

    # --- Branding ---
    add_title(ax1, f"{symbol} | Weekly Price & Open Interest",
              f"As of {date.today().strftime('%b %d, %Y')} | Source: yfinance + CFTC")
    add_watermark(fig)
    add_signature_stripe(fig)
    add_source(fig, f"Source: yfinance + CFTC COT | @marketsmanners")

    if save:
        save_chart(fig, f"price_weekly_{symbol}.png")

    return fig
