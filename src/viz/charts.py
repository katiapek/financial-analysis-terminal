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
                     where=df['spec_net'] >=0,
                     color=COLORS['bull'], alpha=0.2)
    ax1.fill_between(df.index, df['spec_net'], 0,
                     where=df['spec_net'] < 0,
                     color=COLORS['bear'], alpha=0.2)
    ax1.plot(df.index, df['spec_net'], color=COLORS['spec_net'],
             linewidth=1.5, label="Managed Money")
    ax1.plot(df.index, df['comm_net'], color=COLORS['comm_net'],
             linewidth=1.5, label="Commercials")

    # Inline labels at right edge
    for col, color, label in [
        ('spec_net', COLORS['spec_net'], "Managed Money"),
        ('comm_net', COLORS['comm_net'], "Commercials"),
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
                     color=COLORS['bull'], alpha=0.3)
    ax2.fill_between(df.index, df[z_col], 0,
                     where=df[z_col] < 0,
                     color=COLORS['bear'], alpha=0.3)
    ax2.plot(df.index, df[z_col],color=COLORS['spec_net'], linewidth=1.0)

    # Z-score bands
    for level in [1,2,-1,-2]:
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
        for w,v in zip(sm.index, sm[f'mean_return_{label}'])
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
