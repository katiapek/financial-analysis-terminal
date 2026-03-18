"""
Chart branding — colors, fonts, dimensions, and matplotlib style config.

Single source of truth for all visual output. Every chart function
imports from here to ensure consistent branding across X posts,
Substack reports, and dashboard views.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from pathlib import Path

# --- Dimensions (X/Twitter optimized, 16:9) ---
CHART_WIDTH = 1200
CHART_HEIGHT = 675
CHART_DPI = 150
CHART_FIGSIZE = (CHART_WIDTH / CHART_DPI, CHART_HEIGHT / CHART_DPI)  # 8 x 4.5 inches

OUTPUT_DIR = Path("output/charts")

# --- Color Palette ---
COLORS = {
    # Background & text
    "bg": "#0a0e17",
    "bg_panel": "#111827",
    "text": "#e0e0e0",
    "text_muted": "#8b949e",
    "grid": "#1f2937",

    # Directional
    "bull": "#22c55e",
    "bear": "#ef4444",
    "neutral": "#6b7280",

    # Data series (COT positioning)
    "spec": "#3b82f6",
    "comm": "#f59e0b",
    "asset_mgr": "#8b5cf6",
    "swap": "#ec4899",

    # Price & indicators
    "price": "#e0e0e0",
    "sma_20": "#facc15",
    "sma_50": "#fb923c",
    "sma_100": "#38bdf8",
    "sma_200": "#a855f7",
    "donchian": "#06b6d4",
    "atr": "#f43f5e",

    # Seasonal
    "seasonal_positive": "#22c55e",
    "seasonal_negative": "#ef4444",
    "current_week": "#facc15",

    # Branding
    "accent": "#3b82f6",
    "watermark": "#2d3a4a",
}

# --- Watermark ---
WATERMARK_TEXT = "@marketsmanners"

# --- Font Sizes ---
FONT = {
    "title": 14,
    "subtitle": 10,
    "label": 10,
    "tick": 8,
    "watermark": 24,
    "annotation": 9,
    "source": 7,
}


def apply_style():
    """Apply Markets Manners chart style globally."""
    mpl.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],

        # Figure
        "figure.facecolor": COLORS["bg"],
        "figure.figsize": CHART_FIGSIZE,
        "figure.dpi": CHART_DPI,
        "figure.subplot.left": 0.10,
        "figure.subplot.right": 0.92,
        "figure.subplot.top": 0.88,
        "figure.subplot.bottom": 0.12,

        # Axes
        "axes.facecolor": COLORS["bg_panel"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "axes.grid": True,
        "axes.formatter.use_mathtext": True,

        # Grid
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.5,

        # Text & ticks
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text_muted"],
        "ytick.color": COLORS["text_muted"],
        "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"],

        # Legend
        "legend.facecolor": COLORS["bg_panel"],
        "legend.edgecolor": COLORS["grid"],
        "legend.fontsize": FONT["annotation"],

        # Save
        "savefig.dpi": CHART_DPI,
        "savefig.facecolor": COLORS["bg"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


def add_watermark(fig):
    """Add branded attribution to bottom-right corner of figure."""
    fig.text(
        0.98, 0.02,
        WATERMARK_TEXT,
        fontsize=FONT["annotation"],
        color=COLORS["text_muted"],
        alpha=0.6,
        ha="right", va="bottom",
        fontweight="medium",
        fontstyle="italic",
    )


def add_anti_theft_watermark(ax):
    """Add large centered watermark for teaser/gated content."""
    ax.text(
        0.5, 0.5,
        WATERMARK_TEXT,
        transform=ax.transAxes,
        fontsize=FONT["watermark"],
        color=COLORS["watermark"],
        alpha=0.3,
        ha="center", va="center",
        fontweight="bold",
    )


def add_signature_stripe(fig):
    """Add accent blue stripe across bottom edge — brand signature."""
    fig.patches.append(mpatches.Rectangle(
        (0, 0), 1, 0.004,
        transform=fig.transFigure,
        facecolor=COLORS["accent"],
        clip_on=False,
        zorder=10,
    ))


def add_title(ax, title: str, subtitle: str = ""):
    """Add styled title and optional subtitle."""
    ax.set_title(title, fontsize=FONT["title"], fontweight="bold",
                 color=COLORS["text"], pad=22)
    if subtitle:
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes,
                fontsize=FONT["subtitle"], color=COLORS["text_muted"],
                ha="center", va="bottom")


def add_source(fig, source_text: str = "Source: CFTC COT | @marketsmanners"):
    """Add source attribution line at bottom-left."""
    fig.text(
        0.10, 0.02,
        source_text,
        fontsize=FONT["source"],
        color=COLORS["text_muted"],
        alpha=0.6,
        ha="left", va="bottom",
    )


def save_chart(fig, filename: str):
    """Save chart to output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath)
    plt.close(fig)
    return filepath
