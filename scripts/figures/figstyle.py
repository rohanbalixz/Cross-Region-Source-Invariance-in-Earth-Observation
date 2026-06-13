"""Shared house style for every figure: one coherent visual
identity, colourblind-safe (Okabe--Ito), no chartjunk, editable-text PDFs.
Import and call use_style() at the top of each figure script."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe--Ito colourblind-safe qualitative palette
C = {
    "blue":   "#0072B2", "orange": "#E69F00", "green":  "#009E73",
    "red":    "#D55E00", "purple": "#CC79A7", "sky":    "#56B4E9",
    "yellow": "#F0E442", "grey":   "#7F7F7F", "ink":    "#222222",
}
# semantic roles used across figures
HARMONISED = C["green"]      # globally-harmonised representation
MEASUREMENT = C["orange"]    # uniform index / measurement
RAWSENSOR = C["red"]         # raw sensor
TEMPORAL = C["blue"]         # temporal-history family
IMAGERY = C["red"]           # imagery family
BASELINE = C["grey"]


def use_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,                 # editable text
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "TeX Gyre Heros", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.titleweight": "regular", "axes.labelcolor": C["ink"],
        "text.color": C["ink"], "axes.edgecolor": "#555555",
        "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.color": "#555555", "ytick.color": "#555555",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "legend.fontsize": 7, "legend.frameon": False, "legend.handlelength": 1.1,
        "legend.borderpad": 0.25, "legend.labelspacing": 0.3, "legend.columnspacing": 1.0,
        "lines.linewidth": 1.4, "lines.solid_capstyle": "round",
        "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def panel(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="bottom", ha="left", color=C["ink"])


def despine(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
