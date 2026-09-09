"""Figure style: colourblind-safe, print-ready, journal column widths.

Palette is Okabe-Ito plus two accents.  Typography and spine rules follow the
common Nature/Cell house style: 7 pt ticks, 8 pt labels, 9 pt titles, 10 pt bold
panel letters, 0.8 pt axes, no top/right spines, fonts embedded as TrueType so
the PDF is editable.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

PAL = {
    "hill": "#0072B2",        # the proposed model
    "scalar": "#D55E00",      # ScalarHILL
    "naive": "#999999",
    "baseline": "#56B4E9",
    "moli": "#009E73",
    "superfelt": "#E69F00",
    "elasticnet": "#CC79A7",
    "randomforest": "#8C6D31",
    "m2": "#D55E00",          # two-parameter (official) fit
    "m3": "#0072B2",          # three-parameter (proposed) fit
    "efficacy": "#AA4499",
    "potency": "#117733",
    "censored": "#C44536",
    "accent": "#D62828",
    "text": "#222222",
    "grid": "#DDDDDD",
    "gray": "#9A9A9A",
    "gray_light": "#D9D9D9",
}

METHOD_COLOR = {
    "step0": PAL["scalar"], "step1": "#5A9BD5", "step2": PAL["hill"],
    "step3": "#2E5E8A", "step4": "#1F4E79", "step5": "#123A5C",
    "naive": PAL["naive"], "elasticnet": PAL["elasticnet"],
    "randomforest": PAL["randomforest"], "moli": PAL["moli"], "superfelt": PAL["superfelt"],
}

MM = 1 / 25.4
W_SINGLE = 89 * MM
W_ONEHALF = 120 * MM
W_DOUBLE = 183 * MM


def apply_rc() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": PAL["text"],
            "axes.labelcolor": PAL["text"],
            "axes.titlepad": 4.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.color": PAL["text"],
            "ytick.color": PAL["text"],
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.default": "regular",
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
        }
    )


def panel(ax, letter: str, size: int = 10, **_ignored) -> None:
    """Bold panel letter, placed as a left-aligned title so it never collides."""
    ax.set_title(letter, loc="left", fontsize=size, fontweight="bold", color="#111111", pad=3)


def despine(ax, left: bool = True, bottom: bool = True) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def sig_stars(p: float) -> str:
    if p != p:
        return "n/a"
    return "***" if p < 1e-3 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def p_text(p: float) -> str:
    if p != p:
        return "p = n/a"
    if p < 1e-4:
        return f"p = {p:.0e}".replace("e-0", "e-")
    return f"p = {p:.3f}"


def watermark_synthetic(fig, active: bool) -> None:
    """Stamp SYNTHETIC across a figure built from simulated data (project rule R2)."""
    if not active:
        return
    fig.text(0.5, 0.5, "SYNTHETIC", fontsize=40, color="#C44536", alpha=0.16,
             ha="center", va="center", rotation=30, zorder=100, fontweight="bold")
