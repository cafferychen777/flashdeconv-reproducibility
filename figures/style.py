"""Shared Nature-style settings for FlashDeconv main figures.

Import in every figure script:

    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from style import *          # noqa
    apply_style()
    fig = new_figure(height_mm=170)

Conventions
-----------
* Double-column width 180 mm, height <= 230 mm.
* Arial, 5-7 pt text, 8 pt bold lowercase panel letters.
* 0.5 pt spines, no top/right spines, outward ticks.
* Okabe-Ito colours for methods; a fixed lineage palette shared by all figures.
* Spatial maps: rasterized scatter/imshow, equal aspect, no ticks, plain scale bar.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(__file__).resolve().parents[2]
RESULTS = PROJ / "results"
FIGDATA = RESULTS / "figdata"
FIGDIR = PROJ / "paper" / "figures"

MM = 1 / 25.4
FIG_W_MM = 180.0
FIG_H_MAX_MM = 230.0

# ---------------------------------------------------------------------------
# Font sizes (pt)
# ---------------------------------------------------------------------------
FS_TINY = 5.0
FS_SMALL = 5.5
FS = 6.0
FS_LARGE = 7.0
FS_PANEL = 8.0

LW = 0.5  # spine / tick / default line width

# ---------------------------------------------------------------------------
# Okabe-Ito
# ---------------------------------------------------------------------------
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}

FD_COLOR = OI["vermillion"]

# Method colours (fixed across all figures)
METHOD_COLORS = {
    "FlashDeconv": FD_COLOR,
    "RCTD": OI["blue"],
    "RCTD (doublet)": OI["blue"],
    "RCTD (full)": OI["blue"],
    "RCTD (full, UMI ≥ 20)": OI["blue"],
    "Cell2location": OI["green"],
    "CARD": OI["orange"],
    "NNLS": OI["skyblue"],
    "Marker scoring": OI["purple"],
    "DestVI": "#8C6D31",
    "Other": "#B8B8B8",
}
OTHER_METHOD_COLOR = "#B8B8B8"

# RCTD settings share the RCTD colour and differ by marker / line style (all figures)
RCTD_MARKERS = {"doublet": "o", "full": "s", "full20": "^"}
RCTD_LS = {"doublet": "-", "full": "--", "full20": ":"}

# Orthogonal / ground-truth measurements
TRUTH_COLOR = "#333333"
ORTHO_COLORS = {"Xenium": "#333333", "CODEX": "#333333", "Truth": "#333333"}

# Gene-weighting schemes (panels on leverage weighting)
WEIGHT_COLORS = {
    "leverage": FD_COLOR,        # FlashDeconv default
    "equal": "#8A8A8A",
    "variance": OI["blue"],
}
WEIGHT_LABELS = {
    "leverage": "Expected leverage",
    "equal": "Equal",
    "variance": "Between-type variance",
}

# Reference-diagnostic colours
FLAG_COLOR = OI["vermillion"]
UNFLAG_COLOR = "#D9D9D9"
REF_COLORS = {"incomplete": "#5A5A5A", "complete": OI["blue"]}

# ---------------------------------------------------------------------------
# Lineage palette (shared by every figure)
# ---------------------------------------------------------------------------
LINEAGE_COLORS = {
    "Epithelial": "#E69F00",       # tumour / epithelium
    "Tumor": "#E69F00",
    "Fibroblast": "#009E73",
    "Stromal": "#009E73",
    "Pericyte/SMC": "#8FBC8F",
    "Endothelial": "#56B4E9",
    "Macrophage/Mono": "#0072B2",
    "Macrophage": "#0072B2",
    "DC": "#6A3D9A",
    "mRegDC": "#6A3D9A",
    "Neutrophil": "#D55E00",
    "Mast": "#8C510A",
    "CD4 T": "#CC79A7",
    "CD8 T": "#B2182B",
    "NK": "#F4A582",
    "B": "#F0E442",
    "Plasma": "#BFA600",
    "Immune": "#CC79A7",
    "Other": "#BDBDBD",
}

# Fine CRC cell type -> display lineage used in maps
CRC_TYPE_TO_LINEAGE = {
    **{f"Tumor {r}": "Tumor" for r in ["I", "II", "III", "IV", "V"]},
    "Enterocyte": "Epithelial", "Goblet": "Epithelial", "Tuft": "Epithelial",
    "Epithelial": "Epithelial", "Neuroendocrine": "Epithelial",
    "CAF": "Fibroblast", "Myofibroblast": "Fibroblast", "Fibroblast": "Fibroblast",
    "Proliferating Fibroblast": "Fibroblast", "Vascular Fibroblast": "Fibroblast",
    "Smooth Muscle": "Pericyte/SMC", "SM Stress Response": "Pericyte/SMC",
    "vSM": "Pericyte/SMC", "Pericytes": "Pericyte/SMC", "Unknown III (SM)": "Pericyte/SMC",
    "Endothelial": "Endothelial", "Lymphatic Endothelial": "Endothelial",
    "Macrophage": "Macrophage/Mono", "Proliferating Macrophages": "Macrophage/Mono",
    "mRegDC": "DC", "cDC I": "DC", "pDC": "DC",
    "Neutrophil": "Neutrophil", "Mast": "Mast",
    "CD4 T cell": "CD4 T", "CD8 T cell": "CD8 T", "NK": "NK",
    "Mature B": "B", "Memory B": "B", "Plasma": "Plasma",
    "Proliferating Immune II": "Other", "Enteric Glial": "Other", "Adipocyte": "Other",
}


def lineage_color(name: str) -> str:
    return LINEAGE_COLORS.get(name, LINEAGE_COLORS.get(CRC_TYPE_TO_LINEAGE.get(name, "Other"), "#BDBDBD"))


def method_color(name: str) -> str:
    return METHOD_COLORS.get(name, OTHER_METHOD_COLOR)


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.titleweight": "normal",
        "axes.titlepad": 3,
        "axes.labelsize": FS,
        "axes.labelpad": 2,
        "axes.linewidth": LW,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black",
        "axes.facecolor": "none",
        "xtick.labelsize": FS_SMALL,
        "ytick.labelsize": FS_SMALL,
        "xtick.major.width": LW,
        "ytick.major.width": LW,
        "xtick.minor.width": LW * 0.8,
        "ytick.minor.width": LW * 0.8,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": FS_SMALL,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.2,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.25,
        "legend.columnspacing": 0.8,
        "lines.linewidth": 0.8,
        "lines.markersize": 3,
        "patch.linewidth": LW,
        "errorbar.capsize": 0,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "image.interpolation": "nearest",
        "mathtext.default": "regular",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    })
    # Larger, less raised super/subscripts: exponents stay >= ~5 pt at 5.5-6 pt text
    import matplotlib._mathtext as _mt
    _mt.SHRINK_FACTOR = 0.9
    _mt.FontConstantsBase.sup1 = 0.5


def new_figure(height_mm: float, width_mm: float = FIG_W_MM):
    assert height_mm <= FIG_H_MAX_MM, "Nature max height is 230 mm"
    return plt.figure(figsize=(width_mm * MM, height_mm * MM))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def panel_label(ax_or_fig, letter: str, x: float = None, y: float = None,
                dx_mm: float = -6.0, dy_mm: float = 1.5):
    """Bold lowercase 8-pt panel letter placed at the top-left of an axes.

    If x, y are given they are figure-fraction coordinates; otherwise the
    letter is placed dx_mm left / dy_mm above the axes' top-left corner.
    """
    if hasattr(ax_or_fig, "get_position") and x is None:
        ax = ax_or_fig
        fig = ax.figure
        bb = ax.get_position()
        W, H = fig.get_size_inches() / MM
        x = bb.x0 + dx_mm / W
        y = bb.y1 + dy_mm / H
    else:
        fig = ax_or_fig.figure if hasattr(ax_or_fig, "figure") and ax_or_fig.figure else ax_or_fig
    fig.text(x, y, letter, fontsize=FS_PANEL, fontweight="bold", ha="left",
             va="bottom", transform=fig.transFigure)


def scale_bar(ax, length_um: float, units_per_um: float = 1.0, label: str = None,
              loc: str = "lower right", pad: float = 0.04, color: str = "black",
              lw: float = 1.0, fontsize: float = FS_SMALL):
    """Plain line scale bar with a length label (no box) in data coordinates."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    L = length_um * units_per_um
    w, h = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    if "right" in loc:
        xe = x1 - sx * pad * w
        xs = xe - sx * L
    else:
        xs = x0 + sx * pad * w
        xe = xs + sx * L
    yb = (y0 + sy * pad * h) if "lower" in loc else (y1 - sy * pad * h)
    ax.plot([xs, xe], [yb, yb], color=color, lw=lw, solid_capstyle="butt",
            clip_on=False, zorder=10)
    if label is None:
        label = f"{length_um / 1000:g} mm" if length_um >= 1000 else f"{length_um:g} µm"
    ty = yb + sy * 0.015 * h
    ax.text((xs + xe) / 2, ty, label, ha="center",
            va="bottom", fontsize=fontsize, color=color, zorder=10)


def clean_spatial(ax):
    """Spatial-map axes: equal aspect, no ticks, no spines."""
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def spatial_scatter(ax, x, y, c=None, s=0.05, cmap=None, vmin=None, vmax=None,
                    color=None, rasterized=True, **kw):
    """Rasterized scatter for spatial maps (vector PDF, raster points)."""
    return ax.scatter(x, y, c=c if color is None else None,
                      color=color, s=s, cmap=cmap, vmin=vmin, vmax=vmax,
                      linewidths=0, marker="s", rasterized=rasterized, **kw)


def despine(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def mean_ci(x, axis=None, level=0.95):
    """Mean and t-based 95% CI half-width."""
    from scipy import stats
    x = np.asarray(x, float)
    x = x[np.isfinite(x)] if axis is None else x
    n = x.shape[0] if axis is None else x.shape[axis]
    m = np.nanmean(x, axis=axis)
    se = np.nanstd(x, axis=axis, ddof=1) / np.sqrt(n)
    return m, stats.t.ppf(0.5 + level / 2, n - 1) * se


def format_p(p: float) -> str:
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        m = p / 10 ** e
        return f"P = {m:.0f}×10$^{{{e}}}$" if m >= 1 else f"P < 10$^{{{e + 1}}}$"
    if p < 0.01:
        return f"P = {p:.3f}"
    return f"P = {p:.2f}"


def colorbar_small(fig, mappable, rect, label="", orientation="horizontal", ticks=None):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation, ticks=ticks)
    cb.outline.set_linewidth(LW * 0.8)
    cb.ax.tick_params(labelsize=FS_TINY, width=LW * 0.8, length=1.5, pad=1)
    if label:
        cb.set_label(label, fontsize=FS_TINY, labelpad=1)
    return cb


def save(fig, name: str):
    """Save paper/figures/{name}.pdf (vector) and .png (300 dpi)."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf", dpi=600)
    fig.savefig(FIGDIR / f"{name}.png", dpi=300)
    print("saved", FIGDIR / f"{name}.pdf")
