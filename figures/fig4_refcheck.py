"""Figure 4: the reference diagnostic detects and names missing cell types.

Reads result tables directly and draws the whole figure on one canvas.
Output: paper/figures/fig4_refcheck.{pdf,png}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from style import *  # noqa: E402,F401,F403

apply_style()

RF = RESULTS / "rerun_final" / "refdiag"          # final package (v0.2.0 defaults), final rerun
RV2 = RESULTS / "reference_diagnostic_v2"          # package calibration runs (depth deciles)
B1 = RESULTS / "b1_pilot"                          # lung refcheck (fd_final env)

C_INC = REF_COLORS["incomplete"]   # epithelium-only / Flex reference
C_COM = REF_COLORS["complete"]     # composite / LUAD reference
REGION_COLORS = {"follicle": "#CC79A7",
                 "muscle": LINEAGE_COLORS["Fibroblast"],
                 "epithelium": LINEAGE_COLORS["Epithelial"]}
REGION_LABELS = {"follicle": "Follicle", "muscle": "Muscularis", "epithelium": "Epithelium"}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
perbin = pd.read_parquet(RF / "intestine_perbin_default.parquet")
PX2UM = 0.27383668  # from fig_a_spatial_flags_haber.csv.gz (x_um / x_px)
X = perbin.x_px.to_numpy() * PX2UM
Y = perbin.y_px.to_numpy() * PX2UM
flag_h = perbin.haber_flag_pooled.to_numpy()

reg = pd.read_csv(RF / "fig_b_flagged_fraction_by_region.csv")
gt = pd.read_csv(RF / "fig_c_genes_types.csv")
genes_def = pd.read_csv(RF / "intestine_unexplained_genes_default.csv")
comp_fol = (genes_def[(genes_def.ref == "composite") & (genes_def.set == "follicle_region")]
            .sort_values("rank").head(10))
# Spotless controlled removal and complete-reference calibration: lambda=0, because the
# silver-standard pseudo-spots carry no spatial layout (validation/controls_editor/refdiag_spotless_lam0.py)
RL0 = RESULTS / "editor_revision" / "refdiag_lam0"
rem = pd.read_csv(RL0 / "spotless_removal_auroc.csv")
rem = rem[np.isclose(rem.thr, 0.3) & (rem.n_pos >= 5) & (rem.n_neg >= 5)].copy()
cref = pd.read_csv(RL0 / "spotless_complete_reference_flag_rate.csv")
cref = cref.rename(columns={"flag_rate_complete_ref": "flag_rate"})   # package default null (auto)
print(f"refdiag lam0: {len(rem)} removals; median AUROC distinct>0.2 = "
      f"{rem[rem.distinct > 0.2].auroc.median():.3f}; complete-ref median flag = "
      f"{100 * cref.flag_rate.median():.1f}% (mean {100 * cref.flag_rate.mean():.1f}%)")
esc = pd.read_csv(RV2 / "fig_e_raw_score_vs_depth_P5.csv")
fpr = pd.read_csv(RV2 / "fig_f_complete_reference_fpr_by_depth.csv")
# Spotless depth deciles at lambda=0 (validation/controls_editor/refdiag_spotless_lam0_extra.py)
fl0 = pd.read_csv(RL0 / "spotless_fpr_by_depth_decile.csv").assign(decile=lambda d: d.decile + 1,
                                                                   fpr_pct=lambda d: 100 * d.fpr)
fpr = pd.concat([fpr[fpr.set != "spotless"], fl0[["set", "key", "null", "decile", "fpr_pct"]]])
lung = {s: json.load(open(B1 / f"refcheck_summary_{s}.json")) for s in ("LUNG_X1", "LUNG_X5K")}
lung_genes = {s: pd.read_csv(B1 / f"refcheck_topgenes_{s}_flex.csv") for s in ("LUNG_X1", "LUNG_X5K")}

# ---------------------------------------------------------------------------
# Canvas (mm-based placement)
# ---------------------------------------------------------------------------
H = 205.0
fig = new_figure(H)


def ax_mm(x, y, w, h):
    """Axes at (x, y) = top-left corner in mm from the figure's top-left."""
    return fig.add_axes([x / FIG_W_MM, 1 - (y + h) / H, w / FIG_W_MM, h / H])


def letter(ch, x, y):
    x = 2 if x < 10 else (63 if x < 100 else 123)
    fig.text(x / FIG_W_MM, 1 - y / H, ch, fontsize=FS_PANEL, fontweight="bold",
             ha="left", va="top")


def rot(x, y):
    # display orientation: image y downwards
    return x, -y


# ---- a/b: rasterized maps on a 12-um pixel grid (majority value per pixel) --
PIX = 12.0
ix = np.floor((X - X.min()) / PIX).astype(int)
iy = np.floor((Y - Y.min()) / PIX).astype(int)
NX, NY = ix.max() + 1, iy.max() + 1
lin = iy * NX + ix
cnt = np.bincount(lin, minlength=NX * NY)
occ = (cnt > 0).reshape(NY, NX)
EXT = [0, NX * PIX, NY * PIX, 0]  # image y downwards


def frac_img(mask):
    f = np.bincount(lin, weights=mask.astype(float), minlength=NX * NY)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (f / cnt).reshape(NY, NX)


def rgba(hexcol):
    import matplotlib.colors as mc
    return np.array(mc.to_rgba(hexcol))


def draw_map(ax):
    clean_spatial(ax)
    ax.set_xlim(-40, NX * PIX + 40)
    ax.set_ylim(NY * PIX + 420, -40)
    scale_bar(ax, 1000, loc="lower right", pad=0.02)


# a: flagged bins, epithelium-only reference
ax = ax_mm(4, 6, 52, 47)
img = np.zeros((NY, NX, 4))
img[occ] = rgba(UNFLAG_COLOR)
img[frac_img(flag_h) >= 0.5] = rgba(FLAG_COLOR)
ax.imshow(img, extent=EXT, interpolation="nearest", rasterized=True)
draw_map(ax)
ax.set_title("Epithelium-only reference", pad=2)
ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=3.5, color=FLAG_COLOR, label="Flagged"),
                   Line2D([], [], marker="s", ls="", ms=3.5, color=UNFLAG_COLOR, label="Not flagged")],
          loc="lower left", bbox_to_anchor=(0.0, -0.01), handletextpad=0.2)
letter("a", 0, 3)

# b: region masks from raw marker genes
ax = ax_mm(66, 6, 52, 47)
img = np.zeros((NY, NX, 4))
img[occ] = rgba("#EFEFEF")
for r in ("epithelium", "muscle", "follicle"):
    img[frac_img(perbin[r].to_numpy()) >= 0.5] = rgba(REGION_COLORS[r])
ax.imshow(img, extent=EXT, interpolation="nearest", rasterized=True)
draw_map(ax)
ax.set_title("Regions (raw marker genes)", pad=2)
ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=3.5, color=REGION_COLORS[r],
                          label=REGION_LABELS[r]) for r in ("follicle", "muscle", "epithelium")],
          loc="lower left", bbox_to_anchor=(0.0, -0.01), handletextpad=0.2)
letter("b", 63, 3)

# ---- c: flagged fraction by region, two references -------------------------
ax = ax_mm(132, 10, 44, 36)
order = ["follicle", "muscle", "epithelium"]
w = 0.38
for i, (ref, c, lab) in enumerate((("haber", C_INC, "Epithelium-only"),
                                   ("composite", C_COM, "Composite"))):
    v = [100 * reg[(reg.reference == ref) & (reg.region == o)].flagged_fraction.iloc[0] for o in order]
    ax.bar(np.arange(3) + (i - 0.5) * w, v, w, color=c, lw=0, label=lab)
ax.axhline(5, color="black", lw=0.5, ls=(0, (1, 1.5)))
ax.set_xticks(range(3), [REGION_LABELS[o] for o in order])
ax.set_ylabel("Bins flagged (%)")
ax.set_ylim(0, 22)
ax.set_yticks([0, 5, 10, 15, 20])
ax.set_xlim(-0.6, 2.6)
ax.legend(loc="upper right", title="Reference", title_fontsize=FS_SMALL, bbox_to_anchor=(1.02, 1.04))
letter("c", 122, 3)

# ---- d: unexplained genes and suggested types (epithelium-only reference) ---
ROW2_Y = 58


def gene_bars(ax, names, scores, color, xlabel=True, height=0.7):
    yy = np.arange(len(names))[::-1]
    ax.barh(yy, scores, color=color, height=height, lw=0)
    ax.set_yticks(yy, names, fontstyle="italic")
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.tick_params(axis="y", length=0, pad=1.5)
    if xlabel:
        ax.set_xlabel("Unexplained score (log$_2$ O/E)")


def type_bars(ax, names, scores):
    yy = np.arange(len(names))[::-1]
    ax.barh(yy, scores, color="#ABABAB", height=0.6, lw=0)
    ax.set_yticks(yy, [s[0].upper() + s[1:] for s in names])
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.axvline(0, color="black", lw=LW)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlabel("Marker score")


# Two matched columns keep genes and candidate types together.
for x0, region in ((25, "follicle"), (86, "muscle")):
    g = gt[(gt.region == region) & (gt.kind == "gene")].sort_values("rank")
    t = gt[(gt.region == region) & (gt.kind == "type")].sort_values("rank")
    axg = ax_mm(x0, ROW2_Y + 5, 35, 26)
    gene_bars(axg, g.name.tolist(), g.score.to_numpy(), C_INC)
    axg.set_xlim(0, 9)
    axg.set_xticks([0, 3, 6, 9])
    axt = ax_mm(x0, ROW2_Y + 42, 35, 13)
    type_bars(axt, t.name.tolist(), t.score.to_numpy())
    axt.set_xlim(min(-0.5, t.score.min() - 0.2), 3.6)
    axt.set_xticks([0, 1.5, 3])
    fig.text((x0 + 17.5) / FIG_W_MM, 1 - (ROW2_Y - 1) / H,
             REGION_LABELS[region], fontsize=FS, ha="center", va="top")
    axt.set_title("Atlas candidates", fontsize=FS_SMALL, color="#595959", pad=3)
letter("d", 0, ROW2_Y - 3)

# ---- e: composite reference, follicle residual genes -----------------------
axe = ax_mm(141, ROW2_Y + 5, 35, 26 * 10.2 / 8.2)
gene_bars(axe, comp_fol.gene.tolist(), comp_fol.score.to_numpy(), C_COM)
axe.set_xlim(0, 9)
axe.set_xticks([0, 3, 6, 9])
fig.text(158.5 / FIG_W_MM, 1 - (ROW2_Y - 1) / H, "Follicle, composite reference",
         fontsize=FS, ha="center", va="top")
letter("e", 122, ROW2_Y - 3)

# ---- f: controlled removal AUROC vs distinctness ---------------------------
ROW3_Y, ROW3_H = 128, 28
ax = ax_mm(12, ROW3_Y, 44, ROW3_H)
cls_col = {"rare": "#BDBDBD", "moderate": "#7F7F7F", "abundant": "#252525"}
for cls in ("rare", "moderate", "abundant"):
    d = rem[rem["class"] == cls]
    ax.scatter(d.distinct, d.auroc, s=3, c=cls_col[cls], linewidths=0, alpha=0.85,
               label=f"{cls.capitalize()}", rasterized=True)
qb = np.quantile(rem.distinct, np.linspace(0, 1, 7))
mid = rem.groupby(pd.cut(rem.distinct, qb, include_lowest=True), observed=True).agg(
    x=("distinct", "median"), y=("auroc", "median"))
ax.plot(mid.x, mid.y, "-", color=OI["orange"], lw=1.1, label="Binned median")
ax.axhline(0.5, color="black", lw=0.5, ls=(0, (1, 1.5)))
ax.set_xscale("log")
ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5], ["0.02", "0.05", "0.1", "0.2", "0.5"])
ax.minorticks_off()
ax.set_ylim(0, 1.02)
ax.set_xlabel("Distinctness of removed type")
ax.set_ylabel("AUROC")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.035), ncol=2,
          markerscale=1.6, handlelength=1.0, borderaxespad=0)
letter("f", 0, ROW3_Y - 9)

# ---- g: raw score vs depth, observed vs simulated (CRC P5, complete ref) ---
ax = ax_mm(72, ROW3_Y, 40, ROW3_H)
for kind, c, lab in (("sim_raw", "#9A9A9A", "Simulated from fit"), ("obs_raw", "#252525", "Observed")):
    d = esc[esc.kind == kind].sort_values("median_depth")
    ax.fill_between(d.median_depth, d.q16, d.q84, color=c, alpha=0.18, lw=0)
    ax.plot(d.median_depth, d["median"], "-o", color=c, lw=0.8, ms=1.8, label=lab)
ax.set_xscale("log")
ax.axhline(0, color="black", lw=0.5, ls=(0, (1, 1.5)))
ax.set_xlabel("UMIs per bin (selected genes)")
ax.set_ylabel("Raw pooled score")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.035), borderaxespad=0)
letter("g", 60, ROW3_Y - 9)

# ---- h: complete-reference flag rate: per data set + by depth decile -------
axs = ax_mm(130, ROW3_Y, 10, ROW3_H)
rng = np.random.default_rng(0)
v = 100 * cref.flag_rate.to_numpy()
axs.scatter(rng.uniform(-0.25, 0.25, len(v)), v, s=3, c="#7F7F7F", linewidths=0)
axs.plot([-0.38, 0.38], [np.median(v)] * 2, color="black", lw=1.0)
axs.axhline(5, color="black", lw=0.5, ls=(0, (1, 1.5)))
axs.set_xlim(-0.6, 0.6)
axs.set_ylim(0, 14)
axs.set_xticks([0], ["Spotless\ndata sets"])
axs.tick_params(axis="x", length=0)
axs.set_ylabel("Bins flagged (%)")

ax = ax_mm(145, ROW3_Y, 32, ROW3_H)
for (st, key), c, lab in ((("spotless", "score"), "#7F7F7F", "Spotless"),
                          (("c2_8um", "score_pooled"), C_COM, "Xenium CRC 8 µm")):
    d = fpr[(fpr.set == st) & (fpr.key == key) & (fpr.null == "auto")].sort_values("decile")
    ax.plot(d.decile, d.fpr_pct, "-o", color=c, lw=0.8, ms=1.8, label=lab)
ax.axhline(5, color="black", lw=0.5, ls=(0, (1, 1.5)))
ax.set_ylim(0, 14)
ax.tick_params(axis="y", left=False, labelleft=False)
ax.spines["left"].set_visible(False)
ax.set_xticks([1, 5, 10])
ax.set_xlabel("UMI depth decile")
ax.legend(loc="lower right", bbox_to_anchor=(1, 1.035), borderaxespad=0)
letter("h", 119, ROW3_Y - 9)

# ---- i: lung cancer, Flex vs LUAD reference ------------------------------
ROW4_Y, ROW4_H = 173, 26
ax = ax_mm(12, ROW4_Y, 42, ROW4_H)
secs = [("LUNG_X1", "Section 1"), ("LUNG_X5K", "Section 2")]
for i, (ref, c, lab) in enumerate((("flex", C_INC, "Chromium Flex"), ("luad", C_COM, "LUAD atlas"))):
    vals = [100 * lung[s][ref]["frac_flag_pooled"] for s, _ in secs]
    ax.bar(np.arange(2) + (i - 0.5) * 0.38, vals, 0.38, color=c, lw=0, label=lab)
ax.axhline(5, color="black", lw=0.5, ls=(0, (1, 1.5)))
ax.set_xticks([0, 1], [s[1] for s in secs])
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(0, 25)
ax.set_ylabel("Bins flagged (%)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2, borderaxespad=0)
letter("i", 0, ROW4_Y - 7)

# lung unexplained genes (Flex reference), grouped by lineage of origin
groups = [("Neuroendocrine", ["GRP", "CHGA", "PTPRN"]),
          ("Secretory", ["MUC5B", "MSMB"]),
          ("Ciliated", ["SNTN", "CFAP157", "CDHR3"])]
genes = [g for _, gl in groups for g in gl]
axl = ax_mm(80, ROW4_Y, 78, ROW4_H)
bw = 0.38
sec_col = {"LUNG_X1": "#252525", "LUNG_X5K": "#9A9A9A"}
for i, (s, lab) in enumerate(secs):
    tg = lung_genes[s].set_index("gene").score
    vals = [tg.get(g, np.nan) for g in genes]
    axl.bar(np.arange(len(genes)) + (i - 0.5) * bw, vals, bw, color=sec_col[s], lw=0, label=lab)
axl.set_xticks(range(len(genes)), genes, fontstyle="italic")
axl.tick_params(axis="x", length=0)
axl.set_ylabel("Unexplained score (log$_2$ O/E)")
axl.set_ylim(0, 10)
axl.set_xlim(-0.6, len(genes) - 0.4)
k = 0
for name, gl in groups:
    xa, xb = k - 0.4, k + len(gl) - 0.6
    axl.plot([xa, xb], [10.4, 10.4], color="black", lw=0.5, clip_on=False)
    axl.text((xa + xb) / 2, 10.7, name, ha="center", va="bottom", fontsize=FS_SMALL)
    k += len(gl)
axl.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0)

save(fig, "fig4_refcheck")
