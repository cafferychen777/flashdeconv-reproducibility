"""Supplementary figure: independent validation of the CRC neutrophil microdomains.

One matplotlib canvas; reads result files directly. Complements main Fig. 5
(which shows mean lineage r across bin sizes, 4-um AUPR for mRegDC/neutrophil,
kNN self-enrichment, all-bin marker fold changes and RCTD classes).

Data
  results/rerun_final/crc/xenium/virtual_binning/virtual_binning_metrics.csv
  results/rerun_final/crc/xenium/pseudo_vhd/benchmark_per_type.csv
  results/rerun_final/crc/xenium/visium_hd_p1/props_final/global_proportion_comparison.csv
  analysis/cross_cohort_evidence/results_local/marteau_per_patient_results.csv
      (analysis/cross_cohort_evidence/01_marteau_xenium_neutrophil_mregdc.py)
  results/rerun_final/crc/figdata_FINAL/neutrophil_multiresolution_enrichment.csv
  results/rerun_final/crc/P*_CRC/markers.csv (fit == FINAL)
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import *  # noqa: F401,F403

apply_style()

CRC = RESULTS / "rerun_final" / "crc"
XEN = CRC / "xenium"
MARTEAU = PROJ / "analysis" / "cross_cohort_evidence" / "results_local" / "marteau_per_patient_results.csv"
SAMPLES = ["P1_CRC", "P2_CRC", "P5_CRC"]
PLAB = {"P1_CRC": "P1", "P2_CRC": "P2", "P5_CRC": "P5"}
PT_MARK = {"P1_CRC": "o", "P2_CRC": "s", "P5_CRC": "^"}
PT_SHADE = {"P1_CRC": "#4D4D4D", "P2_CRC": "#8C8C8C", "P5_CRC": "#BDBDBD"}
RCTD_COL = METHOD_COLORS["RCTD"]
GREY = "#8A8A8A"

H = 214.0
W = FIG_W_MM
fig = new_figure(height_mm=H)


def rect(l, b, w, h):
    return [l / W, b / H, w / W, h / H]


def log2_axis(ax, which="x", ticks=(0.5, 1, 2, 4)):
    lab = [f"{t:g}" for t in ticks]
    if which == "x":
        ax.set_xscale("log", base=2); ax.set_xticks(ticks); ax.set_xticklabels(lab)
    else:
        ax.set_yscale("log", base=2); ax.set_yticks(ticks); ax.set_yticklabels(lab)
    ax.minorticks_off()


report = {}

# =====================================================================
# a: per-type and per-lineage Pearson r across bin sizes (heatmap)
# =====================================================================
vb = pd.read_csv(XEN / "virtual_binning" / "virtual_binning_metrics.csv")
gp = pd.read_csv(XEN / "visium_hd_p1" / "props_final" / "global_proportion_comparison.csv")
BINS = [8, 16, 32, 64, 128]
type_lin = gp[~gp.cell_type.str.startswith("[")].set_index("cell_type").lineage
LIN_ORDER = ["Tumor", "Epithelial", "Stromal", "Endothelial", "Immune", "Other"]
LIN_NAME = {"Tumor": "Tumour", "Epithelial": "Epithelial", "Stromal": "Stromal",
            "Endothelial": "Endo.", "Immune": "Immune", "Other": "Other"}


def vb_val(metric, b):
    s = vb[(vb.metric == metric) & (vb.bin_size_um == b)]
    return s.value.iloc[0] if len(s) else np.nan


rt = pd.DataFrame({b: {t: vb_val(f"per_type_r__{t}", b) for t in type_lin.index} for b in BINS})
rl = pd.DataFrame({b: {l: vb_val(f"per_lineage_r__{l}", b) for l in LIN_ORDER[:5]} for b in BINS})
mean_lin = [vb_val("mean_per_lineage_r", b) for b in BINS]
report["a mean lineage r 8-128"] = np.round(mean_lin, 3)
report["a per-lineage r min"] = round(float(rl.values.min()), 3)

rows = []  # (label, values, group)
for l in LIN_ORDER:
    ts = [t for t in type_lin.index if type_lin[t] == l]
    ts = sorted(ts, key=lambda t: -rt.loc[t].mean())
    rows += [(t, rt.loc[t].to_numpy(), l) for t in ts]

row_h = 1.72   # mm per row
gap = 1.1      # mm between lineage groups
hm_left, hm_w = 27, 17
top_a = H - 7
cmap = mpl.colormaps["viridis"]
norm = mpl.colors.Normalize(0, 1)

# lineage block (5 rows) on top, then 38 types grouped by lineage
ax_a = fig.add_axes(rect(hm_left, 0, hm_w, 1))  # placeholder, repositioned below
ax_a.remove()
y = top_a
blocks = []
blocks.append(("lin", [(LIN_NAME[l] if l != "Endothelial" else "Endothelial", rl.loc[l].to_numpy()) for l in LIN_ORDER[:5]]))
for l in LIN_ORDER:
    blocks.append((l, [(t, v) for t, v, g in rows if g == l]))

axes_a = []
for bi, (key, items) in enumerate(blocks):
    h = row_h * len(items)
    ax = fig.add_axes(rect(hm_left, y - h, hm_w, h))
    M = np.vstack([v for _, v in items])
    ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, _ in items], fontsize=FS_TINY)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if key != "lin":
        ax.text(1.04, 0.5, LIN_NAME[key], transform=ax.transAxes, rotation=90 if len(items) > 3 else 0,
                ha="left", va="center", fontsize=FS_TINY, color="#4D4D4D")
    else:
        ax.set_title("Lineages", fontsize=FS_TINY, pad=1.5, loc="left", x=-0.0)
    axes_a.append(ax)
    y -= h + (gap * 2.6 if key == "lin" else gap)
    if key == "lin":
        ax_types_title_y = y + gap * 2.6 - 0.2
last = axes_a[-1]
last.set_xticks(range(len(BINS)))
last.set_xticklabels([str(b) for b in BINS], fontsize=FS_TINY)
last.tick_params(axis="x", length=1.5, pad=1)
last.set_xlabel("Bin size (µm)", fontsize=FS_SMALL, labelpad=1)
axes_a[1].set_title("Cell types", fontsize=FS_TINY, pad=1.5, loc="left")
bottom_a = last.get_position().y0 * H
cb = colorbar_small(fig, mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    rect(hm_left, bottom_a - 9.5, hm_w, 1.6), label="Pearson r vs Xenium",
                    ticks=[0, 0.5, 1])
panel_label(fig, "a", x=2 / W, y=(top_a + 3.5) / H)

# =====================================================================
# b: per-type AUPR at 4 um, FlashDeconv vs comparators
# =====================================================================
pt = pd.read_csv(XEN / "pseudo_vhd" / "benchmark_per_type.csv")
w4 = pt[pt.bin_size_um == 4].pivot(index="cell_type", columns="method", values="auprc")
row1_top = H - 9
b_l, b_s = 67, 34
axb = fig.add_axes(rect(b_l, row1_top - b_s, b_s, b_s))
axb.plot([0, 1], [0, 1], color="black", lw=0.4, ls=(0, (2, 2)), zorder=1)
for mk, lab, mrk in [("NNLS", "NNLS", "o"), ("MarkerScoring", "Marker scoring", "^")]:
    axb.scatter(w4.FlashDeconv_auto, w4[mk], s=6, marker=mrk, color=method_color(lab), lw=0,
                alpha=0.85, zorder=2, label=f"{lab} ({int((w4.FlashDeconv_auto > w4[mk]).sum())}/38)")
    for t in ["mRegDC", "Neutrophil"]:
        axb.scatter(w4.loc[t, "FlashDeconv_auto"], w4.loc[t, mk], s=12, marker=mrk,
                    facecolor=method_color(lab), edgecolor="black", lw=0.5, zorder=3)
    report[f"b 4um AUPR FD>{mk}"] = f"{int((w4.FlashDeconv_auto > w4[mk]).sum())}/{len(w4)}"
for t, (tx, ty) in [("mRegDC", (0.05, 0.56)), ("Neutrophil", (0.05, 0.72))]:
    axb.annotate(t, (w4.loc[t, "FlashDeconv_auto"], w4.loc[t, "NNLS"]), xytext=(tx, ty),
                 fontsize=FS_TINY, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.4, color="black", shrinkA=1, shrinkB=2.5))
axb.set_xlim(0, 1); axb.set_ylim(0, 1)
axb.set_xticks([0, 0.5, 1]); axb.set_yticks([0, 0.5, 1])
axb.set_xlabel("FlashDeconv AUPR")
axb.set_ylabel("Comparator AUPR")
axb.set_aspect("equal")
axb.legend(loc="upper left", fontsize=FS_TINY, handletextpad=0.1, borderaxespad=0.1)
axb.set_title("Per type, 4-µm bins", fontsize=FS_SMALL, pad=2)
panel_label(axb, "b", dx_mm=-9, dy_mm=2.5)
report["b 4um AUPR mRegDC FD/NNLS/marker"] = w4.loc["mRegDC", ["FlashDeconv_auto", "NNLS", "MarkerScoring"]].round(3).tolist()
report["b 4um AUPR Neutrophil FD/NNLS/marker"] = w4.loc["Neutrophil", ["FlashDeconv_auto", "NNLS", "MarkerScoring"]].round(3).tolist()

# =====================================================================
# c: AUPR across bin sizes (mean over types, mRegDC, neutrophil)
# =====================================================================
PB = [4, 8, 16, 32]
METHS = [("FlashDeconv_auto", "FlashDeconv", "o"), ("NNLS", "NNLS", "o"),
         ("MarkerScoring", "Marker scoring", "^")]
c_l0, c_w, c_gap = 116, 18.5, 3.5
axc = []
for k, (what, title) in enumerate([("mean", "Mean of 38 types"), ("mRegDC", "mRegDC"),
                                   ("Neutrophil", "Neutrophil")]):
    ax = fig.add_axes(rect(c_l0 + k * (c_w + c_gap), row1_top - b_s, c_w, b_s))
    for mkey, mlab, mrk in METHS:
        vals = []
        for b in PB:
            s = pt[(pt.bin_size_um == b) & (pt.method == mkey)]
            vals.append(s.auprc.mean() if what == "mean" else s.set_index("cell_type").auprc[what])
        ax.plot(PB, vals, marker=mrk, ms=2.3, color=method_color(mlab), lw=0.8, label=mlab)
        report[f"c AUPR {what} {mlab} 4/8/16/32"] = np.round(vals, 3)
    ax.set_xscale("log", base=2); ax.set_xticks(PB); ax.set_xticklabels([str(b) for b in PB])
    ax.minorticks_off()
    ax.set_ylim(0, 0.8); ax.set_yticks([0, 0.4, 0.8])
    ax.set_title(title, fontsize=FS_SMALL, pad=2)
    if k == 0:
        ax.set_ylabel("AUPR")
    else:
        ax.set_yticklabels([])
    if k == 1:
        ax.set_xlabel("Bin size (µm)")
    axc.append(ax)
axc[0].legend(loc="lower left", fontsize=FS_TINY, handlelength=1.0, borderaxespad=0.1)
panel_label(axc[0], "c", dx_mm=-9, dy_mm=2.5)

# =====================================================================
# d, e: whole-section frequencies (P1) vs Xenium
# =====================================================================
gpl = gp[gp.cell_type.str.startswith("[LINEAGE]")].copy()
gpt = gp[~gp.cell_type.str.startswith("[LINEAGE]")].copy()
row2_top = row1_top - b_s - 16
d_s = 34
for key, dd, left, label in [("d", gpl, 67, "Lineages (n = 6)"), ("e", gpt, 128, "Cell types (n = 38)")]:
    ax = fig.add_axes(rect(left, row2_top - d_s, d_s, d_s))
    r_fd = stats.pearsonr(dd.xenium_prop, dd.flashdeconv_prop)[0]
    r_rc = stats.pearsonr(dd.xenium_prop, dd.rctd_prop)[0]
    report[f"{key} r FD / RCTD singlets"] = (round(r_fd, 3), round(r_rc, 3))
    if key == "d":
        lo, hi = 0, 0.45
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.4, ls=(0, (2, 2)), zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xticks([0, 0.2, 0.4]); ax.set_yticks([0, 0.2, 0.4])
    else:
        lo, hi = 5e-6, 0.6
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.4, ls=(0, (2, 2)), zorder=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e-4, hi); ax.set_ylim(lo, 12)
        ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1]); ax.set_yticks([1e-5, 1e-3, 1e-1])
        ax.minorticks_off()
    ax.scatter(dd.xenium_prop, dd.rctd_prop, s=8 if key == "d" else 5, marker="s", color=RCTD_COL,
               lw=0, alpha=0.85, zorder=2, label=f"RCTD singlets (r = {r_rc:.2f})")
    ax.scatter(dd.xenium_prop, dd.flashdeconv_prop, s=9 if key == "d" else 6, marker="o", color=FD_COLOR,
               lw=0, alpha=0.9, zorder=3, label=f"FlashDeconv (r = {r_fd:.2f})")
    ax.set_xlabel("Xenium cell fraction")
    ax.set_ylabel("Estimated fraction")
    if key == "d":
        ax.set_aspect("equal")
    ax.set_title(label, fontsize=FS_SMALL, pad=2)
    h_, l_ = ax.get_legend_handles_labels()
    ax.legend(h_[::-1], l_[::-1], loc="upper left", fontsize=FS_TINY, handletextpad=0.1,
              borderaxespad=0.1)
    panel_label(ax, key, dx_mm=-9, dy_mm=2.5)

# =====================================================================
# f: Marteau Xenium atlas, per-patient enrichment around neutrophils (50 um)
# =====================================================================
mt = pd.read_csv(MARTEAU)
mt = mt[~mt.skipped.astype(bool)]
n_pat = len(mt)
TARGETS = [("mregdc", "LAMP3$^+$ DC", LINEAGE_COLORS["mRegDC"]),
           ("macrophage", "Macrophage", LINEAGE_COLORS["Macrophage"]),
           ("endothelial", "Endothelial", LINEAGE_COLORS["Endothelial"]),
           ("tumor", "Tumour/epithelial", LINEAGE_COLORS["Tumor"]),
           ("mast", "Mast", LINEAGE_COLORS["Mast"])]
row3_top = row2_top - d_s - 19
f_h = 36
axf = fig.add_axes(rect(22, row3_top - f_h, 62, f_h))
rng = np.random.default_rng(3)
for i, (key, lab, col) in enumerate(TARGETS):
    fo = mt[f"perm_r50_{key}_fold_over_expected"].to_numpy()
    pv = mt[f"perm_r50_{key}_p_value"].to_numpy()
    sig = pv < 0.05
    yj = i + rng.uniform(-0.2, 0.2, len(fo))
    axf.scatter(fo[sig], yj[sig], s=9, color=col, lw=0, zorder=3)
    axf.scatter(fo[~sig], yj[~sig], s=9, facecolor="white", edgecolor=col, lw=0.6, zorder=3)
    axf.plot([np.median(fo)] * 2, [i - 0.32, i + 0.32], color="black", lw=0.9, zorder=4)
    axf.text(1.02, i, f"{sig.sum()}/{n_pat}", transform=mpl.transforms.blended_transform_factory(
        axf.transAxes, axf.transData), fontsize=FS_SMALL, va="center", ha="left")
    report[f"f r50 {key}: sig/median/range"] = (int(sig.sum()), round(float(np.median(fo)), 3),
                                               round(float(fo.min()), 2), round(float(fo.max()), 2))
axf.axvline(1, color="black", lw=0.4, zorder=1)
log2_axis(axf, "x", ticks=(0.5, 1, 2, 3))
axf.set_xlim(0.45, 3.2)
axf.set_yticks(range(len(TARGETS)))
axf.set_yticklabels([t[1] for t in TARGETS])
axf.set_ylim(len(TARGETS) - 0.5, -0.5)
axf.set_xlabel("Observed / expected within 50 µm of neutrophils")
axf.text(1.02, -0.85, "P < 0.05", transform=mpl.transforms.blended_transform_factory(
    axf.transAxes, axf.transData), fontsize=FS_TINY, va="center", ha="left")
axf.legend(handles=[Line2D([], [], marker="o", ls="", color="#4D4D4D", ms=2.8, label="P < 0.05"),
                    Line2D([], [], marker="o", ls="", markerfacecolor="white",
                           markeredgecolor="#4D4D4D", markeredgewidth=0.6, ms=2.8, label="n.s."),
                    Line2D([], [], color="black", lw=0.9, label="Median")],
           loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=FS_TINY, columnspacing=0.8)
panel_label(axf, "f", dx_mm=-20, dy_mm=5)

# =====================================================================
# g: radius robustness (50 vs 100 um) for LAMP3+ DC and macrophage
# =====================================================================
g_w = 17
axg = []
for k, (key, lab, col) in enumerate(TARGETS[:2]):
    ax = fig.add_axes(rect(112 + k * (g_w + 6), row3_top - f_h, g_w, f_h))
    f50 = mt[f"perm_r50_{key}_fold_over_expected"].to_numpy()
    f100 = mt[f"perm_r100_{key}_fold_over_expected"].to_numpy()
    p50 = mt[f"perm_r50_{key}_p_value"].to_numpy() < 0.05
    p100 = mt[f"perm_r100_{key}_p_value"].to_numpy() < 0.05
    for a, b in zip(f50, f100):
        ax.plot([0, 1], [a, b], color="#BDBDBD", lw=0.5, zorder=1)
    for xpos, v, s in [(0, f50, p50), (1, f100, p100)]:
        ax.scatter(np.full(s.sum(), xpos), v[s], s=7, color=col, lw=0, zorder=3)
        ax.scatter(np.full((~s).sum(), xpos), v[~s], s=7, facecolor="white", edgecolor=col,
                   lw=0.6, zorder=3)
        ax.text(xpos, 3.3, f"{s.sum()}/{n_pat}", ha="center", va="bottom", fontsize=FS_TINY)
    report[f"g {key} r100 sig/median"] = (int(p100.sum()), round(float(np.median(f100)), 3))
    ax.axhline(1, color="black", lw=0.4, zorder=0)
    log2_axis(ax, "y", ticks=(0.5, 1, 2, 3))
    ax.set_ylim(0.45, 3.2)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["50", "100"])
    ax.set_title(lab, fontsize=FS_SMALL, pad=8)
    if k == 0:
        ax.set_ylabel("Observed / expected")
    else:
        ax.set_yticklabels([])
    axg.append(ax)
fig.text((axg[0].get_position().x0 + axg[1].get_position().x1) / 2,
         axg[0].get_position().y0 - 4.6 / H, "Radius (µm)", ha="center", va="top", fontsize=FS)
panel_label(axg[0], "g", dx_mm=-10, dy_mm=5)

# =====================================================================
# h: Visium HD multi-resolution neighbourhood enrichment (8-64 um)
# =====================================================================
mr = pd.read_csv(CRC / "figdata_FINAL" / "neutrophil_multiresolution_enrichment.csv")
RES = [8, 16, 32, 64]
row4_top = row3_top - f_h - 20
h_h, h_w = 30, 20
HT = [("Neutrophil", "Neutrophil", (1, 2, 4, 8, 16, 32, 64)), ("mRegDC", "mRegDC", (0.5, 1, 2, 4)),
      ("Macrophage", "Macrophage", (0.5, 1, 2, 4))]
axh = []
for k, (t, lab, ticks) in enumerate(HT):
    ax = fig.add_axes(rect(14 + k * (h_w + 8), row4_top - h_h, h_w, h_h))
    for s in SAMPLES:
        d = mr[mr["sample"] == s].sort_values("resolution_um")
        ax.plot(d.resolution_um, d[f"ratio_{t}"], marker=PT_MARK[s], ms=2.4, lw=0.7,
                color=PT_SHADE[s], markerfacecolor=lineage_color(t), markeredgewidth=0,
                label=PLAB[s])
        report[f"h {t} {PLAB[s]} 8/16/32/64"] = np.round(d[f"ratio_{t}"].to_numpy(), 2)
    ax.axhline(1, color="black", lw=0.4, zorder=0)
    ax.set_xscale("log", base=2); ax.set_xticks(RES); ax.set_xticklabels([str(r) for r in RES])
    ax.set_yscale("log", base=2)
    if t == "Neutrophil":
        ax.set_ylim(1, 90); ax.set_yticks([1, 4, 16, 64]); ax.set_yticklabels(["1", "4", "16", "64"])
    else:
        ax.set_ylim(0.35, 5); ax.set_yticks([0.5, 1, 2, 4]); ax.set_yticklabels(["0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_title(lab, fontsize=FS_SMALL, pad=2)
    if k == 0:
        ax.set_ylabel("Neighbourhood / section mean")
    if k == 1:
        ax.set_xlabel("Bin size (µm)")
    axh.append(ax)
axh[0].legend(handles=[Line2D([], [], marker=PT_MARK[s], ls="-", lw=0.7, color=PT_SHADE[s],
                              markerfacecolor="#4D4D4D", markeredgewidth=0, ms=2.4, label=PLAB[s])
                       for s in SAMPLES],
              loc="lower left", bbox_to_anchor=(0, 1.1), ncol=3, fontsize=FS_TINY, columnspacing=0.8)
panel_label(axh[0], "h", dx_mm=-11, dy_mm=6)

# =====================================================================
# i: marker fold change, all hotspot bins vs high-UMI (>=200 UMI) bins
# =====================================================================
mk = pd.concat([pd.read_csv(CRC / s / "markers.csv") for s in SAMPLES])
mk = mk[mk.fit == "FINAL"]
axi = fig.add_axes(rect(112, row4_top - h_h - 1, 32, 32))
lo, hi = 1 / 48, 512
axi.plot([lo, hi], [lo, hi], color="black", lw=0.4, ls=(0, (2, 2)), zorder=1)
axi.axhline(1, color="#BDBDBD", lw=0.4, zorder=0); axi.axvline(1, color="#BDBDBD", lw=0.4, zorder=0)
for s in SAMPLES:
    d = mk[mk["sample"] == s]
    for cls, col in [("neutrophil", LINEAGE_COLORS["Neutrophil"]), ("negative", GREY)]:
        dd = d[d.marker_class == cls]
        axi.scatter(dd.fold_change, dd.fold_change_high_umi, marker=PT_MARK[s], s=8, color=col,
                    lw=0, alpha=0.9, zorder=3)
neu = mk[mk.marker_class == "neutrophil"]
report["i neutrophil markers FC all (min-max)"] = (round(neu.fold_change.min(), 1), round(neu.fold_change.max(), 1))
report["i neutrophil markers FC high-UMI (min-max)"] = (round(neu.fold_change_high_umi.min(), 1),
                                                        round(neu.fold_change_high_umi.max(), 1))
report["i neutrophil markers high>all"] = f"{int((neu.fold_change_high_umi > neu.fold_change).sum())}/{len(neu)}"
report["i n_hot / n_hot_high_umi"] = mk.groupby("sample")[["n_hot", "n_hot_high_umi"]].first().to_dict("index")
neg = mk[mk.marker_class == "negative"]
report["i negative FC all / high (range)"] = ((round(neg.fold_change.min(), 2), round(neg.fold_change.max(), 2)),
                                              (round(neg.fold_change_high_umi.min(), 2), round(neg.fold_change_high_umi.max(), 2)))
axi.set_xscale("log", base=2); axi.set_yscale("log", base=2)
tk = [1 / 16, 1, 16, 256]
axi.set_xticks(tk); axi.set_yticks(tk)
tl = ["1/16", "1", "16", "256"]; axi.set_xticklabels(tl); axi.set_yticklabels(tl)
axi.minorticks_off()
axi.set_xlim(lo, hi); axi.set_ylim(lo, hi)
axi.set_aspect("equal")
axi.set_xlabel("Fold change, all hotspot bins")
axi.set_ylabel("Fold change, ≥200-UMI bins")
axi.legend(handles=[Line2D([], [], marker="o", ls="", color=LINEAGE_COLORS["Neutrophil"], ms=2.6,
                           label="Neutrophil markers (6)"),
                    Line2D([], [], marker="o", ls="", color=GREY, ms=2.6, label="Control markers (4)")]
                   + [Line2D([], [], marker=PT_MARK[s], ls="", color="#4D4D4D", ms=2.6, label=PLAB[s])
                      for s in SAMPLES],
           loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=FS_TINY)
panel_label(axi, "i", dx_mm=-11, dy_mm=3)

save(fig, "supp_crc_validation")
for k, v in report.items():
    print(k, ":", v)
