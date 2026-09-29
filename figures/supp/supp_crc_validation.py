"""Supplementary figure: independent validation of the CRC neutrophil microdomains.

One matplotlib canvas; reads result files directly. Complements main Fig. 5
(which shows mean lineage r across bin sizes, 4-um AUPR for mRegDC/neutrophil,
kNN self-enrichment, all-bin marker fold changes and RCTD classes).

Data
  results/rerun_final/crc/xenium/virtual_binning/virtual_binning_metrics.csv
  results/rerun_final/benchmarks/c2/c2_per_type_ap_final.csv  per-type average precision
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
PT_SHADE = {"P1_CRC": "#4D4D4D", "P2_CRC": "#8C8C8C", "P5_CRC": "#9A9A9A"}
RCTD_COL = METHOD_COLORS["RCTD"]
GREY = "#8A8A8A"

H = 204.0
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
            "Endothelial": "Endothelial", "Immune": "Immune", "Other": "Other"}


# Display names for the reference (Level2) cell types: sentence case, British 'Tumour'.
TYPE_NAME = {"Unknown III (SM)": "Unknown III (SM-like)", "SM Stress Response": "SM stress response",
             "Smooth Muscle": "Smooth muscle", "Proliferating Fibroblast": "Proliferating fibroblast",
             "Vascular Fibroblast": "Vascular fibroblast", "Lymphatic Endothelial": "Lymphatic endothelial",
             "Enteric Glial": "Enteric glia", "Proliferating Macrophages": "Proliferating macrophage",
             "Proliferating Immune II": "Proliferating immune II", "vSM": "Vascular SM"}


def type_name(t):
    return TYPE_NAME.get(t, t.replace("Tumor ", "Tumour "))


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

# Four aligned columns: lineage summary and three cell-type blocks.
row_h = 2.15
cmap = mpl.colormaps["viridis"]
norm = mpl.colors.Normalize(0, 1)
axes_a = []
column_groups = [(20, ["lin"]), (64, ["Tumor", "Epithelial", "Other"]),
                 (111, ["Stromal", "Endothelial"]), (158, ["Immune"])]
for left, groups in column_groups:
    y = H - 11
    for key in groups:
        if key == "lin":
            items = [(LIN_NAME[l], rl.loc[l].to_numpy()) for l in LIN_ORDER[:5]]
            title = "Lineages"
        else:
            items = [(type_name(t), v) for t, v, group in rows if group == key]
            title = LIN_NAME[key] if key != "Endothelial" else "Endothelial"
        height = row_h * len(items)
        ax = fig.add_axes(rect(left, y - height, 20, height))
        ax.imshow(np.vstack([v for _, v in items]), aspect="auto", cmap=cmap,
                  norm=norm, interpolation="nearest")
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([name for name, _ in items], fontsize=FS_TINY)
        ax.tick_params(axis="y", length=0, pad=1)
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=FS_SMALL, loc="left", pad=2)
        axes_a.append(ax)
        y -= height + 5
    ax.set_xticks(range(len(BINS)), [str(v) for v in BINS], fontsize=FS_TINY)
    ax.tick_params(axis="x", length=1.5, pad=1)
    ax.set_xlabel("Bin size (µm)", fontsize=FS_SMALL, labelpad=1)
colorbar_small(fig, mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
               rect(20, H - 38, 20, 1.6), label="Pearson r vs Xenium", ticks=[0, 0.5, 1])
panel_label(fig, "a", x=2 / W, y=(H - 4) / H)

# =====================================================================
# b: per-type AUPR at 4 um, FlashDeconv vs comparators
# =====================================================================
# per-type average precision (same estimator as the standardized C2 metrics / Supp. tables)
pt = pd.read_csv(RESULTS / "rerun_final" / "benchmarks" / "c2" / "c2_per_type_ap_final.csv")
pt["method"] = pt.method.replace({"FlashDeconv": "FlashDeconv_auto"})
w4 = pt[pt.bin_size_um == 4].pivot(index="cell_type", columns="method", values="ap")
row1_top = H - 66
b_l, b_s = 12, 25
axb = fig.add_axes(rect(b_l, row1_top - b_s, b_s, b_s))
axb.plot([0, 1], [0, 1], color="black", lw=0.4, ls=(0, (2, 2)), zorder=1)
for mk, lab, mrk in [("NNLS", "NNLS", "o"), ("MarkerScoring", "Marker scoring", "^")]:
    axb.scatter(w4.FlashDeconv_auto, w4[mk], s=6, marker=mrk, color=method_color(lab), lw=0,
                alpha=0.85, zorder=2, label=f"{lab} ({int((w4.FlashDeconv_auto > w4[mk]).sum())}/38)")
    for t in ["mRegDC", "Neutrophil"]:
        axb.scatter(w4.loc[t, "FlashDeconv_auto"], w4.loc[t, mk], s=12, marker=mrk,
                    facecolor=method_color(lab), edgecolor="black", lw=0.5, zorder=3)
    report[f"b 4um AUPR FD>{mk}"] = f"{int((w4.FlashDeconv_auto > w4[mk]).sum())}/{len(w4)}"
for t, (tx, ty) in [("mRegDC", (0.42, 0.48)), ("Neutrophil", (0.42, 0.65))]:
    axb.annotate(t, (w4.loc[t, "FlashDeconv_auto"], w4.loc[t, "NNLS"]), xytext=(tx, ty),
                 fontsize=FS_TINY, ha="right", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.4, color="black", shrinkA=1, shrinkB=2.5))
axb.set_xlim(0, 1); axb.set_ylim(0, 1)
axb.set_xticks([0, 0.5, 1]); axb.set_yticks([0, 0.5, 1])
axb.set_xlabel("FlashDeconv AUPR")
axb.set_ylabel("Comparator AUPR")
axb.set_aspect("equal")
axb.legend(loc="lower left", bbox_to_anchor=(0, 1.005), fontsize=FS_TINY,
           handletextpad=0.1, borderaxespad=0, labelspacing=0.25)
axb.set_title("Per type, 4-µm bins", fontsize=FS_SMALL, pad=19)
panel_label(fig, "b", x=2 / W, y=(row1_top + 9) / H)
report["b 4um AUPR mRegDC FD/NNLS/marker"] = w4.loc["mRegDC", ["FlashDeconv_auto", "NNLS", "MarkerScoring"]].round(3).tolist()
report["b 4um AUPR Neutrophil FD/NNLS/marker"] = w4.loc["Neutrophil", ["FlashDeconv_auto", "NNLS", "MarkerScoring"]].round(3).tolist()

# =====================================================================
# c: AUPR across bin sizes (mean over types, mRegDC, neutrophil)
# =====================================================================
PB = [4, 8, 16, 32]
METHS = [("FlashDeconv_auto", "FlashDeconv", "o"), ("NNLS", "NNLS", "o"),
         ("MarkerScoring", "Marker scoring", "^")]
c_l0, c_w, c_gap = 47, 16, 3
axc = []
for k, (what, title) in enumerate([("mean", "Mean of 38"), ("mRegDC", "mRegDC"),
                                   ("Neutrophil", "Neutrophil")]):
    ax = fig.add_axes(rect(c_l0 + k * (c_w + c_gap), row1_top - b_s, c_w, b_s))
    for mkey, mlab, mrk in METHS:
        vals = []
        for b in PB:
            s = pt[(pt.bin_size_um == b) & (pt.method == mkey)]
            vals.append(s.ap.mean() if what == "mean" else s.set_index("cell_type").ap[what])
        ax.plot(PB, vals, marker=mrk, ms=2.3, color=method_color(mlab), lw=0.8, label=mlab)
        report[f"c AUPR {what} {mlab} 4/8/16/32"] = np.round(vals, 3)
    ax.set_xscale("log", base=2); ax.set_xticks(PB); ax.set_xticklabels([str(b) for b in PB])
    ax.minorticks_off()
    ax.set_ylim(0, 0.8); ax.set_yticks([0, 0.4, 0.8])
    ax.set_title(title, fontsize=FS_SMALL, pad=19)
    if k == 0:
        ax.set_ylabel("AUPR")
    else:
        ax.set_yticklabels([])
    if k == 1:
        ax.set_xlabel("Bin size (µm)")
    axc.append(ax)
fig.legend(*axc[0].get_legend_handles_labels(), loc="upper center",
           bbox_to_anchor=(74 / W, (row1_top + 4.5) / H), ncol=3,
               fontsize=FS_TINY, handlelength=1.0, columnspacing=0.9, borderaxespad=0)
panel_label(fig, "c", x=40 / W, y=(row1_top + 9) / H)

# =====================================================================
# d, e: whole-section frequencies (P1) vs Xenium
# =====================================================================
gpl = gp[gp.cell_type.str.startswith("[LINEAGE]")].copy()
gpt = gp[~gp.cell_type.str.startswith("[LINEAGE]")].copy()
row2_top = row1_top
d_s = 25
for key, dd, left, label in [("d", gpl, 114, "Lineages (n = 6)"), ("e", gpt, 152, "Cell types (n = 38)")]:
    ax = fig.add_axes(rect(left, row2_top - d_s, d_s, d_s))
    r_fd = stats.pearsonr(dd.xenium_prop, dd.flashdeconv_prop)[0]
    r_rc = stats.pearsonr(dd.xenium_prop, dd.rctd_prop)[0]
    report[f"{key} r FD / RCTD singlets"] = (round(r_fd, 3), round(r_rc, 3))
    if key == "d":
        lo, hi = 0, 0.55
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.4, ls=(0, (2, 2)), zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xticks([0, 0.2, 0.4]); ax.set_yticks([0, 0.2, 0.4])
    else:
        lo, hi = 5e-6, 0.6
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.4, ls=(0, (2, 2)), zorder=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xticks([1e-5, 1e-3, 1e-1]); ax.set_yticks([1e-5, 1e-3, 1e-1])
        ax.minorticks_off()
    # published RCTD calls are doublet mode -> shared RCTD doublet marker
    ax.scatter(dd.xenium_prop, dd.rctd_prop, s=8 if key == "d" else 5, marker=RCTD_MARKERS["doublet"],
               color=RCTD_COL, lw=0, alpha=0.85, zorder=2, label="RCTD singlets")
    ax.scatter(dd.xenium_prop, dd.flashdeconv_prop, s=9 if key == "d" else 6, marker="o", color=FD_COLOR,
               lw=0, alpha=0.9, zorder=3, label="FlashDeconv")
    ax.set_xlabel("Xenium cell fraction")
    ax.set_ylabel("Estimated fraction")
    if key == "d":
        ax.set_aspect("equal")
    ax.set_title(label, fontsize=FS_SMALL, pad=19)
    h_, _ = ax.get_legend_handles_labels()
    ax.legend(h_[::-1], [f"FlashDeconv: $r$ = {r_fd:.2f}",
                        f"RCTD singlets: $r$ = {r_rc:.2f}"],
              loc="lower left", bbox_to_anchor=(0, 1.005),
              fontsize=FS_TINY, handlelength=0.8, handletextpad=0.4,
              labelspacing=0.25, borderaxespad=0)
    panel_label(ax, key, dx_mm=-8, dy_mm=9)

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
row3_top = 92
f_h = 34
axf = fig.add_axes(rect(23, row3_top - f_h, 65, f_h))
rng = np.random.default_rng(3)
for i, (key, lab, col) in enumerate(TARGETS):
    fo = mt[f"perm_r50_{key}_fold_over_expected"].to_numpy()
    pv = mt[f"perm_r50_{key}_p_value"].to_numpy()
    sig = pv < 0.05
    yj = i + rng.uniform(-0.2, 0.2, len(fo))
    axf.scatter(fo[sig], yj[sig], s=9, color=col, lw=0, zorder=3)
    axf.scatter(fo[~sig], yj[~sig], s=9, facecolor="white", edgecolor=col, lw=0.6, zorder=3)
    axf.plot([np.median(fo)] * 2, [i - 0.32, i + 0.32], color="black", lw=0.9, zorder=4)
    axf.text(1.06, i, f"{sig.sum()}/{n_pat}", transform=mpl.transforms.blended_transform_factory(
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
axf.text(1.06, -0.85, "Significant\npatients", transform=mpl.transforms.blended_transform_factory(
    axf.transAxes, axf.transData), fontsize=FS_TINY, va="center", ha="left")
axf.legend(handles=[Line2D([], [], marker="o", ls="", color="#4D4D4D", ms=2.8, label="P < 0.05"),
                    Line2D([], [], marker="o", ls="", markerfacecolor="white",
                           markeredgecolor="#4D4D4D", markeredgewidth=0.6, ms=2.8, label="n.s."),
                    Line2D([], [], color="black", marker="|", ms=6, mew=0.9, ls="", label="Median")],
           loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=FS_TINY, columnspacing=0.8)
panel_label(axf, "f", dx_mm=-20, dy_mm=5)

# =====================================================================
# g: radius robustness (50 vs 100 um) for LAMP3+ DC and macrophage
# =====================================================================
g_w = 25
axg = []
for k, (key, lab, col) in enumerate(TARGETS[:2]):
    ax = fig.add_axes(rect(113 + k * (g_w + 8), row3_top - f_h, g_w, f_h))
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
row4_top = 40
h_h, h_w = 30, 22
HT = [("Neutrophil", "Neutrophil", (1, 2, 4, 8, 16, 32, 64)), ("mRegDC", "mRegDC", (0.5, 1, 2, 4)),
      ("Macrophage", "Macrophage", (0.5, 1, 2, 4))]
axh = []
for k, (t, lab, ticks) in enumerate(HT):
    ax = fig.add_axes(rect(23 + k * (h_w + 4), row4_top - h_h, h_w, h_h))
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
axh[1].legend(handles=[Line2D([], [], marker=PT_MARK[s], ls="-", lw=0.7, color=PT_SHADE[s],
                              markerfacecolor="#4D4D4D", markeredgewidth=0, ms=2.4, label=PLAB[s])
                       for s in SAMPLES],
              loc="lower center", bbox_to_anchor=(0.5, 1.09), ncol=3, fontsize=FS_TINY, columnspacing=0.8)
panel_label(fig, "h", x=3 / W, y=(row4_top + 6) / H)

# =====================================================================
# i: marker fold change, all hotspot bins vs high-UMI (>=200 UMI) bins
# =====================================================================
mk = pd.concat([pd.read_csv(CRC / s / "markers.csv") for s in SAMPLES])
mk = mk[mk.fit == "FINAL"]
axi = fig.add_axes(rect(113, row4_top - h_h, 30, 30))
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
marker_handles = [Line2D([], [], marker="o", ls="", color=col, ms=2.6, label=lab)
                  for col, lab in [(LINEAGE_COLORS["Neutrophil"], "Neutrophil (6)"),
                                   (GREY, "Control (4)")]]
class_legend = axi.legend(handles=marker_handles, title="Markers", title_fontsize=FS_TINY,
                         loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=FS_TINY)
axi.add_artist(class_legend)
axi.legend(handles=[Line2D([], [], marker=PT_MARK[s], ls="", color="#4D4D4D", ms=2.6,
                           label=PLAB[s]) for s in SAMPLES], title="Patient", title_fontsize=FS_TINY,
           loc="upper left", bbox_to_anchor=(1.02, 0.58), fontsize=FS_TINY)
panel_label(fig, "i", x=103 / W, y=(row4_top + 6) / H)

save(fig, "supp_crc_validation")
for k, v in report.items():
    print(k, ":", v)
