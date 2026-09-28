"""Figure 5: whole-cohort CRC deconvolution resolves neutrophil microdomains.

One matplotlib canvas; reads result files directly.

Data (FlashDeconv 0.2.0 final package run unless stated):
  results/reference_diagnostic_v2/crc_P*_CRC_meta.npz   38-type proportions + coords
      (package run identical to results/rerun_final/crc/P*_CRC/figdata.npz FINAL,
       max |diff| 2.4e-4 = float16 rounding)
  results/reference_diagnostic_v2/pkg_crc_P*_CRC.npz    diagnostic pooled score
  results/reference_diagnostic_v2/summary_crc.csv       flagged fraction in programme regions
  results/b2_crc_refcheck/stage2/program_smoothed.npz   IFN-gamma programme score
  results/rerun_final/crc/P*_CRC/{knn_enrichment,markers,rctd}.csv
  results/rerun_final/crc/figdata_FINAL/neutrophil_microdomains_summary.csv
  results/rerun_final/crc/xenium/virtual_binning/virtual_binning_metrics.csv
  results/rerun_final/crc/xenium/pseudo_vhd/benchmark_per_type.csv
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *  # noqa: F401,F403

apply_style()

R = RESULTS
CRC = R / "rerun_final" / "crc"
RD2 = R / "reference_diagnostic_v2"
SAMPLES = ["P1_CRC", "P2_CRC", "P5_CRC"]
PLAB = {"P1_CRC": "P1", "P2_CRC": "P2", "P5_CRC": "P5"}
UM_PER_PX = 8.0 / 29.21  # measured 8-um bin pitch = 29.21 px
PT_MARK = {"P1_CRC": "o", "P2_CRC": "s", "P5_CRC": "^"}
PT_SHADE = {"P1_CRC": "#4D4D4D", "P2_CRC": "#8C8C8C", "P5_CRC": "#BDBDBD"}

DISPLAY_LINEAGES = ["Epithelial", "Fibroblast", "Pericyte/SMC", "Endothelial",
                    "Macrophage/Mono", "DC", "Neutrophil", "Mast", "CD4 T",
                    "CD8 T", "NK", "B", "Plasma", "Other"]
LIN_LABEL = {"Epithelial": "Epithelial/tumour", "Macrophage/Mono": "Macrophage",
             "Pericyte/SMC": "Pericyte/SMC"}
SR_COL = LINEAGE_COLORS["Fibroblast"]
TP_COL = LINEAGE_COLORS["Epithelial"]


def to_lineage(t):
    lin = CRC_TYPE_TO_LINEAGE.get(t, "Other")
    return "Epithelial" if lin == "Tumor" else lin


def load_section(sid):
    m = np.load(RD2 / f"crc_{sid}_meta.npz")
    types = [str(t) for t in m["types"]]
    P = m["proportions"]
    lin_idx = np.array([DISPLAY_LINEAGES.index(to_lineage(t)) for t in types])
    L = np.zeros((P.shape[0], len(DISPLAY_LINEAGES)), np.float32)
    for j, li in enumerate(lin_idx):
        L[:, li] += P[:, j]
    return dict(x=m["x"], y=m["y"], P=P, types=types, L=L)


def lineage_rgb(L):
    cols = np.array([to_rgb(LINEAGE_COLORS[l]) for l in DISPLAY_LINEAGES])
    dom = L.argmax(1)
    s = np.clip(L.max(1), 0, 1)
    s = 0.35 + 0.65 * s
    return 1 - s[:, None] * (1 - cols[dom])


def orient(ax, x, y, pad=0.01):
    """Set limits with image orientation (y downwards)."""
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.max(), y.min())



def scale_bar_below(ax, length_um, label, y=-0.035):
    """Scale bar just below a spatial map (outside tissue), label to its right."""
    tr = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    x0 = min(ax.get_xlim())
    L = length_um / UM_PER_PX
    ax.plot([x0, x0 + L], [y, y], transform=tr, color="black", lw=1.0,
            solid_capstyle="butt", clip_on=False)
    ax.text(x0 + L * 1.15, y, label, transform=tr, ha="left", va="center",
            fontsize=FS_SMALL, clip_on=False)


# ---------------------------------------------------------------------------
fig = new_figure(height_mm=222)
W, H = FIG_W_MM, 222.0


def rect(l, b, w, h):
    """Axes rect from mm (left, bottom, width, height)."""
    return [l / W, b / H, w / W, h / H]


secs = {sid: load_section(sid) for sid in SAMPLES}

# ---------------------------------------------------------------- a: lineage maps
top = H - 6
map_h = 47
ax_a = []
for i, sid in enumerate(SAMPLES):
    d = secs[sid]
    ax = fig.add_axes(rect(4 + i * 46, top - map_h, 45, map_h))
    rgb = lineage_rgb(d["L"])
    ax.scatter(d["x"], d["y"], c=rgb, s=0.06, marker="s", linewidths=0, rasterized=True)
    clean_spatial(ax)
    orient(ax, d["x"], d["y"])
    ax.set_title(f"{PLAB[sid]} ({len(d['x']):,} bins)", fontsize=FS, pad=1)
    scale_bar_below(ax, 1000, "1 mm")
    ax_a.append(ax)
panel_label(ax_a[0], "a", dx_mm=-3.5, dy_mm=2.5)

# legend for lineages
lax = fig.add_axes(rect(143, top - map_h, 36, map_h))
lax.axis("off")
handles = [Patch(facecolor=LINEAGE_COLORS[l], edgecolor="none", label=LIN_LABEL.get(l, l))
           for l in DISPLAY_LINEAGES]
lax.legend(handles=handles, loc="center left", frameon=False, fontsize=FS_SMALL,
           handlelength=0.9, handleheight=0.9, labelspacing=0.3, title="Dominant lineage",
           title_fontsize=FS_SMALL, alignment="left")

# zoom window (P2 stromal aggregate)
summ = pd.read_csv(CRC / "figdata_FINAL" / "neutrophil_microdomains_summary.csv")
ZOOM_SID = "P2_CRC"
cand = summ[(summ["sample"] == ZOOM_SID) & (summ["location"] == "stromal-resident")]
cand = cand.sort_values("n_hotspots", ascending=False)
zc = cand.iloc[0][["centroid_x", "centroid_y"]].to_numpy(float)
ZHALF = 400 / UM_PER_PX  # 800-um window
ax_zoomsrc = ax_a[SAMPLES.index(ZOOM_SID)]
ax_zoomsrc.add_patch(Rectangle((zc[0] - ZHALF, zc[1] - ZHALF), 2 * ZHALF, 2 * ZHALF,
                               fill=False, lw=0.6, ec="black", zorder=5))

# ---------------------------------------------------------------- b: diagnostic
row2_top = top - map_h - 7
row2_h = 44
sid_b = "P2_CRC"
d = secs[sid_b]
pk = np.load(RD2 / f"pkg_crc_{sid_b}.npz")
flag = pk["auto_score_pooled"] > 1.645
prog = np.load(R / "b2_crc_refcheck" / "stage2" / "program_smoothed.npz")[f"{sid_b}__region_1"].astype(np.float32)

axb1 = fig.add_axes(rect(4, row2_top - row2_h, 36, row2_h))
axb1.scatter(d["x"][~flag], d["y"][~flag], color="#CFCFCF", s=0.07, marker="s",
             linewidths=0, rasterized=True)
axb1.scatter(d["x"][flag], d["y"][flag], color=FLAG_COLOR, s=0.06, marker="s",
             linewidths=0, rasterized=True)
clean_spatial(axb1); orient(axb1, d["x"], d["y"])
axb1.set_title(f"P2 flagged bins ({100 * flag.mean():.1f}%)", fontsize=FS, pad=1)
scale_bar_below(axb1, 1000, "1 mm")

axb2 = fig.add_axes(rect(41, row2_top - row2_h, 36, row2_h))
vmax = np.nanpercentile(prog, 99.5)
cm_ifn = LinearSegmentedColormap.from_list("ifn", ["#EFEFEF", "#6A3D9A"])
o = np.argsort(prog)
im = axb2.scatter(d["x"][o], d["y"][o], c=prog[o], cmap=cm_ifn, vmin=0, vmax=vmax,
                  s=0.05, marker="s", linewidths=0, rasterized=True)
clean_spatial(axb2); orient(axb2, d["x"], d["y"])
axb2.set_title("P2 IFN-γ programme", fontsize=FS, pad=1)
bb = axb2.get_position()
colorbar_small(fig, im, [bb.x0 + bb.width * 0.55, bb.y0 - 0.010, bb.width * 0.4, 0.005],
               ticks=[0, vmax])
cbax = fig.axes[-1]
cbax.set_xticklabels(["0", "high"])
panel_label(axb1, "b", dx_mm=-3.5, dy_mm=2.5)

sc = pd.read_csv(RD2 / "summary_crc.csv")
sc = sc[(sc["null"] == "auto") & (sc["key"] == "score_pooled")].set_index("sample")
axb3 = fig.add_axes(rect(86, row2_top - row2_h + 8, 27, row2_h - 10))
cats = [("All bins", "flag_all", "#8A8A8A"),
        ("IFN-γ high", "hot_region_1_flagged (IFN-gamma (R1))", "#6A3D9A"),
        ("Hypoxia high", "hot_region_2_flagged (Hypoxia (R2))", "#56B4E9")]
bw = 0.26
for k, (lab, col, c) in enumerate(cats):
    vals = [100 * sc.loc[s, col] for s in SAMPLES]
    axb3.bar(np.arange(3) + (k - 1) * bw, vals, width=bw, color=c, label=lab, lw=0)
axb3.axhline(5, ls=":", lw=0.6, color="black")
axb3.set_xticks(range(3)); axb3.set_xticklabels([PLAB[s] for s in SAMPLES])
axb3.set_ylabel("Flagged bins (%)")
axb3.set_ylim(0, 40)
axb3.legend(loc="upper left", bbox_to_anchor=(0.0, 1.14), fontsize=FS_TINY, ncol=1,
            handlelength=0.8)

# ---------------------------------------------------------------- c: hotspot zoom
dz = secs[ZOOM_SID]
zm = (np.abs(dz["x"] - zc[0]) < ZHALF) & (np.abs(dz["y"] - zc[1]) < ZHALF)
neut = dz["P"][:, dz["types"].index("Neutrophil")]
zs = 1.1
axc1 = fig.add_axes(rect(121, row2_top - row2_h, 28.5, row2_h))
axc1.scatter(dz["x"][zm], dz["y"][zm], c=lineage_rgb(dz["L"][zm]), s=zs, marker="s",
             linewidths=0, rasterized=True)
axc2 = fig.add_axes(rect(151, row2_top - row2_h, 28.5, row2_h))
cm_neu = LinearSegmentedColormap.from_list("neu", ["#F2F2F2", LINEAGE_COLORS["Neutrophil"]])
imn = axc2.scatter(dz["x"][zm], dz["y"][zm], c=neut[zm], cmap=cm_neu, vmin=0, vmax=0.3,
                   s=zs, marker="s", linewidths=0, rasterized=True)
hot = zm & (neut >= 0.1)
for ax in (axc1, axc2):
    clean_spatial(ax)
    ax.set_xlim(zc[0] - ZHALF, zc[0] + ZHALF)
    ax.set_ylim(zc[1] + ZHALF, zc[1] - ZHALF)
axc1.set_title("Dominant lineage", fontsize=FS, pad=1)
axc2.set_title("Neutrophil proportion", fontsize=FS, pad=1)
scale_bar_below(axc1, 200, "200 µm", y=-0.05)
bb = axc2.get_position()
colorbar_small(fig, imn, [bb.x0 + bb.width * 0.5, bb.y0 - 0.012, bb.width * 0.45, 0.005],
               ticks=[0, 0.1, 0.3])
panel_label(axc1, "c", dx_mm=-3.5, dy_mm=2.5)

# ---------------------------------------------------------------- row 3: d e f
row3_top = row2_top - row2_h - 13
row3_h = 36
# d: kNN self-enrichment
knn = pd.concat([pd.read_csv(CRC / s / "knn_enrichment.csv") for s in SAMPLES])
knn = knn[(knn.fit == "FINAL") & (knn.focal_type == knn.neighbor_type) & (knn.threshold == 0.1)]
axd = fig.add_axes(rect(14, row3_top - row3_h, 34, row3_h))
rng = np.random.default_rng(1)
for i, s in enumerate(SAMPLES):
    k = knn[knn["sample"] == s]
    other = k[k.focal_type != "Neutrophil"]
    axd.scatter(i + rng.uniform(-0.18, 0.18, len(other)), other.log2_enrichment, s=3,
                color="#BDBDBD", lw=0, zorder=2)
    nv = k[k.focal_type == "Neutrophil"].log2_enrichment.iloc[0]
    axd.scatter(i, nv, s=14, color=LINEAGE_COLORS["Neutrophil"], lw=0, zorder=3)
    rank = int((k.log2_enrichment > nv).sum()) + 1
    axd.text(i + 0.22, nv, f"{2 ** nv:.0f}×", fontsize=FS_TINY, va="center",
             color=LINEAGE_COLORS["Neutrophil"])
axd.set_xticks(range(3)); axd.set_xticklabels([PLAB[s] for s in SAMPLES])
axd.set_xlim(-0.5, 2.7)
axd.set_ylabel("Self-enrichment, log$_2$")
axd.legend(handles=[Line2D([], [], marker="o", ls="", color=LINEAGE_COLORS["Neutrophil"], ms=3,
                           label="Neutrophil"),
                    Line2D([], [], marker="o", ls="", color="#BDBDBD", ms=2,
                           label="Other types (37)")],
           loc="upper left", bbox_to_anchor=(0, 1.15), fontsize=FS_TINY, ncol=2,
           columnspacing=0.6)
panel_label(axd, "d", dx_mm=-11, dy_mm=2.5)

# e: marker enrichment
mk = pd.concat([pd.read_csv(CRC / s / "markers.csv") for s in SAMPLES])
mk = mk[mk.fit == "FINAL"]
genes = ["S100A8", "S100A9", "FCGR3B", "CSF3R", "CXCR1", "CXCR2", "CD3D", "CD79A", "KRT20", "COL1A1"]
axe = fig.add_axes(rect(66, row3_top - row3_h, 44, row3_h))
for gi, g in enumerate(genes):
    sub = mk[mk.gene == g]
    col = LINEAGE_COLORS["Neutrophil"] if sub.marker_class.iloc[0] == "neutrophil" else "#8A8A8A"
    v = np.log2(sub.fold_change.to_numpy())
    axe.plot([gi, gi], [v.min(), v.max()], color=col, lw=0.6, zorder=1)
    for s in SAMPLES:
        vv = np.log2(sub[sub["sample"] == s].fold_change.iloc[0])
        axe.scatter(gi, vv, marker=PT_MARK[s], s=7, facecolor=col, edgecolor="none", zorder=2)
axe.axhline(0, color="black", lw=0.4)
axe.set_xticks(range(len(genes)))
axe.set_xticklabels(genes, rotation=45, ha="right", fontstyle="italic", fontsize=FS_TINY)
axe.set_ylabel("Hotspot / background, log$_2$")
axe.set_xlim(-0.6, len(genes) - 0.4)
axe.legend(handles=[Line2D([], [], marker=PT_MARK[s], ls="", color="#4D4D4D", ms=2.5, label=PLAB[s])
                    for s in SAMPLES], loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3,
           fontsize=FS_TINY, columnspacing=0.6, handletextpad=0.2)
panel_label(axe, "e", dx_mm=-11, dy_mm=2.5)

# f: RCTD classification of hotspot bins
rc = pd.concat([pd.read_csv(CRC / s / "rctd.csv") for s in SAMPLES])
rc = rc[rc.fit == "FINAL"].set_index("sample")
rc.loc["Pooled"] = rc.loc[SAMPLES].sum(numeric_only=True)
classes = [("Neutrophil singlet", lambda r: r.neut_singlet_label2, LINEAGE_COLORS["Neutrophil"]),
           ("Other singlet", lambda r: r.singlet - r.neut_singlet_label2, "#4D4D4D"),
           ("Doublet", lambda r: r.doublet, "#8A8A8A"),
           ("Rejected", lambda r: r.reject, "#BDBDBD"),
           ("Not scored", lambda r: r.NA, "#E3E3E3")]
axf = fig.add_axes(rect(128, row3_top - row3_h + 9, 50, row3_h - 9))
rows = SAMPLES + ["Pooled"]
for i, s in enumerate(rows):
    r = rc.loc[s]
    left = 0
    for lab, fn, c in classes:
        v = 100 * fn(r) / r.n_hotspot
        axf.barh(i, v, left=left, color=c, height=0.7, lw=0, label=lab if i == 0 else None)
        left += v
axf.set_yticks(range(len(rows)))
axf.set_yticklabels([PLAB.get(s, s) + (f" (n={int(rc.loc[s].n_hotspot):,})") for s in rows])
axf.invert_yaxis()
axf.set_xlim(0, 100)
axf.set_xlabel("RCTD class of hotspot bins (%)")
axf.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=3, fontsize=FS_TINY,
           handlelength=0.8, columnspacing=0.6)
panel_label(fig, "f", x=113 / W, y=axd.get_position().y1 + 2.5 / H)

# ---------------------------------------------------------------- row 4: g h i
row4_top = row3_top - row3_h - 17
row4_h = 40
types_g = ["mRegDC", "Macrophage", "CD8 T cell", "Mast", "Endothelial"]
axg = fig.add_axes(rect(14, row4_top - row4_h, 66, row4_h))
pvals = {}
for j, t in enumerate(types_g):
    col = f"niche_log2_{t}"
    sr = summ.loc[summ.location == "stromal-resident", col].dropna().to_numpy()
    tp = summ.loc[summ.location == "tumor-proximal", col].dropna().to_numpy()
    pvals[t] = stats.mannwhitneyu(sr, tp, alternative="greater").pvalue
    for off, v, c in [(-0.18, sr, SR_COL), (0.18, tp, TP_COL)]:
        axg.scatter(j + off + rng.uniform(-0.08, 0.08, len(v)), v, s=3, color=c, lw=0,
                    alpha=0.85, zorder=2)
        axg.plot([j + off - 0.13, j + off + 0.13], [np.median(v)] * 2, color="black",
                 lw=0.9, zorder=3)
    ytop = max(sr.max(), tp.max())
    axg.text(j, 4.3, format_p(pvals[t]), ha="center", va="bottom",
             fontsize=FS_TINY)
axg.axhline(0, color="black", lw=0.4, zorder=1)
axg.set_xticks(range(len(types_g)))
axg.set_xticklabels(["mRegDC", "Macrophage", "CD8 T", "Mast", "Endothelial"])
axg.set_ylabel("Niche enrichment, log$_2$")
axg.set_ylim(-5.5, 5.2)
n_sr = (summ.location == "stromal-resident").sum()
n_tp = (summ.location == "tumor-proximal").sum()
axg.legend(handles=[Line2D([], [], marker="o", ls="", color=SR_COL, ms=2.5,
                           label=f"Stromal-resident (n={n_sr})"),
                    Line2D([], [], marker="o", ls="", color=TP_COL, ms=2.5,
                           label=f"Tumour-proximal (n={n_tp})")],
           loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2, fontsize=FS_TINY)
panel_label(axg, "g", dx_mm=-11, dy_mm=6.5)

# h: Xenium lineage r across bin sizes
vb = pd.read_csv(CRC / "xenium" / "virtual_binning" / "virtual_binning_metrics.csv")
axh = fig.add_axes(rect(96, row4_top - row4_h, 33, row4_h))
for metric, lab, c, ls in [("mean_per_lineage_r", "Lineages (mean)", FD_COLOR, "-"),
                           ("global_pearson_r", "All types (pooled)", "#8A8A8A", "--")]:
    s = vb[vb.metric == metric].sort_values("bin_size_um")
    axh.plot(s.bin_size_um, s.value, ls=ls, marker="o", ms=2.5, color=c, label=lab)
axh.set_xscale("log", base=2)
axh.set_xticks([8, 16, 32, 64, 128]); axh.set_xticklabels(["8", "16", "32", "64", "128"])
axh.minorticks_off()
axh.set_xlabel("Bin size (µm)")
axh.set_ylabel("Pearson r vs Xenium")
axh.set_ylim(0.4, 1.0)
axh.legend(loc="lower right", fontsize=FS_TINY)
panel_label(axh, "h", dx_mm=-10, dy_mm=6.5)

# i: 4 um AUPR
pt = pd.read_csv(CRC / "xenium" / "pseudo_vhd" / "benchmark_per_type.csv")
pt = pt[pt.bin_size_um == 4]
meths = [("FlashDeconv_auto", "FlashDeconv"), ("NNLS", "NNLS"), ("MarkerScoring", "Marker scoring")]
axi = fig.add_axes(rect(143, row4_top - row4_h, 36, row4_h))
bw = 0.26
for k, (mkey, mlab) in enumerate(meths):
    vals = [pt[(pt.method == mkey) & (pt.cell_type == t)].auprc.iloc[0] for t in ["mRegDC", "Neutrophil"]]
    axi.bar(np.arange(2) + (k - 1) * bw, vals, width=bw, color=method_color(mlab), lw=0, label=mlab)
axi.set_xticks([0, 1]); axi.set_xticklabels(["mRegDC", "Neutrophil"])
axi.set_ylabel("AUPR (4-µm bins)")
axi.set_ylim(0, 0.8)
axi.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=FS_TINY,
           columnspacing=0.6)
panel_label(axi, "i", dx_mm=-10, dy_mm=6.5)

save(fig, "fig5_crc")
print({t: f"{p:.2g}" for t, p in pvals.items()})
