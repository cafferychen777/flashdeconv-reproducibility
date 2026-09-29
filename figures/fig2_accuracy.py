"""Figure 2: FlashDeconv accuracy (leverage weighting, Spotless, Li et al., Xenium CRC).

Reads result tables directly and draws the whole figure on one canvas.
Output: paper/figures/fig2_accuracy.{pdf,png}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (apply_style, new_figure, panel_label, save, RESULTS, MM,  # noqa: E402
                   FD_COLOR, METHOD_COLORS, WEIGHT_COLORS, OTHER_METHOD_COLOR,
                   FS, FS_SMALL, FS_TINY, LW, mean_ci, RCTD_MARKERS, RCTD_LS)

RF = RESULTS / "rerun_final"
W = RF / "weighting"
SP = RF / "benchmarks" / "spotless"
LI = RF / "benchmarks" / "li2023" / "expected"
# Spotless silver standards: lambda=0 (pseudo-spots have no spatial layout)
ER = RESULTS / "editor_revision"
SPOT_CFG = "final_default_lam0"
C2 = RF / "benchmarks" / "c2" / "c2_standard_metrics_final.csv"

rng = np.random.default_rng(0)
DEPTHS = [1.0, 0.25, 0.10, 0.05]
DEPTH_LAB = ["100", "25", "10", "5"]
VARIANTS = {"EXP_LEV": "leverage", "UNIFORM": "equal", "VAR_REF": "variance"}
VLAB = {"EXP_LEV": "Expected leverage", "UNIFORM": "Equal", "VAR_REF": "Between-type variance"}

METHOD_NAMES = {
    "FlashDeconv": "FlashDeconv", "rctd": "RCTD", "cell2location": "Cell2location",
    "spatialdwls": "SpatialDWLS", "stereoscope": "Stereoscope", "music": "MuSiC",
    "nnls": "NNLS", "seurat": "Seurat", "destvi": "DestVI", "spotlight": "SPOTlight",
    "stride": "STRIDE", "tangram": "Tangram", "dstg": "DSTG",
}
TISSUE_LAB = {"brain_cortex": "Brain cortex", "cerebellum_cell": "Cerebellum (cell)",
              "cerebellum_nucleus": "Cerebellum (nucleus)", "hippocampus": "Hippocampus",
              "kidney": "Kidney", "scc_p5": "SCC"}
TISSUE_COL = dict(zip(TISSUE_LAB, ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#555555"]))


def jitter(n, w=0.28):
    return rng.uniform(-w, w, n)


def wilcox_p(x, y):
    return stats.wilcoxon(x, y).pvalue


def p_text(p):
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        return f"$P$ = {p / 10**e:.0f}×10$^{{{e}}}$"
    return f"$P$ = {p:.2g}" if p >= 0.01 else f"$P$ = {p:.3f}"


# ---------------------------------------------------------------------------
apply_style()
fig = new_figure(height_mm=178)
H_MM, W_MM = 178.0, 180.0


def ax_mm(x, y, w, h):
    """Axes from mm offsets (x from left, y from top)."""
    return fig.add_axes([x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM])


# ===========================================================================
# a  Spotless depth series: mean Pearson and JSD for three weightings
# ===========================================================================
acc = pd.read_csv(ER / "weighting_lam0" / "spotless_acc_lam0.csv")
ax_a1 = ax_mm(12, 14, 24, 35)
ax_a2 = ax_mm(46, 14, 24, 35)
xs = np.arange(len(DEPTHS))
ci_rng = {ax_a1: [np.inf, -np.inf], ax_a2: [np.inf, -np.inf]}
for v in ["UNIFORM", "EXP_LEV"]:
    c = WEIGHT_COLORS[VARIANTS[v]]
    for ax, met in [(ax_a1, "pearson"), (ax_a2, "jsd")]:
        m, h = [], []
        for d in DEPTHS:
            vals = acc.loc[(acc.variant == v) & np.isclose(acc.frac, d), met].to_numpy()
            mm_, hh = mean_ci(vals)
            m.append(mm_); h.append(hh)
        m, h = np.array(m), np.array(h)
        ci_rng[ax] = [min(ci_rng[ax][0], (m - h).min()), max(ci_rng[ax][1], (m + h).max())]
        ax.fill_between(xs, m - h, m + h, color=c, alpha=0.11, lw=0)
        ax.plot(xs, m, "-o", color=c, ms=2.2, lw=0.9, zorder=3 if v == "EXP_LEV" else 2)
for ax, lab in [(ax_a1, "Pearson r"), (ax_a2, "JSD")]:
    ax.set_xticks(xs, DEPTH_LAB)
    ax.set_xlabel("Depth (%)")
    ax.set_ylabel(lab)
    ax.set_xlim(-0.3, 3.3)
for ax in (ax_a1, ax_a2):
    lo_, hi_ = ci_rng[ax]
    pad = 0.03 * (hi_ - lo_)
    ax.set_ylim(lo_ - pad, hi_ + pad)
handles = [Line2D([], [], color=WEIGHT_COLORS[VARIANTS[v]], marker="o", ms=2.2, lw=0.9, label=VLAB[v])
           for v in ["EXP_LEV", "UNIFORM"]]
fig.legend(handles=handles, loc="upper left",
           bbox_to_anchor=(12 / W_MM, 1 - 5 / H_MM), ncol=2,
           fontsize=FS_SMALL, handlelength=1.5, columnspacing=1.1, borderaxespad=0)
panel_label(fig, "a", x=2 / W_MM, y=1 - 7 / H_MM)

# ===========================================================================
# b  Paired per-data-set difference, leverage minus equal (AUPR, rare-type AUPR)
# ===========================================================================
ax_b = ax_mm(82, 14, 34, 35)
piv = acc.pivot_table(index=["dataset", "frac"], columns="variant",
                      values=["aupr", "rare_auprc"]).reset_index()
groups = [("aupr", "All types", "#555555"), ("rare_auprc", "Rare types", FD_COLOR)]
for gi, (met, glab, col) in enumerate(groups):
    for di, d in enumerate(DEPTHS):
        sub = piv[np.isclose(piv.frac, d)]
        diff = (sub[(met, "EXP_LEV")] - sub[(met, "UNIFORM")]).dropna().to_numpy()
        x0 = di + (gi - 0.5) * 0.36
        ax_b.scatter(x0 + jitter(len(diff), 0.1), diff, s=1.6, color=col, alpha=0.45,
                     lw=0, rasterized=True)
        m, h = mean_ci(diff)
        ax_b.errorbar(x0, m, yerr=h, fmt="o", color="black", mfc=col, mec="black",
                      mew=0.4, ms=2.6, elinewidth=0.7, zorder=5)
ax_b.axhline(0, color="black", lw=0.4, ls=(0, (2, 2)))
ax_b.set_xticks(xs, DEPTH_LAB)
ax_b.set_xlabel("Depth (%)")
ax_b.set_ylabel("ΔAUPR (leverage − equal)")
ax_b.set_ylim(-0.06, 0.2)
ax_b.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=2.6, label=l) for _, l, c in groups],
            loc="upper left", fontsize=FS_TINY)


# ===========================================================================
# c  Xenium CRC per-type RMSE change at 2/4/8 um
# ===========================================================================
ax_c = ax_mm(132, 14, 44, 35)
xp = pd.read_csv(W / "xenium_pertype.csv")
sx = pd.read_csv(W / "stats_xenium.csv")
for i, r in enumerate([2, 4, 8]):
    a = xp[(xp.res_um == r) & (xp.variant == "EXP_LEV")].set_index("cell_type")
    b = xp[(xp.res_um == r) & (xp.variant == "UNIFORM")].set_index("cell_type")
    d = (b.rmse - a.rmse).loc[a.index].to_numpy() * 1e3  # positive = leverage better
    cols = np.where(d > 0, FD_COLOR, "#8A8A8A")
    ax_c.scatter(i + jitter(len(d), 0.22), d, s=2.2, c=cols, lw=0, alpha=0.8, rasterized=True)
    ax_c.plot([i - 0.28, i + 0.28], [np.median(d)] * 2, color="black", lw=0.9)
    st = sx[(sx.res_um == r) & (sx.unit == "cell type (all types)") &
            (sx.comparison == "EXP_LEV vs UNIFORM") & (sx.metric == "type_rmse")].iloc[0]
    nwin = int(round(st.win_frac * st.n))
    ax_c.text(i, 1.02, f"{nwin}/{int(st.n)}\n{p_text(st.p_value)}", transform=ax_c.get_xaxis_transform(),
              ha="center", va="bottom", fontsize=FS_TINY, linespacing=1.25)
ax_c.axhline(0, color="black", lw=0.4, ls=(0, (2, 2)))
ax_c.set_xticks([0, 1, 2], ["2", "4", "8"])
ax_c.set_xlabel("Xenium CRC bin size (µm)")
ax_c.set_ylabel("RMSE reduction (×10$^{-3}$)")
ax_c.set_xlim(-0.6, 2.6)
lim = np.nanpercentile(np.abs(ax_c.get_ylim()), 100)
panel_label(fig, "b", x=120 / W_MM, y=1 - 7 / H_MM)

# ===========================================================================
# d  Spotless 54 silver standards: 13 methods, Pearson
# ===========================================================================
sp = pd.read_csv(ER / "silver_per_dataset_lam0_vs_competitors.csv")
methods = [c for c in sp.columns if c not in ("tissue", "pattern", "metric")]
corr = sp[sp.metric == "corr"]
order = corr[methods].mean().sort_values(ascending=True).index.tolist()
ax_d = ax_mm(19, 65, 39, 44)
for i, m in enumerate(order):
    v = corr[m].to_numpy()
    name = METHOD_NAMES[m]
    col = METHOD_COLORS.get(name, "#7A7A7A") if name in ("FlashDeconv", "RCTD", "Cell2location",
                                                           "NNLS", "DestVI") else "#7A7A7A"
    ax_d.scatter(v, i + jitter(len(v), 0.22), s=1.4, color=col, alpha=0.27, lw=0, rasterized=True)
    mu, h = mean_ci(v)
    ax_d.errorbar(mu, i, xerr=h, fmt="o", color="black", mfc=col, mec="black", mew=0.4,
                  ms=3.3 if m == "FlashDeconv" else 2.6,
                  elinewidth=1.0 if m == "FlashDeconv" else 0.65, zorder=5)
ax_d.set_yticks(range(len(order)), [METHOD_NAMES[m] for m in order])
for t in ax_d.get_yticklabels():
    if t.get_text() == "FlashDeconv":
        t.set_fontweight("bold")
ax_d.set_xlabel("Pearson r (54 data sets)")
ax_d.set_xlim(0.2, 1.0)
ax_d.set_ylim(-0.7, len(order) - 0.3)
ax_d.tick_params(axis="y", length=0)
panel_label(ax_d, "c", dx_mm=-17)

# ===========================================================================
# e  Paired per-data-set comparisons: FlashDeconv vs RCTD (Pearson), vs Cell2location (AUPR)
# ===========================================================================
wil = pd.read_csv(SP / "silver_paired_wilcoxon_final.csv")
wil = wil[wil.config == SPOT_CFG]
ax_e1 = ax_mm(72, 65, 27, 27)
ax_e2 = ax_mm(107, 65, 27, 27)
for ax, comp, met, lab, lo in [(ax_e1, "rctd", "corr", "Pearson r", 0.65),
                               (ax_e2, "cell2location", "aupr", "AUPR", 0.6)]:
    sub = sp[sp.metric == met]
    for t, g in sub.groupby("tissue"):
        ax.scatter(g[comp], g["FlashDeconv"], s=3, color=TISSUE_COL[t], lw=0, alpha=0.85,
                   rasterized=True, label=TISSUE_LAB[t])
    ax.plot([lo, 1], [lo, 1], color="black", lw=0.4, ls=(0, (2, 2)))
    ax.set_xlim(lo, 1.005); ax.set_ylim(lo, 1.005)
    tk = np.arange(np.ceil(lo * 10) / 10, 1.001, 0.1)
    ax.set_xticks(tk, [f"{t:.1f}" for t in tk]); ax.set_yticks(tk, [f"{t:.1f}" for t in tk])
    ax.set_aspect("equal")
    ax.set_xlabel(f"{METHOD_NAMES[comp]} {lab}")
    ax.set_ylabel(f"FlashDeconv {lab}")
    p = wil[(wil.metric == met) & (wil.comparator == comp)].p_value.iloc[0]
    nb = int(wil[(wil.metric == met) & (wil.comparator == comp)].flash_better.iloc[0])
    ax.text(0.5, 1.04, f"{nb}/54\n{p_text(p)}", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=FS_TINY, linespacing=1.25)
tissue_handles, tissue_labels = ax_e1.get_legend_handles_labels()
legend_order = [0, 3, 1, 4, 2, 5]  # Matplotlib fills legend columns first.
ax_e1.legend([tissue_handles[i] for i in legend_order],
             [tissue_labels[i] for i in legend_order], loc="upper left", bbox_to_anchor=(-0.04, -0.31), ncol=3, fontsize=FS_TINY,
             handletextpad=0.1, columnspacing=0.6, markerscale=1.6)
panel_label(ax_e1, "d", dx_mm=-11)

# ===========================================================================
# f  Rare cell types: per-type accuracy by abundance class (Spotless)
# ===========================================================================
pc = pd.read_csv(SP / "fd_per_celltype_silver.csv.gz")
pc = pc[pc.config == SPOT_CFG]
ax_f = ax_mm(145, 122, 32, 46)
cats = [("abundant", "Abund."), ("moderate", "Mod."), ("rare", "Rare")]
for i, (cat, lab) in enumerate(cats):
    for j, (met, col) in enumerate([("pearson", "#555555"), ("auprc", FD_COLOR)]):
        v = pc.loc[pc.category == cat, met].dropna().to_numpy()
        x0 = i + (j - 0.5) * 0.38
        vp = ax_f.violinplot(v, positions=[x0], widths=0.34, showextrema=False)
        for b in vp["bodies"]:
            b.set_alpha(None)
            b.set_facecolor(to_rgba(col, 0.27))
            b.set_edgecolor(to_rgba(col, 0.45))
            b.set_linewidth(0.3)
        ax_f.plot(x0, np.mean(v), "o", ms=2.9, mfc=col, mec="black", mew=0.4, zorder=5)
ax_f.set_xticks(range(3), [l for _, l in cats])
ax_f.set_ylabel("Per-type accuracy")
ax_f.set_ylim(0, 1.02)
ax_f.legend(handles=[Line2D([], [], marker="o", ls="", mfc=c, mec="black", mew=0.4, ms=2.4, label=l)
                     for l, c in [("Pearson r", "#555555"), ("AUPR", FD_COLOR)]],
            loc="lower left", fontsize=FS_TINY)
ax_f.tick_params(axis="x", labelsize=FS_SMALL)
panel_label(ax_f, "h", dx_mm=-10)

# ===========================================================================
# g  Li et al. MERFISH benchmark: RMSE rank vs bin size
# ===========================================================================
ax_g = ax_mm(143, 65, 21, 44)
ranks = {}
for r in [100, 50, 20]:
    t = pd.read_csv(LI / f"merfish_{r}_suppdata1_vs_published.csv").set_index("method")
    ranks[r] = t.rank_RMSE
rk = pd.DataFrame(ranks)
hl = ["FlashDeconv", "DestVI", "Cell2location", "RCTD", "CARD"]
for m in rk.index:
    if m in hl:
        continue
    ax_g.plot([0, 1, 2], rk.loc[m], color="#C8C8C8", lw=0.6, zorder=1)
for m in hl[::-1]:
    c = METHOD_COLORS[m]
    ax_g.plot([0, 1, 2], rk.loc[m], "-o", color=c, ms=2.4, lw=1.3 if m == "FlashDeconv" else 0.9, zorder=3)
# right-side labels with collision avoidance
end = rk.loc[hl, 20].sort_values()
ypos, last = {}, -10
for m, y in end.items():
    y2 = max(y, last + 1.25)
    ypos[m] = y2; last = y2
for m in hl:
    ax_g.annotate(m, xy=(2, rk.loc[m, 20]), xytext=(2.16, ypos[m]),
                  color=METHOD_COLORS[m], va="center", fontsize=FS_TINY,
                  fontweight="bold" if m == "FlashDeconv" else "normal",
                  annotation_clip=False,
                  arrowprops=dict(arrowstyle="-", color=METHOD_COLORS[m],
                                  lw=0.4, shrinkA=1, shrinkB=2))
ax_g.set_ylim(19.6, 0.4)
ax_g.set_yticks([1, 5, 10, 15, 19])
ax_g.set_xticks([0, 1, 2], ["100", "50", "20"])
ax_g.set_xlim(-0.2, 2.1)
ax_g.set_xlabel("MERFISH bin (µm)")
ax_g.set_ylabel("RMSE rank (of 19)")
ax_g.tick_params(axis="x", labelsize=FS_TINY)
panel_label(ax_g, "e", dx_mm=-9)

# ===========================================================================
# h  Xenium pseudo-Visium HD: all bins, FlashDeconv vs NNLS vs marker scoring
# ===========================================================================
c2 = pd.read_csv(C2)
sizes = [2, 4, 8, 16, 32]
sel = {
    "FlashDeconv": (c2.method == "FlashDeconv") & (c2["mode"] == "final_default_auto"),
    "NNLS": (c2.method == "NNLS"),
    "Marker scoring": (c2.method == "MarkerScoring"),
}
ax_h1 = ax_mm(12, 122, 28, 46)
ax_h2 = ax_mm(50, 122, 28, 46)
xs5 = np.arange(len(sizes))
for name, m in sel.items():
    t = c2[m & (c2.eval_set == "all")].set_index("resolution_um").loc[sizes]
    for ax, met in [(ax_h1, "pearson_flat"), (ax_h2, "jsd")]:
        ax.plot(xs5, t[met], "-o", color=METHOD_COLORS[name], ms=2.4, lw=1.1 if name == "FlashDeconv" else 0.9,
                label=name, zorder=3 if name == "FlashDeconv" else 2)
for ax, lab in [(ax_h1, "Pearson r (all bins)"), (ax_h2, "JSD (all bins)")]:
    ax.set_xticks(xs5, [str(s) for s in sizes])
    ax.set_xlabel("Bin size (µm)")
    ax.set_ylabel(lab)
ax_h1.set_ylim(0.4, 0.92)
ax_h2.set_ylim(0.0, 0.55)
ax_h1.legend(loc="lower right", fontsize=FS_TINY)
panel_label(ax_h1, "f", dx_mm=-10)

# ===========================================================================
# i  FlashDeconv vs RCTD: coverage and accuracy on RCTD-scored bins
# ===========================================================================
ax_i1 = ax_mm(98, 122, 32, 18)
ax_i2 = ax_mm(98, 148, 32, 20)
fdm = (c2.method == "FlashDeconv") & (c2["mode"] == "final_default_auto")
rc = c2[(c2.method == "RCTD") & (c2.eval_set == "all")]
RC_COL = METHOD_COLORS["RCTD"]
# (label, mode, umi_min, FlashDeconv eval set on the same bins, marker, line style, x offset)
RSETS = [("RCTD doublet", "doublet", 100, "common_doublet_umi100", RCTD_MARKERS["doublet"], RCTD_LS["doublet"], -0.22),
         ("RCTD full", "full", 100, "common_full_umi100", RCTD_MARKERS["full"], RCTD_LS["full"], 0.0),
         ("RCTD full, UMI ≥ 20", "full", 20, "common_full_umi20", RCTD_MARKERS["full20"], RCTD_LS["full20"], 0.22)]
ax_i1.plot(xs5, [1.0] * 5, "-o", color=FD_COLOR, ms=2.2, lw=1.1)
for lab, mode, umi, es, mk, ls_, dx in RSETS:
    t = rc[(rc["mode"] == mode) & (rc.umi_min == umi)].set_index("resolution_um")
    have = [s_ for s_ in sizes if s_ in t.index]
    xi = np.array([sizes.index(s_) for s_ in have])
    t = t.loc[have]
    ax_i1.plot(xi, t.coverage, ls=ls_, color=RC_COL, lw=0.9, marker=mk, ms=2.2)
    f = c2[fdm & (c2.eval_set == es)].set_index("resolution_um").loc[have]
    for k, x in enumerate(xi):
        ax_i2.plot([x + dx] * 2, [t.pearson_flat.iloc[k], f.pearson_flat.iloc[k]],
                   color="#AAAAAA", lw=0.5, zorder=1)
    ax_i2.scatter(xi + dx, t.pearson_flat, s=5, color=RC_COL, lw=0, zorder=3, marker=mk)
    ax_i2.scatter(xi + dx, f.pearson_flat, s=5, color=FD_COLOR, lw=0, zorder=4, marker=mk)
ax_i1.set_ylim(0, 1.08)
ax_i1.set_yticks([0, 0.5, 1], ["0", "50", "100"])
ax_i1.set_ylabel("Bins scored (%)")
ax_i1.set_xticks(xs5, [])
ax_i2.set_xticks(xs5, [str(s) for s in sizes])
ax_i2.set_xlabel("Bin size (µm)")
ax_i2.set_ylabel("Pearson r\n(RCTD-scored bins)")
ax_i2.set_ylim(0.84, 1.0)
for ax in (ax_i1, ax_i2):
    ax.set_xlim(-0.4, 4.4)
# Each FlashDeconv marker corresponds to the matching RCTD evaluation set.
fd_handles = tuple(Line2D([], [], color=FD_COLOR, marker=m_, ls="", ms=2.7) for m_ in ("o", "s", "^"))
fig.legend(handles=[fd_handles] + [Line2D([], [], color=RC_COL, marker=mk, ls=ls_, lw=0.8, ms=2.7)
                                   for _, _, _, _, mk, ls_, _ in RSETS],
           labels=["FlashDeconv"] + [r[0] for r in RSETS],
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
           loc="upper left", bbox_to_anchor=(98 / W_MM, 1 - 141.3 / H_MM), ncol=2,
           fontsize=FS_TINY, handlelength=2.0, labelspacing=0.2, columnspacing=0.8, borderaxespad=0)
panel_label(ax_i1, "g", dx_mm=-11)

save(fig, "fig2_accuracy")

# Console check of numbers quoted in the text
print("Spotless FD mean corr", corr["FlashDeconv"].mean().round(3), "RCTD", corr["rctd"].mean().round(3))
for name, m in sel.items():
    t = c2[m & (c2.eval_set == "all") & (c2.resolution_um == 8)]
    print(name, "8um r", t.pearson_flat.round(3).tolist(), "jsd", t.jsd.round(3).tolist())
print(rc[["resolution_um", "mode", "umi_min", "coverage", "pearson_flat", "jsd"]].round(3).to_string())
print(rk.loc[hl])
