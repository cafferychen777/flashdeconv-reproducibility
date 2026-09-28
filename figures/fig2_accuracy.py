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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (apply_style, new_figure, panel_label, save, RESULTS, MM,  # noqa: E402
                   FD_COLOR, METHOD_COLORS, WEIGHT_COLORS, OTHER_METHOD_COLOR,
                   FS, FS_SMALL, FS_TINY, LW, mean_ci)

RF = RESULTS / "rerun_final"
W = RF / "weighting"
SP = RF / "benchmarks" / "spotless"
LI = RF / "benchmarks" / "li2023" / "expected"
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
acc = pd.read_csv(W / "spotless_acc.csv")
ax_a1 = ax_mm(12, 7, 24, 40)
ax_a2 = ax_mm(46, 7, 24, 40)
xs = np.arange(len(DEPTHS))
for v in ["UNIFORM", "VAR_REF", "EXP_LEV"]:
    c = WEIGHT_COLORS[VARIANTS[v]]
    for ax, met in [(ax_a1, "pearson"), (ax_a2, "jsd")]:
        m, h = [], []
        for d in DEPTHS:
            vals = acc.loc[(acc.variant == v) & np.isclose(acc.frac, d), met].to_numpy()
            mm_, hh = mean_ci(vals)
            m.append(mm_); h.append(hh)
        m, h = np.array(m), np.array(h)
        ax.fill_between(xs, m - h, m + h, color=c, alpha=0.18, lw=0)
        ax.plot(xs, m, "-o", color=c, ms=2.2, lw=0.9, zorder=3 if v == "EXP_LEV" else 2)
for ax, lab in [(ax_a1, "Pearson r"), (ax_a2, "JSD")]:
    ax.set_xticks(xs, DEPTH_LAB)
    ax.set_xlabel("Depth (%)")
    ax.set_ylabel(lab)
    ax.set_xlim(-0.3, 3.3)
ax_a1.set_ylim(0.83, 0.96)
ax_a2.set_ylim(0.03, 0.125)
handles = [Line2D([], [], color=WEIGHT_COLORS[VARIANTS[v]], marker="o", ms=2.2, lw=0.9, label=VLAB[v])
           for v in ["EXP_LEV", "VAR_REF", "UNIFORM"]]
ax_a1.legend(handles=handles, loc="lower left", fontsize=FS_TINY, handlelength=1.4)
panel_label(ax_a1, "a", dx_mm=-10)

# ===========================================================================
# b  Paired per-data-set difference, leverage minus equal (AUPR, rare-type AUPR)
# ===========================================================================
ax_b = ax_mm(82, 7, 34, 40)
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
ax_c = ax_mm(132, 7, 44, 40)
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
              ha="center", va="bottom", fontsize=FS_TINY, linespacing=1.1)
ax_c.axhline(0, color="black", lw=0.4, ls=(0, (2, 2)))
ax_c.set_xticks([0, 1, 2], ["2", "4", "8"])
ax_c.set_xlabel("Xenium CRC bin size (µm)")
ax_c.set_ylabel("RMSE reduction (×10$^{-3}$)")
ax_c.set_xlim(-0.6, 2.6)
lim = np.nanpercentile(np.abs(ax_c.get_ylim()), 100)
panel_label(ax_c, "b", dx_mm=-10)

# ===========================================================================
# d  Spotless 54 silver standards: 13 methods, Pearson
# ===========================================================================
sp = pd.read_csv(SP / "silver_per_dataset_final_vs_competitors.csv")
methods = [c for c in sp.columns if c not in ("tissue", "pattern", "metric")]
corr = sp[sp.metric == "corr"]
order = corr[methods].mean().sort_values(ascending=True).index.tolist()
ax_d = ax_mm(19, 62, 45, 50)
for i, m in enumerate(order):
    v = corr[m].to_numpy()
    name = METHOD_NAMES[m]
    col = METHOD_COLORS.get(name, "#7A7A7A") if name in ("FlashDeconv", "RCTD", "Cell2location",
                                                           "NNLS", "DestVI") else "#7A7A7A"
    ax_d.scatter(v, i + jitter(len(v), 0.22), s=1.4, color=col, alpha=0.35, lw=0, rasterized=True)
    mu, h = mean_ci(v)
    ax_d.errorbar(mu, i, xerr=h, fmt="o", color="black", mfc=col, mec="black", mew=0.4,
                  ms=2.8, elinewidth=0.8, zorder=5)
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
wil = wil[wil.config == "final_default"]
ax_e1 = ax_mm(80, 62, 25, 25)
ax_e2 = ax_mm(113, 62, 25, 25)
for ax, comp, met, lab, lo in [(ax_e1, "rctd", "corr", "Pearson r", 0.84),
                               (ax_e2, "cell2location", "aupr", "AUPR", 0.8)]:
    sub = sp[sp.metric == met]
    for t, g in sub.groupby("tissue"):
        ax.scatter(g[comp], g["FlashDeconv"], s=3, color=TISSUE_COL[t], lw=0, alpha=0.85,
                   rasterized=True, label=TISSUE_LAB[t])
    ax.plot([lo, 1], [lo, 1], color="black", lw=0.4, ls=(0, (2, 2)))
    ax.set_xlim(lo, 1.0); ax.set_ylim(lo, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{METHOD_NAMES[comp]} {lab}")
    ax.set_ylabel(f"FlashDeconv {lab}")
    p = wil[(wil.metric == met) & (wil.comparator == comp)].p_value.iloc[0]
    nb = int(wil[(wil.metric == met) & (wil.comparator == comp)].flash_better.iloc[0])
    ax.text(0.03, 0.97, f"{nb}/54\n{p_text(p)}", transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_TINY, linespacing=1.1)
ax_e1.legend(loc="upper left", bbox_to_anchor=(0.0, -0.33), ncol=3, fontsize=FS_TINY,
             handletextpad=0.1, columnspacing=0.6, markerscale=1.6)
panel_label(ax_e1, "d", dx_mm=-11)

# ===========================================================================
# f  Rare cell types: per-type accuracy by abundance class (Spotless)
# ===========================================================================
pc = pd.read_csv(SP / "fd_per_celltype_silver.csv.gz")
pc = pc[pc.config == "final_default"]
ax_f = ax_mm(145, 128, 32, 42)
cats = [("abundant", "Abund."), ("moderate", "Mod."), ("rare", "Rare")]
for i, (cat, lab) in enumerate(cats):
    for j, (met, col) in enumerate([("pearson", "#555555"), ("auprc", FD_COLOR)]):
        v = pc.loc[pc.category == cat, met].dropna().to_numpy()
        x0 = i + (j - 0.5) * 0.38
        vp = ax_f.violinplot(v, positions=[x0], widths=0.34, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(col); b.set_alpha(0.35); b.set_edgecolor("none")
        ax_f.plot(x0, np.mean(v), "o", ms=2.4, mfc=col, mec="black", mew=0.4, zorder=5)
ax_f.set_xticks(range(3), [l for _, l in cats])
ax_f.set_ylabel("Per-type accuracy")
ax_f.set_ylim(0, 1.02)
ax_f.legend(handles=[Line2D([], [], marker="o", ls="", mfc=c, mec="black", mew=0.4, ms=2.4, label=l)
                     for l, c in [("Pearson r", "#555555"), ("AUPR", FD_COLOR)]],
            loc="lower left", fontsize=FS_TINY)
ax_f.tick_params(axis="x", labelsize=FS_TINY)
panel_label(ax_f, "h", dx_mm=-10)

# ===========================================================================
# g  Li et al. MERFISH benchmark: RMSE rank vs bin size
# ===========================================================================
ax_g = ax_mm(148, 62, 18, 50)
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
    ax_g.text(2.12, ypos[m], m, color=METHOD_COLORS[m], va="center", fontsize=FS_TINY,
              fontweight="bold" if m == "FlashDeconv" else "normal")
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
ax_h1 = ax_mm(12, 128, 28, 42)
ax_h2 = ax_mm(50, 128, 28, 42)
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
ax_i1 = ax_mm(98, 128, 30, 16)
ax_i2 = ax_mm(98, 150, 30, 20)
fdm = (c2.method == "FlashDeconv") & (c2["mode"] == "final_default_auto")
rc = c2[(c2.method == "RCTD") & (c2.umi_min == 100) & (c2.eval_set == "all")]
off = {"doublet": -0.12, "full": 0.12}
ax_i1.plot(xs5, [1.0] * 5, "-o", color=FD_COLOR, ms=2.2, lw=1.1)
for mode, cname, es in [("doublet", "RCTD (doublet)", "common_doublet_umi100"),
                        ("full", "RCTD (full)", "common_full_umi100")]:
    t = rc[rc["mode"] == mode].set_index("resolution_um").loc[sizes]
    ax_i1.plot(xs5, t.coverage, "-", color=METHOD_COLORS[cname], lw=0.9,
               marker="o" if mode == "doublet" else "s", ms=2.2)
    f = c2[fdm & (c2.eval_set == es)].set_index("resolution_um").loc[sizes]
    for k, x in enumerate(xs5):
        ax_i2.plot([x + off[mode]] * 2, [t.pearson_flat.iloc[k], f.pearson_flat.iloc[k]],
                   color="#AAAAAA", lw=0.5, zorder=1)
    ax_i2.scatter(xs5 + off[mode], t.pearson_flat, s=6, color=METHOD_COLORS[cname], lw=0, zorder=3,
                  marker="o" if mode == "doublet" else "s")
    ax_i2.scatter(xs5 + off[mode], f.pearson_flat, s=6, color=FD_COLOR, lw=0, zorder=4,
                  marker="o" if mode == "doublet" else "s")
ax_i1.set_ylim(0, 1.08)
ax_i1.set_yticks([0, 0.5, 1], ["0", "50", "100"])
ax_i1.set_ylabel("Bins scored (%)")
ax_i1.set_xticks(xs5, [])
ax_i2.set_xticks(xs5, [str(s) for s in sizes])
ax_i2.set_xlabel("Bin size (µm)")
ax_i2.set_ylabel("Pearson r\n(RCTD-scored bins)")
ax_i2.set_ylim(0.86, 1.0)
for ax in (ax_i1, ax_i2):
    ax.set_xlim(-0.4, 4.4)
ax_i2.legend(handles=[Line2D([], [], color=FD_COLOR, marker="o", ms=2.2, label="FlashDeconv"),
                      Line2D([], [], color=METHOD_COLORS["RCTD (doublet)"], marker="o", ms=2.2, label="RCTD doublet"),
                      Line2D([], [], color=METHOD_COLORS["RCTD (full)"], marker="s", ms=2.2, label="RCTD full")],
             loc="lower left", fontsize=FS_TINY)
panel_label(ax_i1, "g", dx_mm=-11)

save(fig, "fig2_accuracy")

# Console check of numbers quoted in the text
print("Spotless FD mean corr", corr["FlashDeconv"].mean().round(3), "RCTD", corr["rctd"].mean().round(3))
for name, m in sel.items():
    t = c2[m & (c2.eval_set == "all") & (c2.resolution_um == 8)]
    print(name, "8um r", t.pearson_flat.round(3).tolist(), "jsd", t.jsd.round(3).tolist())
print(rc[["resolution_um", "mode", "coverage", "pearson_flat"]].round(3).to_string())
print(rk.loc[hl])
