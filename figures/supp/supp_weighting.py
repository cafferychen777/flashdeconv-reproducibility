"""Supplementary figure: expected-leverage gene weighting across sequencing depth.

Full detail behind Fig. 2a,b (Spotless depth series; Xenium CRC pseudo-Visium HD).
Reads results/rerun_final/weighting/*.csv and draws one canvas.
Output: paper/figures/supp_weighting.{pdf,png}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import (apply_style, new_figure, panel_label, save, RESULTS,  # noqa: E402
                   WEIGHT_COLORS, WEIGHT_LABELS, OI, FS_SMALL, FS_TINY, LW,
                   mean_ci, colorbar_small)

W = RESULTS / "rerun_final" / "weighting"
DEPTHS = [1.0, 0.25, 0.10, 0.05]
DEPTH_LAB = ["100", "25", "10", "5"]
VKEY = {"EXP_LEV": "leverage", "UNIFORM": "equal", "VAR_REF": "variance"}
ORDER = ["UNIFORM", "VAR_REF", "EXP_LEV"]  # draw leverage on top
TISSUE_LAB = {"brain_cortex": "Brain cortex", "cerebellum_cell": "Cerebellum (cell)",
              "cerebellum_nucleus": "Cerebellum (nucleus)", "hippocampus": "Hippocampus",
              "kidney": "Kidney", "scc_p5": "SCC"}
STRATA = [("rare", "Rare (<5%)", OI["purple"]),
          ("moderate", "Moderate (5–15%)", OI["green"]),
          ("abundant", "Abundant (≥15%)", OI["orange"])]
RES = [2, 4, 8]

rng = np.random.default_rng(0)
apply_style()
H_MM, W_MM = 188.0, 180.0
fig = new_figure(height_mm=H_MM)


def ax_mm(x, y, w, h):
    return fig.add_axes([x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM])


def p_text(p):
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        return f"$P$ = {p / 10**e:.0f}×10$^{{{e}}}$"
    return f"$P$ = {p:.2f}" if p >= 0.01 else f"$P$ = {p:.3f}"


acc = pd.read_csv(W / "spotless_acc.csv")
pt = pd.read_csv(W / "spotless_pertype.csv")
xa = pd.read_csv(W / "xenium_acc.csv")
xp = pd.read_csv(W / "xenium_pertype.csv")
sx = pd.read_csv(W / "stats_xenium.csv")

# ===========================================================================
# a  Spotless: four metrics vs depth, three weightings (mean +/- 95% CI, n = 54)
# ===========================================================================
xs = np.arange(len(DEPTHS))
mets_a = [("pearson", "Pearson r"), ("rmse", "RMSE"), ("jsd", "JSD"), ("aupr", "AUPR")]
axes_a = [ax_mm(12 + i * 43, 10, 31, 30) for i in range(4)]
for ax, (met, lab) in zip(axes_a, mets_a):
    for v in ORDER:
        c = WEIGHT_COLORS[VKEY[v]]
        m, h = zip(*[mean_ci(acc.loc[(acc.variant == v) & np.isclose(acc.frac, d), met])
                     for d in DEPTHS])
        m, h = np.array(m), np.array(h)
        ax.fill_between(xs, m - h, m + h, color=c, alpha=0.18, lw=0)
        ax.plot(xs, m, "-o", color=c, ms=2.2, lw=0.9, zorder=3 if v == "EXP_LEV" else 2)
    ax.set_xticks(xs, DEPTH_LAB)
    ax.set_xlim(-0.3, 3.3)
    ax.set_xlabel("Depth (% of UMIs)")
    ax.set_ylabel(lab)
panel_label(axes_a[0], "a", dx_mm=-10, dy_mm=3.5)
fig.legend(handles=[Line2D([], [], color=WEIGHT_COLORS[VKEY[v]], marker="o", ms=2.2, lw=0.9,
                           label=WEIGHT_LABELS[VKEY[v]]) for v in ["EXP_LEV", "VAR_REF", "UNIFORM"]],
           loc="lower center", bbox_to_anchor=(0.5, 1 - 7.0 / H_MM), ncol=3, fontsize=FS_SMALL)

# ===========================================================================
# b  Per-tissue mean paired difference (leverage - equal), n = 9 per tissue
# ===========================================================================
piv = acc.pivot_table(index=["tissue", "dataset", "frac"], columns="variant",
                      values=["pearson", "jsd", "aupr"]).reset_index()
tissues = list(TISSUE_LAB)
mets_b = [("pearson", "ΔPearson r", 1), ("aupr", "ΔAUPR", 1), ("jsd", "−ΔJSD", -1)]
axes_b = [ax_mm(33 + i * 50, 56, 30, 30) for i in range(3)]
for ax, (met, lab, sgn) in zip(axes_b, mets_b):
    M = np.zeros((len(tissues), len(DEPTHS)))
    K = np.zeros_like(M, dtype=int)
    for i, t in enumerate(tissues):
        for j, d in enumerate(DEPTHS):
            s = piv[(piv.tissue == t) & np.isclose(piv.frac, d)]
            diff = sgn * (s[(met, "EXP_LEV")] - s[(met, "UNIFORM")]).to_numpy()
            M[i, j] = diff.mean()
            K[i, j] = int((diff > 0).sum())
    vmax = np.ceil(np.abs(M).max() * 100) / 100
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    for i in range(len(tissues)):
        for j in range(len(DEPTHS)):
            ax.text(j, i, f"{K[i, j]}", ha="center", va="center", fontsize=FS_TINY,
                    color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
    ax.set_xticks(range(len(DEPTHS)), DEPTH_LAB)
    ax.set_yticks(range(len(tissues)), [TISSUE_LAB[t] for t in tissues] if ax is axes_b[0] else [])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Depth (% of UMIs)")
    ax.set_title(lab + " (leverage − equal)", fontsize=FS_SMALL)
    bb = ax.get_position()
    colorbar_small(fig, im, [bb.x1 + 1.5 / W_MM, bb.y0, 1.5 / W_MM, bb.height],
                   orientation="vertical", ticks=[-vmax, 0, vmax])
panel_label(axes_b[0], "b", dx_mm=-31, dy_mm=3.5)

# ===========================================================================
# c  Per cell type x data set differences by abundance stratum
# ===========================================================================
ppiv = pt.pivot_table(index=["dataset", "frac", "cell_type", "category"], columns="variant",
                      values=["pearson", "auprc", "rmse"]).reset_index()
mets_c = [("pearson", "ΔPearson r (per type)", 1), ("auprc", "ΔAUPR (per type)", 1),
          ("rmse", "−ΔRMSE (per type)", -1)]
axes_c = [ax_mm(12 + i * 58, 101, 44, 30) for i in range(3)]
STRAT_STATS = {}
for ax, (met, lab, sgn) in zip(axes_c, mets_c):
    for si, (cat, clab, col) in enumerate(STRATA):
        for di, d in enumerate(DEPTHS):
            s = ppiv[(ppiv.category == cat) & np.isclose(ppiv.frac, d)]
            diff = (sgn * (s[(met, "EXP_LEV")] - s[(met, "UNIFORM")])).dropna().to_numpy()
            x0 = di + (si - 1) * 0.26
            m, h = mean_ci(diff)
            ax.errorbar(x0, m, yerr=h, fmt="o", color=col, mfc=col, mec="none",
                        ms=2.4, elinewidth=0.8, zorder=5)
            STRAT_STATS[(met, cat, d)] = (m, h, len(diff), int((diff > 0).sum()),
                                          stats.wilcoxon(diff).pvalue)
    ax.axhline(0, color="black", lw=0.4, ls=(0, (2, 2)))
    ax.set_xticks(xs, DEPTH_LAB)
    ax.set_xlim(-0.5, 3.5)
    ax.set_xlabel("Depth (% of UMIs)")
    ax.set_ylabel(lab)
axes_c[0].legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=2.4, label=l)
                          for _, l, c in STRATA], loc="upper left", fontsize=FS_TINY)
panel_label(axes_c[0], "c", dx_mm=-10, dy_mm=3.5)

# ===========================================================================
# d  Xenium CRC pseudo-Visium HD: flattened metrics per bin size (single fit)
# ===========================================================================
mets_d = [("pearson", "ΔPearson r"), ("jsd", "ΔJSD"), ("rmse", "ΔRMSE"), ("ap", "ΔAP")]
axes_d = [ax_mm(12 + i * 28, 150, 17, 30) for i in range(4)]
xr = np.arange(len(RES))
for ax, (met, lab) in zip(axes_d, mets_d):
    base = np.array([xa.loc[(xa.res_um == r) & (xa.variant == "UNIFORM"), met].iloc[0] for r in RES])
    for v in ["VAR_REF", "EXP_LEV"]:
        y = np.array([xa.loc[(xa.res_um == r) & (xa.variant == v), met].iloc[0] for r in RES])
        ax.plot(xr, (y - base) * 1e3, "-o", color=WEIGHT_COLORS[VKEY[v]], ms=2.4, lw=0.8,
                mec="none", zorder=3 if v == "EXP_LEV" else 2)
    ax.axhline(0, color=WEIGHT_COLORS["equal"], lw=0.8, zorder=1)
    ax.set_xticks(xr, [str(r) for r in RES])
    ax.set_xlim(-0.4, 2.4)
    ax.set_xlabel("Bin size (µm)")
    ax.set_ylabel(lab + " (×10$^{-3}$)")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
panel_label(axes_d[0], "d", dx_mm=-10, dy_mm=5.5)

# ===========================================================================
# e  Xenium per-type RMSE reduction: leverage vs equal / vs variance (n = 38 types)
# ===========================================================================
ax_e = ax_mm(128, 150, 50, 30)
cmp_ = [("UNIFORM", "EXP_LEV vs UNIFORM"), ("VAR_REF", "EXP_LEV vs VAR_REF")]
for ri, r in enumerate(RES):
    a = xp[(xp.res_um == r) & (xp.variant == "EXP_LEV")].set_index("cell_type")
    for ci, (other, cname) in enumerate(cmp_):
        b = xp[(xp.res_um == r) & (xp.variant == other)].set_index("cell_type")
        d = (b.rmse - a.rmse).loc[a.index].to_numpy() * 1e3  # positive = leverage better
        x0 = ri + (ci - 0.5) * 0.4
        col = WEIGHT_COLORS[VKEY[other]]
        ax_e.scatter(x0 + rng.uniform(-0.1, 0.1, len(d)), d, s=1.8, color=col, alpha=0.55,
                     lw=0, rasterized=True)
        ax_e.plot([x0 - 0.14, x0 + 0.14], [np.median(d)] * 2, color="black", lw=0.9, zorder=5)
        st = sx[(sx.res_um == r) & (sx.unit == "cell type (all types)") &
                (sx.comparison == cname) & (sx.metric == "type_rmse")].iloc[0]
        nwin = int(round(st.win_frac * st.n))
        ptxt = f"{st.p_value:.3f}" if st.p_value < 0.1 else f"{st.p_value:.2f}"
        for yy, t in [(1.13, f"{nwin}/{int(st.n)}"), (1.02, ptxt)]:
            ax_e.text(x0, yy, t, transform=ax_e.get_xaxis_transform(),
                      ha="center", va="bottom", fontsize=FS_TINY)
for yy, t in [(1.13, "Improved"), (1.02, "$P$")]:
    ax_e.text(-0.62, yy, t, transform=ax_e.get_xaxis_transform(), ha="right", va="bottom",
              fontsize=FS_TINY)
ax_e.axhline(0, color="black", lw=0.4, ls=(0, (2, 2)))
ax_e.set_xticks(xr, [str(r) for r in RES])
ax_e.set_xlim(-0.55, 2.55)
ax_e.set_ylim(-14, 11)
ax_e.set_xlabel("Bin size (µm)")
ax_e.set_ylabel("RMSE reduction (×10$^{-3}$)")
ax_e.legend(handles=[Line2D([], [], marker="o", ls="", color=WEIGHT_COLORS[VKEY[o]], ms=2.4,
                            label="vs " + WEIGHT_LABELS[VKEY[o]].lower()) for o, _ in cmp_],
            loc="lower left", fontsize=FS_TINY, handletextpad=0.1)
panel_label(ax_e, "e", dx_mm=-10, dy_mm=5.5)

save(fig, "supp_weighting")

# ---------------------------------------------------------------------------
# Console summary for the legend / text
# ---------------------------------------------------------------------------
print("\nStratum (leverage - equal; sign so + = leverage better): mean, CI, n, wins, P")
for k, (m, h, n, w, p) in STRAT_STATS.items():
    print(k, f"{m:+.4f} ±{h:.4f} n={n} win={w} P={p:.2g}")
print("\nPer-tissue mean diff (leverage - equal) at 5% depth")
s5 = piv[np.isclose(piv.frac, 0.05)]
for t in tissues:
    s = s5[s5.tissue == t]
    print(t, {m: round(float((s[(m, 'EXP_LEV')] - s[(m, 'UNIFORM')]).mean()), 4)
              for m in ["pearson", "aupr", "jsd"]})
