"""Supplementary figure: the Spotless benchmark in full.

Silver standards (a-c): FlashDeconv with lambda=0 (config final_default_lam0; the pseudo-spots
have no spatial layout). Gold standards and case studies (d-f): package defaults (config
final_default; real coordinates).

Panels
  a  Silver standard: 13 methods x 4 metrics, colour = rank of the mean, text = mean
  b  Silver standard per tissue: mean Pearson per method, colour = rank within tissue
  c  Silver standard: paired per-dataset Pearson difference FlashDeconv - comparator
  d  Gold standards (seqFISH+ cortex/SVZ, seqFISH+ OB, STARmap): 4 metrics per method
  e  Liver case study: JSD, AUPR and reference stability per method
  f  Melanoma case study: JSD per method (FlashDeconv default and Pearson residuals)

Liver and melanoma competitor values are read from the Spotless result objects
(validation/spotless/raw_results/metrics/*.rds) through Rscript.

Output: paper/figures/supp_spotless.{pdf,png}
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import (apply_style, new_figure, panel_label, save, RESULTS, PROJ,  # noqa: E402
                   FD_COLOR, METHOD_COLORS, OTHER_METHOD_COLOR, OI,
                   FS, FS_SMALL, FS_TINY, LW, colorbar_small)

SP = RESULTS / "rerun_final" / "benchmarks" / "spotless"
RDS = PROJ / "validation" / "spotless" / "raw_results" / "metrics"
CFG = "final_default"
SILVER_CFG = "final_default_lam0"

METHOD_NAMES = {
    "FlashDeconv": "FlashDeconv", "rctd": "RCTD", "cell2location": "Cell2location",
    "spatialdwls": "SpatialDWLS", "stereoscope": "Stereoscope", "music": "MuSiC",
    "nnls": "NNLS", "seurat": "Seurat", "destvi": "DestVI", "spotlight": "SPOTlight",
    "stride": "STRIDE", "tangram": "Tangram", "dstg": "DSTG",
}
COMPETITORS = [k for k in METHOD_NAMES if k != "FlashDeconv"]
TISSUE_LAB = {"brain_cortex": "Brain cortex", "cerebellum_cell": "Cerebellum (cell)",
              "cerebellum_nucleus": "Cerebellum (nucleus)", "hippocampus": "Hippocampus",
              "kidney": "Kidney", "scc_p5": "SCC"}
METRICS = ["corr", "rmse", "jsd", "aupr"]
METRIC_LAB = {"corr": "Pearson r", "rmse": "RMSE", "jsd": "JSD", "aupr": "AUPR"}
LOWER_BETTER = {"rmse", "jsd"}
GOLD = {"seqfish_cortex_svz": "seqFISH+ cortex/SVZ", "seqfish_ob": "seqFISH+ OB",
        "starmap": "STARmap"}
GOLD_MARK = {"seqfish_cortex_svz": "o", "seqfish_ob": "s", "starmap": "^"}


def mcolor(key):
    return METHOD_COLORS.get(METHOD_NAMES.get(key, key), OTHER_METHOD_COLOR)


def rank_of(values: pd.Series, lower_better: bool) -> pd.Series:
    return values.rank(ascending=lower_better, method="min")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
silver = pd.read_csv(RESULTS / "editor_revision" / "silver_per_dataset_lam0_vs_competitors.csv")
methods_all = COMPETITORS + ["FlashDeconv"]

sil_mean = silver.groupby("metric")[methods_all].mean().T  # method x metric (NaN skipped)
sil_rank = pd.DataFrame({m: rank_of(sil_mean[m], m in LOWER_BETTER) for m in METRICS})
ORDER = sil_mean["corr"].sort_values(ascending=False, kind="stable").index.tolist()  # by mean Pearson, best first

corr = silver[silver.metric == "corr"]
tis_mean = corr.groupby("tissue")[methods_all].mean().T  # method x tissue
tis_rank = tis_mean.rank(ascending=False, method="min")

wil = pd.read_csv(SP / "silver_paired_wilcoxon_final.csv")
wil = wil[(wil.config == SILVER_CFG) & (wil.metric == "corr")].set_index("comparator")

# gold standards: competitors rescored with identical metric code
gcomp = pd.read_csv(SP / "gold_competitors_recomputed.csv")
gcomp = gcomp.groupby(["benchmark", "method"])[METRICS].mean().reset_index()
# FlashDeconv on the measured spot coordinates (validation/controls_editor/gold_realxy.py)
gfd = pd.read_csv(RESULTS / "editor_revision" / "gold_realxy" / "fd_aggregate_gold.csv")
gfd = gfd[gfd.config == "final_default_realxy"].groupby("benchmark")[METRICS].mean().reset_index()
gfd["method"] = "FlashDeconv"
gold = pd.concat([gcomp, gfd], ignore_index=True)

# liver / melanoma competitor values from the Spotless result objects
R_CODE = r"""
a <- readRDS(file.path(d, "liver_all_metrics.rds")); a <- a[a$digest == "all", c("metric","method","value")]
s <- readRDS(file.path(d, "liver_metrics_ref_sensitivity.rds")); s <- aggregate(jsd ~ method, s, mean)
s <- data.frame(metric = "stability", method = s$method, value = s$jsd)
m <- readRDS(file.path(d, "melanoma_metrics.rds"))$jsd
m <- data.frame(metric = "melanoma_jsd", method = as.character(m$method), value = m$jsd)
write.csv(rbind(a, s, m), out, row.names = FALSE)
"""
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "cases.csv"
    subprocess.run(["Rscript", "-e", f'd <- "{RDS}"; out <- "{out}";' + R_CODE], check=True)
    cases = pd.read_csv(out)

liver_fd = pd.read_csv(SP / "liver_case_study.csv")
liver_fd = liver_fd[liver_fd.config == CFG]
stab_fd = pd.read_csv(SP / "liver_stability.csv")
stab_fd = stab_fd[stab_fd.config == CFG]
mel_fd = pd.read_csv(SP / "melanoma_fixed.csv")

liver = {
    "jsd": dict(cases[cases.metric == "jsd"].set_index("method").value, FlashDeconv=liver_fd.jsd.mean()),
    "aupr": dict(cases[cases.metric == "aupr"].set_index("method").value, FlashDeconv=liver_fd.aupr_mean.mean()),
    "stability": dict(cases[cases.metric == "stability"].set_index("method").value, FlashDeconv=stab_fd.jsd.mean()),
}
mel = dict(cases[cases.metric == "melanoma_jsd"].set_index("method").value)
mel["FlashDeconv"] = mel_fd[mel_fd.config == CFG].jsd.mean()
mel["FlashDeconv_pearson"] = mel_fd[mel_fd.config == "final_default_pearson"].jsd.mean()

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
apply_style()
H_MM, W_MM = 196.0, 180.0
fig = new_figure(height_mm=H_MM)


def ax_mm(x, y, w, h):
    """Axes from mm offsets (x from left, y from top)."""
    return fig.add_axes([x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM])


NM = len(ORDER)
RANK_CMAP = ListedColormap(plt.get_cmap("cividis_r")(np.linspace(0.0, 0.92, NM)))
RANK_NORM = BoundaryNorm(np.arange(0.5, NM + 1.5), NM)


def rank_heatmap(ax, ranks, values, fmt, ylabels=True):
    r = ranks.loc[ORDER].values
    ax.imshow(r, cmap=RANK_CMAP, norm=RANK_NORM, aspect="auto")
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            v = values.loc[ORDER].values[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=FS_TINY,
                        color="white" if r[i, j] >= 7 else "black",
                        fontweight="bold" if ORDER[i] == "FlashDeconv" else "normal")
    ax.set_yticks(range(NM))
    if ylabels:
        ax.set_yticklabels([METHOD_NAMES[m] for m in ORDER])
        for t, m in zip(ax.get_yticklabels(), ORDER):
            if m == "FlashDeconv":
                t.set_color(FD_COLOR)
                t.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)


# ---- a: silver metrics -----------------------------------------------------
ax_a = ax_mm(19, 12, 30, 58)
rank_heatmap(ax_a, sil_rank[METRICS], sil_mean[METRICS], lambda v: f"{v:.3f}")
ax_a.set_xticks(range(4))
ax_a.set_xticklabels([METRIC_LAB[m].replace(" r", "") for m in METRICS])
panel_label(ax_a, "a", dx_mm=-18, dy_mm=6)

# ---- b: silver per tissue ---------------------------------------------------
tissues = list(TISSUE_LAB)
ax_b = ax_mm(53, 12, 45, 58)
rank_heatmap(ax_b, tis_rank[tissues], tis_mean[tissues], lambda v: f"{v:.2f}", ylabels=False)
ax_b.set_xticks(range(len(tissues)))
ax_b.set_xticklabels([TISSUE_LAB[t].replace(" (", "\n(") for t in tissues], rotation=40, ha="left",
                     rotation_mode="anchor", linespacing=1.0)
panel_label(ax_b, "b", dx_mm=-2.5, dy_mm=6)

cb = colorbar_small(fig, plt.cm.ScalarMappable(norm=RANK_NORM, cmap=RANK_CMAP),
                    [19 / W_MM, 1 - 76.5 / H_MM, 30 / W_MM, 1.6 / H_MM],
                    label="Rank among 13 methods", ticks=[1, 4, 7, 10, 13])

# ---- c: paired Pearson differences -----------------------------------------
comp_order = wil.sort_values("comparator_mean", ascending=False).index.tolist()
ax_c = ax_mm(124, 12, 34, 58)
fd_c = corr.set_index(["tissue", "pattern"])["FlashDeconv"]
rng = np.random.default_rng(0)
for i, c in enumerate(comp_order):
    d = fd_c.values - corr.set_index(["tissue", "pattern"])[c].values
    y = i + rng.uniform(-0.22, 0.22, d.size)
    ax_c.scatter(d, y, s=2.2, color=mcolor(c), alpha=0.75, linewidths=0, rasterized=True)
    ax_c.plot([np.median(d)] * 2, [i - 0.36, i + 0.36], color="black", lw=0.8)
    p = wil.loc[c, "p_value"]
    if p >= 0.01:
        ptxt = f"{p:.2f}"
    elif p >= 1e-3:
        ptxt = f"{p:.3f}"
    else:
        mant, ex = f"{p:.0e}".split("e")
        ptxt = f"{mant}×10$^{{{int(ex)}}}$"
    ax_c.text(1.03, i, ptxt, transform=ax_c.get_yaxis_transform(), fontsize=FS_TINY,
              va="center", ha="left")
ax_c.text(1.03, -1.1, "$P$", transform=ax_c.get_yaxis_transform(), fontsize=FS_TINY,
          va="center", ha="left", style="italic")
ax_c.axvline(0, color="#777777", lw=LW, ls=(0, (2, 2)), zorder=0)
ax_c.set_ylim(len(comp_order) - 0.5, -0.5)
ax_c.set_yticks(range(len(comp_order)))
ax_c.set_yticklabels([METHOD_NAMES[c] for c in comp_order])
ax_c.set_xscale("symlog", linthresh=0.05, linscale=1.0)
ax_c.set_xticks([-0.05, 0, 0.05, 0.5])
ax_c.set_xticklabels(["−0.05", "0", "0.05", "0.5"])
allc = np.concatenate([fd_c.values - corr[c].values for c in comp_order])
ax_c.set_xlim(min(-0.07, np.nanmin(allc) * 1.2), np.nanmax(allc) * 1.15)
ax_c.set_xlabel("ΔPearson (FlashDeconv − method)")
panel_label(ax_c, "c", dx_mm=-18, dy_mm=6)

# ---- d: gold standards ------------------------------------------------------
Y_D = 90
ax_d = []
xw, gap = 30.5, 9.0
for k, m in enumerate(METRICS):
    ax = ax_mm(19 + k * (xw + gap), Y_D, xw, 44)
    ax_d.append(ax)
    for i, meth in enumerate(ORDER):
        sub = gold[gold.method == meth].set_index("benchmark")
        vals = [sub.loc[b, m] for b in GOLD]
        ax.plot([min(vals), max(vals)], [i, i], color="#D0D0D0", lw=0.6, zorder=1)
        for b in GOLD:
            is_fd = meth == "FlashDeconv"
            ax.scatter(sub.loc[b, m], i, marker=GOLD_MARK[b], s=9 if is_fd else 7,
                       facecolor=mcolor(meth), edgecolor="black" if is_fd else "none",
                       linewidths=0.4, zorder=3 if is_fd else 2)
    ax.axhspan(ORDER.index("FlashDeconv") - 0.5, ORDER.index("FlashDeconv") + 0.5,
               color=FD_COLOR, alpha=0.10, lw=0, zorder=0)
    ax.set_ylim(NM - 0.5, -0.5)
    ax.set_yticks(range(NM))
    if k == 0:
        ax.set_yticklabels([METHOD_NAMES[mm] for mm in ORDER])
        for t, mm in zip(ax.get_yticklabels(), ORDER):
            if mm == "FlashDeconv":
                t.set_color(FD_COLOR)
                t.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
    ax.set_xlabel(METRIC_LAB[m] + (" (lower is better)" if m in LOWER_BETTER else ""))
panel_label(ax_d[0], "d", dx_mm=-18, dy_mm=5)
handles = [Line2D([], [], marker=GOLD_MARK[b], ls="", color="#606060", markersize=3,
                  label=GOLD[b]) for b in GOLD]
fig.legend(handles=handles, loc="lower left", ncol=3, frameon=False, fontsize=FS_SMALL,
           bbox_to_anchor=(19 / W_MM, 1 - (Y_D - 0.8) / H_MM), borderaxespad=0,
           handletextpad=0.2, columnspacing=1.2)


# ---- e / f: case studies (sorted lollipops) --------------------------------
def lollipop(ax, vals: dict, lower_better: bool, labels=None, xlab=""):
    keys = sorted(vals, key=lambda k: vals[k], reverse=not lower_better)
    for i, k in enumerate(keys):
        is_fd = k.startswith("FlashDeconv")
        col = FD_COLOR if is_fd else mcolor(k)
        ax.plot([0, vals[k]], [i, i], color=col, lw=0.8 if is_fd else 0.6,
                solid_capstyle="butt")
        open_mk = k == "FlashDeconv_pearson"
        ax.scatter(vals[k], i, s=10 if is_fd else 7, facecolor="white" if open_mk else col,
                   edgecolor=col, linewidths=0.7 if open_mk else 0, zorder=3)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.set_yticks(range(len(keys)))
    labs = [(labels or {}).get(k, METHOD_NAMES.get(k, k)) for k in keys]
    ax.set_yticklabels(labs)
    for t, k in zip(ax.get_yticklabels(), keys):
        if k.startswith("FlashDeconv"):
            t.set_color(FD_COLOR)
            t.set_fontweight("bold")
    ax.set_xlim(left=0)
    ax.set_xlabel(xlab)
    ax.tick_params(axis="y", length=0)
    return keys


Y_E, H_E = 152, 38
ax_e1 = ax_mm(19, Y_E, 22, H_E)
lollipop(ax_e1, liver["jsd"], True, xlab="JSD")
ax_e1.set_title("Liver, composition")
ax_e2 = ax_mm(62, Y_E, 22, H_E)
lollipop(ax_e2, liver["aupr"], False, xlab="AUPR")
ax_e2.set_xlim(0, 1)
ax_e2.set_title("Liver, zonation")
ax_e3 = ax_mm(105, Y_E, 22, H_E)
lollipop(ax_e3, liver["stability"], True, xlab="JSD between references")
ax_e3.set_title("Liver, reference stability")
panel_label(ax_e1, "e", dx_mm=-18, dy_mm=3)

ax_f = ax_mm(155, Y_E, 22, H_E)
lollipop(ax_f, mel, True, labels={"FlashDeconv_pearson": "FlashDeconv (PR)"}, xlab="JSD")
ax_f.set_title("Melanoma")
panel_label(ax_f, "f", dx_mm=-24, dy_mm=3)
for ax in (ax_e1, ax_e2, ax_e3, ax_f):
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))

save(fig, "supp_spotless")

# ---------------------------------------------------------------------------
# Console summary of the numbers shown
# ---------------------------------------------------------------------------
print("\nSilver means / ranks\n", pd.concat([sil_mean[METRICS], sil_rank[METRICS]], axis=1).loc[ORDER].round(4))
print("\nPer-tissue Pearson\n", tis_mean.loc[ORDER, tissues].round(3))
print("\nPer-tissue FD rank\n", tis_rank.loc["FlashDeconv", tissues])
for b in GOLD:
    sub = gold[gold.benchmark == b].set_index("method")
    for m in METRICS:
        r = rank_of(sub[m], m in LOWER_BETTER)
        best = sub[m].drop("FlashDeconv")
        bk = best.idxmin() if m in LOWER_BETTER else best.idxmax()
        print(f"gold {b} {m}: FD={sub.loc['FlashDeconv', m]:.4f} rank={int(r['FlashDeconv'])}/13 "
              f"best competitor {bk}={best[bk]:.4f}")
for m, lb in [("jsd", True), ("aupr", False), ("stability", True)]:
    s = pd.Series(liver[m])
    r = rank_of(s, lb)
    comp = s.drop("FlashDeconv")
    bk = comp.idxmin() if lb else comp.idxmax()
    print(f"liver {m}: FD={s['FlashDeconv']:.4f} rank={int(r['FlashDeconv'])}/13 best competitor {bk}={comp[bk]:.4f}")
s = pd.Series(mel)
comp = s.drop(["FlashDeconv", "FlashDeconv_pearson"])
for k in ["FlashDeconv", "FlashDeconv_pearson"]:
    r = rank_of(pd.concat([comp, s[[k]]]), True)
    print(f"melanoma {k}: JSD={s[k]:.4f} rank={int(r[k])}/13 best competitor {comp.idxmin()}={comp.min():.4f}")
