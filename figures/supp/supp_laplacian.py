"""Supplementary figure: spatial-regularization ablation (lambda = 0 vs automatic lambda).

Sources (results/rerun_final):
  a  benchmarks/laplacian/final_default/laplacian_ablation_paired.csv
     (validation/ablation_laplacian.py; Spotless silver, 6 tissues, sample 1)
  b  benchmarks/spotless/fd_per_celltype_silver.csv.gz (54 silver data sets)
  c  benchmarks/spotless/fd_aggregate_silver.csv, fd_aggregate_gold.csv
  d  crc/xenium/lambda_ablation/xenium_crc_lambda_ablation_summary.csv,
     xenium_crc_lambda_paired_{8,16}um.csv
  e  crc/demo/laplacian_ablation_visiumhd_8um_final.csv
Configs: final_default (automatic lambda) vs final_default_lam0 (lambda = 0).
Output: paper/figures/supp_laplacian_ablation.{pdf,png}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import (apply_style, new_figure, panel_label, save, RESULTS,  # noqa: E402
                   FD_COLOR, FS_SMALL, FS_TINY, LW)

RF = RESULTS / "rerun_final"
LAP = RF / "benchmarks" / "laplacian" / "final_default"
SP = RF / "benchmarks" / "spotless"
XE = RF / "crc" / "xenium" / "lambda_ablation"
VHD = RF / "crc" / "demo" / "laplacian_ablation_visiumhd_8um_final.csv"

LAM0_COLOR = "#8A8A8A"
TIE_COLOR = "#D0D0D0"
AUTO, NONE = "final_default", "final_default_lam0"
STRATA = [("rare", "Rare"), ("moderate", "Mod."), ("abundant", "Abund.")]

rng = np.random.default_rng(0)
apply_style()
H_MM, W_MM = 150.0, 180.0
fig = new_figure(height_mm=H_MM)


def ax_mm(x, y, w, h):
    return fig.add_axes([x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM])


def p_text(p):
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        m = p / 10 ** e
        if round(m) >= 10:
            m, e = m / 10, e + 1
        return f"{m:.0f}×10$^{{{e}}}$"
    return f"{p:.2g}" if p >= 0.01 else f"{p:.3f}"


def paired_p(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.all(d == 0):
        return np.nan
    return stats.wilcoxon(a, b).pvalue


def delta_strip(ax, deltas, higher_better=True, labels=None, ms=1.6, w=0.28,
                annotate=True, ann_y=1.0):
    """Jittered strip of paired differences (auto - lambda0), coloured by winner."""
    for i, d in enumerate(deltas):
        d = np.asarray(d, float)
        d = d[np.isfinite(d)]
        good = d > 0 if higher_better else d < 0
        bad = d < 0 if higher_better else d > 0
        col = np.where(good, FD_COLOR, np.where(bad, LAM0_COLOR, TIE_COLOR))
        x = i + rng.uniform(-w, w, d.size)
        ax.scatter(x, d, s=ms ** 2, c=col, linewidths=0, zorder=3, clip_on=False)
        ax.plot([i - w - 0.06, i + w + 0.06], [d.mean()] * 2, color="black", lw=0.8, zorder=4)
        if annotate:
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else np.nan
            ptxt = p_text(p) if np.isfinite(p) else "n.s."
            ax.text(i, ann_y, f"{good.sum()}/{d.size}\n{ptxt}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=FS_TINY, linespacing=1.05)
    ax.axhline(0, color="black", lw=LW * 0.8, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(-0.6, len(deltas) - 0.4)
    ax.set_xticks(range(len(deltas)))
    if labels is not None:
        ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)


# ===========================================================================
# a  Per-cell-type AUPR change, Spotless silver sample 1 (6 data sets, 76 types)
# ===========================================================================
pa = pd.read_csv(LAP / "laplacian_ablation_paired.csv")
ax_a = ax_mm(13, 18, 30, 30)
delta_strip(ax_a, [pa.loc[pa.category == k, "delta_auprc"] for k, _ in STRATA],
            labels=[l for _, l in STRATA], ms=2.0)
ax_a.set_ylabel("ΔAUPR (auto − λ = 0)")
ax_a.set_title("Silver, 6 data sets (per cell type)", pad=18)
panel_label(ax_a, "a", dy_mm=7)

# ===========================================================================
# b  Per-cell-type change in AUPR, precision, recall; 54 silver data sets
# ===========================================================================
ct = pd.read_csv(SP / "fd_per_celltype_silver.csv.gz")
ct = ct[ct.config.isin([AUTO, NONE])]
key = ["tissue", "pattern", "cell_type"]
wide = ct[ct.config == AUTO].merge(ct[ct.config == NONE], on=key, suffixes=("_a", "_n"))
b_axes = []
for j, (m, lab) in enumerate([("auprc", "ΔAUPR"), ("precision", "ΔPrecision"),
                              ("recall", "ΔRecall")]):
    ax = ax_mm(60 + j * 41, 18, 30, 30)
    ds = []
    for k, _ in STRATA:
        s = wide[wide.category_a == k]
        ds.append((s[f"{m}_a"] - s[f"{m}_n"]).values)
    delta_strip(ax, ds, labels=[l for _, l in STRATA], ms=1.1)
    ax.set_ylabel(lab)
    b_axes.append(ax)
b_axes[1].set_title("Silver, 54 data sets (per cell type)", pad=18)
panel_label(b_axes[0], "b", dy_mm=7)

# ===========================================================================
# c  Data-set-level metrics, silver (54) and gold (15)
# ===========================================================================
agg = {}
for bm, f in [("Silver", "fd_aggregate_silver.csv"), ("Gold", "fd_aggregate_gold.csv")]:
    d = pd.read_csv(SP / f)
    d = d[d.config.isin([AUTO, NONE])]
    agg[bm] = d.pivot_table(index=["benchmark", "tissue", "pattern"], columns="config",
                            values=["corr", "rmse", "jsd", "aupr"])
c_axes = []
for j, (m, lab, hb) in enumerate([("corr", "ΔPearson", True), ("rmse", "ΔRMSE", False),
                                  ("jsd", "ΔJSD", False), ("aupr", "ΔAUPR", True)]):
    ax = ax_mm(13 + j * 44, 70, 30, 28)
    ds = [(agg[bm][(m, AUTO)] - agg[bm][(m, NONE)]).values for bm in ["Silver", "Gold"]]
    delta_strip(ax, ds, higher_better=hb, labels=["Silver", "Gold"], ms=1.8, w=0.25)
    ax.set_ylabel(lab)
    c_axes.append(ax)
c_axes[0].set_title("Per data set: silver (54), gold (15)", pad=18, loc="left", x=0.05)
panel_label(c_axes[0], "c", dy_mm=7)

# ===========================================================================
# d  Xenium CRC pseudo-bins (8 and 16 um)
# ===========================================================================
xs = pd.read_csv(XE / "xenium_crc_lambda_ablation_summary.csv")
xp = {b: pd.read_csv(XE / f"xenium_crc_lambda_paired_{b}um.csv") for b in [8, 16]}
d_axes = []
for j, (m, lab) in enumerate([("overall_pearson_r", "Pearson"), ("overall_rmse", "RMSE")]):
    ax = ax_mm(13 + j * 24, 118, 15, 25)
    for i, b in enumerate([8, 16]):
        v0 = xs[(xs.bin_size_um == b) & (xs.condition == "no_spatial")][m].item()
        v1 = xs[(xs.bin_size_um == b) & (xs.condition == "auto")][m].item()
        ax.plot([i, i], [v0, v1], color="#BBBBBB", lw=0.8, zorder=1)
        ax.scatter([i], [v0], s=12, color=LAM0_COLOR, linewidths=0, zorder=3)
        ax.scatter([i], [v1], s=12, color=FD_COLOR, linewidths=0, zorder=3)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["8", "16"])
    ax.set_xlabel("Bin (µm)")
    ax.set_ylabel(lab)
    d_axes.append(ax)
d_axes[0].set_ylim(0.695, 0.72)
d_axes[0].set_yticks([0.70, 0.71, 0.72])
d_axes[1].set_ylim(0.08, 0.094)
d_axes[1].set_yticks([0.08, 0.085, 0.09])
for j, (m, lab, hb) in enumerate([("auprc", "ΔAUPR", True), ("rmse", "ΔRMSE", False)]):
    ax = ax_mm(66 + j * 36, 118, 24, 25)
    delta_strip(ax, [xp[b][f"delta_{m}"].values for b in [8, 16]], higher_better=hb,
                labels=["8", "16"], ms=1.6, w=0.25)
    ax.set_xlabel("Bin (µm)")
    ax.set_ylabel(lab + " per cell type")
    d_axes.append(ax)
panel_label(d_axes[0], "d", dy_mm=7)

# ===========================================================================
# e  Visium HD CRC 8 um: Moran's I of predicted proportions
# ===========================================================================
vh = pd.read_csv(VHD)
w = vh.pivot(index="cell_type", columns="condition", values="morans_i")
w["gain"] = 100 * (w["auto"] - w["no_spatial"]) / w["no_spatial"]
w = w.sort_values("gain")
ax_e = ax_mm(153, 118, 23, 25)
y = np.arange(len(w))
ax_e.barh(y, w["gain"], height=0.62, color=FD_COLOR, linewidth=0)
ax_e.set_yticks(y)
ax_e.set_yticklabels([s.replace(" cells", "").replace("Myeloids", "Myeloid") for s in w.index])
ax_e.tick_params(axis="y", length=0)
ax_e.spines["left"].set_visible(False)
ax_e.axvline(0, color="black", lw=LW)
ax_e.set_xlabel("Moran's I change (%)")
ax_e.set_xlim(0, 4)
ax_e.set_title("Visium HD CRC, 8 µm", pad=18)
panel_label(ax_e, "e", dx_mm=-20, dy_mm=7)
d_axes[0].set_title("Xenium CRC pseudo-bins (overall and per cell type)", pad=18, loc="left", x=0.0)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=FD_COLOR, ms=3, label="Automatic λ (default)"),
                    Line2D([], [], marker="o", ls="", color=LAM0_COLOR, ms=3, label="λ = 0"),
                    Line2D([], [], marker="o", ls="", color=TIE_COLOR, ms=3, label="Tie"),
                    Line2D([], [], color="black", lw=0.8, label="Mean difference")],
           loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=FS_SMALL,
           handletextpad=0.3, columnspacing=1.5)

save(fig, "supp_laplacian_ablation")

# ---------------------------------------------------------------------------
# Console check of numbers
# ---------------------------------------------------------------------------
print("\n[a] 6-data-set paired (laplacian_ablation_paired.csv)")
for k, _ in STRATA:
    s = pa[pa.category == k]
    print(f"  {k}: n={len(s)} auto>lam0={int((s.delta_auprc > 0).sum())} "
          f"<={int((s.delta_auprc < 0).sum())} ties={int((s.delta_auprc == 0).sum())} "
          f"meanD={s.delta_auprc.mean():+.4f} P={paired_p(s.auprc_auto, s.auprc_none):.3g}")
print("[b] 54 silver per type")
for m in ["auprc", "precision", "recall"]:
    for k, _ in STRATA:
        s = wide[wide.category_a == k]
        d = (s[f"{m}_a"] - s[f"{m}_n"]).dropna()
        print(f"  {m} {k}: n={len(d)} up={int((d > 0).sum())} down={int((d < 0).sum())} "
              f"meanD={d.mean():+.4f} P={stats.wilcoxon(d).pvalue:.3g}")
print("[c] data-set level")
for bm in ["Silver", "Gold"]:
    for m in ["corr", "rmse", "jsd", "aupr"]:
        a, n = agg[bm][(m, AUTO)], agg[bm][(m, NONE)]
        print(f"  {bm} {m}: auto={a.mean():.4f} lam0={n.mean():.4f} n={len(a)} "
              f"P={stats.wilcoxon(a, n).pvalue:.3g}")
print("[d] Xenium"); print(xs.to_string())
for b in [8, 16]:
    for m in ["auprc", "pearson", "rmse"]:
        d = xp[b][f"delta_{m}"]
        print(f"  {b}um {m}: meanD={d.mean():+.5f} up={int((d > 0).sum())}/{len(d)} "
              f"P={stats.wilcoxon(d).pvalue:.3g}")
print("[e] Visium HD"); print(w.to_string())
