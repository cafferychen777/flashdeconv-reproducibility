"""Supplementary figure: spatial-regularization ablation (lambda = 0 vs automatic lambda) on
data with real spatial coordinates only. The Spotless silver-standard pseudo-spots have no spatial
layout and are not used here (all silver-standard results use lambda = 0).

Sources:
  a  results/editor_revision/gold_realxy/fd_aggregate_gold.csv
     (validation/controls_editor/gold_realxy.py; seqFISH+ 14 FOVs and STARmap, real spot coordinates)
  b  results/rerun_final/crc/xenium/lambda_ablation/xenium_crc_lambda_ablation_summary.csv,
     xenium_crc_lambda_paired_{8,16}um.csv
  c  results/rerun_final/crc/demo/laplacian_ablation_visiumhd_8um_final.csv
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
GOLD = RESULTS / "editor_revision" / "gold_realxy" / "fd_aggregate_gold.csv"
XE = RF / "crc" / "xenium" / "lambda_ablation"
VHD = RF / "crc" / "demo" / "laplacian_ablation_visiumhd_8um_final.csv"

LAM0_COLOR = "#8A8A8A"
TIE_COLOR = "#D0D0D0"
AUTO, NONE = "final_default_realxy", "final_default_lam0_realxy"

rng = np.random.default_rng(0)
apply_style()
H_MM, W_MM = 92.0, 180.0
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
# a  Gold standards with real coordinates: per-FOV change (15 FOVs)
# ===========================================================================
g = pd.read_csv(GOLD)
g = g[g.config.isin([AUTO, NONE])].pivot_table(index=["benchmark", "tissue"], columns="config",
                                                values=["corr", "rmse", "jsd", "aupr"])
a_axes = []
for j, (m, lab, hb) in enumerate([("corr", "ΔPearson", True), ("rmse", "ΔRMSE", False),
                                  ("jsd", "ΔJSD", False), ("aupr", "ΔAUPR", True)]):
    ax = ax_mm(13 + j * 44, 17, 26, 25)
    delta_strip(ax, [(g[(m, AUTO)] - g[(m, NONE)]).values], higher_better=hb,
                labels=[""], ms=2.0, w=0.3)
    ax.set_ylabel(lab)
    a_axes.append(ax)
a_axes[0].set_title("Gold standards, 15 FOVs (seqFISH+, STARmap)", pad=18, loc="left", x=0.0)
panel_label(fig, "a", x=2 / W_MM, y=1 - 9 / H_MM)

# ===========================================================================
# b  Xenium CRC pseudo-bins (8 and 16 um)
# ===========================================================================
xs = pd.read_csv(XE / "xenium_crc_lambda_ablation_summary.csv")
xp = {b: pd.read_csv(XE / f"xenium_crc_lambda_paired_{b}um.csv") for b in [8, 16]}
d_axes = []
for j, (m, lab) in enumerate([("overall_pearson_r", "Pearson"), ("overall_rmse", "RMSE")]):
    ax = ax_mm(13 + j * 24, 60, 15, 25)
    for i, b in enumerate([8, 16]):
        v0 = xs[(xs.bin_size_um == b) & (xs.condition == "no_spatial")][m].item()
        v1 = xs[(xs.bin_size_um == b) & (xs.condition == "auto")][m].item()
        ax.plot([i, i], [v0, v1], color="#BBBBBB", lw=0.8, zorder=1)
        ax.scatter([i], [v0], s=12, facecolor="white", edgecolor="black", linewidths=0.6, zorder=3)
        ax.scatter([i], [v1], s=12, color="black", linewidths=0, zorder=3)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["8", "16"])
    ax.set_xlabel("Bin (µm)")
    ax.set_ylabel(lab)
    d_axes.append(ax)
d_axes[1].legend(handles=[Line2D([], [], marker="o", ls="", color="black", ms=3, label="Automatic λ"),
                          Line2D([], [], marker="o", ls="", mfc="white", mec="black", mew=0.6, ms=3,
                                 label="λ = 0")],
                 loc="upper right", bbox_to_anchor=(1.25, 1.02), fontsize=FS_TINY, handletextpad=0.1)
d_axes[0].set_ylim(0.695, 0.72)
d_axes[0].set_yticks([0.70, 0.71, 0.72])
d_axes[1].set_ylim(0.08, 0.094)
d_axes[1].set_yticks([0.08, 0.085, 0.09])
for j, (m, lab, hb) in enumerate([("auprc", "ΔAUPR", True), ("rmse", "ΔRMSE", False)]):
    ax = ax_mm(66 + j * 36, 60, 24, 25)
    delta_strip(ax, [xp[b][f"delta_{m}"].values for b in [8, 16]], higher_better=hb,
                labels=["8", "16"], ms=1.6, w=0.25)
    ax.set_xlabel("Bin (µm)")
    ax.set_ylabel(lab + " per cell type")
    d_axes.append(ax)
panel_label(fig, "b", x=2 / W_MM, y=1 - 52 / H_MM)

# ===========================================================================
# c  Visium HD CRC 8 um: Moran's I of predicted proportions
# ===========================================================================
vh = pd.read_csv(VHD)
w = vh.pivot(index="cell_type", columns="condition", values="morans_i")
w["gain"] = 100 * (w["auto"] - w["no_spatial"]) / w["no_spatial"]
w = w.sort_values("gain")
ax_e = ax_mm(153, 60, 23, 25)
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
panel_label(fig, "c", x=133 / W_MM, y=1 - 52 / H_MM)
d_axes[0].set_title("Xenium CRC pseudo-bins (overall and per cell type)", pad=18, loc="left", x=0.0)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=FD_COLOR, ms=3, label="Automatic λ (default) better"),
                    Line2D([], [], marker="o", ls="", color=LAM0_COLOR, ms=3, label="λ = 0 better"),
                    Line2D([], [], marker="o", ls="", color=TIE_COLOR, ms=3, label="Tie"),
                    Line2D([], [], color="black", lw=0.8, label="Mean difference")],
           loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=FS_SMALL,
           handletextpad=0.3, columnspacing=1.5)

save(fig, "supp_laplacian_ablation")

# ---------------------------------------------------------------------------
# Console check of numbers
# ---------------------------------------------------------------------------
print("[a] gold, real coordinates")
for m in ["corr", "rmse", "jsd", "aupr"]:
    x, y = g[(m, AUTO)], g[(m, NONE)]
    print(f"  {m}: auto={x.mean():.4f} lam0={y.mean():.4f} auto>lam0={int((x > y).sum())}/{len(x)} "
          f"P={stats.wilcoxon(x, y).pvalue:.3g}")
print("[d] Xenium"); print(xs.to_string())
for b in [8, 16]:
    for m in ["auprc", "pearson", "rmse"]:
        d = xp[b][f"delta_{m}"]
        print(f"  {b}um {m}: meanD={d.mean():+.5f} up={int((d > 0).sum())}/{len(d)} "
              f"P={stats.wilcoxon(d).pvalue:.3g}")
print("[e] Visium HD"); print(w.to_string())
