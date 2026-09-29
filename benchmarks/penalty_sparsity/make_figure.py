"""Supplementary figure: benefit of the spatial penalty (auto lambda vs lambda = 0) by bin depth.

Reads results/penalty_sparsity/{summary_by_stratum,lambda_by_dataset}.csv (summarize.py) and
writes paper/figures/supp_penalty_sparsity.{pdf,png}.
dJSD = mean per-bin JSD(auto) - JSD(lambda0) (x 10^3); negative = penalty helps. Error bars:
95% spatial block-bootstrap CI (200 um blocks; FOVs for the gold standards). Strata with < 200
bins are not drawn.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from style import MM, FS, FS_SMALL, LW, OI, RESULTS, apply_style, despine, new_figure, panel_label, save  # noqa: E402

R = RESULTS / "penalty_sparsity"
K = 1e3


def err(ax, x, d, color, marker="o", ls="-", label=None, mfc=None, ms=3.0):
    y, lo, hi = d.d_jsd.to_numpy() * K, d.d_jsd_lo.to_numpy() * K, d.d_jsd_hi.to_numpy() * K
    ax.errorbar(x, y, yerr=[y - lo, hi - y], color=color, marker=marker, ms=ms, lw=0.8, ls=ls,
                elinewidth=0.6, capsize=0, mfc=mfc or color, mew=0.6, label=label)


def depth_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks([10, 30, 100, 300, 1000])
    ax.set_xticklabels(["10", "30", "100", "300", "1,000"])
    ax.minorticks_off()
    ax.axhline(0, color="#999999", lw=LW, zorder=0)


def main():
    apply_style()
    S = pd.read_csv(R / "summary_by_stratum.csv")
    L = pd.read_csv(R / "lambda_by_dataset.csv")
    strata = S[(S.stratum != "all") & (S.n_bins >= 200)]
    fig = new_figure(58)
    w, h, y0 = 32, 38, 12
    xs = [14, 58, 104, 146]
    axes = [fig.add_axes([x * MM / (180 * MM), y0 / 58, w / 180, h / 58]) for x in xs]

    # a: S1 MERFISH gut, three depth arms (benchmark counts and binomial thinning)
    ax = axes[0]
    cols = {1.0: OI["blue"], 0.5: OI["skyblue"], 0.25: OI["orange"]}
    for p, c in cols.items():
        d = strata[(strata.dataset == "S1 MERFISH gut") & (strata.thin_p == p) & (strata.scale == 1e6)]
        err(ax, d.median_umi, d, c, label=f"{p:g}×")
        d5 = strata[(strata.dataset == "S1 MERFISH gut") & (strata.thin_p == p) & (strata.scale == 1e5)]
        err(ax, d5.median_umi * 1.06, d5, c, ls=":", mfc="white", ms=2.6)
    depth_axis(ax)
    ax.set_xlabel("Bin UMI (stratum median)")
    ax.set_ylabel("ΔJSD, auto λ − (λ = 0) (×10$^{-3}$)")
    ax.set_title("MERFISH gut, 8 µm bins", fontsize=FS, pad=3)
    ax.legend(frameon=False, fontsize=FS_SMALL, loc="lower right", handlelength=1.2,
              title="Counts retained", title_fontsize=FS_SMALL)

    # b: C2 Xenium CRC pseudo-Visium HD by bin size
    ax = axes[1]
    ccol = {2: OI["vermillion"], 4: OI["orange"], 8: OI["green"], 16: OI["blue"]}
    for r, c in ccol.items():
        d = strata[(strata.dataset == "C2 Xenium CRC") & (strata.bin_um == r)]
        err(ax, d.median_umi, d, c, label=f"{r} µm")
    depth_axis(ax)
    ax.set_xlabel("Bin UMI (stratum median)")
    ax.set_title("Xenium CRC pseudo-Visium HD", fontsize=FS, pad=3)
    ax.legend(frameon=False, fontsize=FS_SMALL, loc="lower right", handlelength=1.2)

    # c: whole-dataset effect vs median depth (S1 and C2 arms)
    ax = axes[2]
    A = S[S.stratum == "all"]
    for (ds, sc), g in A.groupby(["dataset", "scale"], dropna=False):
        if ds == "S1 MERFISH gut":
            c, m, lab = OI["blue"], "o" if sc == 1e6 else "s", f"MERFISH gut ({int(sc):,} bins)"
        elif ds == "C2 Xenium CRC":
            c, m, lab = OI["vermillion"], "D", "Xenium CRC (2–16 µm)"
        else:
            continue  # gold (233 spots, >=400 UMI): CI spans ~20e-3; reported in the CSV / legend text
        g = g.sort_values("median_umi")
        err(ax, g.median_umi, g, c, marker=m, ls="none", label=lab)
    ax.set_xscale("log")
    ax.set_xticks([10, 30, 100, 300])
    ax.set_xticklabels(["10", "30", "100", "300"])
    ax.minorticks_off()
    ax.axhline(0, color="#999999", lw=LW, zorder=0)
    ax.set_ylim(-5.6, 0.9)
    ax.set_xlabel("Median bin UMI")
    ax.set_ylabel("ΔJSD, all bins (×10$^{-3}$)")
    ax.legend(frameon=False, fontsize=FS_SMALL - 0.5, loc="lower center", handlelength=1.0,
              borderaxespad=0.2, bbox_to_anchor=(0.56, 0.0))

    # d: automatically chosen lambda vs median depth
    ax = axes[3]
    for ds, c, m in [("S1 MERFISH gut", OI["blue"], "o"), ("C2 Xenium CRC", OI["vermillion"], "D"),
                     ("Spotless gold", OI["black"], "^")]:
        g = L[L.dataset == ds]
        if ds == "Spotless gold":
            g = g[g.arm == "real coords"]
            ax.errorbar(g.median_umi, g.lambda_auto,
                        yerr=[g.lambda_auto - g.lambda_min, g.lambda_max - g.lambda_auto],
                        color=c, marker=m, ms=3, ls="none", elinewidth=0.6, capsize=0,
                        label="Spotless gold (15 FOVs)")
        else:
            ax.plot(g.median_umi, g.lambda_auto, color=c, marker=m, ms=3, ls="none",
                    label="MERFISH gut" if ds.startswith("S1") else "Xenium CRC")
    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1,000", "10,000"])
    ax.minorticks_off()
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("Median bin UMI")
    ax.set_ylabel("Automatic λ")
    ax.legend(frameon=False, fontsize=FS_SMALL - 0.5, loc="lower right", handlelength=1.0)

    for a_, l_ in zip(axes, "abcd"):
        despine(a_)
        panel_label(a_, l_, dx_mm=-11)
    save(fig, "supp_penalty_sparsity")


if __name__ == "__main__":
    main()
