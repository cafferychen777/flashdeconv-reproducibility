"""Supplementary figure: Xenium-derived CRC pseudo-Visium HD benchmark (C2).

Patient P1, 38 cell types, square bins of 2/4/8/16/32 um. Draws every panel on
one canvas and writes the companion table
results/rerun_final/benchmarks/c2/c2_supp_table.csv.

Configurations used
-------------------
* FlashDeconv            : final_default_auto (package defaults, automatic lambda)
* FlashDeconv (lambda=0) : final_default_l0   (same run without spatial smoothing)
* RCTD (doublet / full)  : official spacexr, UMI_min = 100 (package default)
* NNLS, Marker scoring   : default
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import *  # noqa: E402,F401,F403

C2DIR = RESULTS / "rerun_final" / "benchmarks" / "c2"
SRC = C2DIR / "c2_standard_metrics_final.csv"  # same source as Supplementary Table S9 (AP)
OUT_TABLE = C2DIR / "c2_supp_table.csv"

SIZES = [2, 4, 8, 16, 32]
SHORT = {"pearson_flat": "Pearson r", "pearson_type_mean": "Per-type Pearson r", "ap_type_mean": "Per-type AP"}
FD_L0 = "FlashDeconv (λ = 0)"

# display name -> row selector
METHODS = {
    "FlashDeconv": lambda d: (d.method == "FlashDeconv") & (d["mode"] == "final_default_auto"),
    FD_L0: lambda d: (d.method == "FlashDeconv") & (d["mode"] == "final_default_l0"),
    "RCTD (doublet)": lambda d: (d.method == "RCTD") & (d["mode"] == "doublet") & (d.umi_min == 100),
    "RCTD (full)": lambda d: (d.method == "RCTD") & (d["mode"] == "full") & (d.umi_min == 100),
    "NNLS": lambda d: d.method == "NNLS",
    "Marker scoring": lambda d: d.method == "MarkerScoring",
}
EVAL_SETS = {
    "all": "all predicted bins",
    "common_doublet_umi100": "common bins with RCTD doublet",
    "common_full_umi100": "common bins with RCTD full",
}
METRICS = [  # column, axis label, higher is better
    ("pearson_flat", "Pearson r (flattened)", True),
    ("pearson_type_mean", "Mean per-type Pearson r", True),
    ("rmse", "RMSE", False),
    ("jsd", "JSD", False),
    ("ap_flat", "AP (flattened)", True),
    ("ap_type_mean", "Mean per-type AP", True),
]


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def build_supp_table(src: Path = SRC) -> pd.DataFrame:
    d = pd.read_csv(src)
    n_total = (d[(d.method == "FlashDeconv") & (d["mode"] == "final_default_auto")
                 & (d.eval_set == "all")].set_index("resolution_um").n_bins)
    rows = []
    for size in SIZES:
        for name, sel in METHODS.items():
            for es, es_label in EVAL_SETS.items():
                t = d[sel(d) & (d.resolution_um == size) & (d.eval_set == es)]
                if t.empty:
                    continue  # e.g. RCTD doublet is not defined on RCTD-full bins
                r = t.iloc[0]
                rows.append({
                    "bin_size_um": size, "method": name, "eval_set": es_label,
                    "n_bins": int(r.n_bins),
                    # fraction of all bins at this size that the method scored
                    "coverage": r.coverage,
                    "pearson_flat": r.pearson_flat, "pearson_type_mean": r.type_r,
                    "rmse": r.rmse, "jsd": r.jsd, "ap_flat": r.ap_flat,
                    "ap_type_mean": r.ap_type_mean,
                })
    tab = pd.DataFrame(rows)
    tab.insert(1, "n_bins_total", tab.bin_size_um.map(n_total).astype(int))
    return tab


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
STYLE = {
    "FlashDeconv": dict(color=FD_COLOR, ls="-", marker="o", mfc=FD_COLOR, zorder=6),
    FD_L0: dict(color="#EFA36B", ls=(0, (3, 1.5)), marker="o", mfc="white", zorder=5),
    "RCTD (doublet)": dict(color=METHOD_COLORS["RCTD (doublet)"], ls=RCTD_LS["doublet"],
                           marker=RCTD_MARKERS["doublet"], mfc=METHOD_COLORS["RCTD (doublet)"], zorder=4),
    "RCTD (full)": dict(color=METHOD_COLORS["RCTD (full)"], ls=RCTD_LS["full"],
                        marker=RCTD_MARKERS["full"], mfc=METHOD_COLORS["RCTD (full)"], zorder=4),
    "NNLS": dict(color=METHOD_COLORS["NNLS"], ls="-", marker="^", mfc=METHOD_COLORS["NNLS"], zorder=3),
    "Marker scoring": dict(color=METHOD_COLORS["Marker scoring"], ls="-", marker="v",
                           mfc=METHOD_COLORS["Marker scoring"], zorder=3),
}


def bin_axis(ax, labels=True):
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES] if labels else [])
    ax.minorticks_off()
    ax.set_xlim(2 / 1.3, 32 * 1.3)
    if labels:
        ax.set_xlabel("Bin size (µm)")


def draw_line(ax, x, y, name):
    s = STYLE[name]
    ax.plot(x, y, color=s["color"], ls=s["ls"], lw=0.8, marker=s["marker"], ms=2.6,
            mfc=s["mfc"], mec=s["color"], mew=0.6, zorder=s["zorder"], clip_on=False,
            label=name)


def main():
    apply_style()
    tab = build_supp_table()
    tab.to_csv(OUT_TABLE, index=False)
    print("wrote", OUT_TABLE)

    allb = tab[tab.eval_set == "all predicted bins"]

    fig = new_figure(height_mm=146)
    W, H = fig.get_size_inches() / MM

    def add_ax(x_mm, y_top_mm, w_mm, h_mm):
        return fig.add_axes([x_mm / W, 1 - (y_top_mm + h_mm) / H, w_mm / W, h_mm / H])

    # ---- a-f: six metrics, all predicted bins; g: coverage ----------------
    left, gap, w, h = 13.0, 13.5, 30.5, 30.0
    rows_y = [8.0, 52.0]
    panels = METRICS + [("coverage", "Bins scored (%)", True)]
    letters = "abcdefg"
    axes = []
    for k, (col, lab, _) in enumerate(panels):
        r, c = divmod(k, 4)
        ax = add_ax(left + c * (w + gap), rows_y[r], w, h)
        for name in METHODS:
            t = allb[allb.method == name].set_index("bin_size_um").reindex(SIZES)
            draw_line(ax, SIZES, t[col], name)
        bin_axis(ax)
        ax.set_ylabel(lab)
        if col == "coverage":
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0", "25", "50", "75", "100"])
        panel_label(ax, letters[k], dx_mm=-11.5)
        axes.append(ax)

    # legend in the free slot of row 2
    handles, labels = axes[0].get_legend_handles_labels()
    lax = add_ax(left + 3 * (w + gap), rows_y[1], w, h)
    lax.axis("off")
    lax.legend(handles, labels, loc="center left", fontsize=FS, handlelength=2.4,
               labelspacing=0.6, borderaxespad=0)

    # ---- h: FlashDeconv vs RCTD on common bins ------------------------------
    y_h = 101.0
    w_h, gap_h = 20.5, 8.3
    off = {"doublet": -0.13, "full": 0.13}
    xs = np.log2(SIZES)
    for k, (col, lab, _) in enumerate(METRICS):
        ax = add_ax(left + k * (w_h + gap_h), y_h, w_h, 28.0)
        for mode, rname, es in [("doublet", "RCTD (doublet)", "common bins with RCTD doublet"),
                                ("full", "RCTD (full)", "common bins with RCTD full")]:
            sub = tab[tab.eval_set == es]
            f = sub[sub.method == "FlashDeconv"].set_index("bin_size_um").reindex(SIZES)[col]
            rr = sub[sub.method == rname].set_index("bin_size_um").reindex(SIZES)[col]
            x = xs + off[mode]
            mk = STYLE[rname]["marker"]
            for i in range(len(SIZES)):
                ax.plot([x[i]] * 2, [rr.iloc[i], f.iloc[i]], color="#9A9A9A", lw=0.5, zorder=1)
            ax.scatter(x, rr, s=7, marker=mk, color=STYLE[rname]["color"], lw=0, zorder=3)
            ax.scatter(x, f, s=7, marker=mk, color=FD_COLOR, lw=0, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(s) for s in SIZES])
        ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
        ax.set_xlabel("Bin size (µm)")
        ax.set_title(SHORT.get(col, lab), fontsize=FS)
        ax.locator_params(axis="y", nbins=4)
        if k == 0:
            panel_label(ax, "h", dx_mm=-11.5, dy_mm=4.0)

    # legend for h (marker = RCTD mode; colour = method)
    from matplotlib.lines import Line2D
    mk = lambda m, c, l: Line2D([], [], ls="none", marker=m, ms=3.2, mfc=c, mec=c, label=l)
    md, mf = STYLE["RCTD (doublet)"]["marker"], STYLE["RCTD (full)"]["marker"]
    hh = [mk(md, FD_COLOR, "FlashDeconv, RCTD-doublet bins"),
          mk(md, STYLE["RCTD (doublet)"]["color"], "RCTD (doublet)"),
          mk(mf, FD_COLOR, "FlashDeconv, RCTD-full bins"),
          mk(mf, STYLE["RCTD (full)"]["color"], "RCTD (full)")]
    fig.legend(handles=hh, loc="upper left", ncol=4, fontsize=FS,
               bbox_to_anchor=(left / W, 1 - (y_h + 28.0 + 10.5) / H), borderaxespad=0,
               handletextpad=0.2, columnspacing=1.6)

    save(fig, "supp_pseudo_vhd")

    # ---- console verification ---------------------------------------------
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
