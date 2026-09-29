"""Supplementary Figure: S1 accuracy with the external scRNA-seq reference.

MERFISH-derived mouse intestine benchmark (S1), deconvolved with an external, independently
annotated scRNA-seq reference (harmonised to the 18 S1 types) instead of the MERFISH self-reference.
  a  Pearson r with the true cell fractions vs bins (bins estimated by every completed method)
  b  mean per-bin JSD vs bins (same bins)
  c  fraction of bins with estimates vs bins (not-completed runs: 'X' at 0 %)
Inputs: results/s1_merfish_benchmark/{s1_summary,s1_runtime}.csv (+ c2l_aces/*runtime*.csv)
Usage: python validation/figures/supp/supp_s1_external_reference.py
"""
import sys
from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import *  # noqa: E402,F401,F403

apply_style()

S1 = RESULTS / "s1_merfish_benchmark"
STY = {
    "FlashDeconv": (METHOD_COLORS["FlashDeconv"], "-", "o"),
    "RCTD (doublet)": (METHOD_COLORS["RCTD (doublet)"], RCTD_LS["doublet"], RCTD_MARKERS["doublet"]),
    "RCTD (full)": (METHOD_COLORS["RCTD (full)"], RCTD_LS["full"], RCTD_MARKERS["full"]),
    "CARD": (METHOD_COLORS["CARD"], "-", "^"),
    "Cell2location": (METHOD_COLORS["Cell2location"], "-", "v"),
}
ORDER = list(STY)
MS = {"o": 3.2, "s": 2.8, "D": 2.5, "^": 3.2, "v": 3.2}
LAB = {("flashdeconv", "default"): "FlashDeconv", ("rctd", "doublet"): "RCTD (doublet)",
       ("rctd", "full"): "RCTD (full)", ("card", "default"): "CARD",
       ("cell2location", "fullbatch"): "Cell2location", ("cell2location", "minibatch"): "Cell2location"}
SCALES = [10_000, 100_000, 210_499]
SCALE_LAB = {10_000: "10$^4$", 100_000: "10$^5$", 210_499: "2.1×10$^5$"}
XOFF = {"CARD": 0.88, "RCTD (full)": 1.0, "RCTD (doublet)": 1.0, "Cell2location": 1.13, "FlashDeconv": 1.0}

summ = pd.read_csv(S1 / "s1_summary.csv")
summ = summ[(summ["gt"] == "cell") & (summ.ref == "extref")]
summ = summ[[(m, v) in LAB for m, v in zip(summ.method, summ.variant)]].copy()
summ["lab"] = [LAB[(m, v)] for m, v in zip(summ.method, summ.variant)]
summ["_pri"] = (summ.variant == "fullbatch").astype(int)
summ = summ.sort_values("_pri").drop_duplicates(["scale", "lab", "binset"], keep="last")

rt_files = [S1 / "s1_runtime.csv"] + sorted((S1 / "c2l_aces").glob("*runtime*.csv"))
rt = pd.concat([pd.read_csv(f) for f in rt_files if f.exists()], ignore_index=True)
rt = rt.drop_duplicates(["method", "mode", "scale"], keep="last")
rt["ref"] = rt["mode"].str.split("-").str[0]
rt["variant"] = rt["mode"].str.split("-", n=1).str[1]
rt = rt[(rt.ref == "extref") & [(m, v) in LAB for m, v in zip(rt.method, rt.variant)]].copy()
rt["lab"] = [LAB[(m, v)] for m, v in zip(rt.method, rt.variant)]


def is_bad(status):
    s = str(status).upper()
    return s.startswith(("OOM", "DNF", "TIMEOUT", "TIME"))


fig = new_figure(55.0)
gs = fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.78, bottom=0.2, wspace=0.42)
axes = {k: fig.add_subplot(gs[0, i]) for i, k in enumerate("abc")}


def line(ax, x, y, lab):
    c, ls, mk = STY[lab]
    ax.plot(x, y, color=c, ls=ls, marker=mk, ms=MS[mk], lw=0.9, mew=0, clip_on=False)


def scale_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(SCALES)
    ax.set_xticklabels([SCALE_LAB[s] for s in SCALES])
    ax.minorticks_off()
    ax.set_xlim(SCALES[0] / 1.6, SCALES[-1] * 1.6)
    ax.set_xlabel("Bins")


for k, metric, ylab in [("a", "flat_pearson", "Pearson r"), ("b", "jsd", "Mean JSD")]:
    d = summ[summ.binset == "common"]
    for lab in ORDER:
        x = d[d.lab == lab].sort_values("scale")
        if len(x):
            line(axes[k], x.scale, x[metric], lab)
    scale_axis(axes[k])
    axes[k].set_ylabel(ylab)
    axes[k].set_title("scRNA-seq reference, shared bins")

ax = axes["c"]
d = summ[summ.binset == "own"]
for lab in ORDER:
    x = d[d.lab == lab].sort_values("scale")
    if len(x):
        line(ax, x.scale, 100 * x.coverage, lab)
    for _, b in rt[(rt.lab == lab)].iterrows():
        if is_bad(b.status):
            ax.plot(b.scale * XOFF[lab], 0, ls="", marker="X", color=STY[lab][0], ms=4.5, mew=0.3,
                    mec="white", clip_on=False, zorder=5)
scale_axis(ax)
ax.set_ylim(0, 105)
ax.set_ylabel("Bins with estimates (%)")
ax.set_title("scRNA-seq reference")

for k, a in axes.items():
    panel_label(a, k, dx_mm=-10.5, dy_mm=2.0)

present = [l for l in ORDER if l in set(summ.lab)]
handles = [Line2D([], [], color=STY[l][0], ls=STY[l][1], marker=STY[l][2], ms=MS[STY[l][2]], lw=0.9,
                  mew=0, label=l) for l in present]
if any(is_bad(s) for s in rt.status):
    handles.append(Line2D([], [], ls="", marker="X", color="0.35", ms=4.5, mew=0.3, mec="white",
                          label="Not completed"))
fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.995),
           frameon=False, handlelength=2.2, columnspacing=1.6, fontsize=FS)
save(fig, "supp_s1_external_reference")

pd.set_option("display.width", 200)
print(summ[["scale", "lab", "binset", "n_bins", "coverage", "flat_pearson", "jsd"]]
      .sort_values(["binset", "scale", "lab"]).round(4).to_string(index=False))
print(rt[["scale", "lab", "status", "fit_seconds"]].sort_values(["scale", "lab"]).to_string(index=False))
