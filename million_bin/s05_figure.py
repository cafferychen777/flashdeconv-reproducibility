"""S1 step 5: draft figure (Nature style) for the million-bin MERFISH-derived benchmark.

Usage: python s05_figure.py <results_dir> <out_prefix>
Panels: accuracy on common bins (flattened Pearson, JSD, mean per-type Pearson, mean per-type AP,
rare-type Pearson) by method and scale for the self- and external references; coverage; runtime
and peak memory; accuracy vs bin depth; completion matrix.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

RES = Path(sys.argv[1])
OUT = sys.argv[2]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6, "axes.labelsize": 6.5, "axes.titlesize": 6.5, "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5, "legend.fontsize": 5.5, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300,
})
STYLE = {  # label: (color, linestyle, marker)
    "FlashDeconv": ("#D55E00", "-", "o"),
    "RCTD (doublet)": ("#0072B2", "-", "s"),
    "RCTD (full)": ("#56B4E9", "-", "D"),
    "CARD": ("#009E73", "-", "^"),
    "Cell2location": ("#CC79A7", "-", "v"),
    "RCTD (doublet, UMI_min=20)": ("#0072B2", ":", "s"),
    "CARD (minCountGene=20)": ("#009E73", ":", "^"),
}
PRIMARY = ["FlashDeconv", "RCTD (doublet)", "RCTD (full)", "CARD", "Cell2location"]
REFS = {"selfref": "MERFISH self-reference", "extref": "External scRNA-seq reference"}
SCALE_LAB = {10_000: "10$^4$", 100_000: "10$^5$", 210_499: "2.1×10$^5$", 1_000_000: "10$^6$"}

summ = pd.read_csv(RES / "s1_summary.csv")
summ = summ[(summ["gt"] == "cell")]
rt = pd.concat([pd.read_csv(RES / "s1_runtime.csv")]
               + [pd.read_csv(f) for f in sorted((RES / "c2l_aces").glob("*runtime*.csv"))],
               ignore_index=True).drop_duplicates(["method", "mode", "scale"], keep="last")
rt["ref"] = rt["mode"].str.split("-").str[0]
rt["variant"] = rt["mode"].str.split("-", n=1).str[1]
VLAB = {("flashdeconv", "default"): "FlashDeconv", ("rctd", "doublet"): "RCTD (doublet)",
        ("rctd", "full"): "RCTD (full)", ("card", "default"): "CARD",
        ("cell2location", "fullbatch"): "Cell2location", ("cell2location", "minibatch"): "Cell2location",
        ("rctd", "doublet_umi20"): "RCTD (doublet, UMI_min=20)", ("card", "default_min20"): "CARD (minCountGene=20)"}
rt["label"] = [VLAB.get((m, v), f"{m}:{v}") for m, v in zip(rt.method, rt.variant)]
depth = pd.read_csv(RES / "s1_depth.csv") if (RES / "s1_depth.csv").exists() else None

fig = plt.figure(figsize=(7.2, 6.9))
gs = fig.add_gridspec(4, 4, left=0.07, right=0.985, top=0.965, bottom=0.105, wspace=0.42, hspace=0.62)
trans = ScaledTranslation(-22 / 72, 6 / 72, fig.dpi_scale_trans)
letters = iter("abcdefghijklmnop")


def label(ax):
    ax.text(0, 1, next(letters), transform=ax.transAxes + trans, fontsize=8, fontweight="bold", va="bottom")


def scale_axis(ax, scales):
    ax.set_xscale("log")
    ax.set_xticks(scales)
    ax.set_xticklabels([SCALE_LAB.get(s, str(s)) for s in scales])
    ax.minorticks_off()
    ax.set_xlim(min(scales) / 1.8, max(scales) * 1.8)


def acc_panel(ax, ref, metric, ylab, binset="common", labels=PRIMARY):
    d = summ[(summ.ref == ref) & (summ.binset == binset)]
    scales = sorted(d.scale.unique())
    for lab in labels:
        x = d[d.label == lab].sort_values("scale")
        if x.empty:
            continue
        c, ls, mk = STYLE[lab]
        ax.plot(x.scale, x[metric], ls=ls, color=c, marker=mk, ms=3, lw=1.0, mew=0)
    scale_axis(ax, scales)
    ax.set_ylabel(ylab)
    ax.set_title(REFS[ref], pad=3)
    label(ax)


row_metrics = [("flat_pearson", "Flattened Pearson r"), ("jsd", "Mean JSD (lower is better)"),
               ("mean_type_pearson", "Mean per-type Pearson r"), ("mean_type_ap", "Mean per-type AP")]
for i, (metric, ylab) in enumerate(row_metrics):
    for j, ref in enumerate(REFS):
        r, c = divmod(i * 2 + j, 4)
        acc_panel(fig.add_subplot(gs[r, c]), ref, metric, ylab)

# row 3: rare-type Pearson (self, ext), coverage (self, ext)
for j, ref in enumerate(REFS):
    acc_panel(fig.add_subplot(gs[2, j]), ref, "rare_type_pearson", "Rare-type mean Pearson r")
for j, ref in enumerate(REFS):
    ax = fig.add_subplot(gs[2, 2 + j])
    d = summ[(summ.ref == ref) & (summ.binset == "own")]
    for lab in STYLE:
        x = d[d.label == lab].sort_values("scale")
        if x.empty:
            continue
        c, ls, mk = STYLE[lab]
        ax.plot(x.scale, 100 * x.coverage, ls=ls, color=c, marker=mk, ms=3, lw=1.0, mew=0)
    scale_axis(ax, sorted(d.scale.unique()))
    ax.set_ylim(0, 105)
    ax.set_ylabel("Bins with a prediction (%)")
    ax.set_title(REFS[ref], pad=3)
    label(ax)

# row 4: runtime, memory, depth, completion
ax_t = fig.add_subplot(gs[3, 0])
ax_m = fig.add_subplot(gs[3, 1])
for lab in STYLE:
    x = rt[(rt.label == lab) & (rt.ref == "selfref") & (rt.scale > 0)].sort_values("scale")
    if x.empty:
        continue
    c, ls, mk = STYLE[lab]
    ok = x[x.status == "OK"]
    ax_t.plot(ok.scale, ok.fit_seconds / 60, ls=ls, color=c, marker=mk, ms=3, lw=1.0, mew=0)
    ax_m.plot(ok.scale, ok.peak_rss_gb,  # host memory (PSS) for every method
              ls=ls, color=c, marker=mk, ms=3, lw=1.0, mew=0)
    bad = x[x.status != "OK"]
    for _, b in bad.iterrows():  # not completed: drawn at the limit that was hit
        if b.status == "DNF":
            ax_t.plot(b.scale, 24 * 60, marker="X", color=c, ms=5, mew=0)
        else:  # OOM (or error): at the 500 GB memory limit
            lim = float(b.mem_limit_gb) if str(b.mem_limit_gb) not in ("", "nan") else 500.0
            ax_m.plot(b.scale, lim, marker="X", color=c, ms=5, mew=0)
for ax, yl in [(ax_t, "Fit time (min)"), (ax_m, "Peak memory (GB)")]:
    ax.set_yscale("log")
    scale_axis(ax, [10_000, 100_000, 1_000_000])
    ax.set_ylabel(yl)
    ax.set_xlabel("Bins")
    ax.set_title("Self-reference runs", pad=3)
    label(ax)
ax_t.axhline(24 * 60, color="0.6", lw=0.5, ls="--")
ax_m.axhline(500, color="0.6", lw=0.5, ls="--")

ax_d = fig.add_subplot(gs[3, 2])
if depth is not None and len(depth):
    dd = depth[(depth.ref == "selfref") & (depth.scale == 100_000)]
    order = ["<50", "50-100", "100-200", "200-400", ">=400"]
    for lab in PRIMARY + ["RCTD (doublet, UMI_min=20)", "CARD (minCountGene=20)"]:
        x = dd[dd.label == lab].set_index("umi_bin").reindex(order)
        if x.jsd.notna().sum() == 0:
            continue
        c, ls, mk = STYLE[lab]
        ax_d.plot(range(len(order)), x.jsd, ls=ls, color=c, marker=mk, ms=3, lw=1.0, mew=0)
    ax_d.set_xticks(range(len(order)))
    ax_d.set_xticklabels(order, rotation=30)
    ax_d.set_xlabel("Bin UMI")
    ax_d.set_ylabel("Mean JSD")
    ax_d.set_title("Self-reference, 10$^5$ bins", pad=3)
label(ax_d)

ax_c = fig.add_subplot(gs[3, 3])
rows = [("FlashDeconv", "flashdeconv", ["default"]), ("RCTD (doublet)", "rctd", ["doublet"]),
        ("RCTD (full)", "rctd", ["full"]), ("CARD", "card", ["default"]),
        ("Cell2location", "cell2location", ["fullbatch", "minibatch"])]
cols = [("selfref", 10_000), ("selfref", 100_000), ("selfref", 1_000_000),
        ("extref", 10_000), ("extref", 100_000), ("extref", 210_499)]
SYM = {"OK": ("o", "#333333"), "OOM": ("X", "#000000"), "DNF": ("X", "#7F7F7F"), "ERROR": ("X", "#BBBBBB")}
for i, (lab, m, vs) in enumerate(rows):
    for j, (ref, sc) in enumerate(cols):
        x = rt[(rt.method == m) & (rt.ref == ref) & (rt.scale == sc) & (rt.variant.isin(vs))]
        if x.empty:
            ax_c.plot(j, i, marker="o", mfc="white", mec="#BBBBBB", ms=4, mew=0.6)
            continue
        mk, col = SYM.get(x.status.iloc[-1], ("X", "#999999"))
        ax_c.plot(j, i, marker=mk, color=col, ms=4.5 if mk == "X" else 4, mew=0)
ax_c.set_yticks(range(len(rows)))
ax_c.set_yticklabels([r[0] for r in rows])
ax_c.set_xticks(range(len(cols)))
ax_c.set_xticklabels([SCALE_LAB[s] + ("\nself" if r == "selfref" else "\next.") for r, s in cols])
ax_c.set_xlim(-0.6, len(cols) - 0.4)
ax_c.set_ylim(len(rows) - 0.5, -0.5)
ax_c.axvline(2.5, color="0.7", lw=0.5)
ax_c.tick_params(length=0)
for s in ["left", "bottom"]:
    ax_c.spines[s].set_visible(False)
ax_c.set_title("Run completion", pad=3)
label(ax_c)

handles = [Line2D([], [], color=STYLE[k][0], ls=STYLE[k][1], marker=STYLE[k][2], ms=3, lw=1.0, mew=0, label=k)
           for k in STYLE]
handles += [Line2D([], [], ls="", marker="o", color="#333333", ms=3.5, label="Completed"),
            Line2D([], [], ls="", marker="X", color="#000000", ms=4, label="Not completed: out of memory"),
            Line2D([], [], ls="", marker="X", color="#7F7F7F", ms=4, label="Not completed: >24 h"),
            Line2D([], [], ls="", marker="o", mfc="white", mec="#BBBBBB", ms=3.5, label="Not run / pending")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0),
           handlelength=2.2, columnspacing=1.2)
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
print("saved", OUT)
