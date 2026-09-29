"""Figure 3: FlashDeconv scales linearly to a million bins without loss of accuracy.

One canvas, three rows.
  Row 1 (S1, MERFISH-derived mouse intestine, MERFISH self-reference):
    a  Pearson r with the true cell fractions vs bins (bins estimated by every completed method)
    b  mean per-bin JSD vs bins (same bins)
    c  fraction of bins with estimates vs bins
    d  mean JSD by bin UMI depth, 10^5 bins (each method's own bins)
  Row 2 (S1):
    e  dominant compartment per 8-um bin in a 1.5 x 0.62 mm window of an SPF ileum section,
       ground truth vs FlashDeconv fitted on all 10^6 bins
    f  accuracy vs fitting time, self-reference, 10^5 bins
  Row 3 (C1, pooled CRC Visium HD bins, 18,082 genes, 38 types):
    g  fitting time vs bins      h  peak memory vs bins      i  fraction of bins scored

The external scRNA-seq-reference accuracy panel is Supplementary Fig.
(validation/figures/supp/supp_s1_external_reference.py).

Not-completed runs are drawn with an 'X' at the limit they hit (500 GB memory line, 24 h time
line, or 0% coverage). Cell2location points are filled for runs on an NVIDIA GH200 GPU and open
for runs on an NVIDIA A30 GPU.

Inputs (all optional except the two main tables; missing methods are skipped):
  results/s1_merfish_benchmark/{s1_summary,s1_runtime,s1_depth}.csv (+ c2l_aces/*runtime*.csv)
  results/s1_merfish_benchmark/fig3_spatial_window_1e6.csv.gz
    (validation/s1_merfish_benchmark/s06_spatial_window.py)
  results/rerun_final/benchmarks/c1/c1_runtime_table_final.csv (all methods, incl. Cell2location)
  results/rerun_final/benchmarks/c1/c1_results_merged_final.csv (bins fitted per run)

Usage:
  python validation/figures/fig3_scale.py            # draw from local tables
  python validation/figures/fig3_scale.py --fetch    # first copy S1 tables from arseven
Prints every number used in the manuscript text.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from style import *  # noqa: E402,F401,F403

apply_style()

S1 = RESULTS / "s1_merfish_benchmark"
C1 = RESULTS / "rerun_final" / "benchmarks" / "c1"
REMOTE_S1 = "arseven:/scratch/user/cafferychen777/FlashDeconv/results/s1_merfish_benchmark/"

if "--fetch" in sys.argv:
    S1.mkdir(parents=True, exist_ok=True)
    for f in ["s1_summary.csv", "s1_runtime.csv", "s1_depth.csv", "s1_paired.csv",
              "s1_nested_1e6_on_subsets.csv", "s1_per_type.csv"]:
        subprocess.run(["scp", "-q", REMOTE_S1 + f, str(S1 / f)], check=False)
    subprocess.run(["rsync", "-a", "--include=*/", "--include=*.csv", "--exclude=*",
                    REMOTE_S1 + "c2l_aces/", str(S1 / "c2l_aces") + "/"], check=False)

MEM_LIMIT_GB = 500
TIME_LIMIT_S = 24 * 3600

# label: (colour, linestyle, marker)
STY = {
    "FlashDeconv": (METHOD_COLORS["FlashDeconv"], "-", "o"),
    "RCTD (doublet)": (METHOD_COLORS["RCTD (doublet)"], RCTD_LS["doublet"], RCTD_MARKERS["doublet"]),
    "RCTD (full)": (METHOD_COLORS["RCTD (full)"], RCTD_LS["full"], RCTD_MARKERS["full"]),
    "CARD": (METHOD_COLORS["CARD"], "-", "^"),
    "Cell2location": (METHOD_COLORS["Cell2location"], "-", "v"),
}
ORDER = list(STY)
MS = {"o": 3.2, "s": 2.8, "D": 2.5, "^": 3.2, "v": 3.2}

S1_LAB = {("flashdeconv", "default"): "FlashDeconv", ("rctd", "doublet"): "RCTD (doublet)",
          ("rctd", "full"): "RCTD (full)", ("card", "default"): "CARD",
          ("cell2location", "fullbatch"): "Cell2location", ("cell2location", "minibatch"): "Cell2location"}
C1_LAB = {("flashdeconv_final", "default"): "FlashDeconv", ("rctd", "doublet"): "RCTD (doublet)",
          ("rctd", "full"): "RCTD (full)", ("card", "default"): "CARD",
          ("cell2location", "fullbatch"): "Cell2location", ("cell2location", "minibatch"): "Cell2location"}
SCALE_LAB = {10_000: "10$^4$", 100_000: "10$^5$", 210_499: "2.1×10$^5$",
             300_000: "3×10$^5$", 1_000_000: "10$^6$"}


def is_done_bad(status):
    """Terminal not-completed statuses (running / pending runs return None)."""
    s = str(status).upper()
    if s.startswith("OOM"):
        return "OOM"
    if s.startswith(("DNF", "TIMEOUT", "TIME")):
        return "TIME"
    return None


# ---------------------------------------------------------------------------
# S1 tables
# ---------------------------------------------------------------------------
summ = pd.read_csv(S1 / "s1_summary.csv")
summ = summ[summ["gt"] == "cell"]
summ = summ[[(m, v) in S1_LAB for m, v in zip(summ.method, summ.variant)]].copy()
summ["lab"] = [S1_LAB[(m, v)] for m, v in zip(summ.method, summ.variant)]
# Cell2location: keep the full-batch run when both modes exist at a scale
summ["_pri"] = (summ.variant == "fullbatch").astype(int)
summ = (summ.sort_values("_pri").drop_duplicates(["ref", "scale", "lab", "binset"], keep="last"))

rt_files = [S1 / "s1_runtime.csv"] + sorted((S1 / "c2l_aces").glob("*runtime*.csv"))
s1rt = pd.concat([pd.read_csv(f) for f in rt_files if f.exists()], ignore_index=True)
s1rt = s1rt.drop_duplicates(["method", "mode", "scale"], keep="last")
s1rt["ref"] = s1rt["mode"].str.split("-").str[0]
s1rt["variant"] = s1rt["mode"].str.split("-", n=1).str[1]
s1rt = s1rt[[(m, v) in S1_LAB for m, v in zip(s1rt.method, s1rt.variant)]].copy()
s1rt["lab"] = [S1_LAB[(m, v)] for m, v in zip(s1rt.method, s1rt.variant)]
S1_GPU = {(r.ref, r.variant, r.scale): r.gpu_model for r in s1rt.itertuples()
          if r.method == "cell2location" and r.status == "OK"}
summ["gpu_model"] = [S1_GPU.get((r, v, sc)) if m == "cell2location" else None
                     for m, r, v, sc in zip(summ.method, summ.ref, summ.variant, summ.scale)]
depth = pd.read_csv(S1 / "s1_depth.csv") if (S1 / "s1_depth.csv").exists() else None

# ---------------------------------------------------------------------------
# C1 table (median over repeats already taken)
# ---------------------------------------------------------------------------
c1 = pd.read_csv(C1 / "c1_runtime_table_final.csv")
c1 = c1[c1.status != "NOT_RUN"]
c1 = c1[[(m, v) in C1_LAB for m, v in zip(c1.method, c1["mode"])] & (c1.scale > 0)].copy()
c1["lab"] = [C1_LAB[(m, v)] for m, v in zip(c1.method, c1["mode"])]
c1 = c1.sort_values("mode").drop_duplicates(["lab", "scale"], keep="first")  # fullbatch before minibatch
# bins actually fitted (RCTD / CARD drop bins below their count thresholds) from the raw C1 log
C1_RAW = C1 / "c1_results_merged_final.csv"
cov_c1 = None
if C1_RAW.exists():
    # Cell2location returns an estimate for every bin (runs listed only in the final table)
    c2l_ok = c1[(c1.method == "cell2location") & (c1.status == "OK")]
    raw = pd.concat([pd.read_csv(C1_RAW),
                     c2l_ok.assign(n_bins_fit=c2l_ok.scale, notes="")[["method", "mode", "scale", "status",
                                                                        "n_bins_fit", "notes"]]],
                    ignore_index=True)
    raw = raw[raw.status == "OK"].copy()
    raw["lab"] = [C1_LAB.get((m, v)) for m, v in zip(raw.method, raw["mode"])]
    raw = raw.dropna(subset=["lab"])
    # RCTD doublet mode: bins classified 'reject' get no estimate (same rule as S1)
    rej = raw.notes.astype(str).str.extract(r"reject=(\d+)")[0].astype(float).fillna(0)
    raw["n_bins_fit"] = raw.n_bins_fit - np.where(raw.lab == "RCTD (doublet)", rej, 0)
    cov_c1 = raw.groupby(["lab", "scale"]).n_bins_fit.max().reset_index()
    cov_c1["frac"] = cov_c1.n_bins_fit / cov_c1.scale

# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
H = 145.0
fig = new_figure(H)
L, R = 0.07, 0.985
gs1 = fig.add_gridspec(1, 4, left=L, right=R, top=0.895, bottom=0.725, wspace=0.55)
gs3 = fig.add_gridspec(1, 3, left=L, right=R, top=0.295, bottom=0.075, wspace=0.42)
axes = {k: fig.add_subplot(gs1[0, i]) for i, k in enumerate("abcd")}
axes.update({k: fig.add_subplot(gs3[0, i]) for i, k in enumerate("ghi")})
# row 2: e = two maps (truth, FlashDeconv) + compartment legend; f = accuracy vs time
f_x0 = axes["d"].get_position().x0
axes["f"] = fig.add_axes([f_x0, 0.435, axes["d"].get_position().width, 0.18])


def is_a30(g):
    return "A30" in str(g)


def line(ax, x, y, lab, gpu=None, **kw):
    """Method line; for Cell2location, markers are open for A30 runs and filled for GH200 runs."""
    c, ls, mk = STY[lab]
    if gpu is None:
        ax.plot(x, y, color=c, ls=ls, marker=mk, ms=MS[mk], lw=0.9, mew=0, clip_on=False, **kw)
        return
    x, y, gpu = np.asarray(x), np.asarray(y), list(gpu)
    ax.plot(x, y, color=c, ls=ls, lw=0.9, clip_on=False, **kw)
    for xi, yi, g in zip(x, y, gpu):
        a30 = is_a30(g)
        ax.plot(xi, yi, ls="", marker=mk, ms=MS[mk] + (0.4 if a30 else 0), mfc="white" if a30 else c,
                mec=c, mew=0.7 if a30 else 0, clip_on=False, zorder=4)


XOFF = {"CARD": 0.72, "RCTD (doublet)": 0.85, "RCTD (full)": 1.0, "Cell2location": 1.18, "FlashDeconv": 1.0}


def xmark(ax, x, y, lab):
    """Not-completed run; small horizontal offsets keep coincident marks visible (log x)."""
    ax.plot(x * XOFF[lab], y, ls="", marker="X", color=STY[lab][0], ms=4.5, mew=0.3, mec="white",
            clip_on=False, zorder=5)


def scale_axis(ax, scales, pad=1.6):
    ax.set_xscale("log")
    ax.set_xticks(scales)
    ax.set_xticklabels([SCALE_LAB.get(s, f"{s:.0e}") for s in scales])
    ax.minorticks_off()
    ax.set_xlim(min(scales) / pad, max(scales) * pad)
    ax.set_xlabel("Bins")


S1_SCALES_SELF = [10_000, 100_000, 1_000_000]


def s1_metric(ax, ref, metric, ylab, binset="common"):
    d = summ[(summ.ref == ref) & (summ.binset == binset)]
    for lab in ORDER:
        x = d[d.lab == lab].sort_values("scale")
        if len(x):
            line(ax, x.scale, x[metric], lab, gpu=x.gpu_model if lab == "Cell2location" else None)
    scale_axis(ax, S1_SCALES_SELF)
    ax.set_ylabel(ylab)


# a, b: accuracy, self-reference, common bins
s1_metric(axes["a"], "selfref", "flat_pearson", "Pearson r")
axes["a"].set_title("MERFISH reference, shared bins")
s1_metric(axes["b"], "selfref", "jsd", "Mean JSD")
axes["b"].set_title("MERFISH reference, shared bins")

# c: coverage, self-reference (OOM / time-out -> X at 0 %)
ax = axes["c"]
d = summ[(summ.ref == "selfref") & (summ.binset == "own")]
for lab in ORDER:
    x = d[d.lab == lab].sort_values("scale")
    if len(x):
        line(ax, x.scale, 100 * x.coverage, lab, gpu=x.gpu_model if lab == "Cell2location" else None)
    for _, b in s1rt[(s1rt.lab == lab) & (s1rt.ref == "selfref")].iterrows():
        if is_done_bad(b.status):
            xmark(ax, b.scale, 0, lab)
scale_axis(ax, S1_SCALES_SELF)
ax.set_ylim(0, 105)
ax.set_ylabel("Bins with estimates (%)")
ax.set_title("MERFISH reference")

# d: accuracy by depth, self-reference, 1e5 bins, own bins
ax = axes["d"]
DEPTH_ORDER = ["<50", "50-100", "100-200", "200-400", ">=400"]
DEPTH_TICK = ["<50", "50–\n100", "100–\n200", "200–\n400", "≥400"]
if depth is not None:
    dd = depth[(depth.ref == "selfref") & (depth.scale == 100_000)]
    dd = dd[[(m, v) in S1_LAB for m, v in zip(dd.method, dd.variant)]].copy()
    dd["lab"] = [S1_LAB[(m, v)] for m, v in zip(dd.method, dd.variant)]
    for lab in ORDER:
        x = dd[dd.lab == lab].drop_duplicates("umi_bin", keep="last").set_index("umi_bin").reindex(DEPTH_ORDER)
        if x.jsd.notna().any():
            line(ax, np.arange(len(DEPTH_ORDER)), x.jsd.to_numpy(), lab)
ax.set_xticks(range(len(DEPTH_ORDER)))
ax.set_xticklabels(DEPTH_TICK)
ax.set_xlim(-0.4, len(DEPTH_ORDER) - 0.6)
ax.set_xlabel("UMIs per bin")
ax.set_ylabel("Mean JSD")
ax.set_title("MERFISH reference, 10$^5$ bins")

# f: accuracy vs fitting time, self-reference, 1e5 bins (common bins)
ax = axes["f"]
SC_F = 100_000
dcom = summ[(summ.ref == "selfref") & (summ.binset == "common") & (summ.scale == SC_F)]
for lab in ORDER:
    r = s1rt[(s1rt.lab == lab) & (s1rt.ref == "selfref") & (s1rt.scale == SC_F) & (s1rt.status == "OK")]
    a = dcom[dcom.lab == lab]
    if len(r) and len(a):
        c, _, mk = STY[lab]
        ax.plot(r.fit_seconds.iloc[-1], a.flat_pearson.iloc[-1], ls="", marker=mk, color=c,
                ms=MS[mk] * 1.5, mew=0, clip_on=False)
ax.set_xscale("log")
ax.set_xlabel("Fitting time (s)")
ax.set_ylabel("Pearson r")
ax.set_title("MERFISH reference, 10$^5$ bins")
lo, hi = ax.get_xlim()
ax.set_xlim(lo / 1.6, hi * 1.6)

# e: dominant compartment per bin, truth vs FlashDeconv at 10^6 bins (one window)
COMP = {  # display name: (member types, colour)
    "Enterocyte": (["Enterocyte"], "#E69F00"),
    "Stem/TA": (["Stem/TA"], "#8C510A"),
    "Secretory": (["Goblet", "Paneth", "Tuft", "EEC"], "#CC79A7"),
    "Immune": (["B cell", "Plasma cell", "T/ILC/NK", "Myeloid"], "#0072B2"),
    "Fibroblast": (["Fibroblast"], "#009E73"),
    "Smooth muscle/ICC": (["Smooth muscle", "ICC"], "#B2182B"),
    "Vascular": (["Endothelial", "Pericyte"], "#56B4E9"),
    "Neural": (["Enteric neuron", "Enteric glia"], "#6A3D9A"),
    "Mesothelium": (["Mesothelium"], "#999999"),
}
WIN = S1 / "fig3_spatial_window_1e6.csv.gz"
win_stats = None
e_axes = []
if WIN.exists():
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    w = pd.read_csv(WIN)
    BIN_UM = 8.0
    ix = (w.x // BIN_UM).astype(int).to_numpy()
    iy = (w.y // BIN_UM).astype(int).to_numpy()
    nx, ny = ix.max() + 1, iy.max() + 1

    def dominant(prefix):
        m = np.column_stack([w[[f"{prefix}_{t}" for t in ts]].sum(1) for ts, _ in COMP.values()])
        return m.argmax(1)

    dom = {"Ground truth": dominant("gt"), "FlashDeconv, 10$^6$ bins": dominant("fd")}
    cmap = ListedColormap([c for _, c in COMP.values()])
    cmap.set_bad("white")
    map_w, map_gap, map_top = 0.325, 0.015, 0.615
    h_map = map_w * (FIG_W_MM / H) * (ny / nx)
    y_map = map_top - h_map
    for j, (title, v) in enumerate(dom.items()):
        ax = fig.add_axes([L + j * (map_w + map_gap), y_map, map_w, h_map])
        g = np.full((ny, nx), np.nan)
        g[iy, ix] = v
        ax.imshow(g, cmap=cmap, vmin=-0.5, vmax=len(COMP) - 0.5, interpolation="nearest",
                  extent=[0, nx * BIN_UM, ny * BIN_UM, 0], rasterized=True)
        clean_spatial(ax)
        ax.set_title(title, pad=2)
        e_axes.append(ax)
    scale_bar(e_axes[0], 200, loc="lower left", pad=0.02)
    handles = [Patch(facecolor=c, edgecolor="none", label=n) for n, (_, c) in COMP.items()]
    fig.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=FS_SMALL,
               bbox_to_anchor=(L - 0.006, y_map - 0.008), handlelength=0.9, handleheight=0.9,
               columnspacing=1.2, handletextpad=0.4)
    agree = (dom["Ground truth"] == dom["FlashDeconv, 10$^6$ bins"]).mean()
    win_stats = (len(w), agree)

# g, h: C1 time and memory
C1_SCALES = [10_000, 100_000, 300_000, 1_000_000]
axg, axh = axes["g"], axes["h"]
for lab in ORDER:
    x = c1[c1.lab == lab].sort_values("scale")
    ok = x[x.status == "OK"]
    if len(ok):
        gpu = ok.gpu_model if lab == "Cell2location" else None
        line(axg, ok.scale, ok.median_fit_s, lab, gpu=gpu)
        line(axh, ok.scale, ok.peak_rss_gb, lab, gpu=gpu)
    for _, b in x.iterrows():
        kind = is_done_bad(b.status)
        if kind == "OOM":
            xmark(axh, b.scale, MEM_LIMIT_GB, lab)
        elif kind == "TIME":
            xmark(axg, b.scale, TIME_LIMIT_S, lab)
axg.axhline(TIME_LIMIT_S, color="0.6", lw=0.5, ls=(0, (2, 2)), zorder=0)
axh.axhline(MEM_LIMIT_GB, color="0.6", lw=0.5, ls=(0, (2, 2)), zorder=0)
axg.set_yscale("log")
axg.set_yticks([1, 60, 3600, 86400])
axg.set_yticklabels(["1 s", "1 min", "1 h", "24 h"])
axg.set_ylim(0.7, 2.5e5)
axg.set_ylabel("Fitting time")
axh.set_yscale("log")
axh.set_yticks([1, 10, 100, 500])
axh.set_yticklabels(["1", "10", "100", "500"])
axh.set_ylim(0.8, 900)
axh.set_ylabel("Peak memory (GB)")
for ax in (axg, axh):
    scale_axis(ax, C1_SCALES, pad=1.5)
    ax.set_title("CRC Visium HD")

# i: C1 fraction of bins scored
ax = axes["i"]
if cov_c1 is not None:
    for lab in ORDER:
        x = cov_c1[cov_c1.lab == lab].sort_values("scale")
        if len(x):
            gpu = None
            if lab == "Cell2location":
                gm = c1[c1.lab == lab].set_index("scale").gpu_model
                gpu = [gm.get(sc) for sc in x.scale]
            line(ax, x.scale, 100 * x.frac, lab, gpu=gpu)
for lab in ORDER:  # not completed -> X at 0 %
    for _, b in c1[c1.lab == lab].iterrows():
        if is_done_bad(b.status):
            xmark(ax, b.scale, 0, lab)
scale_axis(ax, C1_SCALES, pad=1.5)
ax.set_ylim(0, 105)
ax.set_ylabel("Bins with estimates (%)")
ax.set_title("CRC Visium HD")

for k, ax in axes.items():
    panel_label(ax, k, dx_mm=-10.5, dy_mm=2.0)
if e_axes:
    panel_label(e_axes[0], "e", dx_mm=-10.5, dy_mm=2.0)

# shared legend (methods + not-completed symbol)
present = [l for l in ORDER if (l in set(summ.lab)) or (l in set(c1[c1.status == "OK"].lab))]
handles = [Line2D([], [], color=STY[l][0], ls=STY[l][1], marker=STY[l][2], ms=MS[STY[l][2]], lw=0.9,
                  mew=0, label=l) for l in present]
handles.append(Line2D([], [], ls="", marker="X", color="0.35", ms=4.5, mew=0.3, mec="white",
                      label="Not completed"))
c2c = STY["Cell2location"][0]
handles.append(Line2D([], [], ls="", marker="v", color=c2c, ms=MS["v"], mew=0, label="GH200 GPU"))
handles.append(Line2D([], [], ls="", marker="v", mfc="white", mec=c2c, ms=MS["v"] + 0.4, mew=0.7,
                      label="A30 GPU"))
fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.995),
           frameon=False, handlelength=2.2, columnspacing=1.6, fontsize=FS)

save(fig, "fig3_scale")

# ---------------------------------------------------------------------------
# Numbers quoted in the text
# ---------------------------------------------------------------------------
pd.set_option("display.width", 200)
cols = ["ref", "scale", "lab", "binset", "n_bins", "coverage", "flat_pearson", "rmse", "jsd",
        "mean_type_pearson"]
print("\n== S1 accuracy (cell-fraction truth) ==")
print(summ[cols].sort_values(["ref", "binset", "scale", "lab"]).round(4).to_string(index=False))
if win_stats:
    print(f"\n== Panel e: {win_stats[0]} bins in window; dominant-compartment agreement {win_stats[1]:.4f} ==")
print("\n== S1 runtime ==")
print(s1rt[["ref", "scale", "lab", "status", "fit_seconds", "peak_rss_gb", "peak_gpu_gb", "n_bins_fit",
            "gpu_model"]].sort_values(["ref", "scale", "lab"]).to_string(index=False))
print("\n== C1 ==")
print(c1[[c for c in ["lab", "scale", "status", "median_fit_s", "peak_rss_gb", "peak_gpu_gb", "gpu_model"] if c in c1]].sort_values(["lab", "scale"])
      .to_string(index=False))
if cov_c1 is not None:
    print(cov_c1.round(4).to_string(index=False))
