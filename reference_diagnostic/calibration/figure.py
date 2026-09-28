"""Supplementary figure v2: incomplete-reference diagnostic with the depth-robust null
(package default null='auto'). Writes paper/figures/supp_reference_diagnostic_v2.{pdf,png} and the
plotted numbers as results/reference_diagnostic_v2/fig_*.csv. Panels a-d as in
validation/rerun_final/refdiag/figure.py (recomputed with null='auto'); e-g calibration."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

ROOT = Path("/Users/apple/Research/FlashDeconv")
R = ROOT / "results/reference_diagnostic_v2"
RF = ROOT / "results/rerun_final/refdiag"
FIG = ROOT / "paper/figures/supp_reference_diagnostic_v2"
UM_PER_PX = 8.0 / 29.2145
N_GENES, N_TYPES = 8, 4

plt.rcParams.update({
    "font.family": "Arial", "font.size": 6, "axes.linewidth": 0.5, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False,
    "axes.spines.right": False, "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300,
})
C_HABER, C_COMP = "#D55E00", "#0072B2"
C_OLD, C_NEW = "#999999", "#0072B2"
C_CLASS = {"rare": "#E69F00", "moderate": "#009E73", "abundant": "#CC79A7"}

pb = pd.read_parquet(R / "pkg_intestine_perbin.parquet")
reg = pd.read_csv(R / "pkg_intestine_region_flags.csv")
reg = reg[(reg.null == "auto") & (reg.key == "score_pooled")]
gen = pd.read_csv(RF / "intestine_unexplained_genes_default.csv")
typ = pd.read_csv(RF / "intestine_suggested_types_default.csv")
v1 = pd.read_csv(R / "pkg_spotless_v1.csv")
dist = pd.read_csv(RF / "spotless_removal_auroc.csv")
dist = dist[dist.thr == 0.3][["ds", "pattern", "removed", "distinct"]]
sp = v1[v1.null == "auto"].merge(dist, on=["ds", "pattern", "removed"])
sp["class"] = np.where(sp.mean_prop < 0.05, "rare", np.where(sp.mean_prop <= 0.15, "moderate", "abundant"))
v3 = pd.read_csv(R / "summary_v3_fpr_by_depth_decile.csv")
crc = pd.read_csv(R / "summary_crc.csv")
diag = pd.read_csv(R / "summary_diagnosis_raw_score_by_depth.csv")

fig = plt.figure(figsize=(7.2, 8.4))
gs_top = fig.add_gridspec(1, 2, left=0.10, right=0.80, top=0.975, bottom=0.66,
                          width_ratios=[1.5, 0.75], wspace=0.25)
gs_mid = fig.add_gridspec(1, 5, left=0.085, right=0.985, top=0.57, bottom=0.40,
                          width_ratios=[1, 1, 1, 1, 1.45], wspace=1.0)
gs_bot = fig.add_gridspec(1, 3, left=0.07, right=0.985, top=0.30, bottom=0.06, wspace=0.45)
letters = []

# a: spatial map of pooled flags (Haber reference)
ax = fig.add_subplot(gs_top[0])
x, y = pb.x_px.to_numpy() * UM_PER_PX / 1000, pb.y_px.to_numpy() * UM_PER_PX / 1000
f = pb.haber_auto_flag_pooled.to_numpy(bool)
ax.scatter(x[~f], -y[~f], s=0.05, c="#D9D9D9", linewidths=0, rasterized=True)
ax.scatter(x[f], -y[f], s=0.05, c=C_HABER, linewidths=0, rasterized=True)
ax.set_aspect("equal")
ax.axis("off")
x0, y0 = x.min(), -y.max()
ax.plot([x0, x0 + 1.0], [y0 - 0.12, y0 - 0.12], color="black", lw=1, solid_capstyle="butt")
ax.text(x0 + 0.5, y0 - 0.2, "1 mm", ha="center", va="top", fontsize=6)
ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=4, color=C_HABER, label="Flagged"),
                   Line2D([], [], marker="s", ls="", ms=4, color="#D9D9D9", label="Not flagged")],
          frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02), handletextpad=0.2)
ax.set_title("Epithelium-only reference (Haber)", fontsize=7, pad=2)
letters.append((ax, "a", -4))
pd.DataFrame({"x_um": x * 1000, "y_um": y * 1000, "flag_pooled_haber": f}).to_csv(
    R / "fig_a_spatial_flags_haber.csv.gz", index=False, float_format="%.4g")

# b: flagged fraction by region
ax = fig.add_subplot(gs_top[1])
order = [("follicle", "Follicle"), ("muscle", "Muscle"), ("epithelium", "Epithelium")]
w = 0.38
brow = []
for i, (ref, c, lab) in enumerate((("haber", C_HABER, "Epithelium-only"), ("composite", C_COMP, "Composite"))):
    r = reg[reg.ref == ref].iloc[0]
    vals = [100 * r[f"flag_{o[0]}"] for o in order]
    ax.bar(np.arange(3) + (i - 0.5) * w, vals, w, color=c, label=lab, lw=0)
    brow += [{"reference": ref, "region": o[0], "flagged_pct": v} for o, v in zip(order, vals)]
pd.DataFrame(brow).to_csv(R / "fig_b_flagged_fraction_by_region.csv", index=False)
ax.set_xticks(range(3), [o[1] for o in order])
ax.set_ylabel("Bins flagged (%)")
ax.axhline(5, color="grey", lw=0.5, ls=":")
ax.set_ylim(0, 25)
ax.legend(frameon=False, loc="upper right", title="Reference", title_fontsize=6)
ax.set_xlim(-0.6, 2.6)
letters.append((ax, "b", -26))

# c: unexplained genes and suggested types (region-defined bin sets)
crow = []
for j, (region, ttl) in enumerate((("follicle", "Follicle"), ("muscle", "Muscle"))):
    st = f"{region}_region"
    g = gen[(gen.ref == "haber") & (gen.set == st)].sort_values("rank").head(N_GENES)
    t = typ[(typ.ref == "haber") & (typ.set == st)].sort_values("rank").head(N_TYPES)
    axg = fig.add_subplot(gs_mid[2 * j])
    axg.barh(np.arange(len(g))[::-1], g.score, color=C_HABER, height=0.7, lw=0)
    axg.set_yticks(np.arange(len(g))[::-1], g.gene, fontstyle="italic")
    axg.set_xlabel("Unexplained score\n(log$_2$ O/E, in − out)", fontsize=6)
    axg.set_title(f"{ttl}: top genes", fontsize=7, pad=3)
    if j == 0:
        letters.append((axg, "c", -44))
    axt = fig.add_subplot(gs_mid[2 * j + 1])
    axt.barh(np.arange(len(t))[::-1], t.mean_score, color="#7F7F7F", height=0.6, lw=0)
    axt.set_yticks(np.arange(len(t))[::-1], [s[0].upper() + s[1:] for s in t.type])
    axt.axvline(0, color="black", lw=0.5)
    axt.set_xlabel("Mean marker score", fontsize=6)
    axt.set_title(f"{ttl}: suggested types", fontsize=7, pad=3)
    crow += [{"region": region, "kind": "gene", "rank": r["rank"], "name": r.gene, "score": r.score}
             for _, r in g.iterrows()]
    crow += [{"region": region, "kind": "type", "rank": r["rank"], "name": r.type, "score": r.mean_score}
             for _, r in t.iterrows()]
pd.DataFrame(crow).to_csv(R / "fig_c_genes_types.csv", index=False)

# d: controlled removal AUROC vs distinctness
ax = fig.add_subplot(gs_mid[4])
for cls in ("rare", "moderate", "abundant"):
    d = sp[sp["class"] == cls]
    ax.scatter(d.distinct, d.auroc, s=5, c=C_CLASS[cls], linewidths=0, alpha=0.8,
               label=f"{cls.capitalize()} ({len(d)})")
bins = np.quantile(sp.distinct, np.linspace(0, 1, 7))
mid = sp.groupby(pd.cut(sp.distinct, bins, include_lowest=True), observed=True).agg(
    x=("distinct", "median"), y=("auroc", "median"))
ax.plot(mid.x, mid.y, "-", color="black", lw=1, label="Median")
ax.axhline(0.5, color="grey", lw=0.5, ls=":")
ax.set_xscale("log")
ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5], ["0.02", "0.05", "0.1", "0.2", "0.5"])
ax.minorticks_off()
ax.set_ylim(-0.55, 1.02)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
ax.spines["left"].set_bounds(0, 1.0)
ax.set_xlabel("Distinctness")
ax.set_ylabel("AUROC")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.22, 0), handletextpad=0.1, borderaxespad=0.1, markerscale=1.5, handlelength=1.2,
          ncol=2, fontsize=5, columnspacing=0.4)
letters.append((ax, "d", -26))
sp[["ds", "pattern", "removed", "class", "mean_prop", "distinct", "auroc"]].to_csv(
    R / "fig_d_removal_auroc_vs_distinctness.csv", index=False)

# e: raw score vs depth, observed vs simulated from the fitted mixture (CRC P5, pooled)
ax = fig.add_subplot(gs_bot[0])
erow = []
for kind, c, lab in (("sim_raw", "#7F7F7F", "Simulated from fit"), ("obs_raw", C_HABER, "Observed")):
    d = diag[(diag.set == "crc_P5_CRC") & (diag.em == 10) & (diag.pooled) & (diag.kind == kind)]
    d = d.sort_values("median_depth")
    ax.fill_between(d.median_depth, d.q16, d.q84, color=c, alpha=0.25, lw=0)
    ax.plot(d.median_depth, d["median"], "-o", color=c, lw=1, ms=2, label=lab)
    erow += [{"kind": kind, "median_depth": r.median_depth, "median": r["median"], "q16": r.q16,
              "q84": r.q84} for _, r in d.iterrows()]
pd.DataFrame(erow).to_csv(R / "fig_e_raw_score_vs_depth_P5.csv", index=False)
ax.set_xscale("log")
ax.axhline(0, color="grey", lw=0.5, ls=":")
ax.set_xlabel("Pooled UMI depth (selected genes)")
ax.set_ylabel("Raw pooled score")
ax.legend(frameon=False, loc="lower left")
ax.set_title("CRC P5, Flex reference", fontsize=7, pad=3)
letters.append((ax, "e", -26))

# f: complete-reference false-positive rate by depth decile
ax = fig.add_subplot(gs_bot[1])
frow = []
styles = {("c2_8um", "score_pooled"): ("-", "Xenium pseudo-bins, pooled"),
          ("c2_8um", "score"): ("--", "Xenium pseudo-bins"),
          ("spotless", "score"): (":", "Spotless")}
for (st, key), (ls, lab) in styles.items():
    for meth, c in (("left_half", C_OLD), ("auto", C_NEW)):
        d = v3[(v3.set == st) & (v3.key == key) & (v3.null == meth)].sort_values("decile")
        ax.plot(d.decile + 1, 100 * d.fpr, ls, color=c, lw=1)
        frow += [{"set": st, "key": key, "null": meth, "decile": r.decile + 1, "fpr_pct": 100 * r.fpr}
                 for _, r in d.iterrows()]
pd.DataFrame(frow).to_csv(R / "fig_f_complete_reference_fpr_by_depth.csv", index=False)
ax.axhline(5, color="black", lw=0.5, ls=":")
ax.set_ylim(0, 12)
ax.set_xticks([1, 5, 10])
ax.set_xlabel("UMI depth decile")
ax.set_ylabel("Bins flagged, complete reference (%)")
h = [Line2D([], [], color=C_OLD, lw=1, label="Left-half null"),
     Line2D([], [], color=C_NEW, lw=1, label="Auto null")]
h += [Line2D([], [], color="black", lw=1, ls=ls, label=lab) for ls, lab in styles.values()]
ax.legend(handles=h, frameon=False, loc="lower center", ncol=1, fontsize=5, handlelength=1.8)
letters.append((ax, "f", -26))

# g: CRC flagged fraction (pooled) overall and in IFN-gamma hotspots
ax = fig.add_subplot(gs_bot[2])
cp = crc[crc.key == "score_pooled"].set_index(["sample", "null"])
samples = ["P1_CRC", "P2_CRC", "P5_CRC"]
ifn = [c for c in crc.columns if c.startswith("hot_region_1_flagged")][0]
grow = []
w = 0.2
for i, (meth, c) in enumerate((("left_half", C_OLD), ("auto", C_NEW))):
    a = [100 * cp.loc[(s, meth), "flag_all"] for s in samples]
    b = [100 * cp.loc[(s, meth), ifn] for s in samples]
    xs = np.arange(3)
    ax.bar(xs + (2 * i - 1.5) * w, a, w, color=c, lw=0)
    ax.bar(xs + (2 * i - 0.5) * w, b, w, color=c, lw=0, hatch="////", edgecolor="white")
    grow += [{"sample": s, "null": meth, "all_bins_pct": u, "ifn_hotspot_pct": v}
             for s, u, v in zip(samples, a, b)]
pd.DataFrame(grow).to_csv(R / "fig_g_crc_flagged.csv", index=False)
ax.set_xticks(range(3), ["P1", "P2", "P5"])
ax.set_ylabel("Bins flagged, pooled (%)")
ax.axhline(5, color="grey", lw=0.5, ls=":")
h = [Line2D([], [], marker="s", ls="", ms=4, color=C_OLD, label="Left-half null"),
     Line2D([], [], marker="s", ls="", ms=4, color=C_NEW, label="Auto null"),
     plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", lw=0.3, label="All bins"),
     plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", lw=0.3, hatch="////",
                   label="IFN-$\\gamma$ hotspots")]
ax.legend(handles=h, frameon=False, loc="upper left", fontsize=5)
ax.set_title("CRC, Flex reference", fontsize=7, pad=3)
letters.append((ax, "g", -26))

for a, l, dx in letters:
    a.text(0, 1, l, transform=a.transAxes + ScaledTranslation(dx / 72, 7 / 72, fig.dpi_scale_trans),
           fontsize=8, fontweight="bold", va="bottom")
fig.savefig(str(FIG) + ".pdf")
fig.savefig(str(FIG) + ".png")
print("saved", FIG)
