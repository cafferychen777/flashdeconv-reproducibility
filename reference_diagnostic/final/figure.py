"""Supplementary figure: incomplete-reference diagnostic (final package, default settings).
Writes paper/figures/supp_reference_diagnostic.{pdf,png} and the plotted numbers as CSVs in
results/rerun_final/refdiag/fig_*.csv."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path("/Users/apple/Research/FlashDeconv")
R = ROOT / "results/rerun_final/refdiag"
FIG = ROOT / "paper/figures/supp_reference_diagnostic"
SET = "default"
GENE_SET = {"follicle": "follicle_region", "muscle": "muscle_region"}
N_GENES, N_TYPES = 8, 4
UM_PER_PX = 8.0 / 29.2145  # 8 um bin pitch = 29.21 px (median nearest-neighbour distance)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 6, "axes.linewidth": 0.5, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False,
    "axes.spines.right": False, "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300,
})
C_HABER, C_COMP = "#D55E00", "#0072B2"
C_CLASS = {"rare": "#E69F00", "moderate": "#009E73", "abundant": "#CC79A7"}

# ---------------- data ----------------
pb = pd.read_parquet(R / f"intestine_perbin_{SET}.parquet")
reg = pd.read_csv(R / f"intestine_region_flags_{SET}.csv")
gen = pd.read_csv(R / f"intestine_unexplained_genes_{SET}.csv")
typ = pd.read_csv(R / f"intestine_suggested_types_{SET}.csv")
sp = pd.read_csv(R / "spotless_removal_auroc.csv")
sp = sp[(sp.thr == 0.3) & sp.auroc.notna()].copy()

fig = plt.figure(figsize=(7.2, 5.0))
gs_top = fig.add_gridspec(1, 3, left=0.01, right=0.985, top=0.895, bottom=0.50,
                          width_ratios=[1.05, 0.62, 1.0], wspace=0.36)
gs_bot = fig.add_gridspec(1, 4, left=0.10, right=0.985, top=0.35, bottom=0.095, wspace=1.05)

# ---------------- a: spatial map ----------------
ax = fig.add_subplot(gs_top[0])
x, y = pb.x_px.to_numpy() * UM_PER_PX / 1000, pb.y_px.to_numpy() * UM_PER_PX / 1000
f = pb.haber_flag_pooled.to_numpy(bool)
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
ax.set_title("Epithelium-only reference (Haber)", fontsize=7, pad=8)
pd.DataFrame({"x_um": x * 1000, "y_um": y * 1000, "flag_pooled_haber": f,
              "score_pooled_haber": pb.haber_score_pooled.to_numpy()}).to_csv(
    R / "fig_a_spatial_flags_haber.csv.gz", index=False, float_format="%.4g")

# ---------------- b: flagged fraction by region ----------------
ax = fig.add_subplot(gs_top[1])
order = [("follicle", "Follicle"), ("muscle", "Muscle"), ("epithelium", "Epithelium")]
w = 0.38
rows = []
for i, (ref, c, lab) in enumerate((("haber", C_HABER, "Epithelium-only"),
                                  ("composite", C_COMP, "Composite"))):
    v = reg[reg.ref == ref].set_index("region").reindex([o[0] for o in order])
    ax.bar(np.arange(3) + (i - 0.5) * w, 100 * v.flag_pooled, w, color=c, label=lab, lw=0)
    for rn, row in v.iterrows():
        rows.append({"reference": ref, "region": rn, "n_bins": row.n_bins,
                     "flagged_fraction": row.flag_pooled})
pd.DataFrame(rows).to_csv(R / "fig_b_flagged_fraction_by_region.csv", index=False)
ax.set_xticks(range(3), [o[1] for o in order])
ax.set_ylabel("Bins flagged (%)")
ax.axhline(5, color="grey", lw=0.5, ls=":")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1,
          handlelength=0.8, handletextpad=0.3, borderaxespad=0.0)
ax.set_xlim(-0.6, 2.6)
ax.tick_params(axis="x", labelrotation=35)

# ---------------- d: controlled removal ----------------
ax = fig.add_subplot(gs_top[2])
for cls in ("rare", "moderate", "abundant"):
    d = sp[sp["class"] == cls]
    ax.scatter(d.distinct, d.auroc, s=5, c=C_CLASS[cls], linewidths=0, alpha=0.8,
               label=f"{cls.capitalize()} ({len(d)})")
bins = np.quantile(sp.distinct, np.linspace(0, 1, 7))
mid = sp.groupby(pd.cut(sp.distinct, bins, include_lowest=True), observed=True).agg(
    x=("distinct", "median"), y=("auroc", "median"))
ax.plot(mid.x, mid.y, "-", color="black", lw=1)
ax.axhline(0.5, color="grey", lw=0.5, ls=":")
ax.set_xscale("log")
ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5], ["0.02", "0.05", "0.1", "0.2", "0.5"])
ax.minorticks_off()
ax.set_ylim(0.15, 1.02)
ax.set_xlabel("Distinctness of removed type")
ax.set_ylabel("AUROC")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3,
          handletextpad=0.0, columnspacing=0.6, borderaxespad=0.0, markerscale=1.5)
sp[["ds", "pattern", "removed", "class", "mean_prop", "distinct", "n_pos", "n_neg", "auroc"]].to_csv(
    R / "fig_d_removal_auroc_vs_distinctness.csv", index=False)
mid.to_csv(R / "fig_d_binned_median.csv")
rho, p = spearmanr(sp.distinct, sp.auroc)
print(f"panel d: n={len(sp)} Spearman rho={rho:.3f} p={p:.2g}")

# ---------------- c: unexplained genes and suggested types ----------------
crow = []
for j, (region, ttl) in enumerate((("follicle", "Follicle"), ("muscle", "Muscle"))):
    st = GENE_SET[region]
    g = gen[(gen.ref == "haber") & (gen.set == st)].sort_values("rank").head(N_GENES)
    t = typ[(typ.ref == "haber") & (typ.set == st)].sort_values("rank").head(N_TYPES)
    axg = fig.add_subplot(gs_bot[2 * j])
    axg.barh(np.arange(len(g))[::-1], g.score, color=C_HABER, height=0.7, lw=0)
    axg.set_yticks(np.arange(len(g))[::-1], g.gene, fontstyle="italic")
    axg.set_xlabel("Unexplained score\n(log$_2$ O/E, in − out)", fontsize=6)
    axg.set_title(f"{ttl}: top genes", fontsize=7, pad=3)
    axt = fig.add_subplot(gs_bot[2 * j + 1])
    axt.barh(np.arange(len(t))[::-1], t.mean_score, color="#7F7F7F", height=0.6, lw=0)
    axt.set_yticks(np.arange(len(t))[::-1], [s[0].upper() + s[1:] for s in t.type])
    axt.axvline(0, color="black", lw=0.5)
    axt.set_xlabel("Mean marker score", fontsize=6)
    axt.set_title(f"{ttl}: suggested types", fontsize=7, pad=3)
    for _, r in g.iterrows():
        crow.append({"region": region, "bin_set": st, "kind": "gene", "rank": r["rank"],
                     "name": r.gene, "score": r.score, "observed": r.observed, "expected": r.expected})
    for _, r in t.iterrows():
        crow.append({"region": region, "bin_set": st, "kind": "type", "rank": r["rank"],
                     "name": r.type, "score": r.mean_score, "n_markers_scored": r.n_markers_scored})
pd.DataFrame(crow).to_csv(R / "fig_c_genes_types.csv", index=False)

# ---------------- panel letters ----------------
axes = fig.axes
letters = {axes[0]: ("a", 0), axes[1]: ("b", -28), axes[2]: ("c", -28), axes[3]: ("d", -52)}
for a, (l, dx) in letters.items():
    a.text(0, 1, l, transform=a.transAxes + ScaledTranslation(dx / 72, (7 if l == "d" else 26) / 72, fig.dpi_scale_trans),
           fontsize=8, fontweight="bold", va="bottom")
fig.savefig(str(FIG) + ".pdf")
fig.savefig(str(FIG) + ".png")
print("saved", FIG)
