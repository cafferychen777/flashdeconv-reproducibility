"""Supplementary Figure: calibration, removal detail, composite reference, lung and CRC detail
for the reference diagnostic (complements main Fig. 4).

Reads result tables directly and draws the whole figure on one canvas.
Output: paper/figures/supp_refdiag.{pdf,png}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import *  # noqa: E402,F401,F403

sys.path.insert(0, str(PROJ))
from flashdeconv.core.refcheck import _central_null  # noqa: E402

apply_style()

RF = RESULTS / "rerun_final" / "refdiag"        # final package runs
RV2 = RESULTS / "reference_diagnostic_v2"        # package calibration runs (null comparison)
B1 = RESULTS / "b1_pilot"                        # lung refcheck
B2 = RESULTS / "b2_crc_refcheck"                 # CRC unexplained-region analysis
COMP = PROJ / "validation/intestine_reference_v2/results/reference/reference_composition.csv"
Z95 = 1.6448536269514722

NULL_STYLE = {  # colour, marker, label
    "left_half": ("#9A9A9A", "o", "Left-half"),
    "central": (OI["skyblue"], "s", "Central matching"),
    "auto": ("#252525", "D", "Auto (smaller scale)"),
}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# complete-reference flag rates under the three nulls (Spotless, Xenium-derived CRC)
nc = pd.read_csv(RF / "complete_reference_null_comparison.csv")
# CRC Visium HD P1/P2/P5 (complete 38-type reference): pooled scores per bin
SAMPLES = ["P1_CRC", "P2_CRC", "P5_CRC"]
crc_rates = {}
for s in SAMPLES:
    d = np.load(RV2 / f"pkg_crc_{s}.npz")
    a = d["auto_score_pooled"]
    mu, sd = _central_null(a)
    crc_rates[s] = {"left_half": np.mean(d["left_half_score_pooled"] > Z95),
                    "central": np.mean((a - mu) / sd > Z95),
                    "auto": np.mean(a > Z95)}

dec = pd.read_csv(RV2 / "summary_v3_fpr_by_depth_decile.csv")
crc_dec = pd.read_csv(RV2 / "summary_crc_by_depth_decile.csv")

rem = pd.read_csv(RF / "spotless_removal_auroc.csv")
nam = pd.read_csv(RF / "spotless_removal_naming.csv")
dist = rem[np.isclose(rem.thr, 0.3)][["ds", "pattern", "removed", "distinct"]]
nam = nam.merge(dist, on=["ds", "pattern", "removed"], how="inner")

comp = pd.read_csv(COMP)
lung = {s: json.load(open(B1 / f"refcheck_summary_{s}.json")) for s in ("LUNG_X1", "LUNG_X5K")}
lung_luad = {s: pd.read_csv(B1 / f"refcheck_topgenes_{s}_luad.csv") for s in ("LUNG_X1", "LUNG_X5K")}
scrc = pd.read_csv(RV2 / "summary_crc.csv")
hall = pd.read_csv(B2 / "hallmark_enrichment.csv")

DBINS = [0, 0.05, 0.1, 0.2, 1.0]
DLAB = ["<0.05", "0.05–0.1", "0.1–0.2", ">0.2"]

# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
H = 214.0
fig = new_figure(H)


def ax_mm(x, y, w, h):
    """Axes with top-left corner at (x, y) mm from the figure's top-left."""
    return fig.add_axes([x / FIG_W_MM, 1 - (y + h) / H, w / FIG_W_MM, h / H])


def letter(ch, x, y):
    fig.text(x / FIG_W_MM, 1 - y / H, ch, fontsize=FS_PANEL, fontweight="bold",
             ha="left", va="top")


def ref5(ax):
    ax.axhline(5, color="black", lw=0.5, ls=(0, (1, 1.5)), zorder=0)


# ---- a: complete-reference flag rate under three nulls --------------------
ROW1_Y, ROW1_H = 3, 38
ax = ax_mm(13, ROW1_Y + 4, 100, ROW1_H - 4)
cats = []
for ds in range(1, 7):
    cats.append((f"DS{ds}", ("spotless", ds)))
cats += [("8 µm", ("c2_8um", None)), ("16 µm", ("c2_16um", None))]
cats += [(s.split("_")[0], ("crc", s)) for s in SAMPLES]
off = {"left_half": -0.22, "central": 0.0, "auto": 0.22}
for i, (_, (st, key)) in enumerate(cats):
    for nl, (c, mk, _) in NULL_STYLE.items():
        if st == "spotless":
            v = 100 * nc[(nc.set == "spotless") & (nc.ds == key) & (nc["null"] == nl)].flag_rate
            ax.plot([i + off[nl]] * 2, [v.min(), v.max()], color=c, lw=0.6, zorder=1)
            y = v.median()
        elif st.startswith("c2"):
            y = 100 * nc[(nc.set == st) & (nc.key == "score_pooled") & (nc["null"] == nl)].flag_rate.iloc[0]
        else:
            y = 100 * crc_rates[key][nl]
        ax.scatter(i + off[nl], y, s=9, marker=mk, color=c, lw=0, zorder=2)
ref5(ax)
ax.set_xticks(range(len(cats)), [c[0] for c in cats])
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.6, len(cats) - 0.4)
ax.set_ylim(0, 14)
ax.set_ylabel("Bins flagged (%)")
for xa, xb, lab in ((-0.4, 5.4, "Spotless silver standards"), (5.6, 7.4, "Xenium-derived CRC"),
                    (7.6, 10.4, "CRC Visium HD")):
    ax.plot([xa, xb], [14.6, 14.6], color="black", lw=0.5, clip_on=False)
    ax.text((xa + xb) / 2, 14.9, lab, ha="center", va="bottom", fontsize=FS_SMALL)
for xv in (5.5, 7.5):
    ax.axvline(xv, color="#D9D9D9", lw=0.5, zorder=0)
ax.legend(handles=[Line2D([], [], ls="", marker=mk, color=c, ms=3, label=lab)
                   for c, mk, lab in NULL_STYLE.values()],
          loc="upper right", bbox_to_anchor=(1.0, 0.98), title="Empirical null",
          title_fontsize=FS_SMALL)
letter("a", 0, ROW1_Y)

# ---- b: false-positive rate by UMI depth decile ---------------------------
BX, BW, BH = 126, 24, 13
subs = [("Spotless", "spotless", "score"), ("CRC 8 µm", "c2_8um", "score_pooled"),
        ("CRC 16 µm", "c2_16um", "score_pooled"), ("CRC Visium HD", "crc", "score_pooled")]
for j, (title, st, key) in enumerate(subs):
    r, c_ = divmod(j, 2)
    axb = ax_mm(BX + c_ * (BW + 4), ROW1_Y + 4 + r * (BH + 8), BW, BH)
    for nl in ("left_half", "auto"):
        col, mk, _ = NULL_STYLE[nl]
        if st == "crc":
            for s, ls in zip(SAMPLES, ("-", "--", ":")):
                d = crc_dec[(crc_dec["sample"] == s) & (crc_dec["null"] == nl)
                            & (crc_dec.key == key)].sort_values("decile")
                axb.plot(d.decile + 1, 100 * d.flag_rate, ls=ls, color=col, lw=0.7)
        else:
            d = dec[(dec.set == st) & (dec.key == key) & (dec["null"] == nl)].sort_values("decile")
            axb.plot(d.decile + 1, 100 * d.fpr, "-", marker=mk, color=col, lw=0.7, ms=1.6)
    ref5(axb)
    axb.set_ylim(0, 12)
    axb.set_xlim(0.5, 10.5)
    axb.set_xticks([1, 5, 10])
    axb.set_yticks([0, 5, 10])
    axb.set_title(title, fontsize=FS_SMALL, pad=1.5)
    if c_ == 0:
        axb.set_ylabel("Flagged (%)", fontsize=FS_SMALL)
    else:
        axb.set_yticklabels([])
    if r == 1:
        axb.set_xlabel("UMI depth decile", fontsize=FS_SMALL)
    else:
        axb.set_xticklabels([])
    if st == "crc":
        axb.legend(handles=[Line2D([], [], color="#555555", ls=ls, lw=0.7, label=s.split("_")[0])
                            for s, ls in zip(SAMPLES, ("-", "--", ":"))],
                   loc="upper left", fontsize=FS_TINY, handlelength=1.6, ncol=1,
                   borderaxespad=0.0, labelspacing=0.1)
letter("b", 118, ROW1_Y)

# ---- c: removal AUROC by abundance class and positive-bin threshold -------
ROW2_Y, ROW2_H = 55, 32
ax = ax_mm(13, ROW2_Y + 3, 44, ROW2_H - 3)
classes = ["rare", "moderate", "abundant"]
thr_col = {0.1: "#D0D0D0", 0.3: "#8A8A8A", 0.5: "#3A3A3A"}
for i, cl in enumerate(classes):
    for j, thr in enumerate((0.1, 0.3, 0.5)):
        v = rem[(rem["class"] == cl) & np.isclose(rem.thr, thr)].auroc.dropna().to_numpy()
        bp = ax.boxplot(v, positions=[i + (j - 1) * 0.27], widths=0.22, patch_artist=True,
                        showfliers=False, whis=(5, 95), medianprops=dict(color="black", lw=0.8),
                        boxprops=dict(lw=0.4), whiskerprops=dict(lw=0.4), capprops=dict(lw=0))
        bp["boxes"][0].set_facecolor(thr_col[thr])
ax.axhline(0.5, color="black", lw=0.5, ls=(0, (1, 1.5)))
ax.set_xticks(range(3), [c.capitalize() for c in classes])
ax.tick_params(axis="x", length=0)
ax.set_xlabel("Abundance of removed type")
ax.set_ylabel("AUROC")
ax.set_ylim(0, 1.05)
ax.set_xlim(-0.55, 2.55)
ax.legend(handles=[Patch(fc=c, ec="black", lw=0.4, label=f"> {t:g}") for t, c in thr_col.items()],
          loc="lower left", title="Removed-type fraction", title_fontsize=FS_TINY,
          fontsize=FS_TINY, ncol=3, columnspacing=0.6, handlelength=0.9, bbox_to_anchor=(0, 0.98))
letter("c", 0, ROW2_Y)

# ---- d: flag rate with / without the removed type, by distinctness --------
r3 = rem[np.isclose(rem.thr, 0.3)].copy()
r3["dbin"] = pd.cut(r3.distinct, DBINS, labels=DLAB, include_lowest=True)
ax = ax_mm(72, ROW2_Y + 3, 40, ROW2_H - 3)
for k, (col, lab, c) in enumerate((("flag_rate_neg", "Removed type < 1%", "#9A9A9A"),
                                    ("flag_rate_pos", "Removed type > 30%", FLAG_COLOR))):
    g = r3.groupby("dbin", observed=False)[col]
    med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
    x = np.arange(len(DLAB)) + (k - 0.5) * 0.28
    ax.vlines(x, 100 * q1, 100 * q3, color=c, lw=1.6, alpha=0.45)
    ax.scatter(x, 100 * med, s=10, color=c, lw=0, zorder=3, label=lab)
ref5(ax)
ax.set_xticks(range(len(DLAB)), DLAB)
ax.tick_params(axis="x", length=0)
ax.set_xlabel("Distinctness of removed type")
ax.set_ylabel("Bins flagged (%)")
ax.set_ylim(0, 100)
ax.set_xlim(-0.5, len(DLAB) - 0.5)
ax.legend(loc="upper left", handletextpad=0.2)
letter("d", 59, ROW2_Y)

# ---- e: naming accuracy by distinctness ------------------------------------
na = nam[nam["null"] == "auto"].copy()
na["dbin"] = pd.cut(na.distinct, DBINS, labels=DLAB, include_lowest=True)
ax = ax_mm(136, ROW2_Y + 3, 42, ROW2_H - 3)
g = na.groupby("dbin", observed=False)["rank"]
top1 = g.apply(lambda x: np.mean(x == 1))
top3 = g.apply(lambda x: np.mean(x <= 3))
x = np.arange(len(DLAB))
ax.bar(x - 0.19, 100 * top1, 0.36, color="#252525", lw=0, label="Ranked first")
ax.bar(x + 0.19, 100 * top3, 0.36, color="#9A9A9A", lw=0, label="Ranked in top 3")
chance = 100 * np.median(1 / na.K)
ax.axhline(chance, color="black", lw=0.5, ls=(0, (1, 1.5)))
ax.set_xticks(x, DLAB)
ax.tick_params(axis="x", length=0)
ax.set_xlabel("Distinctness of removed type")
ax.set_ylabel("Removals (%)")
ax.set_ylim(0, 100)
ax.set_xlim(-0.55, len(DLAB) - 0.45)
ax.legend(loc="upper left")
letter("e", 120, ROW2_Y)

# ---- f: composite intestine reference --------------------------------------
ROW3_Y, ROW3_H = 100, 56
SRC = [("Haber2017", "Haber 2017 (epithelium)", LINEAGE_COLORS["Epithelial"]),
       ("Xu2019_immune", "Xu 2019 (immune)", OI["purple"]),
       ("Paerregaard2023_stroma", "Paerregaard 2023 (stroma)", OI["green"]),
       ("Morarach2021_ENS", "Morarach 2021 (enteric NS)", OI["blue"])]
src_rank = {s: i for i, (s, _, _) in enumerate(SRC)}
tot = comp.groupby("celltype1").n_cells.sum()
main_src = comp.sort_values("n_cells").groupby("celltype1").source.last()
order = sorted(tot.index, key=lambda t: (src_rank[main_src[t]], -tot[t]))
ax = ax_mm(40, ROW3_Y + 3, 30, ROW3_H - 3)
yy = np.arange(len(order))
left = np.zeros(len(order))
for s, lab, c in SRC:
    v = comp[comp.source == s].set_index("celltype1").n_cells.reindex(order).fillna(0).to_numpy()
    ax.barh(yy, v, left=left, color=c, lw=0, height=0.75, label=lab)
    left += v
ax.set_yticks(yy, [t[0].upper() + t[1:] for t in order], fontsize=FS_SMALL)
ax.tick_params(axis="y", length=0)
ax.invert_yaxis()
ax.set_xlim(0, 3200)
ax.set_xticks([0, 1000, 2000, 3000], ["0", "1", "2", "3"])
ax.set_xlabel("Cells (×10$^3$)")
ax.set_ylim(len(order) - 0.4, -0.6)
ax.legend(loc="upper left", bbox_to_anchor=(-0.95, -0.13), ncol=2, fontsize=FS_TINY,
          handlelength=0.8, columnspacing=0.6)
letter("f", 0, ROW3_Y)

# ---- g: lung, lineage composition of flagged bins --------------------------
LINS = ["Epithelial", "Plasma", "Macrophage/Mono", "Fibroblast", "DC", "B"]
LCOL = {**{l: LINEAGE_COLORS[l] for l in LINS}, "Other": LINEAGE_COLORS["Other"]}
rows = []
for s, slab in (("LUNG_X1", "S1"), ("LUNG_X5K", "S2")):
    for ref, rlab in (("flex", "Flex"), ("luad", "LUAD")):
        for kind, klab in (("lineage_all", "all"), ("lineage_in_flagged", "flagged")):
            comp_l = lung[s][ref][kind]
            v = {l: comp_l.get(l, 0.0) for l in LINS}
            v["Other"] = 1 - sum(v.values())
            rows.append((f"{slab} {rlab}, {klab}", v))
ax = ax_mm(90, ROW3_Y + 3, 26, ROW3_H - 3)
yy = np.arange(len(rows)) + np.repeat([0, 0.5, 1.0, 1.5], 2)
for i, (lab, v) in enumerate(rows):
    lft = 0.0
    for l in LINS + ["Other"]:
        ax.barh(yy[i], 100 * v[l], left=lft, color=LCOL[l], lw=0, height=0.8)
        lft += 100 * v[l]
ax.set_yticks(yy, [r[0] for r in rows], fontsize=FS_SMALL)
ax.tick_params(axis="y", length=0)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Bins by dominant lineage (%)")
ax.legend(handles=[Patch(fc=LCOL[l], lw=0, label=l.replace("Macrophage/Mono", "Macrophage"))
                   for l in LINS + ["Other"]],
          loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=4, fontsize=FS_TINY,
          handlelength=0.8, columnspacing=0.6)
letter("g", 73, ROW3_Y)

# ---- h: lung, genes still unexplained with the LUAD reference --------------
_s1 = lung_luad["LUNG_X1"].set_index("gene").score
_s2 = lung_luad["LUNG_X5K"].set_index("gene").score
_sh = _s1.index.intersection(_s2.index)  # genes in the top 40 of both sections
top = ((_s1[_sh] + _s2[_sh]) / 2).sort_values(ascending=False).index.tolist()
ax = ax_mm(140, ROW3_Y + 3, 38, ROW3_H - 3)
bw = 0.4
for i, (s, lab, c) in enumerate((("LUNG_X1", "Section 1", "#252525"),
                                 ("LUNG_X5K", "Section 2", "#9A9A9A"))):
    sc = lung_luad[s].set_index("gene").score
    vals = [sc.get(g, np.nan) for g in top]
    ax.barh(np.arange(len(top)) + (i - 0.5) * bw, vals, bw, color=c, lw=0, label=lab)
ax.set_yticks(range(len(top)), top, fontstyle="italic", fontsize=FS_SMALL)
ax.tick_params(axis="y", length=0)
ax.set_ylim(len(top) - 0.4, -0.6)
ax.set_xlim(0, 5)
ax.set_xlabel("Unexplained score (LUAD reference)")
ax.legend(loc="lower right")
letter("h", 124, ROW3_Y)

# ---- i: CRC, flagged fraction in program-high bins under both nulls -------
ROW4_Y, ROW4_H = 174, 30
cats = [("All bins", "flag_all", None),
        ("IFN-γ R0", "hot_region_0_flagged (IFN-gamma (R0))", "hot_region_0_n"),
        ("IFN-γ R1", "hot_region_1_flagged (IFN-gamma (R1))", "hot_region_1_n"),
        ("Hypoxia R2", "hot_region_2_flagged (Hypoxia (R2))", "hot_region_2_n")]
CAT_COL = ["#8A8A8A", "#6A3D9A", "#9E7CC1", OI["skyblue"]]
ax = ax_mm(13, ROW4_Y + 3, 62, ROW4_H - 3)
sp = scrc[scrc.key == "score_pooled"]
for p, s in enumerate(SAMPLES):
    for k, (lab, col, ncol) in enumerate(cats):
        xk = p + (k - 1.5) * 0.2
        for nl in ("left_half", "auto"):
            row = sp[(sp["sample"] == s) & (sp["null"] == nl)].iloc[0]
            if ncol is not None and not row[ncol] > 0:
                continue
            v = 100 * row[col]
            ax.scatter(xk, v, s=11, marker="o", lw=0.6,
                       facecolor=CAT_COL[k] if nl == "auto" else "white",
                       edgecolor=CAT_COL[k], zorder=3)
ref5(ax)
ax.set_xticks(range(3), [s.split("_")[0] for s in SAMPLES])
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-2, 60)
ax.set_ylabel("Bins flagged (%)")
h1 = [Patch(fc=c, lw=0, label=l[0]) for l, c in zip(cats, CAT_COL)]
h2 = [Line2D([], [], ls="", marker="o", mfc="#555555", mec="#555555", ms=3, label="Auto null"),
      Line2D([], [], ls="", marker="o", mfc="white", mec="#555555", ms=3, mew=0.6,
             label="Left-half null")]
leg = ax.legend(handles=h1, loc="upper left", bbox_to_anchor=(1.0, 1.05), fontsize=FS_TINY,
                handlelength=0.8)
ax.add_artist(leg)
ax.legend(handles=h2, loc="lower left", bbox_to_anchor=(1.0, -0.05), fontsize=FS_TINY,
          handletextpad=0.2)
letter("i", 0, ROW4_Y)

# ---- j: CRC P2 unexplained regions, hallmark enrichment --------------------
hr = hall[(hall["sample"] == "P2_CRC") & (hall.kind == "region")].set_index("set")
hp = hall[(hall["sample"] == "P2_CRC") & (hall.kind == "perm")].copy()
hp["set"] = hp["set"].str.replace("perm_", "")
hp = hp.set_index("set")
regs = [f"region_{i}" for i in range(5)]
ax = ax_mm(128, ROW4_Y + 3, 50, ROW4_H - 3)
lq = lambda q: -np.log10(np.clip(q, 1e-300, 1))  # noqa: E731
yy = np.arange(len(regs))
ax.barh(yy - 0.2, [lq(hr.loc[r, "q"]) for r in regs], 0.38, color=FLAG_COLOR, lw=0,
        label="Unexplained region")
ax.barh(yy + 0.2, [lq(hp.loc[r, "q"]) for r in regs], 0.38, color="#BDBDBD", lw=0,
        label="Size-matched random bins")
short = {"Interferon Gamma Response": "IFN-γ response", "Hypoxia": "Hypoxia",
         "Xenobiotic Metabolism": "Xenobiotic metab.",
         "TNF-alpha Signaling via NF-kB": "TNF-α/NF-κB"}
ax.set_yticks(yy, [f"R{i}  {short.get(hr.loc[r, 'best_hallmark'], hr.loc[r, 'best_hallmark'])}"
                   for i, r in enumerate(regs)], fontsize=FS_SMALL)
ax.tick_params(axis="y", length=0)
ax.set_ylim(len(regs) - 0.4, -0.6)
ax.axvline(lq(0.05), color="black", lw=0.5, ls=(0, (1, 1.5)))
ax.set_xlabel("Best hallmark, −log$_{10}$ q")
ax.set_xlim(0, 32)
ax.legend(loc="lower right")
letter("j", 96, ROW4_Y)

save(fig, "supp_refdiag")
