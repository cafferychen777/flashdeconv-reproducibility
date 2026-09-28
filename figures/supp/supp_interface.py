"""Supplementary figure: per-lineage interface gradients and concordance for all sections (companion to Fig. 6).

Reads results/b1_pilot/ directly and draws everything on one canvas.
Output: paper/figures/supp_interface.{pdf,png}

Panels
  a  z-scored gradients, FlashDeconv vs cells, every section with orthogonal data x every non-epithelial lineage
  b  Pearson r per section x lineage (pre-registered statistic) and the section median
  c  same lineages, r of the lineage's share of non-epithelial units (post hoc) vs r of fraction of all units
  d  CRC boundary sensitivity: CEACAM5/CEACAM6/PERP boundary (primary) vs EPCAM/KRT/CDH1 boundary on Visium HD
  e  raw Visium HD marker gradient vs FlashDeconv gradient, both against cells
  f  FlashDeconv fitting time vs number of 8-um bins
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import (FD_COLOR, FS_SMALL, FS_TINY, LINEAGE_COLORS, LW, RESULTS, TRUTH_COLOR,  # noqa: E402
                   apply_style, colorbar_small, new_figure, panel_label, save)

RES = RESULTS / "b1_pilot"
PRIMARY = ["CRC_P1", "CRC_P2", "CRC_P5", "SPATCH_COAD", "SPATCH_OV"]
SECS = PRIMARY + ["SPATCH_HCC", "LUNG_X1", "LUNG_X5K", "OV10X"]
SENS = {s: f"{s}_sensHD-EPCAMKRT" for s in ["CRC_P1", "CRC_P2", "CRC_P5"]}
LABEL = {"CRC_P1": "CRC P1", "CRC_P2": "CRC P2", "CRC_P5": "CRC P5", "SPATCH_COAD": "COAD", "SPATCH_OV": "OV-1",
         "SPATCH_HCC": "HCC", "LUNG_X1": "Lung-1", "LUNG_X5K": "Lung-2", "OV10X": "OV-2"}
LIN = ["Fibroblast", "Pericyte/SMC", "Endothelial", "Macrophage/Mono", "DC", "Neutrophil", "Mast",
       "CD4 T", "CD8 T", "Treg", "NK", "B", "Plasma"]
SHORT = {"Macrophage/Mono": "Macrophage", "Pericyte/SMC": "Peri./SMC", "Fibroblast": "Fibroblast",
         "Endothelial": "Endothelial", "Neutrophil": "Neutrophil"}
CODEX_COMPARE = ["Fibroblast", "Endothelial", "CD4 T", "CD8 T", "Treg", "NK", "B", "Macrophage/Mono"]
MARKER_LIN = {"CD3E+CD3D": "CD8 T", "MS4A1": "B", "CD68": "Macrophage/Mono", "COL1A1": "Fibroblast",
              "PECAM1": "Endothelial"}
MARKER_LAB = {"CD3E+CD3D": "CD3D/E (T)", "MS4A1": "MS4A1 (B)", "CD68": "CD68 (macrophage)",
              "COL1A1": "COL1A1 (fibroblast)", "PECAM1": "PECAM1 (endothelial)"}


def zs(v, ref=None):
    ref = v if ref is None else ref
    return (v - np.nanmean(ref)) / (np.nanstd(ref) + 1e-12)


def pr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return float(np.corrcoef(a, b)[0, 1]) if len(a) >= 8 and a.std() > 0 and b.std() > 0 else np.nan


def load():
    cur = pd.concat([pd.read_csv(RES / "gradient_curves.csv.gz"), pd.read_csv(RES / "gradient_curves_ext.csv.gz")])
    conc = pd.concat([pd.read_csv(RES / "concordance_per_lineage.csv"),
                      pd.read_csv(RES / "concordance_per_lineage_ext.csv")])
    summ = pd.concat([pd.read_csv(RES / "section_summary.csv"), pd.read_csv(RES / "section_summary_ext.csv")])
    summ = summ.drop_duplicates("section").set_index("section")
    mk = pd.concat([pd.read_csv(RES / "marker_diagnostic.csv"), pd.read_csv(RES / "marker_diagnostic_ext.csv")])
    return cur, conc, summ, mk


def tables(cur, sec, mask=False):
    """FD and orthogonal band tables. mask=True blanks bands with <50 units (display only)."""
    sub = cur[cur.section == sec].copy()
    if mask:
        sub.loc[sub.n_units < 50, ["value", "lo", "hi"]] = np.nan
    osrc = [s for s in sub.source.unique() if s not in ("FD", "HDmarker")][0]
    P = lambda src, v="value": sub[sub.source == src].pivot(index="band_mid", columns="name", values=v)
    F, Flo, Fhi, O = P("FD"), P("FD", "lo"), P("FD", "hi"), P(osrc)
    if osrc == "CODEX":  # CODEX fibroblasts are SMA-defined: FD fibroblast + pericyte/SMC
        for T in (F, Flo, Fhi):
            T["Fibroblast"] = T["Fibroblast"] + T["Pericyte/SMC"]
    return F, Flo, Fhi, O, osrc


def comparable(osrc, lin):
    return lin in CODEX_COMPARE if osrc == "CODEX" else True


def within_nonepi(cur, conc, sec, base=None):
    """Post hoc: r of each included lineage's share of non-epithelial units (same bands, same lineages as primary)."""
    base = base or sec
    F, _, _, O, osrc = tables(cur, sec)
    comp = CODEX_COMPARE if osrc == "CODEX" else LIN
    inc = conc[(conc.section == base) & conc.included].lineage.tolist()
    ft, ot = F[comp].sum(1), O[comp].sum(1)
    return {l: pr(F[l] / ft, O[l] / ot) for l in inc}


def main():
    apply_style()
    cur, conc, summ, mk = load()
    H, W = 226.0, 180.0
    fig = new_figure(H)

    def rect(x, y, w, h):
        return [x / W, 1 - (y + h) / H, w / W, h / H]

    # ------------------------------------------------------------------ a: gradient small multiples
    x0, y0, cw, ch, dx, dy = 17.0, 8.0, 11.2, 7.6, 1.2, 1.5
    for i, sec in enumerate(SECS):
        Fm, Flo, Fhi, Om, osrc = tables(cur, sec, mask=True)
        x = Fm.index.values
        yrow = y0 + i * (ch + dy) + (2.0 if i >= 5 else 0)
        for j, lin in enumerate(LIN):
            ax = fig.add_axes(rect(x0 + j * (cw + dx), yrow, cw, ch))
            ax.axvline(0, color="#C8C8C8", lw=0.35)
            fv = Fm[lin].values
            c = conc[(conc.section == sec) & (conc.lineage == lin)]
            inc = len(c) and bool(c.included.iloc[0]) and comparable(osrc, lin)
            if np.nanstd(fv) > 0:
                if inc:
                    ax.fill_between(x, zs(Flo[lin].values, fv), zs(Fhi[lin].values, fv), color=FD_COLOR,
                                    alpha=0.22, lw=0)
                ax.plot(x, zs(fv), color=FD_COLOR if inc else "#C9A48A", lw=0.6)
            if inc:
                ax.plot(x, zs(Om[lin].values), color=TRUTH_COLOR, lw=0.6, ls=(0, (2.2, 0.9)))
            ax.set_xlim(-200, 300)
            ax.set_ylim(-2.6, 3.8)
            ax.set_yticks([])
            ax.set_xticks([-200, 0, 200])
            ax.tick_params(length=1.2, pad=1)
            ax.spines["left"].set_visible(False)
            if i < len(SECS) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(["−200" if j == 0 else "", "0", "200"], fontsize=FS_TINY)
            if not inc:
                ax.set_facecolor("#F4F4F4")
            if i == 0:
                ax.set_title(SHORT.get(lin, lin), fontsize=FS_SMALL, pad=2)
            if j == 0:
                ax.text(-0.1, 0.5, f"{LABEL[sec]}\n{osrc}", transform=ax.transAxes, ha="right", va="center",
                        fontsize=FS_SMALL, linespacing=1.1)
            if i == 0 and j == 0:
                panel_label(ax, "a", dx_mm=-15)
    ya_end = y0 + len(SECS) * (ch + dy) + 2.0
    fig.text((x0 + 6.5 * (cw + dx)) / W, 1 - (ya_end + 2.6) / H, "Signed distance to epithelial boundary (µm)",
             ha="center", va="top")
    fig.legend(handles=[Line2D([], [], color=FD_COLOR, lw=1, label="FlashDeconv, Visium HD (z-score)"),
                        Line2D([], [], color=TRUTH_COLOR, lw=1, ls=(0, (2.2, 0.9)),
                               label="Cells, Xenium or CODEX (z-score)"),
                        Line2D([], [], color="#C9A48A", lw=1, label="FlashDeconv, lineage not compared")],
               loc="upper center", bbox_to_anchor=((x0 + 6.5 * (cw + dx)) / W, 1 - (ya_end + 5.8) / H), ncol=3,
               frameon=False, fontsize=FS_SMALL)

    # ------------------------------------------------------------------ b: r heatmap
    yb = ya_end + 17.5
    hb_row = 4.6
    bx, bw = 22.0, 13 * 6.6
    ax = fig.add_axes(rect(bx, yb, bw, hb_row * len(SECS)))
    V = np.full((len(SECS), len(LIN)), np.nan)
    INC = np.zeros_like(V, bool)
    HAS = np.zeros_like(V, bool)
    for i, sec in enumerate(SECS):
        osrc = summ.loc[sec, "modality"]
        for j, lin in enumerate(LIN):
            c = conc[(conc.section == sec) & (conc.lineage == lin)]
            if len(c) and comparable(osrc, lin):
                HAS[i, j] = True
                V[i, j] = c.r.iloc[0]
                INC[i, j] = bool(c.included.iloc[0])
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    Vs = np.where(INC, V, np.nan)
    im = ax.imshow(np.ma.masked_invalid(Vs), cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    for i in range(len(SECS)):
        for j in range(len(LIN)):
            if INC[i, j]:
                ax.text(j, i, f"{V[i, j]:.2f}".replace("-", "−"), ha="center", va="center", fontsize=FS_TINY,
                        color="white" if abs(V[i, j]) > 0.62 else "black")
            elif HAS[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#EDEDED", lw=0))
    ax.axhline(4.5, color="k", lw=0.6)
    ax.set_xticks(range(len(LIN)))
    ax.set_xticklabels([SHORT.get(l, l) for l in LIN], rotation=90)
    ax.set_yticks(range(len(SECS)))
    ax.set_yticklabels([LABEL[s] for s in SECS])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-0.5, len(LIN) - 0.5)
    ax.set_ylim(len(SECS) - 0.5, -0.5)
    # median column
    axm = fig.add_axes(rect(bx + bw + 1.5, yb, 6.6, hb_row * len(SECS)))
    med = summ.loc[SECS, "median_r"].values
    axm.imshow(med[:, None], cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    for i, m in enumerate(med):
        axm.text(0, i, f"{m:.2f}", ha="center", va="center", fontsize=FS_TINY, fontweight="bold",
                 color="white" if abs(m) > 0.62 else "black")
    axm.axhline(4.5, color="k", lw=0.6)
    axm.set_xticks([0])
    axm.set_xticklabels(["Median"], rotation=90)
    axm.set_yticks([])
    axm.tick_params(length=0)
    for s in axm.spines.values():
        s.set_visible(False)
    colorbar_small(fig, im, rect(bx + bw + 10.5, yb + 8, 1.5, 22), orientation="vertical", ticks=[-1, 0, 1],
                   label="Pearson r")
    panel_label(ax, "b", dx_mm=-14)

    # ------------------------------------------------------------------ c: within non-epithelial compartment
    cx, cw_ = 138.0, 38.0
    ax = fig.add_axes(rect(cx, yb, cw_, hb_row * len(SECS)))
    rng = np.random.default_rng(0)
    for i, sec in enumerate(SECS):
        inc = conc[(conc.section == sec) & conc.included]
        raw = inc.r.values
        wit = np.array(list(within_nonepi(cur, conc, sec).values()))
        ax.scatter(raw, np.full(len(raw), i - 0.17) + rng.uniform(-0.07, 0.07, len(raw)), s=2.2, color=FD_COLOR,
                   alpha=0.45, lw=0, zorder=2)
        ax.scatter(wit, np.full(len(wit), i + 0.17) + rng.uniform(-0.07, 0.07, len(wit)), s=2.2, color="#7F7F7F",
                   alpha=0.55, lw=0, zorder=2)
        ax.plot([np.nanmedian(raw)] * 2, [i - 0.36, i + 0.02], color=FD_COLOR, lw=1.1, zorder=3,
                solid_capstyle="butt")
        ax.plot([np.nanmedian(wit)] * 2, [i - 0.02, i + 0.36], color="#333333", lw=1.1, zorder=3,
                solid_capstyle="butt")
    ax.axvline(0.7, color="k", lw=0.5, ls=(0, (3, 2)))
    ax.axvline(0, color="#C8C8C8", lw=0.4)
    ax.axhline(4.5, color="k", lw=0.6)
    ax.set_ylim(len(SECS) - 0.5, -0.5)
    ax.set_yticks(range(len(SECS)))
    ax.set_yticklabels([LABEL[s] for s in SECS])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-1.02, 1.02)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["−1", "−0.5", "0", "0.5", "1"])
    ax.set_xlabel("Pearson r, FlashDeconv vs cells")
    ax.legend(handles=[Line2D([], [], color=FD_COLOR, lw=1.1, label="Fraction of all units"),
                       Line2D([], [], color="#333333", lw=1.1, label="Share of non-epithelial units")],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, frameon=False, fontsize=FS_SMALL,
              handlelength=1.0)
    panel_label(ax, "c", dx_mm=-11)

    # ------------------------------------------------------------------ d: CRC boundary sensitivity
    yd = yb + hb_row * len(SECS) + 23.0
    hd = 30.0
    ax = fig.add_axes(rect(17, yd, 44, hd))
    for k, (sec, sens) in enumerate(SENS.items()):
        F0, _, _, O0, _ = tables(cur, sec)
        F1, _, _, O1, _ = tables(cur, sens)
        inc = conc[(conc.section == sec) & conc.included].lineage.tolist()
        a = np.array([pr(F0[l], O0[l]) for l in inc])
        b = np.array([pr(F1[l], O1[l]) for l in inc])
        xa, xb = 3 * k, 3 * k + 1
        for l, va, vb in zip(inc, a, b):
            ax.plot([xa, xb], [va, vb], color="#A8A8A8", lw=0.5, zorder=1)
        ax.scatter([xa] * len(a), a, s=4, color=FD_COLOR, lw=0, zorder=2)
        ax.scatter([xb] * len(b), b, s=4, color=FD_COLOR, lw=0, zorder=2)
        for xx, m in [(xa, summ.loc[sec, "median_r"]), (xb, summ.loc[sens, "median_r"])]:
            ax.plot([xx - 0.32, xx + 0.32], [m, m], color="k", lw=1.0, zorder=3)
        ax.text(xa + 0.5, -0.02, LABEL[sec], ha="center", va="top", transform=ax.get_xaxis_transform(),
                fontsize=FS_SMALL)
    ax.axhline(0.7, color="k", lw=0.5, ls=(0, (3, 2)))
    ax.set_xticks([0, 1, 3, 4, 6, 7])
    ax.set_xticklabels(["CEACAM", "EPCAM/KRT"] * 3, rotation=90, fontsize=FS_TINY)
    ax.tick_params(axis="x", length=0, pad=7.5)
    ax.set_xlim(-0.6, 7.6)
    ax.set_ylim(0.2, 1.02)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Pearson r, FlashDeconv vs Xenium")
    ax.set_title("Visium HD boundary genes", fontsize=FS_SMALL, pad=2)
    panel_label(ax, "d", dx_mm=-12)

    # ------------------------------------------------------------------ e: raw marker vs FlashDeconv
    ax = fig.add_axes(rect(80, yd, hd, hd))
    mks = mk[mk.section.isin(PRIMARY)].dropna(subset=["r_marker_vs_orth", "r_fd_vs_orth"])
    ax.plot([-1, 1], [-1, 1], color="#BBBBBB", lw=0.5, zorder=0)
    ax.axhline(0, color="#E0E0E0", lw=0.4, zorder=0)
    ax.axvline(0, color="#E0E0E0", lw=0.4, zorder=0)
    for m, lin in MARKER_LIN.items():
        d = mks[mks.marker == m]
        ax.scatter(d.r_marker_vs_orth, d.r_fd_vs_orth, s=9, color=LINEAGE_COLORS[lin], edgecolor="#333333",
                   lw=0.25, zorder=3, label=MARKER_LAB[m])
    ax.set_xlim(-1.02, 1.05)
    ax.set_ylim(-1.02, 1.05)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xticklabels(["−1", "0", "1"])
    ax.set_yticklabels(["−1", "0", "1"])
    ax.set_xlabel("r, raw HD marker vs cells")
    ax.set_ylabel("r, FlashDeconv vs cells")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.02), frameon=False, fontsize=FS_TINY, handletextpad=0.1,
              borderaxespad=0, labelspacing=0.35)
    panel_label(ax, "e", dx_mm=-10)

    # ------------------------------------------------------------------ f: runtime vs bins
    ax = fig.add_axes(rect(146, yd, 31, hd))
    nb = summ.loc[SECS, "n_hd_bins"].values / 1e3
    t = summ.loc[SECS, "t_deconvolve_s"].values
    it = summ.loc[SECS, "fd_n_iter"].values
    ax.scatter(nb, t, s=8, color=FD_COLOR, lw=0, zorder=3)
    off = {"CRC_P1": (-4, 0.3, "right"), "CRC_P2": (4, 0.6, "left"), "CRC_P5": (4, -1.8, "left"),
           "SPATCH_COAD": (-4, 0, "right"), "SPATCH_OV": (-4, 1.2, "right"), "SPATCH_HCC": (-4, -1.4, "right"),
           "LUNG_X1": (4, -1.3, "left"), "LUNG_X5K": (4, 1.3, "left"), "OV10X": (4, 0.4, "left")}
    for s, a, b in zip(SECS, nb, t):
        ox, oy, ha = off[s]
        ax.text(a + ox, b + oy, LABEL[s], fontsize=FS_TINY, ha=ha, va="center")
    ax.set_xlim(380, 700)
    ax.set_ylim(10, 42)
    ax.set_xticks([400, 500, 600, 700])
    ax.set_yticks([10, 20, 30, 40])
    ax.set_xlabel("Visium HD 8-µm bins (×10$^3$)")
    ax.set_ylabel("FlashDeconv time (s)")
    panel_label(ax, "f", dx_mm=-10)
    print("iterations", dict(zip(SECS, it)))

    save(fig, "supp_interface")

    # numbers for the caption
    print("median r", summ.loc[SECS, "median_r"].round(3).to_dict())
    print("within-nonepi median", {s: round(np.nanmedian(list(within_nonepi(cur, conc, s).values())), 3)
                                   for s in SECS})
    print("sens median", {s: round(summ.loc[v, "median_r"], 3) for s, v in SENS.items()})
    print("total bins", summ.loc[SECS, "n_hd_bins"].sum(), "total min", summ.loc[SECS, "t_deconvolve_s"].sum() / 60)


if __name__ == "__main__":
    main()
