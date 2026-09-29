"""Figure 6: tumour-stroma interface atlas across 4.8 million Visium HD bins.

Reads results/b1_pilot/ directly and draws the whole figure on one canvas.
Output: paper/figures/fig6_interface.{pdf,png}
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from style import (FD_COLOR, FS_SMALL, FS_TINY, LINEAGE_COLORS, LW, MM, RESULTS, TRUTH_COLOR,  # noqa: E402
                   apply_style, clean_spatial, colorbar_small, new_figure, panel_label, save)

RES = RESULTS / "b1_pilot"
PX_UM = 8.0

PRIMARY = ["CRC_P1", "CRC_P2", "CRC_P5", "SPATCH_COAD", "SPATCH_OV"]
EXT = ["LUNG_X1", "LUNG_X5K", "OV10X"]
ORTH_SECS = PRIMARY + EXT
ALL_SECS = PRIMARY + ["SPATCH_HCC"] + EXT
LABEL = {"CRC_P1": "CRC P1", "CRC_P2": "CRC P2", "CRC_P5": "CRC P5", "SPATCH_COAD": "COAD",
         "SPATCH_OV": "OV-1", "SPATCH_HCC": "HCC", "LUNG_X1": "Lung-1", "LUNG_X5K": "Lung-2", "OV10X": "OV-2"}
CANCER = {"CRC_P1": "CRC", "CRC_P2": "CRC", "CRC_P5": "CRC", "SPATCH_COAD": "CRC", "SPATCH_OV": "Ovarian",
          "SPATCH_HCC": "Liver", "LUNG_X1": "Lung", "LUNG_X5K": "Lung", "OV10X": "Ovarian"}
CANCER_SHADE = {"CRC": "#6E6E6E", "Ovarian": "#A0A0A0", "Lung": "#C8C8C8", "Liver": "#E0E0E0"}
LIN = ["Fibroblast", "Pericyte/SMC", "Endothelial", "Macrophage/Mono", "DC", "Neutrophil", "Mast",
       "CD4 T", "CD8 T", "Treg", "NK", "B", "Plasma"]
# Lung sections: T-cell and plasma estimates are reference-limited (Flex FFPE reference); not shown.
LUNG_MASK = {"CD4 T", "CD8 T", "Treg", "Plasma"}
GRAD_LIN = ["Fibroblast", "Macrophage/Mono", "Pericyte/SMC"]


def zs(v, ref=None):
    ref = v if ref is None else ref
    return (v - np.nanmean(ref)) / (np.nanstd(ref) + 1e-12)


CODEX_COMPARE = ["Fibroblast", "Endothelial", "CD4 T", "CD8 T", "Treg", "NK", "B", "Macrophage/Mono"]


def descriptors(cur, sec):
    """Rim/deep-stroma descriptors for one section, as in validation/b1_pilot/posthoc_v2.py."""
    sub = cur[cur.section == sec]
    mod = [s for s in sub.source.unique() if s not in ("FD", "HDmarker")][0]
    F = sub[sub.source == "FD"].pivot(index="band_mid", columns="name", values="value")
    O = sub[sub.source == mod].pivot(index="band_mid", columns="name", values="value")
    if mod == "CODEX":
        F = F.assign(Fibroblast=F["Fibroblast"] + F["Pericyte/SMC"])

    def d(C, l):
        x = C.index.values
        m = lambda lo, hi: C.loc[(x > lo) & (x < hi), l].mean()  # noqa: E731
        return np.log2((m(0, 50) + 1e-4) / (m(150, 300) + 1e-4)), float(C[l].mean())

    rows = []
    for l in LIN:
        fe, fm = d(F, l)
        if l in O.columns and (mod != "CODEX" or l in CODEX_COMPARE):
            oe, om = d(O, l)
        else:
            oe = om = np.nan
        rows.append(dict(section=sec, modality=mod, lineage=l, fd_mean=fm, orth_mean=om, fd_edge=fe,
                         orth_edge=oe, edge_supported=bool(np.isfinite(oe) and om >= 0.003
                                                           and np.sign(fe) == np.sign(oe) and abs(oe) >= 0.3)))
    return pd.DataFrame(rows)


def load():
    cur = pd.concat([pd.read_csv(RES / "gradient_curves.csv.gz"), pd.read_csv(RES / "gradient_curves_ext.csv.gz")])
    cur.loc[cur.n_units < 50, ["value", "lo", "hi"]] = np.nan
    conc = pd.concat([pd.read_csv(RES / "concordance_per_lineage.csv"),
                      pd.read_csv(RES / "concordance_per_lineage_ext.csv")])
    summ = pd.concat([pd.read_csv(RES / "section_summary.csv"), pd.read_csv(RES / "section_summary_ext.csv")])
    summ = summ.drop_duplicates("section").set_index("section")
    desc = pd.read_csv(RES / "cross_cancer_interface_descriptors_v2.csv")
    # HCC is not in the precomputed table; same definitions, computed here from the same curves
    desc = pd.concat([desc, descriptors(cur, "SPATCH_HCC")], ignore_index=True)
    return cur, conc, summ, desc


def curves(cur, sec):
    sub = cur[cur.section == sec]
    osrc = [s for s in sub.source.unique() if s not in ("FD", "HDmarker")][0]
    F = sub[sub.source == "FD"].pivot(index="band_mid", columns="name", values="value")
    Flo = sub[sub.source == "FD"].pivot(index="band_mid", columns="name", values="lo")
    Fhi = sub[sub.source == "FD"].pivot(index="band_mid", columns="name", values="hi")
    O = sub[sub.source == osrc].pivot(index="band_mid", columns="name", values="value")
    if osrc == "CODEX":  # CODEX fibroblasts are SMA-defined: compare with FD fibroblast + pericyte/SMC
        for T in (F, Flo, Fhi):
            T["Fibroblast"] = T["Fibroblast"] + T["Pericyte/SMC"]
    return F, Flo, Fhi, O, osrc


def orth_ok(O, osrc, lin):
    if lin not in O.columns or (osrc == "CODEX" and lin in ("Pericyte/SMC", "Neutrophil", "DC", "Mast", "Plasma")):
        return False
    return np.nanmean(O[lin].values) >= 0.003


def step(T, lin):
    """log2 stroma-side band (+25..+100 um) over tumour-edge band (-50..0 um), as in posthoc.py."""
    x = T.index.values
    edge = T.loc[(x > -50) & (x < 0), lin].mean()
    strom = T.loc[(x > 25) & (x < 100), lin].mean()
    return np.log2((strom + 1e-4) / (edge + 1e-4))


def main():
    apply_style()
    cur, conc, summ, desc = load()
    H = 200
    fig = new_figure(H)
    W = 180.0

    def rect(x_mm, y_top_mm, w_mm, h_mm):
        return [x_mm / W, 1 - (y_top_mm + h_mm) / H, w_mm / W, h_mm / H]

    # ------------------------------------------------------------------ a: sections, bins, time
    ax = fig.add_axes(rect(17, 6, 26, 40))
    secs = ALL_SECS
    y = np.arange(len(secs))
    t = summ.loc[secs, "t_deconvolve_s"].values
    nb = summ.loc[secs, "n_hd_bins"].values
    ax.barh(y, t, color=FD_COLOR, height=0.65, lw=0)
    for yy, n in zip(y, nb):
        ax.text(1.5, yy, f"{n / 1e3:.0f}k", va="center", ha="left", fontsize=FS_TINY, color="white")
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[s] for s in secs])
    ax.set_ylim(len(secs) - 0.4, -0.6)
    ax.set_xlim(0, 42)
    ax.set_xticks([0, 20, 40])
    ax.set_xlabel("FlashDeconv time (s)")
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{nb.sum() / 1e6:.2f}M bins, {t.sum() / 60:.1f} min", fontsize=FS_SMALL)
    panel_label(ax, "a", dx_mm=-16)

    # ------------------------------------------------------------------ b: example boundary + bands
    band_cmap = ListedColormap(["#E69F00", "#0072B2", "#D0D0D0", "#56B4E9", "#EFEFEF"])
    band_names = ["Tumour", "Rim 0–50 µm", "50–150 µm", "Deep 150–300 µm", ">300 µm"]
    bx = [50, 88]
    for k, (sec, title) in enumerate([("CRC_P1", "CRC P1"), ("SPATCH_OV", "OV-1")]):
        ax = fig.add_axes(rect(bx[k], 5, 37, 37))
        g = np.load(RES / f"grid_{sec}.npz")
        tissue = ndimage.binary_closing(g["hd_occ"], iterations=3)
        mask = g["hd_mask"]
        d = (ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)) * PX_UM
        cls = np.full(mask.shape, np.nan)
        cls[d < 0] = 0
        cls[(d >= 0) & (d < 50)] = 1
        cls[(d >= 50) & (d < 150)] = 2
        cls[(d >= 150) & (d < 300)] = 3
        cls[d >= 300] = 4
        cls[~tissue] = np.nan
        ax.imshow(cls.T, cmap=band_cmap, vmin=-0.5, vmax=4.5, origin="lower", interpolation="nearest",
                  rasterized=True)
        clean_spatial(ax)
        n0, n1 = cls.T.shape
        xs = n1 * 0.05
        ax.plot([xs, xs + 1000 / PX_UM], [-n0 * 0.035] * 2, color="k", lw=1, solid_capstyle="butt", clip_on=False)
        ax.text(xs + 500 / PX_UM, -n0 * 0.05, "1 mm", ha="center", va="top", fontsize=FS_SMALL)
        ax.set_title(f"{title}, Visium HD", fontsize=FS_SMALL, pad=2)
        if k == 0:
            panel_label(ax, "b", dx_mm=-4)
    fig.legend(handles=[Patch(color=band_cmap(i), label=band_names[i]) for i in range(5)],
               loc="upper center", bbox_to_anchor=(87.5 / W, 1 - 46.5 / H), fontsize=FS_SMALL, frameon=False,
               handlelength=0.9, handleheight=0.9, ncol=5, columnspacing=0.8)

    # ------------------------------------------------------------------ c: concordance per section
    ax = fig.add_axes(rect(148, 6, 30, 38))
    rng = np.random.default_rng(0)
    for i, sec in enumerate(ORTH_SECS):
        r = conc[(conc.section == sec) & conc.included]
        ax.scatter(np.full(len(r), i) + rng.uniform(-0.18, 0.18, len(r)), r.r, s=4, color=FD_COLOR, alpha=0.7,
                   lw=0, zorder=3)
        med = summ.loc[sec, "median_r"]
        ax.plot([i - 0.32, i + 0.32], [med, med], color="k", lw=1.0, zorder=4)
    ax.axhline(0.7, color="k", lw=0.5, ls=(0, (3, 2)))
    ax.axvline(4.5, color="#BBBBBB", lw=0.4)
    ax.set_xticks(range(len(ORTH_SECS)))
    ax.set_xticklabels([LABEL[s] for s in ORTH_SECS], rotation=90)
    ax.set_xlim(-0.6, len(ORTH_SECS) - 0.4)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel("Pearson r, FlashDeconv vs cells")
    ax.tick_params(axis="x", length=0)
    panel_label(ax, "c", dx_mm=-9)

    # ------------------------------------------------------------------ d: gradient small multiples
    gx0, gy0, gw, gh, gdx, gdy = 17, 60, 22.0, 17.0, 4.0, 5.0
    for j, sec in enumerate(PRIMARY):
        F, Flo, Fhi, O, osrc = curves(cur, sec)
        x = F.index.values
        for i, lin in enumerate(GRAD_LIN):
            ax = fig.add_axes(rect(gx0 + j * (gw + gdx), gy0 + i * (gh + gdy), gw, gh))
            ax.axvline(0, color="#BBBBBB", lw=0.4)
            fv = F[lin].values
            ax.fill_between(x, zs(Flo[lin].values, fv), zs(Fhi[lin].values, fv), color=FD_COLOR, alpha=0.16, lw=0)
            ax.plot(x, zs(fv), color=FD_COLOR, lw=0.8)
            if orth_ok(O, osrc, lin):
                ov = O[lin].values
                ax.plot(x, zs(ov), color=TRUTH_COLOR, lw=0.8, ls=(0, (2.5, 1)))
                rr = conc[(conc.section == sec) & (conc.lineage == lin)].r
                if len(rr):
                    ax.text(0.04, 0.97, f"$r$ = {rr.iloc[0]:.2f}", transform=ax.transAxes, va="top", ha="left",
                            fontsize=FS_TINY)
            ax.set_xlim(-200, 300)
            ax.set_ylim(-3, 3.6)
            ax.set_xticks([-200, 0, 200])
            ax.set_yticks([-2, 0, 2])
            if i < len(GRAD_LIN) - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(lin.replace("Macrophage/Mono", "Macrophage").replace("Pericyte/SMC", "Pericyte/SMC"),
                              fontsize=FS_SMALL)
            if i == 0:
                ax.set_title(f"{LABEL[sec]} ({osrc})", fontsize=FS_SMALL, pad=2)
            if i == 0 and j == 0:
                panel_label(ax, "d", dx_mm=-15)
    fig.text((gx0 + 2.5 * gw + 2 * gdx) / W, 1 - (gy0 + 3 * gh + 2 * gdy + 6.5) / H,
             "Signed distance to epithelial boundary (µm)", ha="center", va="top")
    fig.text(4.5 / W, 1 - (gy0 + 1.5 * gh + gdy) / H, "Proportion (z-score)", rotation=90, ha="center", va="center")
    fig.legend(handles=[Line2D([], [], color=FD_COLOR, lw=1, label="FlashDeconv (Visium HD)"),
                        Line2D([], [], color=TRUTH_COLOR, lw=1, ls=(0, (2.5, 1)), label="Cells (Xenium / CODEX)")],
               loc="upper center", bbox_to_anchor=((gx0 + 178) / (2 * W), 1 - (gy0 + 3 * gh + 2 * gdy + 10.5) / H),
               ncol=2, frameon=False, fontsize=FS_SMALL)

    # ------------------------------------------------------------------ e: CD8 T placement
    ex0 = 153
    eh = (3 * gh + 2 * gdy - 6) / 2
    for k, (secs_, osrc_lab, title) in enumerate([(["SPATCH_OV"], "CODEX", "Ovarian (OV-1)"),
                                                   (["CRC_P1", "CRC_P2", "CRC_P5"], "Xenium", "CRC P1, P2, P5")]):
        ax = fig.add_axes(rect(ex0, gy0 + k * (eh + 6), 25, eh))
        ax.axvline(0, color="#BBBBBB", lw=0.4)
        ax.axvspan(0, 50, color="#0072B2", alpha=0.07, lw=0)
        ax.axvspan(150, 300, color="#56B4E9", alpha=0.07, lw=0)
        for sec in secs_:
            F, Flo, Fhi, O, osrc = curves(cur, sec)
            x = F.index.values
            fv = F["CD8 T"].values
            ov = O["CD8 T"].values
            ax.plot(x, fv / np.nanmax(fv), color=FD_COLOR, lw=0.8)
            ax.plot(x, ov / np.nanmax(ov), color=TRUTH_COLOR, lw=0.8, ls=(0, (2.5, 1)))
        ax.set_xlim(-200, 300)
        ax.set_ylim(0, 1.08)
        ax.set_xticks([-200, 0, 200])
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(f"CD8 T, {title}", fontsize=FS_SMALL, pad=2)
        ax.set_ylabel("Relative to max", labelpad=1.5)
        if k == 0:
            ax.set_xticklabels([])
            panel_label(ax, "e", dx_mm=-9)
        else:
            ax.set_xlabel("Distance to boundary (µm)")

    # ------------------------------------------------------------------ f: rim/deep heatmap
    hx, hy, hw, hh = 23, 141, 52, 44
    ax = fig.add_axes(rect(hx, hy, hw, hh))
    M = desc.pivot(index="lineage", columns="section", values="fd_edge").reindex(LIN)[ALL_SECS]
    Mm = desc.pivot(index="lineage", columns="section", values="fd_mean").reindex(LIN)[ALL_SECS]
    Sp = desc.pivot(index="lineage", columns="section", values="edge_supported").reindex(LIN)[ALL_SECS]
    V = M.values.astype(float).copy()
    V[Mm.values < 0.001] = np.nan
    for j, sec in enumerate(ALL_SECS):
        if sec.startswith("LUNG"):
            for i, lin in enumerate(LIN):
                if lin in LUNG_MASK:
                    V[i, j] = np.nan
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#D6D6D6")
    im = ax.imshow(np.ma.masked_invalid(V), cmap=cmap, vmin=-3, vmax=3, aspect="auto", interpolation="nearest")
    S = Sp.values.astype(bool) & np.isfinite(V)
    yy, xx = np.where(S)
    ax.scatter(xx, yy, s=3, color="k", lw=0)
    ax.set_xticks(range(len(ALL_SECS)))
    ax.set_xticklabels([LABEL[s] for s in ALL_SECS], rotation=90)
    ax.set_yticks(range(len(LIN)))
    ax.set_yticklabels([l.replace("Macrophage/Mono", "Macrophage") for l in LIN])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    colorbar_small(fig, im, rect(hx + hw + 2.5, hy + 8, 2.0, 26), orientation="vertical", ticks=[-3, 0, 3],
                   label="log$_2$ rim / deep stroma")
    panel_label(ax, "f", dx_mm=-21)

    # ------------------------------------------------------------------ g: shared features, FD vs cells
    fx0, fw = 98, 23
    feats = [("Fibroblast", "Fibroblast step\n(log$_2$ stroma / tumour edge)", PRIMARY, "step"),
             ("Macrophage/Mono", "Macrophage rim\n(log$_2$ rim / deep stroma)", PRIMARY + ["LUNG_X1", "LUNG_X5K"], "edge"),
             ("Pericyte/SMC", "Pericyte/SMC rim\n(log$_2$ rim / deep stroma)", ["CRC_P1", "CRC_P2", "CRC_P5",
                                                                                "LUNG_X1", "LUNG_X5K"], "edge")]
    for k, (lin, lab, secs_, kind) in enumerate(feats):
        panel_x, panel_w = [(98, 22), (124.5, 25), (154, 22)][k]
        ax = fig.add_axes(rect(panel_x, hy + 1, panel_w, 34))
        vals_fd, vals_or = [], []
        for sec in secs_:
            if kind == "step":
                F, _, _, O, osrc = curves(cur, sec)
                Fraw = cur[(cur.section == sec) & (cur.source == "FD")].pivot(index="band_mid", columns="name",
                                                                             values="value")
                vals_fd.append(step(Fraw, lin))
                vals_or.append(step(O, lin))
            else:
                r = desc[(desc.section == sec) & (desc.lineage == lin)].iloc[0]
                vals_fd.append(r.fd_edge)
                vals_or.append(r.orth_edge)
        xs = np.arange(len(secs_))
        ax.axhline(0, color="#BBBBBB", lw=0.4)
        for xx_, a, b in zip(xs, vals_fd, vals_or):
            if np.isfinite(b):
                ax.plot([xx_, xx_], [a, b], color="#BBBBBB", lw=0.5, zorder=1)
        ax.scatter(xs, vals_or, s=9, facecolor="white", edgecolor=TRUTH_COLOR, lw=0.6, zorder=3)
        ax.scatter(xs, vals_fd, s=9, color=FD_COLOR, lw=0, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABEL[s] for s in secs_], fontsize=FS_TINY,
                           rotation=55, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(secs_) - 0.4)
        ax.tick_params(axis="x", length=0)
        feature, ratio = lab.split("\n")
        ax.text(0.5, 1.085, feature, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=FS_SMALL)
        ax.text(0.5, 1.015, ratio, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=FS_TINY)
        if k == 0:
            ax.set_ylabel("log$_2$ enrichment")
            panel_label(fig, "g", x=89 / W, y=1 - (hy - 1.5) / H)
        lo = np.nanmin(vals_fd + vals_or)
        hi = np.nanmax(vals_fd + vals_or)
        ax.set_ylim(min(-0.5, lo - 0.5), max(0.5, hi + 0.5))
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=3, color=FD_COLOR, label="FlashDeconv"),
                        Line2D([], [], marker="o", ls="", ms=3, mfc="white", mec=TRUTH_COLOR, mew=0.6,
                               label="Cells (Xenium / CODEX)")],
               loc="upper center", bbox_to_anchor=((98 + 176) / (2 * W), 1 - (hy + 46) / H), ncol=2, frameon=False,
               fontsize=FS_SMALL)

    save(fig, "fig6_interface")


if __name__ == "__main__":
    main()
