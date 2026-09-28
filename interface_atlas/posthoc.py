"""B1 pilot POST-HOC analyses (not pre-registered), computed from gradient_curves.csv.gz.

1. Type-specificity: the pre-registered r is inflated by the shared 'everything non-epithelial rises outside the
   boundary' shape. Here each lineage curve is divided by the band's total non-epithelial fraction (composition within
   the non-epithelial compartment), for FD and the orthogonal modality alike, then Pearson r across bands is recomputed.
   Matched vs mismatched pairs quantify how much agreement is lineage-specific.
2. Cross-cancer descriptive summary: per section and lineage, FD fraction in the stroma-side band (+25..+100 um)
   divided by the tumour-edge band (-50..0 um) (log2), and the peak position of the FD curve.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("/Users/apple/Research/FlashDeconv/results/b1_pilot")
PRIMARY = ["CRC_P1", "CRC_P2", "CRC_P5", "SPATCH_COAD", "SPATCH_OV"]
ALL = PRIMARY + ["SPATCH_HCC"]
CODEX_COMPARE = ["Fibroblast", "Endothelial", "CD4 T", "CD8 T", "Treg", "NK", "B", "Macrophage/Mono"]
NONEPI_FD = ["Fibroblast", "Pericyte/SMC", "Endothelial", "CD4 T", "CD8 T", "Treg", "NK", "B", "Plasma",
             "Macrophage/Mono", "DC", "Neutrophil", "Mast"]


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan


cur = pd.read_csv(RES / "gradient_curves.csv.gz")
conc = pd.read_csv(RES / "concordance_per_lineage.csv")
rows, srows = [], []
for sec in ALL:
    sub = cur[cur.section == sec]
    mod = [s for s in sub.source.unique() if s not in ("FD", "HDmarker")][0]
    W = lambda src: sub[sub.source == src].pivot(index="band_mid", columns="name", values="value")
    F, O = W("FD"), W(mod)
    if mod == "CODEX":
        F = F.assign(Fibroblast=F["Fibroblast"] + F["Pericyte/SMC"])
        fd_tot = F[CODEX_COMPARE].sum(1)
        or_tot = O[CODEX_COMPARE].sum(1)
    else:
        fd_tot = F[NONEPI_FD].sum(1)
        or_tot = O[NONEPI_FD].sum(1)
    inc = conc[(conc.section == sec) & conc.included].lineage.tolist()
    Fs, Os = F[inc].div(fd_tot, axis=0), O[inc].div(or_tot, axis=0)
    for l in inc:
        rows.append(dict(section=sec, lineage=l, r_raw=conc[(conc.section == sec) & (conc.lineage == l)].r.iloc[0],
                         r_within_nonepi=r(Fs[l], Os[l])))
    mm = [r(Fs[a], Os[b]) for a in inc for b in inc if a != b]
    mt = [r(Fs[a], Os[a]) for a in inc]
    srows.append(dict(section=sec, n=len(inc), median_r_within=np.nanmedian(mt), mean_matched_within=np.nanmean(mt),
                      mean_mismatched_within=np.nanmean(mm), r_total_nonepi=r(fd_tot, or_tot)))
ph = pd.DataFrame(rows)
phs = pd.DataFrame(srows)
ph.to_csv(RES / "posthoc_within_nonepi_concordance.csv", index=False)
phs.to_csv(RES / "posthoc_within_nonepi_summary.csv", index=False)
pd.set_option("display.width", 200)
print(ph.round(3).to_string())
print(phs.round(3).to_string())

# cross-cancer descriptive
xr = []
for sec in ALL:
    F = cur[(cur.section == sec) & (cur.source == "FD")].pivot(index="band_mid", columns="name", values="value")
    for l in F.columns:
        if l == "Other":
            continue
        edge = F.loc[(F.index > -50) & (F.index < 0), l].mean()
        strom = F.loc[(F.index > 25) & (F.index < 100), l].mean()
        core = F.loc[F.index < -150, l].mean()
        far = F.loc[F.index > 250, l].mean()
        xr.append(dict(section=sec, lineage=l, mean=F[l].mean(), log2_stroma_vs_edge=np.log2((strom + 1e-4) / (edge + 1e-4)),
                       log2_edge_vs_core=np.log2((edge + 1e-4) / (core + 1e-4)),
                       log2_near_vs_far_stroma=np.log2((strom + 1e-4) / (far + 1e-4)),
                       peak_um=float(F[l].idxmax())))
X = pd.DataFrame(xr)
X.to_csv(RES / "cross_cancer_descriptors.csv", index=False)
for m in ["peak_um", "log2_near_vs_far_stroma", "log2_edge_vs_core"]:
    print(m)
    print(X[X.section.isin(PRIMARY)].pivot(index="lineage", columns="section", values=m).round(2).to_string())

# 3. Same within-compartment test for raw HD marker curves (marker CPM / FD non-epithelial total): does a
#    deconvolution-free HD readout do better than FlashDeconv once the shared shape is removed?
DIAG_TARGET = {"CD3E+CD3D": ["CD4 T", "CD8 T", "Treg"], "MS4A1": ["B"], "CD68": ["Macrophage/Mono"],
               "COL1A1": ["Fibroblast"], "PECAM1": ["Endothelial"]}
mrows = []
for sec in PRIMARY:
    sub = cur[cur.section == sec]
    mod = [s for s in sub.source.unique() if s not in ("FD", "HDmarker")][0]
    W = lambda src: sub[sub.source == src].pivot(index="band_mid", columns="name", values="value")
    F, O, M = W("FD"), W(mod), W("HDmarker")
    if mod == "CODEX":
        F = F.assign(Fibroblast=F["Fibroblast"] + F["Pericyte/SMC"])
        fd_tot, or_tot = F[CODEX_COMPARE].sum(1), O[CODEX_COMPARE].sum(1)
    else:
        fd_tot, or_tot = F[NONEPI_FD].sum(1), O[NONEPI_FD].sum(1)
    for mk, tg in DIAG_TARGET.items():
        tg = [t for t in tg if t in O.columns]
        o = O[tg].sum(1) / or_tot
        if O[tg].sum(1).mean() < 0.005:
            continue
        mrows.append(dict(section=sec, marker=mk, r_marker_within=r(M[mk] / fd_tot, o), r_fd_within=r(F[tg].sum(1) / fd_tot, o)))
MR = pd.DataFrame(mrows)
MR.to_csv(RES / "posthoc_marker_within_nonepi.csv", index=False)
print(MR.round(3).to_string())
print("median marker-within", MR.r_marker_within.median(), "median FD-within", MR.r_fd_within.median())
