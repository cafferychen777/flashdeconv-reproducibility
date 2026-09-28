"""B1 pilot extension (exploratory): all sections combined.

1. Within-non-epithelial concordance and matched-vs-mismatched gap for the extension sections.
2. Interface descriptors per section/lineage, for FlashDeconv AND the orthogonal cells:
     edge  = log2(mean fraction in 0..+50 um / mean in +150..+300 um)   (>0: enriched at the stromal rim)
     inside = log2(mean in -50..0 um / mean in -200..-100 um)            (>0: enriched at the tumour-side edge)
   A pattern is 'supported' in a section when FD and orthogonal descriptors have the same sign and |orth| >= 0.3.
3. Ovarian SPATCH vs 10x: correlation of FD curves (and of orthogonal curves) per lineage between the two tumours.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("/Users/apple/Research/FlashDeconv/results/b1_pilot")
CROSS = ["CRC_P1", "CRC_P2", "CRC_P5", "SPATCH_COAD", "SPATCH_OV", "LUNG_X1", "LUNG_X5K", "OV10X"]
EXT = ["LUNG_X1", "LUNG_X5K", "OV10X", "OV10X_v1"]
LIN = ["Fibroblast", "Pericyte/SMC", "Endothelial", "CD4 T", "CD8 T", "Treg", "NK", "B", "Plasma",
       "Macrophage/Mono", "DC", "Neutrophil", "Mast"]
CODEX_COMPARE = ["Fibroblast", "Endothelial", "CD4 T", "CD8 T", "Treg", "NK", "B", "Macrophage/Mono"]


def r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan


cur = pd.concat([pd.read_csv(RES / "gradient_curves.csv.gz"), pd.read_csv(RES / "gradient_curves_ext.csv.gz")])
conc = pd.concat([pd.read_csv(RES / "concordance_per_lineage.csv"), pd.read_csv(RES / "concordance_per_lineage_ext.csv")])


# Bands with < 50 units in a modality are set to NaN (band_stats zero-fills empty bands). This only affects
# SPATCH_HCC, LUNG_X5K (deepest tumour bands in HD) and OV10X/OV10X_v1 (far-stroma bands); primary sections are unaffected.
cur.loc[cur.n_units < 50, "value"] = np.nan


def tables(sec):
    sub = cur[cur.section == sec]
    mod = [s for s in sub.source.unique() if s not in ("FD", "HDmarker")][0]
    W = lambda src: sub[sub.source == src].pivot(index="band_mid", columns="name", values="value")
    F, O = W("FD"), W(mod)
    if mod == "CODEX":
        F = F.assign(Fibroblast=F["Fibroblast"] + F["Pericyte/SMC"])
    return F, O, mod


# 1. within-compartment concordance for extension sections
rows, srows = [], []
for sec in EXT:
    F, O, mod = tables(sec)
    comp = [l for l in LIN if l in O.columns]
    ft, ot = F[comp].sum(1), O[comp].sum(1)
    inc = [l for l in comp if not (l == "Treg" and sec.startswith("CRC")) and np.nanmean(O[l]) >= 0.005]
    Fs, Os = F[inc].div(ft, axis=0), O[inc].div(ot, axis=0)
    mt = [r(Fs[l], Os[l]) for l in inc]
    mm = [r(Fs[a], Os[b]) for a in inc for b in inc if a != b]
    mt_raw = [r(F[l], O[l]) for l in inc]
    mm_raw = [r(F[a], O[b]) for a in inc for b in inc if a != b]
    for l, a in zip(inc, mt):
        rows.append(dict(section=sec, lineage=l, r_raw=r(F[l], O[l]), r_within_nonepi=a))
    srows.append(dict(section=sec, n=len(inc), included=",".join(inc), median_r_raw=np.nanmedian(mt_raw), matched_raw=np.nanmean(mt_raw),
                      mismatched_raw=np.nanmean(mm_raw), median_r_within=np.nanmedian(mt),
                      matched_within=np.nanmean(mt), mismatched_within=np.nanmean(mm), r_total_nonepi=r(ft, ot)))
pd.DataFrame(rows).to_csv(RES / "posthoc_within_nonepi_concordance_ext.csv", index=False)
S = pd.DataFrame(srows)
S.to_csv(RES / "posthoc_within_nonepi_summary_ext.csv", index=False)
pd.set_option("display.width", 220)
print(S.round(3).to_string())

# 2. interface descriptors
def desc(C, l):
    x = C.index.values
    m = lambda lo, hi: C.loc[(x > lo) & (x < hi), l].mean()
    e = 1e-4
    return (np.log2((m(0, 50) + e) / (m(150, 300) + e)), np.log2((m(-50, 0) + e) / (m(-200, -100) + e)),
            float(C[l].mean()))


drows = []
for sec in CROSS:
    F, O, mod = tables(sec)
    for l in LIN:
        if l not in F.columns:
            continue
        fe, fi, fm = desc(F, l)
        if l in O.columns and (mod != "CODEX" or l in CODEX_COMPARE):
            oe, oi, om = desc(O, l)
        else:
            oe = oi = om = np.nan
        drows.append(dict(section=sec, modality=mod, lineage=l, fd_mean=fm, orth_mean=om, fd_edge=fe, orth_edge=oe,
                          fd_inside=fi, orth_inside=oi,
                          edge_supported=bool(np.isfinite(oe) and om >= 0.003 and np.sign(fe) == np.sign(oe) and abs(oe) >= 0.3)))
D = pd.DataFrame(drows)
D.to_csv(RES / "cross_cancer_interface_descriptors_v2.csv", index=False)
print("\nFD rim enrichment log2(0..50 / 150..300)")
print(D.pivot(index="lineage", columns="section", values="fd_edge").reindex(LIN)[CROSS].round(2).to_string())
print("\northogonal rim enrichment")
print(D.pivot(index="lineage", columns="section", values="orth_edge").reindex(LIN)[CROSS].round(2).to_string())
print("\nsupported (same sign, |orth|>=0.3, orth mean>=0.3%)")
print(D.pivot(index="lineage", columns="section", values="edge_supported").reindex(LIN)[CROSS].to_string())

# 3. ovarian SPATCH vs 10x, and lung X1 vs X5K (same block) as a within-tumour reference
prow = []
for a, b in [("SPATCH_OV", "OV10X"), ("LUNG_X1", "LUNG_X5K"), ("CRC_P1", "CRC_P2"), ("CRC_P1", "CRC_P5"), ("CRC_P2", "CRC_P5")]:
    Fa, Oa, ma = tables(a)
    Fb, Ob, mb = tables(b)
    Fa0 = cur[(cur.section == a) & (cur.source == "FD")].pivot(index="band_mid", columns="name", values="value")
    Fb0 = cur[(cur.section == b) & (cur.source == "FD")].pivot(index="band_mid", columns="name", values="value")
    for l in LIN:
        prow.append(dict(pair=f"{a} vs {b}", lineage=l, r_fd=r(Fa0[l], Fb0[l]),
                         r_orth=r(Oa[l], Ob[l]) if (l in Oa.columns and l in Ob.columns and Oa[l].mean() > 0.003 and Ob[l].mean() > 0.003) else np.nan,
                         fd_mean_a=Fa0[l].mean(), fd_mean_b=Fb0[l].mean()))
P = pd.DataFrame(prow)
P.to_csv(RES / "section_pair_similarity_v2.csv", index=False)
print("\npairwise curve similarity")
print(P.pivot(index="lineage", columns="pair", values="r_fd").round(2).to_string())
print(P.groupby("pair")[["r_fd", "r_orth"]].median().round(3).to_string())
