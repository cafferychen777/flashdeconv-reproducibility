"""Control 2b summary: interface-atlas descriptors for the default fit (results/b1_pilot) and the
lambda_spatial=0 refit (results/controls_editor/b1), with the definitions of posthoc_v2.py
(rim = log2 0..50 / 150..300 um; supported = same sign, |orth| >= 0.3, orth mean >= 0.3%) and the
Fig. 6g fibroblast step (log2 +25..+100 um over -50..0 um)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/apple/Research/FlashDeconv/results")
DIRS = {"default": ROOT / "b1_pilot", "lam0": ROOT / "controls_editor" / "b1"}
CROSS = ["CRC_P1", "CRC_P2", "CRC_P5", "SPATCH_COAD", "SPATCH_OV", "LUNG_X1", "LUNG_X5K", "OV10X"]
LIN = ["Fibroblast", "Pericyte/SMC", "Endothelial", "CD4 T", "CD8 T", "Treg", "NK", "B", "Plasma",
       "Macrophage/Mono", "DC", "Neutrophil", "Mast"]
CODEX_COMPARE = ["Fibroblast", "Endothelial", "CD4 T", "CD8 T", "Treg", "NK", "B", "Macrophage/Mono"]
E = 1e-4


def m(C, l, lo, hi):
    x = C.index.values
    return C.loc[(x > lo) & (x < hi), l].mean()


rows, secs = [], []
for lab, R in DIRS.items():
    cur = pd.concat([pd.read_csv(R / "gradient_curves.csv.gz"), pd.read_csv(R / "gradient_curves_ext.csv.gz")])
    cur.loc[cur.n_units < 50, "value"] = np.nan
    sm = pd.concat([pd.read_csv(R / "section_summary.csv"), pd.read_csv(R / "section_summary_ext.csv")])
    for r in sm.itertuples():
        secs.append(dict(fit=lab, section=r.section, median_r=r.median_r, matched=r.mean_matched_r,
                         mismatched=r.mean_mismatched_r, passes=r.passes))
    for sec in CROSS:
        sub = cur[cur.section == sec]
        mod = [s for s in sub.source.unique() if s not in ("FD", "HDmarker")][0]
        W = lambda src: sub[sub.source == src].pivot(index="band_mid", columns="name", values="value")
        F, O = W("FD"), W(mod)
        F0 = F.copy()
        if mod == "CODEX":
            F = F.assign(Fibroblast=F["Fibroblast"] + F["Pericyte/SMC"])
        for l in LIN:
            fe = np.log2((m(F, l, 0, 50) + E) / (m(F, l, 150, 300) + E))
            fstep = np.log2((m(F0, l, 25, 100) + E) / (m(F0, l, -50, 0) + E))
            ok = l in O.columns and (mod != "CODEX" or l in CODEX_COMPARE)
            oe = np.log2((m(O, l, 0, 50) + E) / (m(O, l, 150, 300) + E)) if ok else np.nan
            ostep = np.log2((m(O, l, 25, 100) + E) / (m(O, l, -50, 0) + E)) if ok else np.nan
            om = float(O[l].mean()) if ok else np.nan
            rows.append(dict(fit=lab, section=sec, modality=mod, lineage=l, fd_mean=float(F[l].mean()), orth_mean=om,
                             fd_rim=fe, orth_rim=oe, fd_step=fstep, orth_step=ostep,
                             testable=bool(np.isfinite(oe) and om >= 0.003),
                             supported=bool(np.isfinite(oe) and om >= 0.003 and np.sign(fe) == np.sign(oe) and abs(oe) >= 0.3)))
D, S = pd.DataFrame(rows), pd.DataFrame(secs)
out = ROOT / "controls_editor"
D.to_csv(out / "c2_b1_descriptors_default_vs_lam0.csv", index=False)
S.to_csv(out / "c2_b1_concordance_default_vs_lam0.csv", index=False)
pd.set_option("display.width", 250)
print(S.pivot(index="section", columns="fit", values=["median_r", "matched", "mismatched"]).round(3).to_string())
for l, col in [("Fibroblast", "fd_step"), ("Macrophage/Mono", "fd_rim"), ("Pericyte/SMC", "fd_rim"), ("CD8 T", "fd_rim")]:
    print(f"\n{l} {col}")
    print(D[D.lineage == l].pivot(index="section", columns="fit", values=[col, "supported"]).reindex(CROSS).round(2).to_string())
    for f in DIRS:
        d = D[(D.fit == f) & (D.lineage == l)]
        print(f, "positive FD rim in primary 5:", int((d[d.section.isin(CROSS[:5])].fd_rim > 0).sum()),
              "| supported/testable:", int(d.supported.sum()), "/", int(d.testable.sum()))
print("\north fibroblast step:", D[(D.fit == "default") & (D.lineage == "Fibroblast")].set_index("section").orth_step.round(2).to_dict())
# overall agreement of rim signs across all lineage x sections
a = D.pivot_table(index=["section", "lineage"], columns="fit", values="fd_rim")
print("\nrim default vs lam0: r =", round(a.corr().iloc[0, 1], 3), "sign agreement =", round((np.sign(a["default"]) == np.sign(a["lam0"])).mean(), 3),
      "max |diff| =", round((a["default"] - a["lam0"]).abs().max(), 2))
print("supported total default/lam0:", D.groupby("fit").supported.sum().to_dict())
