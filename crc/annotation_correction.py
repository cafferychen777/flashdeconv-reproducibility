"""Apply the P2 pathologist-annotation correction to the final claim tables.

The SpaceHack annotation (8um_squares_annotation.csv) belongs to P2 (exact match of
all 545,913 in-tissue 8 um barcodes). Its only consumer is pathologist_concordance()
in xenium_crc_validation.py (Supp. Note, Supp. Fig. xenium_crc_validation d). The
boundary-distance claims use the per-patient published Periphery annotations
(P{1,2,5}CRC_periphery.csv.gz; bin counts match each patient exactly) and are unaffected.
Run after merge_final.py, build_claims_final.py and part_summary.py.
"""
from pathlib import Path
import pandas as pd

R = Path("/Users/apple/Research/FlashDeconv/results/rerun_final/crc")
fix = pd.read_csv(R / "xenium/pathologist_claims_P1wrong_vs_P2correct.csv", index_col=0)

# 1) claims_final.csv: no claim there uses the SpaceHack annotation
c = pd.read_csv(R / "claims_final.csv", index_col=0)
c["annotation_patient_corrected"] = "no - does not use the SpaceHack pathologist annotation"
bd = [i for i in c.index if i.startswith(("tumor %", "Plasma max", "CAF ", "Macrophage max", "CD8 %", "total immune"))]
c.loc[bd, "annotation_patient_corrected"] = ("no - boundary distance uses per-patient published Periphery "
                                              "annotation (P1/P2/P5 files match each patient's bins); unaffected")
c.to_csv(R / "claims_final.csv")

# 2) Xenium claim summary: P1-joined pathologist rows are invalid; add P2 rows
x = pd.read_csv(R / "xenium/claims_summary_final.csv")
x = x[~x.claim.str.startswith("Patho (P2")]
x["annotation_patient_corrected"] = "no - does not use the SpaceHack pathologist annotation"
isp = x.claim.str.startswith("Patho")
x.loc[isp, "annotation_patient_corrected"] = ("INVALID - annotation (P2) was joined to P1 bins; "
                                              "superseded by the 'Patho (P2 ...)' rows")
man = {"Patho matched annotated bins (excl. Outside) (456,107)": "456,107",
       "Patho Neoplasm share of bins (58%)": "58%", "Patho Vessel share of bins (0.7%)": "0.7%",
       "Patho Neoplasm Tumor/Stromal/Immune mean (27.6/31.3/21.7%)": "27.6/31.3/21.7%",
       "Patho Vessel Tumor mean (61.7%)": "61.7%",
       "Patho fold vs tissue mean Neoplasm Tumor / Vessel Tumor (0.8x / 1.8x)": "0.8x / 1.8x",
       "Patho dominant lineage matches expected (of 6 categories)": ""}
new = []
for k, m in man.items():
    new.append(dict(claim="Patho (P2, correct patient) " + k.replace("Patho ", ""), manuscript=m,
                    ORIG=fix.loc[k, "ORIG [P2_correct]"], V020=fix.loc[k, "V020 [P2_correct]"],
                    FINAL=fix.loc[k, "FINAL [P2_correct]"], holds="",
                    note=f"before (P1 join): ORIG {fix.loc[k, 'ORIG [P1_wrong]']}; FINAL {fix.loc[k, 'FINAL [P1_wrong]']}",
                    annotation_patient_corrected="yes - recomputed on P2 deconvolution (545,913/545,913 barcodes matched)"))
x = pd.concat([x, pd.DataFrame(new)], ignore_index=True)
x.to_csv(R / "xenium/claims_summary_final.csv", index=False)

# 3) part summary
p = pd.read_csv(R / "part_summary_crc.csv")
p = p[~p.benchmark.str.startswith("Pathologist concordance")]
k = "Patho fold vs tissue mean Neoplasm Tumor / Vessel Tumor (0.8x / 1.8x)"
k2 = "Patho Neoplasm Tumor/Stromal/Immune mean (27.6/31.3/21.7%)"
p = pd.concat([p, pd.DataFrame([
    dict(benchmark="Pathologist concordance P2 (corrected patient)", metric="tumor fold vs tissue mean Neoplasm / Vessel",
         manuscript_value="0.8x / 1.8x (computed on P1 - wrong patient)", v020_value=fix.loc[k, "V020 [P2_correct]"],
         final_value=fix.loc[k, "FINAL [P2_correct]"], rank="", p_value="",
         notes="annotation belongs to P2; P1 join was invalid. Direction reverses: Neoplasm tumor-enriched, Vessel tumor-depleted"),
    dict(benchmark="Pathologist concordance P2 (corrected patient)", metric="Neoplasm Tumor/Stromal/Immune mean",
         manuscript_value="27.6/31.3/21.7% (P1 - wrong patient)", v020_value=fix.loc[k2, "V020 [P2_correct]"],
         final_value=fix.loc[k2, "FINAL [P2_correct]"], rank="", p_value="", notes="ORIG on P2: " + fix.loc[k2, "ORIG [P2_correct]"])])],
    ignore_index=True)
p.to_csv(R / "part_summary_crc.csv", index=False)
print(x[x.claim.str.startswith("Patho")][["claim", "manuscript", "ORIG", "FINAL", "annotation_patient_corrected"]].to_string())
