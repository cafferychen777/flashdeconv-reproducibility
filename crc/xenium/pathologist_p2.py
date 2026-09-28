"""Pathologist-region concordance on the CORRECT patient.

data/visium_hd_crc_cohort/pathologist_annotations/8um_squares_annotation.csv
(SpaceHack) annotates patient P2 (545,913 barcodes = P2's in-tissue 8 um bins);
the earlier analysis joined it with P1 because 8 um barcodes encode only the array
position and are shared across slides. This recomputes pathologist_concordance()
(unchanged function of xenium_crc_validation.py, no cache) for P2 with the
ORIG (archived), V020 and FINAL proportions, and also records the barcode match
of the annotation with every patient.

Usage: python pathologist_p2.py <v020_P2_npz> <final_P2_npz>
"""
import os
import sys
from pathlib import Path

B = Path("/Users/apple/Research/FlashDeconv")
OUT = B / "results/rerun_final/crc/xenium/visium_hd_p2"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("RERUN_OUT_DIR", str(OUT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import anndata as ad  # noqa: E402
import xenium_crc_validation as xv  # noqa: E402

patho = pd.read_csv(xv.PATHO_CSV, sep="\t", header=None, names=["barcode", "category"]).set_index("barcode")
match = []
for sid in ["P1_CRC", "P2_CRC", "P5_CRC"]:
    a = ad.read_h5ad(B / f"analysis/crc_cohort_results/{sid}_deconv.h5ad", backed="r")
    bc = pd.Index(a.obs_names)
    match.append(dict(sample=sid, n_bins=len(bc), n_annotation=len(patho),
                      n_matched=int(bc.isin(patho.index).sum()),
                      exact_same_set=bool(len(bc) == len(patho) and bc.isin(patho.index).all())))
    a.file.close()
pd.DataFrame(match).to_csv(OUT / "annotation_barcode_match_by_patient.csv", index=False)
print(pd.DataFrame(match).to_string())


def wrap(npz, path):
    z = np.load(npz, allow_pickle=True)
    df = pd.DataFrame(z["P"].astype(np.float64), index=[str(b) for b in z["barcodes"]],
                      columns=[str(t) for t in z["types"]])
    a = ad.AnnData(obs=pd.DataFrame(index=df.index))
    a.obsm["flashdeconv"] = df
    a.write_h5ad(path)
    return path


fits = {"orig": B / "analysis/crc_cohort_results/P2_CRC_deconv.h5ad",
        "v020": wrap(sys.argv[1], OUT / "_tmp_v020.h5ad"),
        "final": wrap(sys.argv[2], OUT / "_tmp_final.h5ad")}
for name, h5 in fits.items():
    sub = OUT / f"props_{name}"
    sub.mkdir(exist_ok=True)
    xv.pathologist_concordance(fd_h5ad=h5, patho_csv=xv.PATHO_CSV, output_dir=sub)
for name in ["v020", "final"]:
    os.remove(fits[name])
