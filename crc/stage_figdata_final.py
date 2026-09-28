"""Stage figure input tables (same file names/columns as analysis/crc_cohort_results)
for one fit (ORIG, V020 or FINAL) from the rerun outputs.

Usage: python stage_figdata.py <results_dir> <FIT> <out_dir>
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

R, FIT, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES = ["P1_CRC", "P2_CRC", "P5_CRC"]


def load(name):
    d = pd.concat([pd.read_csv(R / s / f"{name}.csv") for s in SAMPLES], ignore_index=True)
    return d[d.fit == FIT].drop(columns="fit")


load("aggregates").to_csv(OUT / "neutrophil_microdomains_summary.csv", index=False)
load("markers").to_csv(OUT / "neutrophil_marker_validation.csv", index=False)
load("lineage_markers").to_csv(OUT / "marker_gene_validation.csv", index=False)
b = load("boundary")
b["distance_mid_um"] = (b.distance_min_um + b.distance_max_um) / 2
b.to_csv(OUT / "tumor_boundary_gradient.csv", index=False)
if FIT in ("V020", "FINAL"):
    pd.concat([pd.read_csv(R / s / "multires_enrichment.csv") for s in SAMPLES]).to_csv(
        OUT / "neutrophil_multiresolution_enrichment.csv", index=False)
else:
    shutil.copy("/Users/apple/Research/FlashDeconv/analysis/crc_cohort_results/"
                "neutrophil_multiresolution_enrichment.csv", OUT)
for s in SAMPLES:
    dst = OUT / f"{s}_figdata.npz"
    if not dst.exists():
        dst.symlink_to((R / s / "figdata.npz").resolve())
print("staged", FIT, "->", OUT)
