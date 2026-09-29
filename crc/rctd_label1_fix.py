"""Recompute the RCTD-dependent CRC tables after the singlet-label fix.

The Oliveira et al. RCTD release stores the singlet call in DeconvolutionLabel1
(DeconvolutionLabel2 is the runner-up type). lineage_marker_validation() used
Label2; it now uses Label1 (crc_common.py). rctd.csv already holds
neut_singlet_label1, so only lineage_markers.csv is rewritten here, from the saved
proportions (ORIG: archived h5ad; FINAL: float16 npz copied from arseven).
Also writes rctd_na_diagnostics.csv: what the 'NA' class (no RCTD call) is, and
rctd_lineage_agreement.csv: on RCTD singlet bins, agreement between the lineage of the
FlashDeconv argmax type and the lineage of the RCTD singlet type (both via LINEAGE_MAP),
for FINAL and the lambda=0 control (LAM0).

Usage: python rctd_label1_fix.py <final_props_dir> <lam0_props_dir>
(lam0_props_dir holds <sample>_lam0_8um.npz, written by controls_editor/c2_crc_lam0.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crc_common as cc  # noqa: E402

PROJ = Path("/Users/apple/Research/FlashDeconv")
ST_DIR = PROJ / "analysis" / "crc_cohort_results"
RCTD_DIR = PROJ / "data" / "visium_hd_crc_cohort" / "metadata"
R = PROJ / "results" / "rerun_final" / "crc"
PROPS = Path(sys.argv[1])
LAM0_PROPS = Path(sys.argv[2])
SAMPLES = ["P1_CRC", "P2_CRC", "P5_CRC"]


def load_rctd(sid, barcodes):
    rc = pd.read_csv(RCTD_DIR / f"DeconvolutionResults_{sid.replace('_CRC', '')}CRC.csv.gz",
                     usecols=["barcode", "DeconvolutionClass", "DeconvolutionLabel1", "DeconvolutionLabel2"],
                     dtype="string", keep_default_na=False, low_memory=False).set_index("barcode")
    n_file = len(rc)
    rc = rc.reindex(barcodes)
    return n_file, {"cls": rc["DeconvolutionClass"].fillna("missing").astype(str).to_numpy(),
                    "l1": rc["DeconvolutionLabel1"].fillna("NA").astype(str).to_numpy(),
                    "l2": rc["DeconvolutionLabel2"].fillna("NA").astype(str).to_numpy()}


diag, agree = [], []
for sid in SAMPLES:
    st = sc.read_h5ad(ST_DIR / f"{sid}_deconv.h5ad")
    barcodes = np.asarray(st.obs_names).astype(str)
    Xraw = st.X.tocsr() if sparse.issparse(st.X) else sparse.csr_matrix(st.X)
    umi = np.asarray(Xraw.sum(1)).ravel()
    lin_expr = cc.gene_columns(Xraw, np.asarray(st.var_names).astype(str), cc.IMMUNE_MARKERS + cc.STROMAL_MARKERS)
    z = np.load(PROPS / f"{sid}_final_8um.npz", allow_pickle=True)
    types = [str(t) for t in z["types"]]
    assert np.array_equal(z["barcodes"].astype(str), barcodes)
    fits = {"ORIG": st.obsm["flashdeconv"][types].to_numpy(np.float32), "FINAL": z["P"].astype(np.float32)}
    n_file, rctd = load_rctd(sid, barcodes)

    zl = np.load(LAM0_PROPS / f"{sid}_lam0_8um.npz", allow_pickle=True)
    assert [str(t) for t in zl["types"]] == types and np.array_equal(zl["barcodes"].astype(str), barcodes)
    sing = rctd["cls"] == "singlet"
    rl = pd.Series(rctd["l1"]).map(cc.LINEAGE_MAP).fillna("Other").to_numpy()
    for name, P in [("FINAL", fits["FINAL"]), ("LAM0", zl["P"].astype(np.float32))]:
        fl = pd.Series(np.array(types)[P.argmax(1)]).map(cc.LINEAGE_MAP).fillna("Other").to_numpy()
        agree.append({"sample": sid, "fit": name, "n_rctd_singlet_bins": int(sing.sum()),
                      "n_lineage_agree": int((fl[sing] == rl[sing]).sum()),
                      "pct_lineage_agree": float(100 * (fl[sing] == rl[sing]).mean())})

    old = pd.read_csv(R / sid / "lineage_markers.csv")
    rows = []
    for name, P in fits.items():
        rows += [{"sample": sid, "fit": name, **r} for r in cc.lineage_marker_validation(P, types, rctd, lin_expr)]
    new = pd.DataFrame(rows)[old.columns]
    new.to_csv(R / sid / "lineage_markers.csv", index=False)

    # what 'NA' (no RCTD class) means: UMI distribution of bins by RCTD class
    cls = rctd["cls"]
    for c in ["singlet", "doublet_certain", "doublet_uncertain", "reject", "NA", "missing"]:
        m = cls == c
        if not m.any():
            continue
        diag.append({"sample": sid, "rctd_class": c, "n_bins": int(m.sum()), "frac_bins": float(m.mean()),
                     "umi_min": float(umi[m].min()), "umi_median": float(np.median(umi[m])),
                     "umi_max": float(umi[m].max()), "frac_umi_lt100": float((umi[m] < 100).mean()),
                     "n_bins_h5ad": len(barcodes), "n_barcodes_rctd_file": n_file})
    print(sid, "done", flush=True)

a = pd.DataFrame(agree)
a.to_csv(R / "rctd_lineage_agreement.csv", index=False)
print(a.round(2).to_string())
d = pd.DataFrame(diag)
d.to_csv(R / "rctd_na_diagnostics.csv", index=False)
pd.set_option("display.width", 200)
print(d.to_string())
