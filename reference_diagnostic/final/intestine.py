"""Final-package rerun of the incomplete-reference diagnostic on Visium HD mouse small intestine
(8 um official bins): Haber epithelium-only vs composite reference.

Usage: FD_FITLOG=... python intestine.py [default|manuscript]
  default    : FlashDeconv() package defaults (lambda_spatial='auto', max_iter=1000, tol=1e-4)
  manuscript : manuscript solver settings of the prototype (lambda_spatial=5000) with the final
               max_iter/tol defaults (sensitivity analysis)
Regions are the raw-marker masks of validation/intestine_pg_robustness (as in the prototype
v2b_package_regions.py): follicle_buf, muscle, epi_low, clean (= outside mask_full, epithelium).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

ROOT = Path("/Users/apple/Research/FlashDeconv")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation/rerun_final"))
sys.path.insert(0, str(ROOT / "validation/tuft_investigation"))
import fdfinal  # noqa: E402,F401
from common import load_raw  # noqa: E402
import flashdeconv  # noqa: E402
from flashdeconv import FlashDeconv  # noqa: E402
from flashdeconv.io.loader import prepare_data  # noqa: E402

SETTING = sys.argv[1] if len(sys.argv) > 1 else "default"
OUT = ROOT / "results/rerun_final/refdiag"
REFS = {"haber": ROOT / "validation/visium_hd_data/haber_intestine_matched.h5ad",
        "composite": ROOT / "validation/intestine_reference_v2/results/reference/composite_reference.h5ad"}
KW = {"default": dict(verbose=False),
      "manuscript": dict(sketch_dim=512, lambda_spatial=5000.0, rho_sparsity=0.01, n_hvg=2000,
                         n_markers_per_type=50, k_neighbors=6, verbose=False)}[SETTING]

ad = load_raw("008um")
mask = pd.read_csv(ROOT / "validation/intestine_pg_robustness/results/mask_8um.csv.gz",
                   index_col=0).reindex(ad.obs_names)
assert mask.mask_full.notna().all()
regions = {"follicle": mask.follicle_buf.to_numpy(bool), "muscle": mask.muscle.to_numpy(bool),
           "epi_low": mask.epi_low.to_numpy(bool), "epithelium": ~mask.mask_full.to_numpy(bool)}

comp = sc.read_h5ad(REFS["composite"])
names = sorted(comp.obs.celltype1.unique())
Xc = sparse.csr_matrix(comp.X)
atlas = np.vstack([np.asarray(Xc[(comp.obs.celltype1 == t).to_numpy()].mean(0)).ravel() for t in names])
agenes = list(comp.var_names)
del comp, Xc

rows, grows, trows, perbin = [], [], [], {}
for name, path in REFS.items():
    os.environ["FD_CONTEXT"] = f"intestine8um_{SETTING}_{name}"
    ref = sc.read_h5ad(path)
    Y, X, crd, ct, genes = prepare_data(ad, ref, cell_type_key="celltype1")
    del ref
    m = FlashDeconv(**KW)
    t0 = time.time()
    m.fit(sparse.csr_matrix(Y).astype(float), X, crd, cell_type_names=np.array(ct))
    tf = time.time() - t0
    t0 = time.time()
    s = flashdeconv.reference_fit_scores(m)
    td = time.time() - t0
    info = m.info_ or {}
    print(f"[{SETTING}/{name}] K={len(ct)} fit {tf:.0f}s diag {td:.0f}s iters={info.get('n_iterations')} "
          f"converged={info.get('converged')} lambda={m.lambda_used_}", flush=True)
    perbin[f"{name}_score_pooled"] = s["score_pooled"].astype(np.float32)
    perbin[f"{name}_flag_pooled"] = s["flag_pooled"]
    for rn, r in regions.items():
        rows.append({"setting": SETTING, "ref": name, "region": rn, "n_bins": int(r.sum()),
                     "flag": s["flag"][r].mean(), "flag_pooled": s["flag_pooled"][r].mean(),
                     "median_score_pooled": float(np.median(s["score_pooled"][r])),
                     "n_iterations": info.get("n_iterations"), "converged": info.get("converged"),
                     "lambda_used": m.lambda_used_, "fit_s": tf, "diag_s": td})
    rows.append({"setting": SETTING, "ref": name, "region": "all", "n_bins": len(s["flag"]),
                 "flag": s["flag"].mean(), "flag_pooled": s["flag_pooled"].mean(),
                 "median_score_pooled": float(np.median(s["score_pooled"])),
                 "n_iterations": info.get("n_iterations"), "converged": info.get("converged"),
                 "lambda_used": m.lambda_used_, "fit_s": tf, "diag_s": td})
    sets = {"all_flagged": s["flag_pooled"],
            "flagged_in_follicle": s["flag_pooled"] & regions["follicle"],
            "flagged_in_muscle": s["flag_pooled"] & regions["muscle"],
            "follicle_region": regions["follicle"], "muscle_region": regions["muscle"]}
    for sn, R in sets.items():
        if R.sum() < 20:
            print(f"  {sn}: only {R.sum()} bins, skipped", flush=True)
            continue
        g = flashdeconv.unexplained_genes(m, R, gene_names=genes)
        t = flashdeconv.suggest_missing_types(g, atlas, agenes, names)
        for i in range(30):
            grows.append({"setting": SETTING, "ref": name, "set": sn, "n_bins": int(R.sum()),
                          "rank": i + 1, "gene": g["gene"][i], "score": g["score"][i],
                          "observed": g["observed"][i], "expected": g["expected"][i]})
        for i in range(len(t["type"])):
            trows.append({"setting": SETTING, "ref": name, "set": sn, "n_bins": int(R.sum()),
                          "rank": i + 1, "type": t["type"][i], "mean_score": t["mean_score"][i],
                          "n_markers_scored": t["n_markers_scored"][i]})
        print(f"  {sn} n={int(R.sum())} genes={list(g['gene'][:12])} types={list(t['type'][:4])}",
              flush=True)
    del m, Y, X, s

pd.DataFrame(rows).to_csv(OUT / f"intestine_region_flags_{SETTING}.csv", index=False)
pd.DataFrame(grows).to_csv(OUT / f"intestine_unexplained_genes_{SETTING}.csv", index=False)
pd.DataFrame(trows).to_csv(OUT / f"intestine_suggested_types_{SETTING}.csv", index=False)
if SETTING == "default":
    df = pd.DataFrame(perbin)
    df["x_px"] = ad.obsm["spatial"][:, 0].astype(np.float32)
    df["y_px"] = ad.obsm["spatial"][:, 1].astype(np.float32)
    for rn, r in regions.items():
        df[rn] = r
    df.to_parquet(OUT / "intestine_perbin_default.parquet", index=False)
print(pd.DataFrame(rows)[["ref", "region", "n_bins", "flag", "flag_pooled"]].round(3).to_string())
