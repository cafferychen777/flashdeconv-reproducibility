"""Spotless gold standard (seqFISH+ cortex_svz/ob, STARmap VISp) -> npz (run locally).
Signature = mean counts per cell type from the matching gold reference, exactly as
validation/benchmark_gold_correct_reference.py builds it. Real spot coordinates kept.

  python prep_gold.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_gold_correct_reference as g  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "benchmark_data" / "converted"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
REF = {"cortex_svz": "gold_ref_1", "ob": "gold_ref_2", "visp": "gold_ref_3"}
names = ([f"Eng2019_cortex_svz_fov{i}" for i in range(7)] + [f"Eng2019_ob_fov{i}" for i in range(7)]
         + ["Wang2018_visp_rep0410"])
refs = {k: g.load_reference(str(DATA / v)) for k, v in REF.items()}
for nm in names:
    key = "cortex_svz" if "cortex_svz" in nm else ("ob" if "_ob_" in nm else "visp")
    t = g.load_test_data(str(DATA / nm))
    X, gi, common = g.build_signature_matrix(refs[key], t["genes"], t["cell_types"])
    Y = sparse.csr_matrix(t["counts"][:, gi])
    assert np.allclose(Y.data, np.round(Y.data))
    np.savez_compressed(OUT / f"gold_{nm}.npz", Y_data=Y.data.astype(np.float32),
                        Y_indices=Y.indices, Y_indptr=Y.indptr, Y_shape=np.array(Y.shape),
                        X=X, cell_types=np.array(t["cell_types"]), gt=t["proportions"],
                        gt_cols=np.array(t["cell_types"]), genes=np.array(common),
                        coords=t["coords"], tissue=f"gold_{key}", pattern=nm)
    print(nm, Y.shape, X.shape, flush=True)
