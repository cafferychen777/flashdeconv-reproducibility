"""Build compact per-dataset inputs for the Spotless Silver Standard (54 datasets).

Uses exactly the loading / reference-signature / gene-alignment code of
validation/comprehensive_per_celltype_evaluation.py (the harness behind the
paper's Spotless numbers), then stores the aligned matrices as compressed npz so
the cluster does not need the 3.3 GB of Matrix Market files.

Run locally:  python prep_spotless_local.py OUT_DIR
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import comprehensive_per_celltype_evaluation as ce  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "spotless_npz")


def do_tissue(tid):
    X, cell_types, ref_genes = ce.load_reference_data(tid)
    for pid in sorted(ce.PATTERN_MAP):
        Y, sp_genes, true_props = ce.load_silver_data(tid, pid)
        Y_a, X_a, common = ce.align_genes(Y, sp_genes, X, ref_genes)
        true_cols = [ct for ct in cell_types if ct in true_props.columns]
        gt = true_props[true_cols].values.astype(np.float64)
        Ys = sparse.csr_matrix(Y_a)
        assert np.all(Ys.data == np.round(Ys.data)) and Ys.data.max() < 2 ** 24
        np.savez_compressed(
            OUT / f"silver_{tid}_{pid}.npz",
            Y_data=Ys.data.astype(np.float32), Y_indices=Ys.indices, Y_indptr=Ys.indptr,
            Y_shape=np.array(Ys.shape), X=X_a, cell_types=np.array(cell_types),
            gt=gt, gt_cols=np.array(true_cols), genes=np.array(common),
            tissue=ce.TISSUE_NAMES[tid], pattern=ce.PATTERN_MAP[pid])
        print(tid, pid, Y_a.shape, X_a.shape, len(true_cols), flush=True)
    return tid


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(6) as p:
        print(p.map(do_tissue, sorted(ce.TISSUE_NAMES)))
