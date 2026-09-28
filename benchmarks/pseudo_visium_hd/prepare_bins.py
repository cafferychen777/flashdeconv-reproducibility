"""Materialize pseudo-Visium HD bins (identical to the original table) for all methods.

Reuses create_bins / build_reference_from_xenium from the original benchmark.
The original run processed bin sizes [2, 4, 8, 16, 32] with a single
np.random.default_rng(42); only 2 um was downsampled (it came first), so a fresh
default_rng(42) per resolution reproduces the exact same counts.

Outputs (OUT/bins/):
  bins_{res}um.npz          counts (csr), gt proportions, centers, stats
  signature.npz             Xenium self-reference signature (38 types x 422 genes)
  ref_cells_all.npz         all annotated Xenium cells (TACCO reference)
  rctd/ref_counts.mtx, ref_meta.csv   per-type capped (10,000, spacexr default
                                       n_max_cells) reference for spacexr
  rctd/bins_{res}um.mtx, bins_{res}um_coords.csv, genes.txt
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import io as sio
from scipy import sparse

from c2_common import BIN_DIR, RESOLUTIONS
import xenium_pseudo_visiumhd_benchmark as ob

RCTD_REF_CAP = 10000  # spacexr Reference() default n_max_cells per type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", nargs="+", type=int, default=RESOLUTIONS)
    args = ap.parse_args()

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    (BIN_DIR / "rctd").mkdir(exist_ok=True)

    adata = ob.load_xenium()
    cell_types = sorted([ct for ct in adata.obs["Level2"].unique() if ct != "Unassigned"])
    genes = list(adata.var_names)
    X_sig, _, _ = ob.build_reference_from_xenium(adata, cell_types)
    np.savez_compressed(BIN_DIR / "signature.npz", X_sig=X_sig,
                        cell_types=np.array(cell_types), genes=np.array(genes))
    (BIN_DIR / "rctd" / "genes.txt").write_text("\n".join(genes) + "\n")

    # Reference cells (annotated only)
    X_raw = adata.layers["counts"] if "counts" in adata.layers else adata.X
    X_raw = sparse.csr_matrix(X_raw)
    labels = adata.obs["Level2"].astype(str).values
    keep = labels != "Unassigned"
    Xr = X_raw[keep].astype(np.float32)
    lab = labels[keep]
    np.savez_compressed(BIN_DIR / "ref_cells_all.npz", X_data=Xr.data,
                        X_indices=Xr.indices, X_indptr=Xr.indptr,
                        X_shape=np.array(Xr.shape), labels=lab, genes=np.array(genes))
    counts_per_type = pd.Series(lab).value_counts()
    print("Reference cells per type (min/max):",
          counts_per_type.min(), counts_per_type.max())
    print(counts_per_type.to_string())

    # spacexr reference: cap each type at 10,000 cells (rng 42)
    rng = np.random.default_rng(42)
    idx = []
    for ct in cell_types:
        ii = np.where(lab == ct)[0]
        if len(ii) > RCTD_REF_CAP:
            ii = np.sort(rng.choice(ii, RCTD_REF_CAP, replace=False))
        idx.append(ii)
    idx = np.concatenate(idx)
    Xs = Xr[idx]
    Xs.data = np.rint(Xs.data)
    sio.mmwrite(str(BIN_DIR / "rctd" / "ref_counts.mtx"), Xs.T.tocoo().astype(np.int32))
    pd.DataFrame({"barcode": [f"cell_{i}" for i in idx], "cell_type": lab[idx],
                  "nUMI": np.asarray(Xs.sum(axis=1)).ravel().astype(int)}
                 ).to_csv(BIN_DIR / "rctd" / "ref_meta.csv", index=False)
    print(f"spacexr reference: {Xs.shape[0]:,} cells (cap {RCTD_REF_CAP}/type)")

    for res in args.res:
        rng = np.random.default_rng(42)
        target = ob.VHD_TARGET_MEDIAN_UMI.get(res)
        Y, gt_props, gt_counts, centers, ct_list, stats = ob.create_bins(
            adata, res, 1.0, target_median_umi=target, rng=rng)
        assert list(ct_list) == cell_types
        Ys = sparse.csr_matrix(Y.astype(np.float32))
        umi = np.asarray(Ys.sum(axis=1)).ravel()
        stats = {k: float(v) for k, v in stats.items()}
        stats.update({"frac_umi_ge_100": float((umi >= 100).mean()),
                      "frac_umi_ge_20": float((umi >= 20).mean()),
                      "frac_umi_eq_0": float((umi == 0).mean()),
                      "median_umi": float(np.median(umi))})
        print(json.dumps(stats))
        np.savez_compressed(BIN_DIR / f"bins_{res}um.npz",
                            Y_data=Ys.data, Y_indices=Ys.indices, Y_indptr=Ys.indptr,
                            Y_shape=np.array(Ys.shape), gt_props=gt_props,
                            centers=centers, cell_types=np.array(cell_types),
                            genes=np.array(genes), stats=json.dumps(stats))
        sio.mmwrite(str(BIN_DIR / "rctd" / f"bins_{res}um.mtx"),
                    Ys.T.tocoo().astype(np.int32))
        pd.DataFrame(centers, columns=["x", "y"],
                     index=[f"bin_{i}" for i in range(Ys.shape[0])]
                     ).to_csv(BIN_DIR / "rctd" / f"bins_{res}um_coords.csv")
    print("PREP_DONE")


if __name__ == "__main__":
    main()
