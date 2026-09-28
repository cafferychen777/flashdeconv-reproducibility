"""
Driver for the final rerun of the Xenium CRC P1 validations
(Supplementary Note 7): virtual binning, global proportion comparison and
pathologist concordance. All logic lives in the copied original script
xenium_crc_validation.py; this driver only routes inputs/outputs.

Steps:
  binning  -> virtual_binning_validation() on the cached Xenium annotation
  global   -> global_proportion_comparison() + three_way_comparison()
  patho    -> pathologist_concordance()

For global/patho, --props selects the Visium HD P1 proportions:
  old -> obsm['flashdeconv'] of the archived P1_CRC_deconv.h5ad
  new -> v0.2.0 npz (P, barcodes, types), wrapped into a light h5ad so the
         original functions run unchanged.

Env: RERUN_OUT_DIR (output dir), FLASHDECONV_PROJECT_ROOT (data root).
"""

import sys as _sys  # final rerun: fit-logging hook (validation/rerun_final/fdfinal.py)
_sys.path.insert(0, "/scratch/user/cafferychen777/fd_final/code")
import fdfinal  # noqa: E402,F401
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import xenium_crc_validation as xv


def props_h5ad_from_npz(npz_path: Path, out_path: Path) -> Path:
    """Wrap v0.2.0 npz proportions into an h5ad with obsm['flashdeconv']."""
    import anndata as ad
    z = np.load(npz_path, allow_pickle=True)
    P = z["P"].astype(np.float64)
    barcodes = [str(b) for b in z["barcodes"]]
    types = [str(t) for t in z["types"]]
    df = pd.DataFrame(P, index=barcodes, columns=types)
    a = ad.AnnData(obs=pd.DataFrame(index=barcodes))
    a.obsm["flashdeconv"] = df
    a.write_h5ad(out_path)
    print(f"  Wrapped {P.shape} proportions -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["binning", "global", "patho"], required=True)
    parser.add_argument("--props", choices=["old", "new"], default="new")
    parser.add_argument("--npz", type=str, default=None,
                        help="v0.2.0 P1 proportions npz (for --props new)")
    parser.add_argument("--rctd-csv", type=str, default=str(xv.RCTD_CSV))
    parser.add_argument("--patho-csv", type=str, default=str(xv.PATHO_CSV))
    args = parser.parse_args()

    out_dir = xv.OUTPUT_DIR
    print(f"flashdeconv {xv.fd.__version__} from {xv.fd.__file__}")

    import scanpy as sc
    t0 = time.time()
    if args.step == "binning":
        adata_xen = sc.read_h5ad(xv.XENIUM_ANNOT)
        print(f"  {adata_xen.n_obs:,} cells, {adata_xen.obs['Level2'].nunique()} types")
        df = xv.virtual_binning_validation(adata_xen, output_dir=out_dir)
        print(f"  binning wall time: {time.time() - t0:.1f}s")
        return

    if args.props == "old":
        fd_h5ad = xv.FD_H5AD
    else:
        fd_h5ad = props_h5ad_from_npz(Path(args.npz), out_dir / "_P1_final_props_tmp.h5ad")

    sub = out_dir / f"props_{args.props}"
    sub.mkdir(parents=True, exist_ok=True)
    if args.step == "global":
        adata_xen = sc.read_h5ad(xv.XENIUM_ANNOT)
        gdf = xv.global_proportion_comparison(
            adata_xen, fd_h5ad=fd_h5ad, rctd_csv=Path(args.rctd_csv), output_dir=sub)
        xv.three_way_comparison(gdf, output_dir=sub)
    else:
        xv.pathologist_concordance(fd_h5ad=fd_h5ad, patho_csv=Path(args.patho_csv),
                                   output_dir=sub)
    if args.props == "new":
        os.remove(fd_h5ad)


if __name__ == "__main__":
    main()
