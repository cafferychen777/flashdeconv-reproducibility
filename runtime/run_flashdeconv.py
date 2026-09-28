"""C1 runner: FlashDeconv (CPU) with the settings used for the CRC cohort in the paper."""
import argparse
import sys
from pathlib import Path

import anndata as ad

from c1_common import SAVE_PROPS_MAX_SCALE, Timer, run_guarded, save_props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--props-dir", required=True)
    a = ap.parse_args()

    def job():
        import flashdeconv
        from flashdeconv import FlashDeconv
        from flashdeconv.io import prepare_data

        with Timer() as tl:
            st = ad.read_h5ad(Path(a.data_dir) / f"st_{a.scale}.h5ad")
            ref = ad.read_h5ad(Path(a.data_dir) / "ref.h5ad")
        with Timer() as tf:
            # Reference signatures + gene alignment are part of the method cost.
            Y, X, coords, types, genes = prepare_data(st, ref, cell_type_key="cell_type")
            model = FlashDeconv(
                sketch_dim=512, lambda_spatial="auto", rho_sparsity=0.01, n_hvg=2000,
                n_markers_per_type=50, k_neighbors=6, max_iter=100, tol=1e-4,
                preprocess="log_cpm", random_state=42, verbose=False,
            )
            props = model.fit_transform(Y, X, coords, cell_type_names=types)
        if a.scale <= SAVE_PROPS_MAX_SCALE and a.rep == 1:
            save_props(props, st.obs_names, types, Path(a.props_dir) / f"flashdeconv_default_{a.scale}.csv.gz")
        return dict(
            fit_seconds=round(tf.s, 3), load_seconds=round(tl.s, 3), n_bins=st.n_obs,
            n_bins_fit=props.shape[0], n_genes=len(genes), n_types=len(types),
            version=f"flashdeconv {flashdeconv.__version__}",
            notes=f"converged={model.info_.get('converged')}; iters={model.info_.get('n_iterations')}; "
                  f"n_selected_genes={len(model.gene_idx_)}; seed=42",
        )

    return run_guarded(job)


if __name__ == "__main__":
    sys.exit(main())
