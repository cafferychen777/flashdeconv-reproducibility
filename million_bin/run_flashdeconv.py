"""S1 runner: FlashDeconv final package (v0.2.0), package defaults (max_iter=1000), seed 0.

Usage: python run_flashdeconv.py --st ST.h5ad --ref-dir DIR --props OUT.csv.gz
Reference signatures and gene alignment (prepare_data) are timed as part of the fit.
"""
import argparse
import sys

import anndata as ad

from s1_common import Timer, run_guarded, save_props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--props", required=True)
    a = ap.parse_args()

    def job():
        import flashdeconv
        from flashdeconv import FlashDeconv
        from flashdeconv.io import prepare_data

        with Timer() as tl:
            st = ad.read_h5ad(a.st)
            ref = ad.read_h5ad(f"{a.ref_dir}/ref.h5ad")
        with Timer() as tf:
            Y, X, coords, types, genes = prepare_data(st, ref, cell_type_key="cell_type")
            model = FlashDeconv(random_state=0)
            props = model.fit_transform(Y, X, coords, cell_type_names=types)
        save_props(props, st.obs_names, types, a.props)
        return dict(
            fit_seconds=round(tf.s, 3), load_seconds=round(tl.s, 3), n_bins=st.n_obs,
            n_bins_fit=props.shape[0], n_genes=len(genes), n_types=len(types),
            version=f"flashdeconv {flashdeconv.__version__}",
            notes=f"defaults; max_iter={model.max_iter}; converged={model.info_.get('converged')}; "
                  f"iters={model.info_.get('n_iterations')}; n_selected_genes={len(model.gene_idx_)}; seed=0",
        )

    return run_guarded(job)


if __name__ == "__main__":
    sys.exit(main())
