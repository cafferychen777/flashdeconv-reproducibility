"""C1 runner for the final FlashDeconv package (defaults: max_iter=1000, tol=1e-4,
gene_weighting="expected").

Same as validation/rerun_v020/benchmarks/c1/run_flashdeconv_v020.py; the fdfinal
hook is imported first (fit log via FD_FITLOG), and n_iterations / converged /
max_iter / lambda are returned as separate result fields (monitor.py keeps only its
fixed columns, so merge_c1_final.py reads them back from the per-task result JSON).
"""
import argparse
import sys
from pathlib import Path

import fdfinal  # noqa: F401
import anndata as ad

from c1_common import SAVE_PROPS_MAX_SCALE, Timer, run_guarded, save_props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--props-dir", required=True)
    ap.add_argument("--random-state", type=int, default=42)
    a = ap.parse_args()

    def job():
        import flashdeconv
        from flashdeconv import FlashDeconv
        from flashdeconv.io import prepare_data

        with Timer() as tl:
            st = ad.read_h5ad(Path(a.data_dir) / f"st_{a.scale}.h5ad")
            ref = ad.read_h5ad(Path(a.data_dir) / "ref.h5ad")
        with Timer() as tf:
            Y, X, coords, types, genes = prepare_data(st, ref, cell_type_key="cell_type")
            model = FlashDeconv(random_state=a.random_state)
            props = model.fit_transform(Y, X, coords, cell_type_names=types)
        if a.scale <= SAVE_PROPS_MAX_SCALE and a.rep == 1:
            save_props(props, st.obs_names, types,
                       Path(a.props_dir) / f"flashdeconv_final_{a.scale}_seed{a.random_state}.csv.gz")
        info = model.info_ or {}
        return dict(
            fit_seconds=round(tf.s, 3), load_seconds=round(tl.s, 3), n_bins=st.n_obs,
            n_bins_fit=props.shape[0], n_genes=len(genes), n_types=len(types),
            version=f"flashdeconv {flashdeconv.__version__}",
            n_iterations=info.get("n_iterations"), converged=info.get("converged"),
            max_iter=model.max_iter, tol=model.tol, lambda_used=float(model.lambda_used_),
            random_state=a.random_state,
            notes=f"gene_weighting={model.gene_weighting}; converged={info.get('converged')}; "
                  f"iters={info.get('n_iterations')}; max_iter={model.max_iter}; tol={model.tol}; "
                  f"n_selected_genes={len(model.gene_idx_)}; lambda={model.lambda_used_:.4g}; "
                  f"seed={a.random_state}",
        )

    return run_guarded(job)


if __name__ == "__main__":
    sys.exit(main())
