"""S1 runner: cell2location (GPU); settings identical to the C1 runtime benchmark.

Modes
  ref_train : NB regression reference signatures (one-time cost). Tutorial settings:
              filter_genes(cell_count_cutoff=5, cell_percentage_cutoff2=0.03,
              nonz_mean_cutoff=1.12); train(max_epochs=250, batch_size=2500,
              train_size=1, lr=0.002); export_posterior(num_samples=1000, batch_size=2500).
              Writes inf_aver.csv to the data dir.
  fullbatch : spatial mapping with the documented defaults, train(max_epochs=30000,
              batch_size=None, train_size=1, lr=0.002); all data are placed on the GPU.
  minibatch : spatial mapping with minibatches of 20,000 locations and the same total
              number of optimizer steps as full-batch training (30,000), i.e.
              max_epochs = ceil(30000 / ceil(N / 20000)). Used where full-batch does not fit
              in 24 GB of GPU memory.
Both spatial modes: N_cells_per_location=2 (8 um bins), detection_alpha=20;
export_posterior(num_samples=1000, batch_size=min(N, 20000)).
"""
import argparse
import math
import sys
from pathlib import Path

import anndata as ad
import numpy as np

from s1_common import Timer, run_guarded, save_props

MB_SIZE = 20_000
TOTAL_STEPS = 30_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--st", default="")
    ap.add_argument("--mode", required=True, choices=["ref_train", "fullbatch", "minibatch"])
    ap.add_argument("--props", default="")
    a = ap.parse_args()
    data_dir = Path(a.ref_dir)

    def job():
        from importlib.metadata import version

        import cell2location
        import pandas as pd
        import torch
        from cell2location.models import Cell2location, RegressionModel
        from cell2location.utils.filtering import filter_genes

        ver = f"cell2location {version('cell2location')}; scvi-tools {version('scvi-tools')}; torch {torch.__version__}"
        torch.set_float32_matmul_precision("high")
        partial = {"version": ver}

        if a.mode == "ref_train":
            with Timer() as tl:
                ref = ad.read_h5ad(data_dir / "ref.h5ad")
            with Timer() as tf:
                sel = filter_genes(ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
                ref = ref[:, sel].copy()
                RegressionModel.setup_anndata(ref, batch_key="patient", labels_key="cell_type")
                mod = RegressionModel(ref)
                mod.train(max_epochs=250, batch_size=2500, train_size=1, lr=0.002, accelerator="gpu")
                ref = mod.export_posterior(ref, sample_kwargs={"num_samples": 1000, "batch_size": 2500})
                cols = [f"means_per_cluster_mu_fg_{c}" for c in ref.uns["mod"]["factor_names"]]
                inf_aver = ref.varm["means_per_cluster_mu_fg"][cols].copy()
                inf_aver.columns = ref.uns["mod"]["factor_names"]
            inf_aver.to_csv(data_dir / "c2l_inf_aver.csv")
            return dict(fit_seconds=round(tf.s, 3), load_seconds=round(tl.s, 3), n_bins=0,
                        n_bins_fit=0, n_genes=inf_aver.shape[0], n_types=inf_aver.shape[1],
                        peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, version=ver,
                        notes=f"reference cells={ref.n_obs}; genes after filter_genes={inf_aver.shape[0]}")

        with Timer() as tl:
            st = ad.read_h5ad(a.st)
            inf_aver = pd.read_csv(data_dir / "c2l_inf_aver.csv", index_col=0)
        n = st.n_obs
        partial.update(n_bins=n, load_seconds=round(tl.s, 3))
        if a.mode == "fullbatch":
            batch_size, max_epochs = None, TOTAL_STEPS
        else:
            batch_size = min(n, MB_SIZE)
            max_epochs = math.ceil(TOTAL_STEPS / math.ceil(n / batch_size))
        timings = {}
        genes = []
        try:
            with Timer() as tf:
                genes = np.intersect1d(st.var_names, inf_aver.index)
                st = st[:, genes].copy()
                inf = inf_aver.loc[genes, :]
                Cell2location.setup_anndata(st)
                mod = Cell2location(st, cell_state_df=inf, N_cells_per_location=2, detection_alpha=20)
                with Timer() as tt:
                    mod.train(max_epochs=max_epochs, batch_size=batch_size, train_size=1, lr=0.002,
                              accelerator="gpu")
                timings["train_s"] = round(tt.s, 1)
                with Timer() as te:
                    st = mod.export_posterior(
                        st, sample_kwargs={"num_samples": 1000, "batch_size": min(n, MB_SIZE)})
                timings["export_s"] = round(te.s, 1)
                ab = st.obsm["q05_cell_abundance_w_sf"]
                props = ab.to_numpy() / np.clip(ab.to_numpy().sum(1, keepdims=True), 1e-12, None)
        except BaseException as exc:  # attach context for the failure row
            exc.c1_partial = dict(partial, n_genes=int(len(genes)),
                                  peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
            raise
        types = [c.replace("q05cell_abundance_w_sf_", "") for c in ab.columns]
        save_props(props, st.obs_names, types, a.props)
        hist = mod.history["elbo_train"].to_numpy().ravel()
        return dict(fit_seconds=round(tf.s, 3), load_seconds=round(tl.s, 3), n_bins=n, n_bins_fit=n,
                    n_genes=len(genes), n_types=len(types),
                    peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, version=ver,
                    notes=(f"max_epochs={max_epochs}; batch_size={batch_size}; train_s={timings['train_s']}; "
                           f"export_s={timings['export_s']}; elbo_first={hist[0]:.4g}; elbo_last={hist[-1]:.4g}; "
                           f"N_cells_per_location=2; detection_alpha=20; excludes one-time ref_train"))

    return run_guarded(job)


if __name__ == "__main__":
    sys.exit(main())
