"""Run Python-side methods on C2 bins: FlashDeconv (original-table settings), NNLS,
marker scoring (original code), and TACCO (default annotation method).

Idempotent: skips any (method, mode) whose prediction file already exists.

Usage: python run_py_methods.py --res 32 --methods fd_auto fd_l0 nnls marker tacco
"""

import argparse
import time

import numpy as np

from c2_common import load_bins, load_signature, load_reference_cells, pred_path, save_pred
import xenium_pseudo_visiumhd_benchmark as ob


def fd_version():
    import flashdeconv
    return f"flashdeconv {getattr(flashdeconv, '__version__', '?')}"


def run_fd(Y, X_sig, centers, cell_types, lam):
    """Exact settings used for the original table (cluster script, Feb 2026):
    rho_sparsity=0, max_iter=200, random_state=42, single fit."""
    from flashdeconv.core.deconv import FlashDeconv
    model = FlashDeconv(
        sketch_dim=min(512, Y.shape[1]), lambda_spatial=lam, rho_sparsity=0.0,
        preprocess="log_cpm", n_hvg=min(2000, Y.shape[1]), n_markers_per_type=50,
        max_iter=200, verbose=False, random_state=42,
    )
    return model.fit_transform(Y, X_sig, centers, cell_type_names=cell_types)


def run_tacco(Ys, genes, cell_types):
    import anndata as ad
    import pandas as pd
    import tacco as tc
    Xr, labels, ref_genes = load_reference_cells()
    assert ref_genes == genes
    adata = ad.AnnData(X=Ys.astype(np.float32), var=pd.DataFrame(index=genes),
                       obs=pd.DataFrame(index=[f"bin_{i}" for i in range(Ys.shape[0])]))
    ref = ad.AnnData(X=Xr, var=pd.DataFrame(index=genes),
                     obs=pd.DataFrame({"Level2": pd.Categorical(labels)},
                                      index=[f"cell_{i}" for i in range(Xr.shape[0])]))
    t0 = time.time()
    df = tc.tl.annotate(adata, ref, annotation_key="Level2")  # default method
    fit = time.time() - t0
    props = np.zeros((Ys.shape[0], len(cell_types)), dtype=np.float64)
    row = pd.Series(np.arange(Ys.shape[0]), index=adata.obs_names)
    ri = row.loc[df.index].values
    for j, ct in enumerate(cell_types):
        if ct in df.columns:
            props[ri, j] = df[ct].values
    # TACCO can return NaN rows for degenerate bins: treat them as not predicted.
    nan_rows = ~np.isfinite(props).all(axis=1)
    props[nan_rows] = 0.0
    s = props.sum(axis=1, keepdims=True)
    covered = s[:, 0] > 0
    props[covered] /= s[covered]
    ver = f"tacco {getattr(tc, '__version__', '?')} (tl.annotate defaults)"
    return props, covered, fit, ver, int(nan_rows.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--methods", nargs="+",
                    default=["fd_auto", "fd_l0", "nnls", "marker", "tacco"])
    args = ap.parse_args()
    res = args.res

    b = load_bins(res)
    X_sig, cell_types, genes = load_signature()
    assert cell_types == b["cell_types"] and genes == b["genes"]
    Ys = b["Y"]
    Y = Ys.toarray().astype(np.float32)
    umi = Y.sum(axis=1)
    print(f"{res}um: {Y.shape[0]:,} bins, stats={b['stats']}", flush=True)

    todo = {
        "fd_auto": ("FlashDeconv", "lambda_auto"),
        "fd_l0": ("FlashDeconv", "lambda_0"),
        "nnls": ("NNLS", "default"),
        "marker": ("MarkerScoring", "default"),
        "tacco": ("TACCO", "default"),
    }
    for key in args.methods:
        method, mode = todo[key]
        if pred_path(res, method, mode).exists():
            print(f"  skip {method}/{mode} (exists)")
            continue
        print(f"  running {method}/{mode} ...", flush=True)
        t0 = time.time()
        if key in ("fd_auto", "fd_l0"):
            lam = "auto" if key == "fd_auto" else 0.0
            props = run_fd(Y, X_sig, b["centers"], cell_types, lam)
            fit = time.time() - t0
            covered = np.ones(Y.shape[0], dtype=bool)
            ver, notes = fd_version(), "orig-table settings: rho=0,max_iter=200,seed=42"
        elif key == "nnls":
            props = ob.run_nnls(Y, X_sig)
            fit = time.time() - t0
            covered = props.sum(axis=1) > 0
            ver, notes = "scipy.optimize.nnls", "zero-UMI bins not predicted"
        elif key == "marker":
            props = ob.run_marker_scoring(Y, X_sig, cell_types)
            fit = time.time() - t0
            covered = props.sum(axis=1) > 0
            ver, notes = "original run_marker_scoring", "zero-UMI bins not predicted"
        else:
            props, covered, fit, ver, n_nan = run_tacco(Ys, genes, cell_types)
            notes = ("ref=all annotated Xenium cells; zero-count bins removed by TACCO; "
                     f"{n_nan} NaN-output bins treated as not predicted")
        save_pred(res, method, mode, props, covered, fit, ver, notes,
                  extra={"n_zero_umi_bins": int((umi == 0).sum())})
        m = ob.compute_metrics(props, b["gt_props"], cell_types)
        print(f"    [legacy zero-fill] r={m['global_r']:.4f} type_r={m['mean_per_type_r']:.4f} "
              f"AUPRC={m['global_auprc']:.4f} JSD={m['global_jsd']:.4f} "
              f"coverage={covered.mean():.4f} fit={fit:.1f}s", flush=True)
    print("PY_DONE")


if __name__ == "__main__":
    main()
