"""Gene-weighting comparison on the Spotless silver standard with lambda=0 (no spatial
information). Local copy of validation/rerun_final/weighting/run_weighting_final.py
(spotless part): identical pipeline, thinning seeds and metrics; only the spatial
penalty is fixed to 0 instead of auto-tuned on the file-order lattice.

  python weighting_spotless_lam0.py [lam]     lam = 0 (default) or auto (reproduction check)
Output: results/editor_revision/weighting_lam0/spotless_{acc,pertype}_lam{lam}.csv
"""
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation")
import comprehensive_per_celltype_evaluation as ce  # noqa: E402
from flashdeconv import FlashDeconv, __version__  # noqa: E402
from flashdeconv.core.sketching import apply_gene_weights, expected_countsketch_weights  # noqa: E402
from flashdeconv.core.solver import bcd_solve, normalize_proportions  # noqa: E402
from flashdeconv.core.spatial import auto_tune_lambda  # noqa: E402
from flashdeconv.utils.genes import select_informative_genes  # noqa: E402
from flashdeconv.utils.graph import coords_to_adjacency  # noqa: E402

LAM = sys.argv[1] if len(sys.argv) > 1 else "0"
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None
NPZ = Path("/Users/apple/Research/FlashDeconv/results/tacco_gap/spotless_npz")
OUT = Path("/Users/apple/Research/FlashDeconv/results/editor_revision/weighting_lam0")
OUT.mkdir(parents=True, exist_ok=True)
VARIANTS = ["EXP_LEV", "UNIFORM", "VAR_REF"]
DEPTHS = [1.0, 0.25, 0.10, 0.05]
FINAL = dict(rho=0.01, max_iter=1000, tol=1e-4, d=512, n_hvg=2000, n_markers=50, k=6)


def fit(Y, X, coords, variant):
    t0 = time.perf_counter()
    gene_idx, lev = select_informative_genes(Y, X, n_hvg=FINAL["n_hvg"],
                                             n_markers_per_type=FINAL["n_markers"])
    Yt, Xt = FlashDeconv._preprocess_data(None, Y[:, gene_idx], X[:, gene_idx], "log_cpm")
    scores = {"EXP_LEV": lev, "UNIFORM": None,
              "VAR_REF": np.var(np.asarray(Xt, dtype=np.float64), axis=0) if variant == "VAR_REF" else None}[variant]
    w = expected_countsketch_weights(scores, len(gene_idx), sketch_dim=FINAL["d"])
    Yw, Xw = apply_gene_weights(Yt, Xt, w)
    A = coords_to_adjacency(coords, method="knn", k=FINAL["k"], radius=None)
    lam = auto_tune_lambda(Yw, Xw, A) if LAM == "auto" else 0.0
    beta, info = bcd_solve(Yw, Xw, A, lambda_=lam, rho=FINAL["rho"],
                           max_iter=FINAL["max_iter"], tol=FINAL["tol"])
    return normalize_proportions(beta), dict(
        n_iterations=int(info["n_iterations"]), converged=bool(info["converged"]),
        lambda_used=float(lam), fit_seconds=round(time.perf_counter() - t0, 3))


def thin(Y, frac, seed):
    if frac >= 1.0:
        return Y
    rng = np.random.default_rng(seed)
    return rng.binomial(np.rint(Y).astype(np.int64), frac).astype(np.float64)


def jsd_safe(pred, true_df, cell_types):
    from scipy.spatial.distance import jensenshannon
    pdf = pd.DataFrame(pred, columns=cell_types)
    common = sorted(set(pdf.columns) & set(true_df.columns))
    P = np.clip(pdf[common].values.astype(float), 0, None) + 1e-10
    P = P / P.sum(axis=1, keepdims=True)
    T = true_df[common].values.astype(float) + 1e-10
    T = T / T.sum(axis=1, keepdims=True)
    v = np.array([jensenshannon(P[i], T[i]) ** 2 for i in range(P.shape[0])])
    return float(np.mean(np.nan_to_num(v, nan=0.0)))


def run(npz):
    name = npz.stem
    z = np.load(npz, allow_pickle=True)
    Y0 = sparse.csr_matrix((z["Y_data"].astype(np.float64), z["Y_indices"], z["Y_indptr"]),
                           shape=tuple(z["Y_shape"])).toarray()
    gt = pd.DataFrame(z["gt"], columns=[str(c) for c in z["gt_cols"]])
    cts = np.array([str(c) for c in z["cell_types"]])
    X = z["X"]
    n = Y0.shape[0]
    g = int(np.ceil(np.sqrt(n)))
    coords = np.array([[i % g, i // g] for i in range(n)], dtype=float)
    acc, per = [], []
    for frac in DEPTHS:
        Y = thin(Y0, frac, zlib.crc32(f"{name}|{frac}".encode()))
        med = float(np.median(Y.sum(axis=1)))
        for v in VARIANTS:
            p, info = fit(Y, X, coords, v)
            a = ce.compute_aggregate_metrics(p, gt, cts)
            pts = ce.compute_per_celltype_metrics(p, gt, cts)
            pdf = pd.DataFrame(pts)
            rare = pdf[pdf["category"] == "rare"]
            row = dict(dataset=name, tissue=str(z["tissue"]), pattern=str(z["pattern"]),
                       variant=v, frac=frac, median_umi=med,
                       pearson=a["corr"], rmse=a["rmse"], jsd=jsd_safe(p, gt, cts), aupr=a["aupr"],
                       mean_type_pearson=float(np.nanmean(pdf["pearson"])),
                       mean_type_auprc=float(np.nanmean(pdf["auprc"])),
                       n_rare=int(len(rare)),
                       rare_pearson=float(np.nanmean(rare["pearson"])) if len(rare) else np.nan,
                       rare_auprc=float(np.nanmean(rare["auprc"])) if len(rare) else np.nan,
                       version=__version__, **info)
            if v == "EXP_LEV" and frac == 1.0:
                kw = {} if LAM == "auto" else dict(lambda_spatial=0.0)
                row["maxdiff_vs_package"] = float(np.abs(FlashDeconv(**kw).fit_transform(Y, X, coords) - p).max())
            acc.append(row)
            per += [dict(dataset=name, variant=v, frac=frac, cell_type=r["cell_type"],
                         mean_abundance=r["mean_abundance"], category=r["category"],
                         pearson=r["pearson"], auprc=r["auprc"], rmse=r["rmse"]) for r in pts]
    print(name, "done", flush=True)
    return acc, per


if __name__ == "__main__":
    from joblib import Parallel, delayed
    files = sorted(NPZ.glob("silver_*_*.npz"))
    if ONLY:
        files = [f for f in files if f.stem in ONLY]
    res = Parallel(n_jobs=8)(delayed(run)(f) for f in files)
    tag = "" if ONLY is None else "_subset"
    pd.DataFrame([r for a, _ in res for r in a]).to_csv(OUT / f"spotless_acc_lam{LAM}{tag}.csv", index=False)
    pd.DataFrame([r for _, p in res for r in p]).to_csv(OUT / f"spotless_pertype_lam{LAM}{tag}.csv", index=False)
