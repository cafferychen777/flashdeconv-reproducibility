"""Gene-weighting evidence under sparse counts (final package defaults).

Three gene-weighting schemes share the final FlashDeconv pipeline (HVG U marker
gene selection, log-CPM, kNN k=6 graph, auto lambda, BCD with rho=0.01,
max_iter=1000, tol=1e-4, sketch_dim=512); only the per-gene scores fed to
expected_countsketch_weights change:

  EXP_LEV   expected leverage-weighted CountSketch weights (package default)
  UNIFORM   the same formula with uniform scores (all weights equal)
  VAR_REF   the same formula with per-gene variance of the log-CPM reference
            signatures across cell types in place of leverage

EXP_LEV is checked against FlashDeconv().fit_transform (final defaults) on every
input (package fit logged through fdfinal).

  python run_weighting_final.py spotless --npz .../silver_1_1.npz
      full depth + binomial UMI thinning to 25/10/5% with the C3b seeds
      (zlib.crc32(f"{name}|{frac}"), validation/sketch_ablation_c3b/run_c3b.py)
  python run_weighting_final.py xenium --res 2
      Xenium CRC P1 pseudo-Visium HD bins (C2), self signature, dense float32
      counts, bin centres (as validation/rerun_v020/benchmarks/c2/run_fd_v020.py)
"""
import argparse
import os
import sys
import time
import zlib
from pathlib import Path

W = Path(os.environ.get("WT_WORK", "/scratch/user/cafferychen777/fd_final/weighting"))
PARTS = W / "parts"
PARTS.mkdir(parents=True, exist_ok=True)


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["spotless", "xenium"])
    ap.add_argument("--npz")
    ap.add_argument("--res", type=int)
    return ap.parse_args()


A_ = _args()
NAME = Path(A_.npz).stem if A_.bench == "spotless" else f"xenium_{A_.res}um"
os.environ["FD_FITLOG"] = str(PARTS / f"fitlog_pkg_{NAME}.csv")
os.environ.setdefault("FD_TAG", f"weighting_pkgcheck_{NAME}")

# Final package first (installed env), then the hook, then project helpers.
import flashdeconv  # noqa: E402
sys.path.insert(0, os.environ.get("FD_CODE", "/scratch/user/cafferychen777/fd_final/code"))
import fdfinal  # noqa: E402,F401
from flashdeconv import FlashDeconv  # noqa: E402
from flashdeconv.core.sketching import apply_gene_weights, expected_countsketch_weights  # noqa: E402
from flashdeconv.core.solver import bcd_solve, normalize_proportions  # noqa: E402
from flashdeconv.core.spatial import auto_tune_lambda  # noqa: E402
from flashdeconv.utils.genes import select_informative_genes  # noqa: E402
from flashdeconv.utils.graph import coords_to_adjacency  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.metrics import auc, average_precision_score, precision_recall_curve  # noqa: E402

PROJ = Path(os.environ.get("FLASHDECONV_PROJECT_ROOT", "/scratch/user/cafferychen777/FlashDeconv"))
VARIANTS = ["EXP_LEV", "UNIFORM", "VAR_REF"]
DEPTHS = [1.0, 0.25, 0.10, 0.05]
FINAL = dict(rho=0.01, max_iter=1000, tol=1e-4, d=512, n_hvg=2000, n_markers=50, k=6)


def fit(Y, X, coords, variant):
    """Package pipeline (FlashDeconv.fit, gene_weighting='expected') with swapped scores."""
    t0 = time.perf_counter()
    gene_idx, lev = select_informative_genes(Y, X, n_hvg=FINAL["n_hvg"],
                                             n_markers_per_type=FINAL["n_markers"])
    Ys = Y[:, gene_idx]
    if sparse.issparse(Ys) and not sparse.isspmatrix_csr(Ys):
        Ys = Ys.tocsr()
    Yt, Xt = FlashDeconv._preprocess_data(None, Ys, X[:, gene_idx], "log_cpm")
    if variant == "EXP_LEV":
        scores = lev
    elif variant == "UNIFORM":
        scores = None
    elif variant == "VAR_REF":
        scores = np.var(np.asarray(Xt, dtype=np.float64), axis=0)
    else:
        raise ValueError(variant)
    w = expected_countsketch_weights(scores, len(gene_idx), sketch_dim=FINAL["d"])
    Yw, Xw = apply_gene_weights(Yt, Xt, w)
    A = coords_to_adjacency(coords, method="knn", k=FINAL["k"], radius=None)
    lam = auto_tune_lambda(Yw, Xw, A)
    beta, info = bcd_solve(Yw, Xw, A, lambda_=lam, rho=FINAL["rho"],
                           max_iter=FINAL["max_iter"], tol=FINAL["tol"])
    cv = float(np.std(w) / np.mean(w))
    return normalize_proportions(beta), dict(
        n_iterations=int(info["n_iterations"]), converged=bool(info["converged"]),
        lambda_used=float(lam), n_genes_used=int(len(gene_idx)), weight_cv=cv,
        fit_seconds=round(time.perf_counter() - t0, 3))


def package_check(Y, X, coords, pred_lev):
    p = FlashDeconv().fit_transform(Y, X, coords)
    return float(np.abs(p - pred_lev).max())


def append(path, rows):
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


# ------------------------------------------------------------------ Spotless silver
def thin(Y, frac, seed):
    """Identical to validation/sketch_ablation_c3b/c3b_core.thin."""
    if frac >= 1.0:
        return Y
    rng = np.random.default_rng(seed)
    return rng.binomial(np.rint(Y).astype(np.int64), frac).astype(np.float64)


def jsd_safe(pred, true_df, cell_types):
    """ce.compute_aggregate_metrics JSD, but a spot whose prediction equals the truth to
    rounding error (scipy returns NaN from sqrt of a tiny negative) contributes 0
    (as validation/rerun_v020/benchmarks/weighting/run_weighting.py)."""
    from scipy.spatial.distance import jensenshannon
    pdf = pd.DataFrame(pred, columns=cell_types)
    common = sorted(set(pdf.columns) & set(true_df.columns))
    P = np.clip(pdf[common].values.astype(float), 0, None) + 1e-10
    P = P / P.sum(axis=1, keepdims=True)
    T = true_df[common].values.astype(float) + 1e-10
    T = T / T.sum(axis=1, keepdims=True)
    v = np.array([jensenshannon(P[i], T[i]) ** 2 for i in range(P.shape[0])])
    return float(np.mean(np.nan_to_num(v, nan=0.0)))


def run_spotless(npz):
    sys.path.insert(1, str(PROJ / "validation"))
    import comprehensive_per_celltype_evaluation as ce
    z = np.load(npz, allow_pickle=True)
    Y0 = sparse.csr_matrix((z["Y_data"].astype(np.float64), z["Y_indices"], z["Y_indptr"]),
                           shape=tuple(z["Y_shape"])).toarray()
    gt = pd.DataFrame(z["gt"], columns=[str(c) for c in z["gt_cols"]])
    cts = np.array([str(c) for c in z["cell_types"]])
    X = z["X"]
    n = Y0.shape[0]
    g = int(np.ceil(np.sqrt(n)))
    coords = np.array([[i % g, i // g] for i in range(n)], dtype=float)
    key = dict(dataset=NAME, tissue=str(z["tissue"]), pattern=str(z["pattern"]))
    acc_p, pt_p, fl_p = (PARTS / f"acc_{NAME}.csv", PARTS / f"pertype_{NAME}.csv",
                         PARTS / f"fitlog_{NAME}.csv")
    done = set()
    if acc_p.exists():
        d = pd.read_csv(acc_p)
        done = set(zip(d["variant"], d["frac"].astype(float)))
    for frac in DEPTHS:
        if all((v, frac) in done for v in VARIANTS):
            continue
        Y = thin(Y0, frac, zlib.crc32(f"{NAME}|{frac}".encode()))
        med = float(np.median(Y.sum(axis=1)))
        preds = {}
        for v in VARIANTS:
            p, info = fit(Y, X, coords, v)
            preds[v] = p
            a = ce.compute_aggregate_metrics(p, gt, cts)
            pts = ce.compute_per_celltype_metrics(p, gt, cts)
            pdf = pd.DataFrame(pts)
            rare = pdf[pdf["category"] == "rare"]
            row = dict(**key, variant=v, frac=frac, median_umi=med,
                       pearson=a["corr"], rmse=a["rmse"], jsd=jsd_safe(p, gt, cts), jsd_ce=a["jsd"],
                       aupr=a["aupr"],
                       mean_type_pearson=float(np.nanmean(pdf["pearson"])),
                       mean_type_auprc=float(np.nanmean(pdf["auprc"])),
                       n_rare=int(len(rare)),
                       rare_pearson=float(np.nanmean(rare["pearson"])) if len(rare) else np.nan,
                       rare_auprc=float(np.nanmean(rare["auprc"])) if len(rare) else np.nan,
                       **info)
            if v == "EXP_LEV":
                row["maxdiff_vs_package"] = package_check(Y, X, coords, p)
            append(acc_p, [row])
            append(pt_p, [dict(dataset=NAME, variant=v, frac=frac, cell_type=r["cell_type"],
                               mean_abundance=r["mean_abundance"], category=r["category"],
                               pearson=r["pearson"], auprc=r["auprc"], rmse=r["rmse"])
                          for r in pts])
            append(fl_p, [dict(tag="weighting", context=f"{NAME}|{v}|{frac}", variant=v,
                               n_spots=n, n_types=X.shape[0], max_iter=FINAL["max_iter"],
                               tol=FINAL["tol"], rho=FINAL["rho"], **info,
                               version=flashdeconv.__version__)])
            print(f"{NAME} f={frac:.2f} {v:8s} r={a['corr']:.4f} it={info['n_iterations']} "
                  f"conv={info['converged']} t={info['fit_seconds']}", flush=True)
    print("WT_TASK_DONE", flush=True)


# ------------------------------------------------------------------ Xenium CRC bins
def xen_metrics(pred, true):
    """C2 legacy metrics (tg_common.legacy_metrics) + average-precision variants."""
    p, g = pred.ravel(), true.ravel()
    nz = (p > 0) | (g > 0)
    out = dict(r_legacy=float(np.corrcoef(p[nz], g[nz])[0, 1]),
               pearson=float(np.corrcoef(p, g)[0, 1]),
               rmse=float(np.sqrt(np.mean((pred - true) ** 2))))
    tb = (g > 0.01).astype(int)
    pr, rc, _ = precision_recall_curve(tb, p)
    out["auprc_trapz"] = float(auc(rc, pr))
    out["ap"] = float(average_precision_score(tb, p))
    P = pred + 1e-10
    P = P / P.sum(1, keepdims=True)
    Q = true + 1e-10
    Q = Q / Q.sum(1, keepdims=True)
    M = 0.5 * (P + Q)
    out["jsd"] = float(np.mean(0.5 * np.sum(P * np.log(P / M), 1)
                               + 0.5 * np.sum(Q * np.log(Q / M), 1)))
    per = []
    for j in range(pred.shape[1]):
        t, q = true[:, j], pred[:, j]
        r = np.corrcoef(q, t)[0, 1] if (q.std() > 0 and t.std() > 0) else np.nan
        b = (t > 0.01).astype(int)
        ap = average_precision_score(b, q) if 0 < b.sum() < len(b) else np.nan
        ma = float(t.mean())
        per.append(dict(type_idx=j, mean_abundance=ma, prevalence=float(b.mean()),
                        category="rare" if ma < 0.05 else ("moderate" if ma < 0.15 else "abundant"),
                        pearson=r, ap=ap, rmse=float(np.sqrt(np.mean((q - t) ** 2)))))
    per = pd.DataFrame(per)
    out["mean_type_pearson"] = float(np.nanmean(per["pearson"]))
    out["mean_type_ap"] = float(np.nanmean(per["ap"]))
    rare = per[per["category"] == "rare"]
    out["n_rare"] = int(len(rare))
    out["rare_pearson"] = float(np.nanmean(rare["pearson"])) if len(rare) else np.nan
    out["rare_ap"] = float(np.nanmean(rare["ap"])) if len(rare) else np.nan
    return out, per


def run_xenium(res):
    sys.path.insert(1, str(PROJ / "validation" / "pseudo_vhd_c2"))
    from c2_common import load_bins, load_signature
    b = load_bins(res)
    X, cts, genes = load_signature()
    assert cts == b["cell_types"] and genes == b["genes"]
    Y = b["Y"].toarray().astype(np.float32)
    gt = b["gt_props"].astype(np.float64)
    coords = b["centers"]
    med = float(np.median(Y.sum(axis=1)))
    acc_p, pt_p, fl_p = (PARTS / f"acc_{NAME}.csv", PARTS / f"pertype_{NAME}.csv",
                         PARTS / f"fitlog_{NAME}.csv")
    done = set(pd.read_csv(acc_p)["variant"]) if acc_p.exists() else set()
    print(f"{NAME}: Y {Y.shape}, median UMI {med}", flush=True)
    for v in VARIANTS:
        if v in done:
            continue
        p, info = fit(Y, X, coords, v)
        m, per = xen_metrics(p, gt)
        row = dict(dataset=NAME, res_um=res, variant=v, n_bins=Y.shape[0], median_umi=med,
                   **m, **info)
        if v == "EXP_LEV":
            row["maxdiff_vs_package"] = package_check(Y, X, coords, p)
        append(acc_p, [row])
        per.insert(0, "cell_type", [cts[i] for i in per["type_idx"]])
        per.insert(0, "variant", v)
        per.insert(0, "res_um", res)
        append(pt_p, per.to_dict("records"))
        append(fl_p, [dict(tag="weighting", context=f"{NAME}|{v}", variant=v,
                           n_spots=Y.shape[0], n_types=X.shape[0], max_iter=FINAL["max_iter"],
                           tol=FINAL["tol"], rho=FINAL["rho"], **info,
                           version=flashdeconv.__version__)])
        print(f"{NAME} {v:8s} r={m['pearson']:.4f} type_r={m['mean_type_pearson']:.4f} "
              f"ap={m['ap']:.4f} it={info['n_iterations']} conv={info['converged']} "
              f"t={info['fit_seconds']}", flush=True)
    print("WT_TASK_DONE", flush=True)


if __name__ == "__main__":
    assert flashdeconv.__version__ == "0.2.0", flashdeconv.__file__
    if A_.bench == "spotless":
        run_spotless(A_.npz)
    else:
        run_xenium(A_.res)
