"""FlashDeconv final-package rerun on the Spotless Silver (54) and Gold (seqFISH+ 14
FOVs, STARMap) standards. Copy of validation/rerun_v020/benchmarks/spotless/run_silver_gold.py;
only output paths, configurations (pure package defaults, lambda=0, random_state=42)
and fit logging (validation/rerun_final/fdfinal.py) changed."""
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.io
from scipy import sparse

sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation")
import comprehensive_per_celltype_evaluation as ce  # noqa: E402
sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/rerun_final")
import fdfinal  # noqa: F401,E402  (fit log)
from flashdeconv import FlashDeconv, __version__  # noqa: E402

OUT = "/Users/apple/Research/FlashDeconv/results/rerun_final/benchmarks/spotless"
os.makedirs(OUT, exist_ok=True)
GOLD_NPZ = "/Users/apple/Research/FlashDeconv/results/rerun_v020/benchmarks/_inputs/gold"


def harness_kwargs(G, gold=False):
    return dict(sketch_dim=min(512, G), lambda_spatial="auto", preprocess="log_cpm",
                n_hvg=min(2000, G) if gold else 2000, rho_sparsity=0, max_iter=200,
                verbose=False, random_state=42)


CONFIGS = {
    "final_default": lambda G, gold: dict(),                        # pure package defaults
    "final_default_lam0": lambda G, gold: dict(lambda_spatial=0.0),  # Laplacian ablation
    "final_default_seed42": lambda G, gold: dict(random_state=42),   # determinism check (default seed 0)
}


def jsd_nan0(pred, true_df, types):
    """Same per-spot squared JS distance as ce.compute_aggregate_metrics, but a spot
    whose prediction equals the truth (scipy returns NaN from sqrt of a -0 rounding
    residue) counts as 0 instead of turning the dataset mean into NaN."""
    from scipy.spatial.distance import jensenshannon
    pdf = pd.DataFrame(pred, columns=types)
    common = sorted(set(pdf.columns) & set(true_df.columns))
    p = np.clip(pdf[common].values.astype(float), 0, None) + 1e-10
    t = true_df[common].values.astype(float) + 1e-10
    p /= p.sum(1, keepdims=True); t /= t.sum(1, keepdims=True)
    v = np.array([jensenshannon(p[i], t[i]) ** 2 for i in range(len(p))])
    return float(np.nan_to_num(v, nan=0.0).mean())


def grid(n):
    s = int(np.ceil(np.sqrt(n)))
    return np.array([[i % s, i // s] for i in range(n)], dtype=float)


def full_reference(prefix):
    """Mean raw counts per cell type over all reference types; R-style names."""
    import re
    counts = scipy.io.mmread(f"{prefix}_counts.mtx").tocsr()  # genes x cells
    cts = np.array([re.sub(r"[^A-Za-z0-9]", ".", l.strip())
                    for l in open(f"{prefix}_celltypes.txt")])
    genes = np.array([l.strip() for l in open(f"{prefix}_genes.txt")])
    if counts.shape[0] != len(genes):
        counts = counts.T.tocsr()
    types = np.unique(cts)
    X = np.vstack([np.asarray(counts[:, cts == t].mean(axis=1)).ravel() for t in types])
    return X, types, genes


def mean_signature(prefix):
    counts = scipy.io.mmread(f"{prefix}_counts.mtx").toarray().T
    cts = np.loadtxt(f"{prefix}_celltypes.txt", dtype=str)
    genes = np.loadtxt(f"{prefix}_genes.txt", dtype=str)
    types = np.unique(cts)
    X = np.vstack([counts[cts == t].mean(axis=0) for t in types])
    return X, types, genes


def iter_datasets(which):
    D = ce.DATA_DIR
    if which in ("silver", "all"):
        for tid, tissue in ce.TISSUE_NAMES.items():
            X, types, rgenes = ce.load_reference_data(tid)
            for pid, pattern in ce.PATTERN_MAP.items():
                Y, g, props = ce.load_silver_data(tid, pid)
                yield dict(benchmark="silver_standard", tissue=tissue, pattern=pattern,
                           Y=Y, genes=g, props=props, X=X, types=types, rgenes=rgenes, gold=False)
    if which in ("gold", "all"):
        # The per-FOV reference files used for the manuscript run
        # (reference_Eng2019_*, Wang2018_visp_*) are no longer on disk; the gold
        # inputs are the npz files built by validation/method_improvement/prep_gold.py
        # (Y counts on the aligned genes, mean-count signature of the matching gold
        # reference, ground truth, real coordinates).
        names = ([f"Eng2019_cortex_svz_fov{i}" for i in range(7)]
                 + [f"Eng2019_ob_fov{i}" for i in range(7)] + ["Wang2018_visp_rep0410"])
        refs = {}
        for nm in names:
            z = np.load(f"{GOLD_NPZ}/gold_{nm}.npz", allow_pickle=True)
            key = "1" if "cortex_svz" in nm else ("2" if "_ob_" in nm else "3")
            if key not in refs:
                refs[key] = full_reference(os.path.join(ce.DATA_DIR, f"gold_ref_{key}"))
            Y = sparse.csr_matrix((z["Y_data"], z["Y_indices"], z["Y_indptr"]),
                                  shape=tuple(z["Y_shape"])).toarray().astype(float)
            genes = np.asarray(z["genes"]).astype(str)
            types = np.asarray(z["cell_types"]).astype(str)
            props = pd.DataFrame(z["gt"], columns=np.asarray(z["gt_cols"]).astype(str))
            bench = ("seqfish_cortex_svz" if "cortex_svz" in nm else
                     "seqfish_ob" if "_ob_" in nm else "starmap")
            # Spotless protocol: deconvolve with ALL reference cell types (17 / 9 / 12),
            # evaluate on the ground-truth types (as the manuscript harness did).
            Xf, tf, rg = refs[key]
            yield dict(benchmark=bench, tissue=nm, pattern="gold", Y=Y, genes=genes,
                       props=props, X=Xf, types=tf, rgenes=rg, gold=True,
                       real_coords=z["coords"])


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    configs = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CONFIGS)
    agg, perct, det = [], [], []
    for d in iter_datasets(which):
        Ya, Xa, common = ce.align_genes(d["Y"], d["genes"], d["X"], d["rgenes"])
        true_cols = [c for c in d["types"] if c in d["props"].columns]
        true_sub = d["props"][true_cols]
        if Ya.shape[0] != len(true_sub):
            print("skip (spot mismatch)", d["tissue"]); continue
        preds = {}
        for cname in configs:
            if cname.endswith("_realxy"):
                if "real_coords" not in d:
                    continue
                coords = d["real_coords"]
            else:
                coords = grid(Ya.shape[0])
            kw = CONFIGS[cname](Ya.shape[1], d["gold"])
            t0 = time.time()
            os.environ["FD_CONTEXT"] = f"{d['benchmark']}|{d['tissue']}|{d['pattern']}|{cname}"
            m = FlashDeconv(**kw)
            pred = m.fit_transform(Ya, Xa, coords)
            dt = time.time() - t0
            preds[cname] = pred
            if d["gold"]:
                # Gold: restrict to the ground-truth types and renormalise, exactly
                # as the evaluator treats the Spotless competitor predictions.
                idx = [list(d["types"]).index(c) for c in true_cols]
                raw_corr = ce.compute_aggregate_metrics(pred, true_sub, d["types"])["corr"]
                pe = np.clip(pred[:, idx], 0, None)
                rs = pe.sum(1, keepdims=True); rs[rs == 0] = 1
                pred_eval, types_eval = pe / rs, np.array(true_cols)
            else:
                pred_eval, types_eval, raw_corr = pred, d["types"], np.nan
            a = ce.compute_aggregate_metrics(pred_eval, true_sub, types_eval)
            a.pop("jsd_contributions")
            a["corr_unrenormalised"] = raw_corr
            a["jsd_paper"] = a["jsd"]
            a["jsd"] = jsd_nan0(pred_eval, true_sub, types_eval)
            meta = dict(config=cname, benchmark=d["benchmark"], tissue=d["tissue"],
                        pattern=d["pattern"], time_s=dt, lambda_used=m.lambda_used_,
                        n_iter=m.info_.get("n_iterations"), converged=m.info_.get("converged"),
                        n_genes_used=len(m.gene_idx_), version=__version__)
            agg.append({**a, **meta})
            for r in ce.compute_per_celltype_metrics(pred_eval, true_sub, types_eval):
                perct.append({**r, **meta})
        if "final_default" in preds and "final_default_seed42" in preds:
            det.append(dict(benchmark=d["benchmark"], tissue=d["tissue"], pattern=d["pattern"],
                            max_abs_diff=float(np.abs(preds["final_default"] - preds["final_default_seed42"]).max())))
        print(d["benchmark"], d["tissue"], d["pattern"],
              {r["config"]: round(r["corr"], 4) for r in agg[-len(preds):]}, flush=True)
    pd.DataFrame(agg).to_csv(f"{OUT}/fd_aggregate_{which}.csv", index=False)
    pd.DataFrame(perct).to_csv(f"{OUT}/fd_per_celltype_{which}.csv.gz", index=False)
    if det:
        pd.DataFrame(det).to_csv(f"{OUT}/determinism_{which}.csv", index=False)


if __name__ == "__main__":
    main()
