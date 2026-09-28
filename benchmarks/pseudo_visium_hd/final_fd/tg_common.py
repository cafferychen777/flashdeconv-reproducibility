"""Shared helpers for the TACCO-gap diagnosis (Xenium CRC pseudo-Visium HD, benchmark C2).

Bins/predictions are the C2 files (validation/pseudo_vhd_c2). Locally they are read from
results/tacco_gap/c2/{bins,preds}; on arseven set TG_C2=<C2 results dir>.
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.metrics import auc, precision_recall_curve

HERE = Path(__file__).resolve().parent
PROJ = Path(os.environ.get("FLASHDECONV_PROJECT_ROOT", HERE.parents[1]))
C2 = Path(os.environ.get("TG_C2", PROJ / "results" / "tacco_gap" / "c2"))
OUT = Path(os.environ.get("TG_OUT", PROJ / "results" / "tacco_gap"))
RES = [2, 4, 8, 16, 32]


def load_bins(res):
    d = np.load(C2 / "bins" / f"bins_{res}um.npz", allow_pickle=True)
    Y = sparse.csr_matrix((d["Y_data"], d["Y_indices"], d["Y_indptr"]),
                          shape=tuple(d["Y_shape"])).astype(np.float64)
    gt = d["gt_props"].astype(np.float64)
    return dict(Y=Y, gt=gt, centers=d["centers"].astype(float),
                cell_types=[str(x) for x in d["cell_types"]],
                genes=[str(x) for x in d["genes"]], ncell=infer_ncells(gt),
                stats=json.loads(str(d["stats"])))


def load_signature():
    d = np.load(C2 / "bins" / "signature.npz", allow_pickle=True)
    return d["X_sig"].astype(np.float64), [str(x) for x in d["cell_types"]]


def load_ref_cells():
    d = np.load(C2 / "bins" / "ref_cells_all.npz", allow_pickle=True)
    X = sparse.csr_matrix((d["X_data"], d["X_indices"], d["X_indptr"]),
                          shape=tuple(d["X_shape"])).astype(np.float64)
    return X, np.array([str(x) for x in d["labels"]])


def load_pred(res, stem):
    z = np.load(C2 / "preds" / f"{res}um" / f"{stem}.npz", allow_pickle=True)
    return z["props"].astype(np.float64), z["covered"].astype(bool)


def infer_ncells(gt, nmax=200):
    """Smallest n such that n * proportions are integers (annotated cells per bin)."""
    n = np.zeros(gt.shape[0], dtype=int)
    todo = np.ones(gt.shape[0], dtype=bool)
    for k in range(1, nmax + 1):
        v = gt[todo] * k
        ok = np.all(np.abs(v - np.rint(v)) < 1e-3, axis=1)
        idx = np.where(todo)[0][ok]
        n[idx] = k
        todo[idx] = False
        if not todo.any():
            break
    return n


# ----------------------------------------------------------------------------- metrics
def legacy_metrics(pred, true):
    """Vectorised xenium_pseudo_visiumhd_benchmark.compute_metrics (C2 table metric):
    global r over entries with pred>0 OR true>0 (joint zeros dropped); mean per-type r;
    flattened AUPRC (presence > 0.01, trapezoid auc of PR curve); mean per-bin JSD^2."""
    p, g = pred.ravel(), true.ravel()
    nz = (p > 0) | (g > 0)
    out = {"r_legacy": float(np.corrcoef(p[nz], g[nz])[0, 1])}
    out["r_full"] = float(np.corrcoef(p, g)[0, 1])  # standard flattened Pearson
    rs = []
    for j in range(pred.shape[1]):
        if pred[:, j].std() > 0 and true[:, j].std() > 0:
            rs.append(np.corrcoef(pred[:, j], true[:, j])[0, 1])
        else:
            rs.append(np.nan)
    rs = np.array(rs)
    out["type_r"] = float(np.nanmean(rs))
    tb = (g > 0.01).astype(int)
    pr, rc, _ = precision_recall_curve(tb, p)
    out["auprc"] = float(auc(rc, pr))
    out["jsd"] = jsd_scipy(pred, true)
    out["rmse"] = float(np.sqrt(np.mean((pred - true) ** 2)))
    # top-1 accuracy against the majority true type (meaningful for 1-cell bins)
    out["top1"] = float(np.mean(pred.argmax(1) == true.argmax(1)))
    out["_per_type_r"] = rs
    return out


def jsd_scipy(pred, true):
    """Exactly scipy.spatial.distance.jensenshannon(p, q)**2 (natural log) per bin, mean."""
    P = pred + 1e-10
    P = P / P.sum(1, keepdims=True)
    Q = true + 1e-10
    Q = Q / Q.sum(1, keepdims=True)
    M = 0.5 * (P + Q)
    js = 0.5 * np.sum(P * np.log(P / M), 1) + 0.5 * np.sum(Q * np.log(Q / M), 1)
    return float(np.mean(js))


def summarize(pred, true, ncell=None):
    m = legacy_metrics(pred, true)
    m.pop("_per_type_r")
    if ncell is not None:
        # NOTE: gt_props alone cannot give the cell count of pure bins; strata are
        # pure (one true type, props max == 1) vs mixed (>= 2 true types).
        pure = true.max(1) > 1 - 1e-6
        for name, msk in (("pure", pure), ("mixed", ~pure)):
            if msk.sum() >= 100:
                p, g = pred[msk].ravel(), true[msk].ravel()
                m[f"r_full_{name}"] = float(np.corrcoef(p, g)[0, 1])
                m[f"top1_{name}"] = float(np.mean(pred[msk].argmax(1) == true[msk].argmax(1)))
                m[f"jsd_{name}"] = jsd_scipy(pred[msk], true[msk])
                m[f"frac_{name}"] = float(msk.mean())
    return m
