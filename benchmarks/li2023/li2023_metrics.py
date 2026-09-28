"""Re-implementation of the Li et al. 2023 (Nat Commun 14:1548) accuracy metrics.

Mirrors `benchmark_performance()` in
STdeconv_benchmark/evaluation & visualization/metrics.R:

* JSD  : per-spot Jensen-Shannon divergence, philentropy::JSD(unit="log2",
         est.prob="empirical") -> rows are normalised to sum 1, result in bits,
         NOT square-rooted. Spots whose ground truth sums to 0 get JSD = 1.
         The reported value is the MEDIAN over spots.
* total_RMSE : sqrt( sum_{spots, types} (pred - gt)^2 / (n_spots * n_types) ).
* per-type RMSE : sqrt( mean_spots (pred_k - gt_k)^2 ).
* PCC  : cor.test(pred, gt) on the two matrices, i.e. Pearson correlation of the
         flattened (spots x types) matrices.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _clean(name: str) -> str:
    # Same column-name normalisation as metrics.R
    return re.sub(r"[^\w]|_", ".", name)


def _row_jsd_log2(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = p / np.clip(p.sum(1, keepdims=True), 1e-300, None)
    q = q / np.clip(q.sum(1, keepdims=True), 1e-300, None)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kp = np.where(p > 0, p * np.log2(p / m), 0.0)
        kq = np.where(q > 0, q * np.log2(q / m), 0.0)
    return 0.5 * kp.sum(1) + 0.5 * kq.sum(1)


def align(pred: pd.DataFrame, gt: pd.DataFrame, by_position: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align rows and (normalised) column names of prediction and ground truth.

    by_position=True reproduces metrics.R, which pairs rows by order (the
    released seqFISH+ predictions are renumbered 0..70 after NA spots were
    dropped from the ground truth).
    """
    pred = pred.copy()
    if by_position:
        if len(pred) != len(gt):
            raise ValueError("row counts differ")
        pred.index = gt.index
    gt = gt.copy()
    pred.columns = [_clean(c) for c in pred.columns]
    gt.columns = [_clean(c) for c in gt.columns]
    pred.index = pred.index.astype(str)
    gt.index = gt.index.astype(str)
    missing = set(gt.columns) - set(pred.columns)
    if missing:
        raise ValueError(f"prediction lacks cell types: {sorted(missing)}")
    pred = pred.loc[gt.index, gt.columns]
    return pred, gt


def li_metrics(pred: pd.DataFrame, gt: pd.DataFrame, by_position: bool = False) -> dict:
    pred, gt = align(pred, gt, by_position)
    P = pred.to_numpy(float)
    G = gt.to_numpy(float)
    jsd = _row_jsd_log2(P, G)
    jsd[G.sum(1) <= 0] = 1.0
    se = (P - G) ** 2
    out = {
        "JSD": float(np.round(np.quantile(jsd, 0.5), 5)),
        "total_RMSE": float(np.sqrt(se.sum() / se.size)),
        "PCC": float(np.corrcoef(P.ravel(), G.ravel())[0, 1]),
    }
    # Per-cell-type JSD as in Supplementary Dataset 1: JSD (log2) between the
    # predicted and true spatial distributions of each cell type (columns
    # normalised over spots). Verified against the released seqFISH+ outputs.
    col_jsd = _row_jsd_log2(P.T, G.T)
    out["mean_typeJSD"] = float(col_jsd.mean())
    for k, c in enumerate(gt.columns):
        out[f"RMSE_{c}"] = float(np.sqrt(se[:, k].mean()))
        out[f"typeJSD_{c}"] = float(col_jsd[k])
    return out


def li_metrics_by_group(pred: pd.DataFrame, gt: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Per-sample metrics (e.g. per MERFISH Bregma section)."""
    rows = {}
    groups = groups.astype(str)
    groups.index = groups.index.astype(str)
    for g in sorted(groups.unique(), key=float):
        idx = groups.index[groups == g]
        rows[g] = li_metrics(pred.loc[idx], gt.loc[idx])
    return pd.DataFrame(rows).T
