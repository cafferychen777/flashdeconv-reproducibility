"""Per-type average precision (the estimator of the standardized C2 metrics,
rescore_standard.py) for the figure panels: FlashDeconv final default (auto lambda),
NNLS and marker scoring, each on all bins it predicted ("all" eval set), presence =
truth > 0.01. The trapezoid PR-AUC (legacy) is kept as a side column. The mean over
types reproduces ap_type_mean of c2_standard_metrics_final.csv.
Usage: python per_type_ap.py 4 8 16 32"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

sys.path.insert(0, "/scratch/user/cafferychen777/FlashDeconv/validation/pseudo_vhd_c2")
import evaluate as c2e  # noqa: E402
from c2_common import load_bins  # noqa: E402

OUT = Path(os.environ.get("FDFIN_C2_OUT", "/scratch/user/cafferychen777/fd_final/results/c2"))
KEEP = {("final", "FlashDeconv", "final_default_auto"): "FlashDeconv",
        ("c2", "NNLS", "default"): "NNLS", ("c2", "MarkerScoring", "default"): "MarkerScoring"}

rows = []
for res in [int(x) for x in sys.argv[1:]]:
    b = load_bins(res)
    gt, cts = b["gt_props"].astype(np.float64), b["cell_types"]
    n = gt.shape[0]
    preds = {("c2", k): v for k, v in c2e.load_preds(res, n, cts).items()}
    orig = c2e.PRED_DIR
    c2e.PRED_DIR = OUT / "preds"
    preds.update({("final", k): v for k, v in c2e.load_preds(res, n, cts).items()})
    c2e.PRED_DIR = orig
    for (src, _), (pred, cov, meta) in preds.items():
        lab = KEEP.get((src, meta["method"], meta["mode"]))
        if lab is None:
            continue
        P, G = pred[cov], gt[cov]
        for j, ct in enumerate(cts):
            t = (G[:, j] > 0.01).astype(int)
            ok = 0 < t.sum() < len(t)
            ap = average_precision_score(t, P[:, j]) if ok else np.nan
            if ok:
                pr, rc, _ = precision_recall_curve(t, P[:, j])
                trap = auc(rc, pr)
            else:
                trap = np.nan
            rows.append(dict(bin_size_um=res, method=lab, cell_type=ct, n_bins=int(cov.sum()),
                             n_present=int(t.sum()), ap=ap, auprc_trapezoid=trap))
    print(res, "done", flush=True)
pd.DataFrame(rows).to_csv(OUT / "c2_per_type_ap_final.csv", index=False)
