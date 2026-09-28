"""Rescore C2 (Xenium CRC pseudo-Visium HD, Xenium panel as SELF-reference) with
standard metrics, identically for every method:
  pearson_flat : Pearson over all bins x types (no entries dropped)
  type_r       : mean over types of per-type Pearson
  rmse         : sqrt(mean squared error) over all entries
  jsd          : mean per-bin squared JS distance (scipy definition, natural log)
  ap_flat      : average precision (sklearn, step-wise; ties handled) on flattened
                 presence (truth > 0.01)
  ap_type_mean : mean per-type average precision
Side columns (traceability only): r_legacy (joint zeros dropped) and auprc_legacy
(trapezoid PR-AUC), as in the original compute_metrics.
Eval sets: "all" = each method on its own covered bins; "common_<RCTD cfg>" = bins
covered by that RCTD config, all methods scored on the same bins. TACCO is flagged
reserve=True (separate table).
Adapted from rerun_v020 for the final package: FlashDeconv final rows are source="final";
all C2 configs present at run time (NNLS, marker scoring, TACCO, RCTD, v0.1.6 FlashDeconv)
are rescored from C2's read-only prediction files. Rerun after the C2 RCTD array has
finished so that every RCTD config and its common set is included."""
import sys
from pathlib import Path

import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, "/scratch/user/cafferychen777/FlashDeconv/validation/pseudo_vhd_c2")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tg_common.py copied from validation/tacco_gap
import evaluate as c2e  # noqa: E402
from c2_common import load_bins  # noqa: E402
from tg_common import jsd_scipy, legacy_metrics  # noqa: E402

OUT = Path(os.environ.get("FDFIN_C2_OUT", "/scratch/user/cafferychen777/fd_final/results/c2"))
DEST = os.environ.get("FDFIN_C2_METRICS", "c2_standard_metrics_final.csv")


def metrics(pred, true):
    p, g = pred.ravel(), true.ravel()
    tb = (g > 0.01).astype(int)
    rs, aps = [], []
    for j in range(pred.shape[1]):
        a, b = pred[:, j], true[:, j]
        rs.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan)
        t = (b > 0.01).astype(int)
        aps.append(average_precision_score(t, a) if 0 < t.sum() < len(t) else np.nan)
    leg = legacy_metrics(pred, true)
    return dict(pearson_flat=float(np.corrcoef(p, g)[0, 1]), type_r=float(np.nanmean(rs)),
                rmse=float(np.sqrt(np.mean((pred - true) ** 2))), jsd=jsd_scipy(np.clip(pred, 0, None), true),  # clip: RCTD full has tiny negatives
                ap_flat=float(average_precision_score(tb, p)), ap_type_mean=float(np.nanmean(aps)),
                r_legacy=leg["r_legacy"], auprc_legacy=leg["auprc"])


rows = []
for res in [int(x) for x in sys.argv[1:]]:
    b = load_bins(res)
    gt, cts = b["gt_props"].astype(np.float64), b["cell_types"]
    n = gt.shape[0]
    preds = {f"c2:{k}": v for k, v in c2e.load_preds(res, n, cts).items()}
    orig = c2e.PRED_DIR
    c2e.PRED_DIR = OUT / "preds"
    preds.update({f"final:{k}": v for k, v in c2e.load_preds(res, n, cts).items()})
    c2e.PRED_DIR = orig
    sets = {"all": None}
    for k, (_, cov, meta) in preds.items():
        if meta["method"] == "RCTD" and cov.sum() >= 10:
            sets[f"common_{meta['mode']}_umi{meta['umi_min']}"] = cov
    for k, (pred, cov, meta) in preds.items():
        for es, m in sets.items():
            mask = cov if m is None else (m & cov)
            if mask.sum() < 10 or (m is not None and (m & ~cov).any()):
                continue
            r = metrics(pred[mask], gt[mask])
            rows.append(dict(resolution_um=res, source=k.split(":")[0], method=meta["method"],
                             mode=meta["mode"], umi_min=meta.get("umi_min", "NA"), eval_set=es,
                             n_bins=int(mask.sum()), coverage=float(cov.mean()),
                             reserve=meta["method"] == "TACCO", **r))
    print(res, "done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / DEST, index=False)
