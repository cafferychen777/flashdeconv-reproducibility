"""Evaluate all C2 predictions with the original compute_metrics().

Eval sets per resolution:
  legacy_zero_fill : all bins, uncovered bins scored as all-zero predictions
                     (the ORIGINAL table protocol; used only to check reproduction
                     and to show how the old protocol penalized filtered bins)
  all              : each method on its own covered bins (coverage reported)
  common           : bins covered by every method config at that resolution
                     (configs with zero coverage excluded from the intersection)
  common_no_umi100 : as common, but excluding RCTD UMI_min=100 configs

Metric mapping to the table: pearson = global_r; type_r = mean_per_type_r;
auprc = global_auprc (flattened; this is what tab:pseudo_vhd reported);
auprc_type_mean = mean per-type AUPRC; jsd = global_jsd.

Usage: python evaluate.py --res 32      (writes eval/c2_metrics_32um.csv, ...)
       python evaluate.py --merge       (writes c2_metrics.csv, c2_per_type.csv)
"""

import argparse
import json

import numpy as np
import pandas as pd

from c2_common import EVAL_DIR, OUT, PRED_DIR, RESOLUTIONS, load_bins
import xenium_pseudo_visiumhd_benchmark as ob


def load_preds(res, n_bins, cell_types):
    preds = {}
    d = PRED_DIR / f"{res}um"
    for p in sorted(d.glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        z = np.load(p, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        preds[p.stem] = (z["props"].astype(np.float64), z["covered"], meta)
    for p in sorted(d.glob("RCTD__*.csv.gz")):
        if p.name.endswith(".tmp.gz"):
            continue
        stem = p.name[:-len(".csv.gz")]
        _, mode, umi = stem.split("__")
        mt = {}
        for line in p.with_name(stem + ".meta.txt").read_text().splitlines():
            k, _, v = line.partition("=")
            mt[k] = v
        props = np.zeros((n_bins, len(cell_types)))
        covered = np.zeros(n_bins, dtype=bool)
        df = pd.read_csv(p)
        if len(df):
            idx = df["barcode"].str.replace("bin_", "", regex=False).astype(int).values
            for j, ct in enumerate(cell_types):
                if ct in df.columns:
                    props[idx, j] = df[ct].values
            covered[idx] = True
            s = props[idx].sum(axis=1, keepdims=True)
            s[s == 0] = 1
            props[idx] /= s
        meta = {"method": "RCTD", "mode": mode, "umi_min": int(umi.replace("umi", "")),
                "fit_seconds": float(mt.get("fit_seconds", "nan")),
                "version": mt.get("version", ""),
                "notes": f"UMI_min_sigma={mt.get('umi_min_sigma')}; {mt.get('notes', '')}"}
        preds[stem] = (props, covered, meta)
    return preds


def score(pred, gt, cell_types, mask):
    if mask.sum() < 10:
        return None
    return ob.compute_metrics(pred[mask], gt[mask], cell_types)


def eval_res(res):
    b = load_bins(res)
    gt, cts = b["gt_props"].astype(np.float64), b["cell_types"]
    n = gt.shape[0]
    preds = load_preds(res, n, cts)
    print(f"{res}um: {n:,} bins; configs: {list(preds)}", flush=True)

    active = {k: v for k, v in preds.items() if v[1].any()}
    common = np.ones(n, dtype=bool)
    for k, v in active.items():
        common &= v[1]
    common2 = np.ones(n, dtype=bool)
    for k, v in active.items():
        if not (v[2]["method"] == "RCTD" and int(v[2]["umi_min"]) == 100):
            common2 &= v[1]

    rows, per_type = [], []
    for key, (pred, cov, meta) in preds.items():
        sets = {"legacy_zero_fill": np.ones(n, dtype=bool), "all": cov,
                "common": common, "common_no_umi100": common2}
        for es, mask in sets.items():
            if es == "legacy_zero_fill":
                m = score(np.where(cov[:, None], pred, 0.0), gt, cts, mask)
            else:
                m = score(pred, gt, cts, mask & cov)
            rows.append({
                "resolution_um": res, "method": meta["method"], "mode": meta["mode"],
                "umi_min": meta.get("umi_min", "NA"), "n_bins": int(n),
                "coverage": float(cov.mean()), "eval_set": es,
                "n_eval_bins": int((mask if es == "legacy_zero_fill" else mask & cov).sum()),
                "pearson": m["global_r"] if m else np.nan,
                "type_r": m["mean_per_type_r"] if m else np.nan,
                "auprc": m["global_auprc"] if m else np.nan,
                "auprc_type_mean": m["mean_per_type_auprc"] if m else np.nan,
                "jsd": m["global_jsd"] if m else np.nan,
                "fit_seconds": meta["fit_seconds"], "version": meta["version"],
                "notes": meta.get("notes", ""),
            })
            if m and es in ("all", "common"):
                for ct in cts:
                    per_type.append({
                        "resolution_um": res, "method": meta["method"],
                        "mode": meta["mode"], "umi_min": meta.get("umi_min", "NA"),
                        "eval_set": es, "cell_type": ct,
                        "lineage": ob.LINEAGE_MAP.get(ct, "Other"),
                        "pearson_r": m["per_type_r"].get(ct, np.nan),
                        "auprc": m["per_type_auprc"].get(ct, np.nan),
                        "rmse": m["per_type_rmse"].get(ct, np.nan),
                    })
            print(f"  {key:40s} {es:18s} cov={cov.mean():.3f} "
                  + (f"r={m['global_r']:.4f} type_r={m['mean_per_type_r']:.4f} "
                     f"AUPRC={m['global_auprc']:.4f} JSD={m['global_jsd']:.4f}" if m else "NA"),
                  flush=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(EVAL_DIR / f"c2_metrics_{res}um.csv", index=False)
    pd.DataFrame(per_type).to_csv(EVAL_DIR / f"c2_per_type_{res}um.csv", index=False)


def merge():
    m = [pd.read_csv(p) for p in sorted(EVAL_DIR.glob("c2_metrics_*um.csv"))]
    t = [pd.read_csv(p) for p in sorted(EVAL_DIR.glob("c2_per_type_*um.csv"))]
    pd.concat(m).sort_values(["resolution_um", "eval_set", "method", "mode", "umi_min"]
                             ).to_csv(OUT / "c2_metrics.csv", index=False)
    pd.concat(t).to_csv(OUT / "c2_per_type.csv", index=False)
    print("merged ->", OUT / "c2_metrics.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs="*", default=[])
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    for r in a.res:
        eval_res(r)
    if a.merge:
        merge()
