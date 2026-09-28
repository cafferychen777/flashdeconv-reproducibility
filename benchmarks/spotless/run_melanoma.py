"""Melanoma case study (tissue-level JSD vs Molecular Cartography) for the final package (copy of the v0.2.0
rerun script; configurations and fit logging changed; grid uses max_iter default 1000).

Data loading, 15-type signature, aggregation to the 7 evaluated classes and JSD are
imported from validation/melanoma_analysis/tune_melanoma_params.py (source of the
manuscript configurations in best_configs.json).

  fixed : the two manuscript configurations (JSD-optimised: pearson, lambda=5000,
          rho=0; accuracy-optimised: log_cpm, lambda=0, rho=0.01), legacy vs v0.2.0,
          plus v0.2.0 package defaults (log_cpm / pearson, auto lambda)
  grid  : the same 108-configuration grid the manuscript configuration was selected
          from, rerun with v0.2.0 (a fixed numeric lambda acts on a representation of
          different scale, so the manuscript's lambda=5000 does not transfer)
"""
import sys
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/melanoma_analysis")
import tune_melanoma_params as tm  # noqa: E402
sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/rerun_final")
import fdfinal  # noqa: F401,E402  (fit log)
import os  # noqa: E402
from flashdeconv import FlashDeconv  # noqa: E402

OUT = "/Users/apple/Research/FlashDeconv/results/rerun_final/benchmarks/spotless"
# The manuscript configurations with every other argument at the package default
# (max_iter 1000, tol 1e-4, expected weighting, random_state 0).
FIXED = {
    "final_default": dict(),                                   # pure package defaults (log-CPM)
    "final_default_seed42": dict(random_state=42),             # determinism check
    "final_default_pearson": dict(preprocess="pearson"),       # defaults + Pearson residuals
    "final_jsdopt": dict(preprocess="pearson", lambda_spatial=5000, rho_sparsity=0),  # manuscript JSD config
    "final_accopt": dict(preprocess="log_cpm", lambda_spatial=0, rho_sparsity=0.01),  # manuscript accuracy config
}
BASE = dict(sketch_dim=512, n_hvg=2000, n_markers_per_type=50, tol=1e-4, verbose=False)


def load():
    rc, rg, rct = tm.load_reference()
    X_full, _, cts = tm.build_full_signature(rc, rct, rg)
    data = {}
    for s in [2, 3, 4]:
        Y, g, xy = tm.load_spatial(s)
        Ya, Xa, _ = tm.align_genes(Y, g, X_full, rg)
        data[s] = (Ya, Xa, xy)
    return data, cts, np.array([tm.MC_GROUND_TRUTH[c] for c in tm.EVAL_CELLTYPES])


def run(data, cts, gt, kw, label=''):
    out = {}
    for s, (Y, X, xy) in data.items():
        os.environ["FD_CONTEXT"] = f"melanoma|{s}|{label}"
        props = FlashDeconv(**kw).fit_transform(Y, X, xy)
        ev = tm.aggregate_to_eval(props.mean(0), cts)
        out[s] = (tm.calculate_jsd(gt, ev), ev[tm.EVAL_CELLTYPES.index("Melanocytic")],
                  ev[tm.EVAL_CELLTYPES.index("Tcell")], props)
    return out


def main(mode):
    data, cts, gt = load()
    rows = []
    if mode == "fixed":
        keep = {}
        for name, kw in FIXED.items():
            r = run(data, cts, gt, kw, name)
            keep[name] = r
            for s, (j, mel, t, _) in r.items():
                rows.append(dict(config=name, sample=s, jsd=j, melanocytic=mel, tcell=t))
            print(name, np.mean([v[0] for v in r.values()]).round(4), flush=True)
        det = [dict(sample=s, max_abs_diff=float(np.abs(keep["final_default"][s][3] - keep["final_default_seed42"][s][3]).max()))
               for s in data]
        pd.DataFrame(det).to_csv(f"{OUT}/determinism_melanoma.csv", index=False)
        pd.DataFrame(rows).to_csv(f"{OUT}/melanoma_fixed.csv", index=False)
    else:
        gw = sys.argv[2] if len(sys.argv) > 2 else "expected"
        grid = dict(n_hvg=[2000, 5000, 10000], n_markers_per_type=[50, 100],
                    lambda_spatial=[0, "auto", 5000], rho_sparsity=[0, 0.01, 0.05],
                    preprocess=["log_cpm", "pearson"])
        for vals in product(*grid.values()):
            cfg = dict(zip(grid, vals))
            kw = dict(BASE, **cfg, gene_weighting=gw)
            r = run(data, cts, gt, kw, 'grid:' + str(cfg))
            rows.append(dict(cfg, gene_weighting=gw, jsd=np.mean([v[0] for v in r.values()]),
                             melanocytic=np.mean([v[1] for v in r.values()])))
            print(cfg, round(rows[-1]["jsd"], 4), flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/melanoma_grid_{gw}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
