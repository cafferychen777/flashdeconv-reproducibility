"""Control 1: does the Spotless silver-standard lattice layout carry structure, and how much
of FlashDeconv's silver-standard performance depends on it?

Part A (structure): for each of the 54 silver data sets, place spots on the same square
lattice (data-set order, row-major, as run_silver_gold.grid) and the same k=6 kNN graph the
package builds; compute Moran's I of each true proportion column, the mean cosine similarity
of true compositions across graph edges vs random pairs, and the same with a random layout.

Part B (rerun): FlashDeconv package defaults with (a) the original lattice, (b) 5 random
permutations of spot positions (seeds 0-4), (c) lambda_spatial=0. Same metric code as
validation/rerun_final/benchmarks/spotless/run_silver_gold.py.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation")
sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/rerun_final/benchmarks/spotless")
import comprehensive_per_celltype_evaluation as ce  # noqa: E402
from run_silver_gold import grid, jsd_nan0  # noqa: E402
from flashdeconv import FlashDeconv, __version__  # noqa: E402
from flashdeconv.utils.graph import build_knn_graph  # noqa: E402

OUT = "/Users/apple/Research/FlashDeconv/results/controls_editor"
os.makedirs(OUT, exist_ok=True)
PERM_SEEDS = [0, 1, 2, 3, 4]


def morans_I(A, x):
    z = x - x.mean()
    den = (z ** 2).sum()
    if den == 0:
        return np.nan
    W = A.sum()
    return len(x) / W * float(z @ (A @ z)) / den


def edge_vs_random_similarity(A, P, rng, n_rand=20000):
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    Au = A.tocoo()
    m = Au.row < Au.col
    e = (Pn[Au.row[m]] * Pn[Au.col[m]]).sum(1).mean()
    i = rng.integers(0, len(P), n_rand); j = rng.integers(0, len(P), n_rand)
    k = i != j
    r = (Pn[i[k]] * Pn[j[k]]).sum(1).mean()
    # index-adjacent pairs (i, i+1)
    a = (Pn[:-1] * Pn[1:]).sum(1).mean()
    return e, r, a


def structure():
    rows = []
    for tid, tissue in ce.TISSUE_NAMES.items():
        for pid, pattern in ce.PATTERN_MAP.items():
            _, _, props = ce.load_silver_data(tid, pid)
            P = props.values.astype(float)
            n = len(P)
            rng = np.random.default_rng(0)
            coords = grid(n)
            A = build_knn_graph(coords, k=6)
            Ip = [morans_I(A, P[:, j]) for j in range(P.shape[1])]
            e, r, a = edge_vs_random_similarity(A, P, rng)
            # the same on a random layout (seed 0)
            Ar = build_knn_graph(coords[rng.permutation(n)], k=6)
            Ir = [morans_I(Ar, P[:, j]) for j in range(P.shape[1])]
            er, _, _ = edge_vs_random_similarity(Ar, P, rng)
            # number of runs of identical presence pattern along the data-set order
            pres = [tuple(np.flatnonzero(p > 0)) for p in P]
            runs = 1 + sum(pres[t] != pres[t - 1] for t in range(1, n))
            rows.append(dict(tissue=tissue, pattern=pattern, n_spots=n, n_types=P.shape[1],
                             moranI_lattice_mean=np.nanmean(Ip), moranI_lattice_max=np.nanmax(Ip),
                             moranI_random_mean=np.nanmean(Ir), moranI_expected=-1 / (n - 1),
                             cos_lattice_edges=e, cos_random_layout_edges=er, cos_random_pairs=r,
                             cos_index_adjacent=a, n_unique_presence_sets=len(set(pres)),
                             n_presence_runs_in_order=runs))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/c1_structure.csv", index=False)
    return df


def settings(n):
    s = [("lattice", grid(n), {})]
    for sd in PERM_SEEDS:
        perm = np.random.default_rng(sd).permutation(n)
        s.append((f"perm{sd}", grid(n)[perm], {}))
    s.append(("lam0", grid(n), dict(lambda_spatial=0.0)))
    return s


def rerun():
    agg = []
    for tid, tissue in ce.TISSUE_NAMES.items():
        X, types, rgenes = ce.load_reference_data(tid)
        for pid, pattern in ce.PATTERN_MAP.items():
            Y, g, props = ce.load_silver_data(tid, pid)
            Ya, Xa, _ = ce.align_genes(Y, g, X, rgenes)
            true_sub = props[[c for c in types if c in props.columns]]
            for name, coords, kw in settings(Ya.shape[0]):
                m = FlashDeconv(**kw)
                pred = m.fit_transform(Ya, Xa, coords)
                a = ce.compute_aggregate_metrics(pred, true_sub, types)
                a.pop("jsd_contributions")
                a["jsd"] = jsd_nan0(pred, true_sub, types)
                agg.append(dict(setting=name, tissue=tissue, pattern=pattern,
                                lambda_used=m.lambda_used_, version=__version__,
                                **{k: a[k] for k in ["corr", "rmse", "jsd", "aupr"]}))
            print(tissue, pattern, {r["setting"]: round(r["corr"], 4) for r in agg[-7:]}, flush=True)
    df = pd.DataFrame(agg)
    df.to_csv(f"{OUT}/c1_fd_aggregate_settings.csv", index=False)
    return df


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("structure", "all"):
        s = structure()
        pd.set_option("display.width", 250)
        print(s.round(3).to_string(index=False))
    if what in ("rerun", "all"):
        rerun()
