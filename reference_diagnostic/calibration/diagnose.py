"""Step 1 (diagnosis): raw score location/scale vs UMI depth on complete references, observed vs
counts simulated from the fitted mixture at the same depths, for EM = 10 (package default) and 0.

Usage: python diagnose.py spotless | c2 | intestine | crc P2_CRC (arseven)
Writes results/reference_diagnostic_v2/diag_<set>.csv and per-bin arrays diag_<set>_<name>.npz.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nullcal as nc  # noqa: E402

OUT = Path(__import__("os").environ.get("RD2_OUT", HERE.parents[1] / "results/reference_diagnostic_v2"))
OUT.mkdir(parents=True, exist_ok=True)


def diagnose(m, name, save=True, rows=None):
    Y, Xbar, P0 = nc.prep(m)
    A = nc.pool_matrix(m)
    arrs = {}
    for em in (10, 0):
        t = time.time()
        o = nc.core(Y, Xbar, P0, n_em=em)
        t_obs = time.time() - t
        t = time.time()
        s = nc.simulate(Xbar, P0, o["P"], o["n"], n_em=em, seed=1)
        t_sim = time.time() - t
        arrs.update({f"dev_em{em}": o["dev"], f"nV_em{em}": o["nV"],
                     f"sdev_em{em}": s["dev"], f"snV_em{em}": s["nV"]})
        arrs["n"] = o["n"]
        for pooled in ((False, True) if A is not None else (False,)):
            AA = A if pooled else None
            z, depth = nc.zscore(o["dev"], o["nV"], o["n"], AA)
            zs, _ = nc.zscore(s["dev"], s["nV"], s["n"], AA)
            keep = o["n"] > 0 if not pooled else np.ones(len(z), bool)
            zg = nc.global_null(z[keep])
            for kind, v in (("obs_raw", z[keep]), ("sim_raw", zs[keep]), ("obs_globalnull", zg)):
                for r in nc.summarize_by_depth(v, depth[keep]):
                    r.update({"set": name, "em": em, "pooled": pooled, "kind": kind,
                              "t_obs": t_obs, "t_sim": t_sim})
                    rows.append(r)
            # per-UMI deviance (location per UMI) by depth
            if not pooled:
                u = -o["dev"][keep] / o["n"][keep]
                us = -s["dev"][keep] / s["n"][keep]
                for kind, v in (("obs_perumi", u), ("sim_perumi", us)):
                    for r in nc.summarize_by_depth(v, depth[keep]):
                        r.update({"set": name, "em": em, "pooled": False, "kind": kind})
                        rows.append(r)
    if save:
        extra = {}
        if A is not None:
            extra = {"A_indptr": A.indptr, "A_indices": A.indices}
        np.savez_compressed(OUT / f"diag_{name}.npz", **{k: v.astype(np.float32) for k, v in arrs.items()},
                            **extra)
    return rows


def main():
    import data
    what = sys.argv[1]
    rows = []
    if what == "spotless":
        for ds in range(1, 7):
            for pat in data.PATTERNS:
                d = data.spotless(ds, pat)
                if d is None:
                    continue
                Yc, Xf, crd, cts, T = d
                m = data.fit(Yc, Xf, crd, cts)
                diagnose(m, f"spotless_{ds}_{pat}", save=False, rows=rows)
                print(ds, pat, flush=True)
    elif what == "c2":
        for res in (8, 16):
            Y, X, C, cts, T = data.c2(res)
            m = data.fit(Y, X, C, cts)
            diagnose(m, f"c2_{res}um", rows=rows)
            print(res, flush=True)
    elif what == "intestine":
        for ref in ("haber", "composite"):
            Y, X, crd, ct, genes, reg, _ = data.intestine(ref)
            m = data.fit(Y, X, crd, ct)
            diagnose(m, f"intestine_{ref}", rows=rows)
            print(ref, flush=True)
    elif what == "crc":
        import crc
        sid = sys.argv[2]
        m, info, extra = crc.fit_sample(sid)
        diagnose(m, f"crc_{sid}", rows=rows)
        np.savez_compressed(OUT / f"crc_{sid}_meta.npz", **extra)
        pd.Series(info).to_json(OUT / f"crc_{sid}_info.json")
        what = f"crc_{sid}"
    pd.DataFrame(rows).to_csv(OUT / f"diag_{what}.csv", index=False)


if __name__ == "__main__":
    main()
