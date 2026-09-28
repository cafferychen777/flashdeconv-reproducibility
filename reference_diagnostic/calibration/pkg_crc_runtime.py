"""Re-validation with the updated package (flashdeconv.reference_fit_scores, null='auto' vs
'left_half'): CRC P1/P2/P5 per-bin scores, and runtime at 1e5 / 1e6 bins (C1 benchmark data).

Usage: python pkg_crc_runtime.py crc P1_CRC | runtime
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import os  # noqa: E402

OUT = Path(os.environ["RD2_OUT"])


def crc(sid):
    import crc as C
    import flashdeconv
    m, info, extra = C.fit_sample(sid)
    res = {}
    for meth in ("auto", "left_half"):
        t = time.time()
        s = flashdeconv.reference_fit_scores(m, null=meth)
        info[f"diag_s_{meth}"] = time.time() - t
        for k in ("score", "score_pooled"):
            res[f"{meth}_{k}"] = s[k].astype(np.float32)
        info[f"null_{meth}"] = {k: v for k, v in s["null"].items()}
        info[f"flag_{meth}"] = float(s["flag"].mean())
        info[f"flag_pooled_{meth}"] = float(s["flag_pooled"].mean())
    res["n_umi"] = s["n_umi"].astype(np.float32)
    res["barcode"] = extra["barcode"]
    np.savez_compressed(OUT / f"pkg_crc_{sid}.npz", **res)
    pd.Series(info).to_json(OUT / f"pkg_crc_{sid}_info.json", default_handler=str)
    print(info, flush=True)


def runtime():
    import anndata as ad
    import flashdeconv
    from flashdeconv import FlashDeconv
    from flashdeconv.io import prepare_data
    data = Path("/scratch/user/cafferychen777/FlashDeconv/data/runtime_benchmark_c1")
    ref = ad.read_h5ad(data / "ref.h5ad")
    rows = []
    for scale in (100000, 1000000):
        st = ad.read_h5ad(data / f"st_{scale}.h5ad")
        Y, X, coords, types, genes = prepare_data(st, ref, cell_type_key="cell_type")
        t0 = time.time()
        m = FlashDeconv()
        m.fit(Y, X, coords, cell_type_names=types)
        tf = time.time() - t0
        flashdeconv.reference_fit_scores(m, max_em_iter=1)  # numba compile outside the timing
        r = {"n_bins": Y.shape[0], "K": len(types), "n_selected": len(m.gene_idx_), "fit_s": tf,
             "fit_iterations": m.info_.get("n_iterations")}
        for meth in ("auto", "left_half"):
            t0 = time.time()
            s = flashdeconv.reference_fit_scores(m, null=meth)
            r[f"diag_s_{meth}"] = time.time() - t0
            r[f"flag_pooled_{meth}"] = float(s["flag_pooled"].mean())
        r["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        r["version"] = flashdeconv.__version__
        rows.append(r)
        print(r, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "pkg_runtime.csv", index=False)
        del st, Y, m, s


if __name__ == "__main__":
    if sys.argv[1] == "crc":
        crc(sys.argv[2])
    else:
        runtime()
