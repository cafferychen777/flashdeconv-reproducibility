"""Timing of the final FlashDeconv package on ONE CRC patient (called sequentially for
P1, P2, P5 inside one SLURM job on one node, see timing.sbatch). Manuscript settings
with max_iter=1000 (final default). Records n_bins, fit_transform seconds (rep 0 cold
in this process incl. numba JIT/cache load, rep 1 warm), total wall seconds from
process start (load ST + reference + prepare_data + rep-0 fit), peak RSS of this
process, n_iterations and converged."""
import resource
import sys
import time

T_START = time.perf_counter()
from pathlib import Path  # noqa: E402

import scanpy as sc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/scratch/user/cafferychen777/fd_final/code")
import fdfinal  # noqa: E402,F401
import crc_common as cc  # noqa: E402
import pandas as pd  # noqa: E402
import flashdeconv  # noqa: E402
from flashdeconv import FlashDeconv  # noqa: E402
from flashdeconv.io import prepare_data  # noqa: E402

cc.FD_KW["max_iter"] = 1000
sid, out = sys.argv[1], Path(sys.argv[2])
st = sc.read_h5ad(cc.ST_DIR / f"{sid}_deconv.h5ad")
st.obs = st.obs[[]]; st.obsm.pop("flashdeconv", None); st.uns.pop("flashdeconv_params", None)
ref = cc.load_reference()
t0 = time.perf_counter()
Y, X, coords, ctn, genes = prepare_data(st, ref, cell_type_key="Level2")
t_prep = time.perf_counter() - t0
del st, ref
rec = {"sample": sid, "n_bins": Y.shape[0], "version": flashdeconv.__version__}
for rep in range(2):
    m = FlashDeconv(verbose=False, **cc.FD_KW)
    t0 = time.perf_counter()
    m.fit_transform(Y, X, coords, cell_type_names=ctn)
    tf = time.perf_counter() - t0
    if rep == 0:
        rec.update(fit_seconds=tf, total_seconds=time.perf_counter() - T_START,
                   prepare_data_seconds=t_prep, n_iterations=m.info_.get("n_iterations"),
                   converged=m.info_.get("converged"), lambda_used=m.lambda_used_,
                   max_iter=m.max_iter, tol=m.tol)
    else:
        rec.update(fit_seconds_warm=tf, n_iterations_warm=m.info_.get("n_iterations"))
rec["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2  # Linux: KiB
rec["bins_per_second"] = rec["n_bins"] / rec["fit_seconds"]
print(rec, flush=True)
pd.DataFrame([rec]).to_csv(out, mode="a", header=not out.exists(), index=False)
