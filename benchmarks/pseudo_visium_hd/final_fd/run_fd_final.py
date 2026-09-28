"""Final FlashDeconv package on the C2 pseudo-Visium HD bins (Xenium CRC P1,
Xenium panel as self-reference). Adapted from rerun_v020/benchmarks/c2/run_fd_v020.py.

Reads C2 bins/signature read-only and writes predictions in C2's npz format to
FDFIN_C2_OUT/preds/<res>um/FlashDeconv__<mode>__umiNA.npz.

Modes (package defaults: max_iter=1000, tol=1e-4, gene_weighting="expected"):
  final_default_auto / final_default_l0 : lambda auto / 0, random_state=42
  final_default_auto_seed0              : determinism check (random_state=0)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Import the installed final package (and the fit-log hook) BEFORE C2 helpers put
# the project root on sys.path.
import fdfinal  # noqa: F401
import flashdeconv
from flashdeconv.core.deconv import FlashDeconv

import numpy as np

sys.path.insert(0, "/scratch/user/cafferychen777/FlashDeconv/validation/pseudo_vhd_c2")
from c2_common import load_bins, load_signature  # noqa: E402

OUT = Path(os.environ.get("FDFIN_C2_OUT", "/scratch/user/cafferychen777/fd_final/results/c2"))


def build(mode):
    lam = 0.0 if "_l0" in mode else "auto"
    seed = 0 if mode.endswith("seed0") else 42
    return FlashDeconv(lambda_spatial=lam, random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--modes", nargs="+", required=True)
    a = ap.parse_args()
    b = load_bins(a.res)
    X_sig, cts, genes = load_signature()
    assert cts == b["cell_types"] and genes == b["genes"]
    Y = b["Y"].toarray().astype(np.float32)
    print(f"{a.res}um: {Y.shape} bins x genes; flashdeconv {flashdeconv.__version__} "
          f"{flashdeconv.__file__}", flush=True)
    for mode in a.modes:
        p = OUT / "preds" / f"{a.res}um" / f"FlashDeconv__{mode}__umiNA.npz"
        if p.exists():
            print("skip", p)
            continue
        os.environ["FD_TAG"] = f"c2_{a.res}um_{mode}"
        t0 = time.time()
        m = build(mode)
        props = m.fit_transform(Y, X_sig, b["centers"], cell_type_names=cts)
        fit = time.time() - t0
        info = m.info_ or {}
        meta = {"method": "FlashDeconv", "mode": mode, "umi_min": "NA", "fit_seconds": fit,
                "version": f"flashdeconv {flashdeconv.__version__} (final)",
                "n_iterations": info.get("n_iterations"), "converged": info.get("converged"),
                "notes": f"gene_weighting={m.gene_weighting}; lambda_used={m.lambda_used_:.4g}; "
                         f"rho={m.rho_sparsity}; max_iter={m.max_iter}; tol={m.tol}; "
                         f"iters={info.get('n_iterations')}; converged={info.get('converged')}; "
                         f"seed={m.random_state}"}
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, props=np.asarray(props, np.float32),
                            covered=np.ones(Y.shape[0], bool), meta=json.dumps(meta))
        print(f"  {mode}: {fit:.1f}s  {meta['notes']}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
