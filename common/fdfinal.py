"""Shared hook for the final-package rerun.

Import this module before any FlashDeconv model is built. It
  1. optionally rewrites FlashDeconv(...) arguments (FD_PROTOCOL, same semantics
     as validation/rerun_v020/benchmarks/fdpatch.py: "default" keeps only
     preprocess / spatial_method / radius / verbose / random_state and an
     explicit lambda_spatial=0, everything else falls back to the package
     defaults; "orig" keeps the script's arguments; unset = "orig");
  2. appends one row per fit (n_iterations, converged, lambda, seconds, shape)
     to the CSV named by FD_FITLOG (if set), tagged with FD_TAG.
"""
import csv
import os
import time

import flashdeconv
from flashdeconv.core.deconv import FlashDeconv

PROTOCOL = os.environ.get("FD_PROTOCOL", "orig")
FITLOG = os.environ.get("FD_FITLOG")
TAG = os.environ.get("FD_TAG", "")
_KEEP_DEFAULT = {"preprocess", "spatial_method", "radius", "verbose", "random_state"}
_orig_init = FlashDeconv.__init__
_orig_fit = FlashDeconv.fit


def _patched_init(self, *args, **kw):
    if args:
        raise TypeError("fdfinal expects keyword arguments only")
    if PROTOCOL == "default":
        lam = kw.get("lambda_spatial", "auto")
        kw = {k: v for k, v in kw.items() if k in _KEEP_DEFAULT}
        if isinstance(lam, (int, float)) and lam == 0:
            kw["lambda_spatial"] = 0.0
    elif PROTOCOL != "orig":
        raise ValueError(f"unknown FD_PROTOCOL {PROTOCOL!r}")
    _orig_init(self, **kw)


def _patched_fit(self, Y, X, coords=None, *args, **kw):
    t0 = time.perf_counter()
    out = _orig_fit(self, Y, X, coords, *args, **kw)
    if FITLOG:
        info = self.info_ or {}
        row = dict(tag=TAG or os.environ.get("FD_CONTEXT", ""),
                   context=os.environ.get("FD_CONTEXT", ""),
                   n_spots=getattr(self, "n_spots_", None),
                   n_types=getattr(self, "n_cell_types_", None),
                   n_iterations=info.get("n_iterations"), converged=info.get("converged"),
                   max_iter=self.max_iter, tol=self.tol,
                   lambda_used=getattr(self, "lambda_used_", None),
                   rho=self.rho_sparsity, preprocess=self.preprocess,
                   gene_weighting=self.gene_weighting, random_state=self.random_state,
                   fit_seconds=round(time.perf_counter() - t0, 3),
                   version=flashdeconv.__version__)
        new = not os.path.exists(FITLOG)
        with open(FITLOG, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
    return out


FlashDeconv.__init__ = _patched_init
FlashDeconv.fit = _patched_fit
print(f"[fdfinal] flashdeconv {flashdeconv.__version__} from {flashdeconv.__file__}; "
      f"protocol={PROTOCOL}; fitlog={FITLOG}", flush=True)
