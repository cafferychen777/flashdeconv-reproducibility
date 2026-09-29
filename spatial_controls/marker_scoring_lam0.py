"""Marker-scoring comparison on Spotless with FlashDeconv at lambda=0 (no spatial
information). Runs validation/rerun_final/benchmarks/marker_scoring/
compare_marker_scoring_rerun.py unchanged under FD_PROTOCOL=default, with every
FlashDeconv(...) forced to lambda_spatial=0.
  RERUN_OUT=... FD_PROTOCOL=default python marker_scoring_lam0.py
"""
import runpy
import sys

D = "/Users/apple/Research/FlashDeconv/validation/rerun_final/benchmarks/marker_scoring"
sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/rerun_final")
import fdfinal  # noqa: E402,F401
from flashdeconv.core.deconv import FlashDeconv  # noqa: E402

_init = FlashDeconv.__init__


def _lam0_init(self, *args, **kw):
    _init(self, *args, **kw)
    self.lambda_spatial = 0.0


FlashDeconv.__init__ = _lam0_init
sys.argv = [f"{D}/compare_marker_scoring_rerun.py"]
runpy.run_path(sys.argv[0], run_name="__main__")
