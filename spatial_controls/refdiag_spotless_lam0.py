"""Spotless controlled-removal validation of the reference diagnostic with lambda=0 (no spatial
information). Runs validation/rerun_final/refdiag/spotless.py unchanged except FlashDeconv(lambda_spatial=0)
and the output directory (results/editor_revision/refdiag_lam0)."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/rerun_final")
import fdfinal  # noqa: E402,F401
from flashdeconv.core.deconv import FlashDeconv  # noqa: E402

_init = FlashDeconv.__init__


def _lam0(self, *a, **kw):
    _init(self, *a, **kw)
    self.lambda_spatial = 0.0


FlashDeconv.__init__ = _lam0
src = Path("/Users/apple/Research/FlashDeconv/validation/rerun_final/refdiag/spotless.py").read_text()
src = src.replace('OUT = ROOT / "results/rerun_final/refdiag"', 'OUT = ROOT / "results/editor_revision/refdiag_lam0"')
assert "editor_revision/refdiag_lam0" in src
exec(compile(src, "spotless_refdiag_lam0", "exec"), {"__name__": "__main__"})
