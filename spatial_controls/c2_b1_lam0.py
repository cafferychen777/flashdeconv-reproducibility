"""Control 2b: interface atlas (validation/b1_pilot) with lambda_spatial=0.

  fit SEC   run_fd.main(SEC) unchanged except fd.tl.deconvolve(..., lambda_spatial=0) and output dir
  grad      gradients.main / main_ext unchanged, with RES pointing to the lambda=0 fits
            (orthogonal orth_*.npz symlinked from the original b1_pilot results)
"""
import functools
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/user/cafferychen777/b1_pilot/code")
CTL = Path("/scratch/user/cafferychen777/controls_editor/b1")
CTL.mkdir(parents=True, exist_ok=True)
ORIG = Path("/scratch/user/cafferychen777/b1_pilot/results")

if __name__ == "__main__":
    if sys.argv[1] == "fit":
        import run_fd
        run_fd.OUT = CTL
        run_fd.fd.tl.deconvolve = functools.partial(run_fd.fd.tl.deconvolve, lambda_spatial=0.0)
        run_fd.main(sys.argv[2])
    else:
        for f in ORIG.glob("orth_*.npz"):
            t = CTL / f.name
            if not t.exists():
                t.symlink_to(f)
        import gradients
        gradients.RES = CTL
        gradients.main(gradients.PRIMARY + gradients.SECONDARY)
        gradients.main_ext(gradients.EXT)
