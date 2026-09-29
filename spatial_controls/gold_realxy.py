"""Gold standards (seqFISH+ 14 FOVs, STARmap) with the real spot coordinates instead of the
index lattice used by run_silver_gold.py (whose non-'_realxy' configs place every data set on
grid(n)). Same code path, metrics and package defaults; output to results/editor_revision/gold_realxy.
"""
import sys
sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/rerun_final/benchmarks/spotless")
import run_silver_gold as rsg  # noqa: E402

rsg.OUT = "/Users/apple/Research/FlashDeconv/results/editor_revision/gold_realxy"
rsg.CONFIGS = {
    "final_default_realxy": lambda G, gold: dict(),
    "final_default_lam0_realxy": lambda G, gold: dict(lambda_spatial=0.0),
    "final_default": lambda G, gold: dict(),  # lattice, reproduction check
}
sys.argv = ["run_silver_gold.py", "gold", ",".join(rsg.CONFIGS)]
rsg.main()
