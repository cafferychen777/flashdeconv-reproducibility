"""Li et al. 2023 (Nat Commun 14:1548) benchmark rerun, final package (defaults).

Runs validation/external_benchmarks/li2023_merfish_eval.py (MERFISH 100/50/20 um)
and the FlashDeconv part of li2023_check_metrics_seqfish.py (seqFISH+ 10,000 genes)
unchanged except for the output directory ($RERUN_OUT) fit logging (fdfinal);
package defaults otherwise. Ranks among the 18 published
methods are computed from the published tables loaded by those scripts."""
import os
import sys
from pathlib import Path

sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/rerun_final")
sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/rerun_final/benchmarks")
import fdfinal  # noqa: F401  (fit log / FD_PROTOCOL)
import seedpatch  # noqa: F401  (FD_SEED)

EXT = "/Users/apple/Research/FlashDeconv/validation/external_benchmarks"
sys.path.insert(0, EXT)
os.chdir(EXT)  # li2023_metrics import
import li2023_merfish_eval as me  # noqa: E402
import li2023_check_metrics_seqfish as sq  # noqa: E402
from li2023_metrics import li_metrics  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(os.environ["RERUN_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
me.OUT = OUT
sys.argv = ["x", "--res", "100", "50", "20"]
me.main()

gt = sq.load_gt()
pub = sq.published_table()
pred = sq.run_flashdeconv(gt)
pred.to_csv(OUT / "seqfish10000_flashdeconv_pred.csv")
m = li_metrics(pred, gt)
tab = pub.rename(columns={"JSD_pub": "JSD", "total_RMSE_pub": "total_RMSE"}).copy()
tab.loc["FlashDeconv"] = [m["JSD"], m["total_RMSE"]]
tab["rank_JSD"] = tab["JSD"].rank().astype(int)
tab["rank_RMSE"] = tab["total_RMSE"].rank().astype(int)
tab.sort_values("JSD").to_csv(OUT / "seqfish10000_vs_published.csv")
print("seqFISH+ FlashDeconv", {k: round(v, 4) for k, v in m.items() if not k.startswith(("RMSE_", "typeJSD_"))})
print(tab.sort_values("JSD").round(4).to_string())
