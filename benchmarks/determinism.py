"""Max abs difference between FlashDeconv predictions fitted with random_state 42
and 0 (final package).
  determinism.py c2 <C2 out dir>   : 16 um, FlashDeconv__final_default_auto(_seed0)
  determinism.py xb <breast out dir>: final_default vs final_default_seed0, fd_auto and
                                      fd_l0 at every saved bin size
Writes determinism_<part>.csv into the given directory."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

part, d = sys.argv[1], Path(sys.argv[2])
rows = []
if part == "c2":
    for f in sorted((d / "preds").glob("*um/FlashDeconv__final_default_auto_seed0__umiNA.npz")):
        a = np.load(f.with_name("FlashDeconv__final_default_auto__umiNA.npz"))["props"].astype(float)
        b = np.load(f)["props"].astype(float)
        rows.append(dict(part="c2", resolution_um=int(f.parent.name[:-2]), config="lambda_auto",
                         seeds="42_vs_0", n_bins=a.shape[0], max_abs_diff=float(np.abs(a - b).max())))
else:
    for f in sorted((d / "final_default_seed0").glob("preds_*um.npz")):
        za, zb = np.load(d / "final_default" / f.name), np.load(f)
        for k in ["fd_auto", "fd_l0", "marker_scoring"]:
            a, b = za[k].astype(float), zb[k].astype(float)
            rows.append(dict(part="xenium_breast", resolution_um=int(f.stem.split("_")[1][:-2]),
                             config=k, seeds="42_vs_0", n_bins=a.shape[0],
                             max_abs_diff=float(np.abs(a - b).max())))
df = pd.DataFrame(rows)
df.to_csv(d / f"determinism_{part}.csv", index=False)
print(df.to_string(index=False))
