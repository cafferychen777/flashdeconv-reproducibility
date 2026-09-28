"""Merge ACES cell2location C1 rows into the C1 runtime table.

Usage: python merge_c1.py <aces_raw_csv>
Writes results/rerun_final/benchmarks/c1/c1_c2l_aces.csv (C1 raw schema incl. gpu_model; notes
prefixed with the cluster) and replaces the cell2location rows of c1_runtime_table_final.csv
(backup kept as c1_runtime_table_final.before_c2l_aces.csv).
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

D = Path("/Users/apple/Research/FlashDeconv/results/rerun_final/benchmarks/c1")
raw = pd.read_csv(sys.argv[1])
raw = raw.drop_duplicates(["method", "mode", "scale", "rep"], keep="last")
prefix = "cluster=ACES (TAMU HPRC), not arseven; GPU=" + raw.gpu_model.fillna("").astype(str)
raw["notes"] = prefix + " | " + raw.notes.fillna("").astype(str)
raw.to_csv(D / "c1_c2l_aces.csv", index=False)

tab_path = D / "c1_runtime_table_final.csv"
backup = D / "c1_runtime_table_final.before_c2l_aces.csv"
if not backup.exists():
    shutil.copy(tab_path, backup)
tab = pd.read_csv(backup)
keep = tab[tab.method != "cell2location"]
rows = []
for _, r in raw.iterrows():
    ok = r.status == "OK"
    rows.append(dict(
        method=r.method, mode=r["mode"], scale=r.scale,
        median_fit_s=r.fit_seconds if ok else None,
        peak_rss_gb=r.peak_rss_gb, n_reps=1 if ok else 0, status=r.status, hosts=r.hostname,
        n_iterations=None, converged=None,
        notes=(f"ACES cluster (not arseven), GPU {r.gpu_model}, 16 CPUs; peak_gpu_gb={r.peak_gpu_gb}; "
               + str(r.notes).split(" | ", 1)[1]),
    ))
out = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
# cell2location runs not yet finished on ACES stay listed as pending
done = {(r["method"], r["mode"], int(r["scale"])) for r in rows}
pend = tab[(tab.method == "cell2location")
           & ~tab.apply(lambda r: (r["method"], r["mode"], int(r["scale"])) in done, axis=1)].copy()
pend["status"] = "PENDING(ACES)"
pend["notes"] = "moved from arseven to ACES; not yet finished"
out = pd.concat([out, pend], ignore_index=True).sort_values(["method", "mode", "scale"])
out.to_csv(tab_path, index=False)
print(out[out.method == "cell2location"].to_string())
