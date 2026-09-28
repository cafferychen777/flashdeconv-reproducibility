"""Merge the final-package FlashDeconv C1 rows with the C1 competitor rows.

Inputs (rsynced to results/rerun_final/benchmarks/c1/):
  c1_results_snapshot.csv   copy of FlashDeconv/results/runtime_benchmark_c1/c1_results.csv
  c1_flashdeconv_final.csv  monitor.py rows of the sequential final job (numba cache
                            shared across the tasks of the job)
  c1_flashdeconv_final_coldcache.csv  same, fresh numba cache per task (as in the
                            original one-job-per-task C1 protocol)
  run_json[_coldcache]/*.json  per-task runner results (n_iterations, converged, ...)
  props/*.csv.gz            1e4 predictions, seed 42 and seed 0
  tasks/*.txt               the C1 task lists (expected method x mode x scale cells)
  queue_status.txt          optional "method mode scale state" lines for C1 cells that
                            are still running/pending in SLURM at merge time
Outputs: c1_results_merged_final.csv, c1_runtime_table_final.csv, c1_determinism_final.csv
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(os.environ.get("C1_FINAL_DIR",
                        "/Users/apple/Research/FlashDeconv/results/rerun_final/benchmarks/c1"))
V020 = Path("/Users/apple/Research/FlashDeconv/results/rerun_v020/benchmarks/c1")

c1 = pd.read_csv(R / "c1_results_snapshot.csv")
c1.loc[c1.method == "flashdeconv", "method"] = "flashdeconv_v016"

def load_final(sfx):
    """Final rows (sfx "" = shared numba cache within the job, "_coldcache" = fresh
    cache per task) with n_iterations / converged etc. from the runner JSONs."""
    fin = pd.read_csv(R / f"c1_flashdeconv_final{sfx}.csv")
    extra = []
    for f in glob.glob(str(R / f"run_json{sfx}" / "flashdeconv_final_*.json")):
        j = json.load(open(f))
        mode, scale, rep = Path(f).stem.split("_")[2:5]  # flashdeconv_final_<mode>_<scale>_r<rep>
        extra.append(dict(method="flashdeconv_final", mode=mode, scale=int(scale), rep=int(rep[1:]),
                          n_iterations=j.get("n_iterations"), converged=j.get("converged"),
                          max_iter=j.get("max_iter"), tol=j.get("tol"),
                          lambda_used=j.get("lambda_used"), random_state=j.get("random_state")))
    fin = fin.merge(pd.DataFrame(extra), on=["method", "mode", "scale", "rep"], how="left")
    fin["method"] = "flashdeconv_final" + sfx
    return fin


fins = [load_final(s) for s in ("", "_coldcache") if (R / f"c1_flashdeconv_final{s}.csv").exists()]
fin = pd.concat(fins, ignore_index=True)

# v0.2.0 rows used in the previous merge (for the side-by-side table only)
v020 = pd.read_csv(V020 / "c1_results_merged_with_v020.csv")
v020 = v020[v020.method == "flashdeconv_v020"]

# Figure file: competitor rows exactly as in c1_results.csv (all statuses) + the final
# FlashDeconv rows (cold numba cache per task, as in the one-job-per-task C1 protocol),
# method="flashdeconv_final", mode="default", with n_iterations / converged columns.
# The v0.1.6 FlashDeconv rows, the warm-cache rows and the seed-0 determinism row are
# kept only in c1_flashdeconv_final_all.csv.
raw = pd.read_csv(R / "c1_results_snapshot.csv")
fig_fd = fin[(fin.method == "flashdeconv_final_coldcache") & (fin["mode"] == "default")].copy()
fig_fd["method"] = "flashdeconv_final"
fig_fd["n_iterations"] = fig_fd["n_iterations"].astype("Int64")
fig = pd.concat([raw[raw.method != "flashdeconv"], fig_fd[list(raw.columns) + ["n_iterations", "converged"]]],
                ignore_index=True)
fig.to_csv(R / "c1_results_merged_final.csv", index=False)
fin.to_csv(R / "c1_flashdeconv_final_all.csv", index=False)
merged = pd.concat([c1, fin], ignore_index=True)

# ---- expected cells from the C1 task lists
exp = set()
for f in glob.glob(str(R / "tasks" / "*.txt")):
    for line in open(f):
        p = line.split()
        if len(p) >= 3:
            m = "flashdeconv_v016" if p[0] == "flashdeconv" else p[0]
            exp.add((m, p[1], int(p[2])))
queue = {}
if (R / "queue_status.txt").exists():
    for line in open(R / "queue_status.txt"):
        p = line.split()
        if len(p) >= 4:
            queue[(p[0], p[1], int(p[2]))] = p[3]

rows = []
allp = pd.concat([merged, v020], ignore_index=True)
for (m, mode, s), g in allp.groupby(["method", "mode", "scale"]):
    ok = g[g.status == "OK"]
    if len(ok):
        st = "OK"
        note = ""
    else:
        bad = g.iloc[-1]
        st = bad.status
        note = (f"mem limit {bad.mem_limit_gb} GB; sampled peak PSS {bad.peak_rss_gb} GB; {bad.notes}" if st == "OOM"
                else f"wall cap; {bad.notes}" if st == "DNF" else str(bad.notes)[:120])
        if (m, mode, s) in queue:
            note += f"; retry {queue[(m, mode, s)]}"
    fo = g if not len(ok) else ok
    rows.append(dict(method=m, mode=mode, scale=int(s),
                     median_fit_s=ok.fit_seconds.median() if len(ok) else np.nan,
                     peak_rss_gb=ok.peak_rss_gb.max() if len(ok) else fo.peak_rss_gb.max(),
                     n_reps=len(ok), status=st,
                     hosts=",".join(sorted(set(map(str, fo.hostname)))),
                     n_iterations=(",".join(str(int(x)) for x in ok.n_iterations.dropna())
                                   if "n_iterations" in ok and ok.n_iterations.notna().any() else ""),
                     converged=(",".join(str(x) for x in ok.converged.dropna())
                                if "converged" in ok and ok.converged.notna().any() else ""),
                     notes=note))
    exp.discard((m, mode, int(s)))
for (m, mode, s) in sorted(exp):
    rows.append(dict(method=m, mode=mode, scale=s, median_fit_s=np.nan, peak_rss_gb=np.nan,
                     n_reps=0, status=queue.get((m, mode, s), "not run"), hosts="", notes=""))
tab = pd.DataFrame(rows).sort_values(["method", "mode", "scale"])
tab.to_csv(R / "c1_runtime_table_final.csv", index=False)
print(tab.to_string(index=False))

# ---- determinism (1e4, seed 42 vs seed 0)
a = pd.read_csv(R / "props" / "flashdeconv_final_10000_seed42.csv.gz", index_col=0)
b = pd.read_csv(R / "props" / "flashdeconv_final_10000_seed0.csv.gz", index_col=0).loc[a.index, a.columns]
d = pd.DataFrame([dict(scale=10000, seeds="42_vs_0", n_bins=len(a),
                       max_abs_diff=float(np.abs(a.values - b.values).max()),
                       note="props saved with float_format %.5g")])
d.to_csv(R / "c1_determinism_final.csv", index=False)
print(d.to_string(index=False))
