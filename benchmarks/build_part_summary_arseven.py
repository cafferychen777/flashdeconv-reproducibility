"""Part summary for the arseven benchmarks (C1 runtime, C2 Xenium CRC pseudo-Visium HD,
Xenium breast) of the final-package rerun.

Writes results/rerun_final/benchmarks/part_summary_arseven.csv
(benchmark,metric,manuscript_value,v020_value,final_value,rank,p_value,notes) plus
c2/c2_final_summary.csv and c2/c2_reserve_tacco.csv.
manuscript/v020 values for C2 and breast: consolidated_v020_benchmarks.csv
(manuscript, v020_default). C2 metrics file: c2/c2_standard_metrics_final_allrctd.csv
if present (rescored after the C2 RCTD array finished), else
c2/c2_standard_metrics_final.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

P = Path("/Users/apple/Research/FlashDeconv/results")
F = P / "rerun_final" / "benchmarks"
CONS = pd.read_csv(P / "rerun_v020" / "benchmarks" / "consolidated_v020_benchmarks.csv").drop_duplicates()
METRICS = ["pearson_flat", "type_r", "rmse", "jsd", "ap_flat", "ap_type_mean"]
LOWER = {"rmse", "jsd"}
rows = []


def cons(bench, metric):
    r = CONS[(CONS.benchmark == bench) & (CONS.metric == metric)]
    if not len(r):
        return np.nan, np.nan
    return r.manuscript.iloc[0], r.v020_default.iloc[0]


def rank_of(vals, key, metric):
    s = pd.Series(vals).sort_values(ascending=metric in LOWER)
    return f"{list(s.index).index(key) + 1}/{len(s)}"


# ---------------------------------------------------------------- C2
c2f = F / "c2" / "c2_standard_metrics_final_allrctd.csv"
rctd_state = "all 20 RCTD configs (rescored after array 2693139 finished)"
if not c2f.exists():
    c2f = F / "c2" / "c2_standard_metrics_final.csv"
    rctd_state = "RCTD configs present at rescoring time only (array 2693139 still running)"
c2 = pd.read_csv(c2f)
c2["cfg"] = np.where(c2.method == "RCTD", "RCTD_" + c2["mode"] + "_umi" + c2.umi_min.fillna(-1).astype(int).astype(str),
                     c2.source + ":" + c2.method + ":" + c2["mode"])
FD, FD0 = "final:FlashDeconv:final_default_auto", "final:FlashDeconv:final_default_l0"
c2.to_csv(F / "c2" / "c2_final_summary_long.csv", index=False)
c2[c2.reserve].to_csv(F / "c2" / "c2_reserve_tacco.csv", index=False)
for res in sorted(c2.resolution_um.unique()):
    a = c2[(c2.resolution_um == res) & (c2.eval_set == "all")].set_index("cfg")
    bench = f"C2 Xenium CRC {res} um (self-ref)"
    for m in METRICS:
        full = {k: a.loc[k, m] for k in [FD, "c2:NNLS:default", "c2:MarkerScoring:default"] if k in a.index}
        notes = [f"lambda0 {a.loc[FD0, m]:.3f}",
                 f"NNLS {a.loc['c2:NNLS:default', m]:.3f}",
                 f"marker {a.loc['c2:MarkerScoring:default', m]:.3f}"]
        if "c2:TACCO:default" in a.index:
            notes.append(f"[reserve] TACCO {a.loc['c2:TACCO:default', m]:.3f}")
        for es in sorted(e for e in c2.eval_set.unique() if e.startswith("common_")):
            b = c2[(c2.resolution_um == res) & (c2.eval_set == es)].set_index("cfg")
            rc = "RCTD_" + es[len("common_"):]
            if rc in b.index and FD in b.index:
                notes.append(f"{rc.replace('_', ' ')} {b.loc[rc, m]:.3f} vs FD final "
                             f"{b.loc[FD, m]:.3f} on its {int(b.loc[FD, 'n_bins'])} bins")
        man, v020 = cons(bench, m)
        rows.append(dict(benchmark=bench, metric=m, manuscript_value=man, v020_value=v020,
                         final_value=round(float(a.loc[FD, m]), 4),
                         rank=rank_of(full, FD, m) + " (FD, NNLS, marker; all bins)",
                         p_value="", notes="; ".join(notes) + f"; {rctd_state}"))

# ---------------------------------------------------------------- Xenium breast
xb = pd.read_csv(F / "xenium_breast" / "xenium_breast_standard_metrics.csv")
xb = xb[xb.run == "final_default"].set_index(["bin_um", "method"])
for b in [16, 32, 64]:
    bench = f"Xenium breast {b} um"
    for m in METRICS:
        fd, ms, l0 = xb.loc[(b, "fd_auto"), m], xb.loc[(b, "marker_scoring"), m], xb.loc[(b, "fd_l0"), m]
        man, v020 = cons(bench, m)
        rows.append(dict(benchmark=bench, metric=m, manuscript_value=man, v020_value=v020,
                         final_value=round(float(fd), 4),
                         rank=rank_of({"fd": fd, "ms": ms}, "fd", m) + " (FD vs marker scoring)",
                         p_value="",
                         notes=f"lambda0 {l0:.3f}; marker scoring {ms:.3f}; old-metric r "
                               f"(joint zeros dropped) final {xb.loc[(b, 'fd_auto'), 'r_legacy']:.3f}"))

# ---------------------------------------------------------------- C1 runtime
tab = pd.read_csv(F / "c1" / "c1_runtime_table_final.csv")
comp = tab[~tab.method.str.startswith("flashdeconv") & (tab.status == "OK")]
for s in [10000, 100000, 300000, 1000000]:
    fin = tab[(tab.method == "flashdeconv_final") & (tab["mode"] == "default") & (tab.scale == s)].iloc[0]
    cold = tab[(tab.method == "flashdeconv_final_coldcache") & (tab["mode"] == "default") & (tab.scale == s)].iloc[0]
    v = tab[(tab.method == "flashdeconv_v020") & (tab.scale == s)].iloc[0]
    old = tab[(tab.method == "flashdeconv_v016") & (tab.scale == s)].iloc[0]
    others = comp[comp.scale == s]
    for metric, col in [("median_fit_seconds", "median_fit_s"), ("peak_rss_gb", "peak_rss_gb")]:
        vals = {f"{r.method}_{r['mode']}": r[col] for _, r in others.iterrows()}
        vals["fd"] = cold[col]
        rk = pd.Series(vals).rank(method="min")["fd"]
        rows.append(dict(
            benchmark=f"C1 runtime {s:.0e} bins (32 CPUs)".replace("+0", ""), metric=metric,
            manuscript_value="", v020_value=round(float(v[col]), 3),
            final_value=round(float(cold[col]), 3),
            rank=f"{int(rk)}/{len(vals)} (completed methods, lower better)", p_value="",
            notes=(f"C1 table v0.1.6 row {old[col]:.3g}; n_reps={int(fin.n_reps)}; iterations {fin.n_iterations}; converged {fin.converged}; "
                   f"final_value = fresh numba cache per task (as in the one-job-per-task C1/v020 protocol); "
                   f"shared warm cache within the job {fin[col]:.3g}; sequential single job on {fin.hosts}; "
                   "competitors: " + ", ".join(f"{k} {x:.4g}" for k, x in vals.items() if k != "fd"))))

out = pd.DataFrame(rows)
out.to_csv(F / "part_summary_arseven.csv", index=False)
print(out.drop(columns="notes").to_string(index=False))
