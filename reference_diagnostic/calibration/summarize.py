"""Summary tables of the re-validation (package function, null='auto' vs 'left_half')."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nullcal as nc  # noqa: E402
from eval_crc import PROGS, hotspots  # noqa: E402
from evaluate import arrays_from_npz  # noqa: E402

R = HERE.parents[1] / "results/reference_diagnostic_v2"
B2 = HERE.parents[1] / "results/b2_crc_refcheck"

# V3: complete references, flag rate per depth decile
sp = pd.read_csv(R / "pkg_spotless_v3_deciles.csv")
sp["n_flag"] = sp.flag_rate * sp.n_bins
sp = sp.groupby(["null", "decile"]).agg(n_bins=("n_bins", "sum"), n_flag=("n_flag", "sum"),
                                         median_depth=("median_depth", "median")).reset_index()
sp["fpr"] = sp.n_flag / sp.n_bins
sp["set"], sp["key"] = "spotless", "score"
c2 = pd.read_csv(R / "pkg_c2_v3_deciles.csv").rename(columns={"flag_rate": "fpr"})
v3 = pd.concat([sp[["set", "key", "null", "decile", "median_depth", "n_bins", "fpr"]],
                c2[["set", "key", "null", "decile", "median_depth", "n_bins", "fpr"]]])
v3.to_csv(R / "summary_v3_fpr_by_depth_decile.csv", index=False)

# V1: Spotless controlled removal (positives > 0.3)
v1 = pd.read_csv(R / "pkg_spotless_v1.csv")
v1s = v1.groupby("null").agg(n=("auroc", "size"), auroc_median=("auroc", "median"),
                             flag_pos_median=("flag_pos", "median"),
                             flag_neg_median=("flag_neg", "median")).reset_index()
v1s.to_csv(R / "summary_v1_spotless.csv", index=False)

# V2: intestine
pd.read_csv(R / "pkg_intestine_region_flags.csv").to_csv(R / "summary_v2_intestine.csv", index=False)

# CRC: flagged fractions, per decile, hotspots
rows, drows = [], []
for sid in ("P1_CRC", "P2_CRC", "P5_CRC"):
    d = np.load(R / f"pkg_crc_{sid}.npz", allow_pickle=True)
    info = json.loads((R / f"pkg_crc_{sid}_info.json").read_text())
    pb = np.load(B2 / f"stage1/{sid}/perbin.npz", allow_pickle=True)
    assert np.array_equal(pb["barcode"], d["barcode"])
    hs = hotspots(sid, np.c_[pb["x"], pb["y"]].astype(np.float64), pb["umi"])
    _, _, _, A = arrays_from_npz(R / f"diag_crc_{sid}.npz")
    n = d["n_umi"].astype(np.float64)
    for meth in ("auto", "left_half"):
        for key, depth in (("score", n), ("score_pooled", A @ n)):
            f = d[f"{meth}_{key}"].astype(np.float64) > nc.Z95
            r = {"sample": sid, "null": meth, "key": key, "flag_all": f.mean(),
                 "median_umi_selected": float(np.median(n)), "diag_s": info[f"diag_s_{meth}"],
                 "fit_s": info["fit_s"]}
            for prog, idx in hs.items():
                r[f"hot_{prog}_n"] = len(idx)
                r[f"hot_{prog}_flagged"] = f[idx].mean() if len(idx) else np.nan
            rows.append(r)
            for x in nc.summarize_by_depth(d[f"{meth}_{key}"].astype(np.float64), depth):
                x.update({"sample": sid, "null": meth, "key": key})
                drows.append(x)
pd.DataFrame(rows).rename(columns={f"hot_{p}_flagged": f"hot_{p}_flagged ({v})" for p, v in PROGS.items()}) \
    .to_csv(R / "summary_crc.csv", index=False)
pd.DataFrame(drows).to_csv(R / "summary_crc_by_depth_decile.csv", index=False)

# Diagnosis: observed vs simulated raw score by depth decile (EM 10)
diag = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "diag_*.csv")))])
diag = diag[(diag.em.isin([0, 10])) & diag.kind.isin(["obs_raw", "sim_raw", "obs_perumi", "sim_perumi"])]
diag[["set", "em", "pooled", "kind", "decile", "median_depth", "median", "q16", "q84", "q01", "q99",
      "left_scale"]].to_csv(R / "summary_diagnosis_raw_score_by_depth.csv", index=False)

pd.set_option("display.width", 250)
print(v3.pivot_table(index=["set", "key", "null"], columns="decile", values="fpr").round(3).to_string())
print(v1s.round(3).to_string())
print(pd.DataFrame(rows).round(4).to_string())
