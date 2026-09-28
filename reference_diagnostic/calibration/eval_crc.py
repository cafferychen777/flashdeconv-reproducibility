"""CRC P1/P2/P5 (Flex reference): flagged fractions per candidate, per depth decile, and in the B2
raw-count program hotspots (DBSCAN clusters >= 50 bins at the P2 source level; b2_stage2.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nullcal as nc  # noqa: E402
from evaluate import arrays_from_npz  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "results/reference_diagnostic_v2"
B2 = ROOT / "results/b2_crc_refcheck"
PROGS = {"region_0": "IFN-gamma (R0)", "region_1": "IFN-gamma (R1)", "region_2": "Hypoxia (R2)"}


def hotspots(sid, xy, umi):
    scan = pd.read_csv(B2 / "stage2/cross_patient_program_scan.csv")
    sm = np.load(B2 / "stage2/program_smoothed.npz")
    out = {}
    for prog in PROGS:
        thr = scan[(scan["sample"] == sid) & (scan.program == prog)].source_level.iloc[0]
        sf = sm[f"{sid}__{prog}"].astype(np.float64)
        hot = np.flatnonzero((sf >= thr) & (umi > 0))
        big = hot[:0]
        if len(hot) >= 5:
            lab = DBSCAN(eps=1.5 * 65.3, min_samples=5).fit_predict(xy[hot])
            sizes = pd.Series(lab[lab >= 0]).value_counts()
            big = hot[np.isin(lab, sizes[sizes >= 50].index)]
        out[prog] = big
    return out


def main():
    rows, drows = [], []
    for sid in ("P1_CRC", "P2_CRC", "P5_CRC"):
        o10, o0, s10, A = arrays_from_npz(OUT / f"diag_crc_{sid}.npz")
        meta = np.load(OUT / f"crc_{sid}_meta.npz", allow_pickle=True)
        pb = np.load(B2 / f"stage1/{sid}/perbin.npz", allow_pickle=True)
        assert np.array_equal(pb["barcode"], meta["barcode"])
        xy = np.c_[pb["x"], pb["y"]].astype(np.float64)
        hs = hotspots(sid, xy, pb["umi"])
        for pooled in (False, True):
            sc = nc.candidates(o10, o0, s10, A if pooled else None)
            depth = o10["n"] if not pooled else A @ o10["n"]
            for name, s in sc.items():
                f = s > nc.Z95
                r = {"sample": sid, "pooled": pooled, "cand": name, "flag_all": f.mean()}
                for prog, idx in hs.items():
                    r[f"n_{prog}"] = len(idx)
                    r[f"flag_{prog}"] = f[idx].mean() if len(idx) else np.nan
                rows.append(r)
                for d in nc.summarize_by_depth(s, depth):
                    d.update({"sample": sid, "pooled": pooled, "cand": name})
                    drows.append(d)
        print(sid, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "eval_crc.csv", index=False)
    pd.DataFrame(drows).to_csv(OUT / "eval_crc_deciles.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
