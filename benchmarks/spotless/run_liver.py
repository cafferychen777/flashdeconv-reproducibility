"""Liver case study (JSD, portal/central EC AUPR) and reference-protocol stability
for the final FlashDeconv package
(copy of the v0.2.0 rerun script; configurations and fit logging changed).

Evaluation code is imported from the scripts that produced the manuscript values:
validation/benchmark_liver.py (JSD vs snRNA-seq frequencies, AUPR on Portal/Central
spots) and validation/benchmark_liver_stability.py (mean per-spot squared JSD between
predictions with the exVivo / inVivo / nuclei references). Per-cell reference MTX files
for liver_ref_9ct and inVivo are no longer on disk, so their mean-count signatures are
read from liver_signature.npz (precompute_signature.py, same mean-per-type definition)
and from the inVivo Seurat object (export_invivo_signature.R, annot_cd45 labels).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation")
import benchmark_liver as bl  # noqa: E402
import benchmark_liver_stability as bs  # noqa: E402
sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/rerun_final")
import fdfinal  # noqa: F401,E402  (fit log)
from flashdeconv import FlashDeconv  # noqa: E402

D = bl.DATA_DIR
OUT = "/Users/apple/Research/FlashDeconv/results/rerun_final/benchmarks/spotless"
INP = "/Users/apple/Research/FlashDeconv/results/rerun_v020/benchmarks/_inputs"

CONFIGS = {
    "final_default": dict(),                        # pure package defaults
    "final_default_lam0": dict(lambda_spatial=0.0),
    "final_default_seed42": dict(random_state=42),  # determinism check (default seed 0)
}
# Stability script's call = package defaults (log_cpm, n_hvg 2000, d 512, auto lambda).
STAB = {"final_default": dict()}


def liver_signature():
    z = np.load(f"{D}/liver_signature.npz", allow_pickle=True)
    types = [str(t) for t in z["celltypes"]]
    order = sorted(types)
    X = z["signature"][[types.index(t) for t in order]]
    return X, order, [str(g) for g in z["genes"]]


def case_study():
    X_all, unique_cts, rgenes = liver_signature()
    rows, det = [], []
    for sample in ["liver_mouseVisium_JB0%d" % i for i in range(1, 5)]:
        sp = bl.load_visium_sample(sample)
        zon = sp["metadata"]["zonationGroup"].values
        common = sorted(set(rgenes) & set(sp["genes"]))
        ri = {g: i for i, g in enumerate(rgenes)}
        ti = {g: i for i, g in enumerate(sp["genes"])}
        X = X_all[:, [ri[g] for g in common]]
        Y = sp["counts"][:, [ti[g] for g in common]]
        preds = {}
        for name, kw in CONFIGS.items():
            os.environ["FD_CONTEXT"] = f"liver_case|{sample}|{name}"
            m = FlashDeconv(**kw)
            pred = m.fit_transform(Y, X, sp["coords"])
            preds[name] = pred
            gt = np.array([bl.SNRNASEQ_PROPORTIONS.get(ct, 0) for ct in unique_cts])
            jsd = bl.calculate_jsd(gt, pred.mean(0))
            mask = (zon == "Portal") | (zon == "Central")
            pi = unique_cts.index("Portal Vein Endothelial cells")
            ci = unique_cts.index("Central Vein Endothelial cells")
            ap = bl.calculate_aupr((zon[mask] == "Portal").astype(int), pred[mask, pi])
            ac = bl.calculate_aupr((zon[mask] == "Central").astype(int), pred[mask, ci])
            rows.append(dict(sample=sample, config=name, jsd=jsd, aupr_portal=ap, aupr_central=ac,
                             aupr_mean=(ap + ac) / 2, lambda_used=m.lambda_used_,
                             **{f"prop_{c}": v for c, v in zip(unique_cts, pred.mean(0))}))
            print(sample, name, round(jsd, 4), round((ap + ac) / 2, 4), flush=True)
        det.append(dict(sample=sample, max_abs_diff=float(np.abs(preds["final_default"] - preds["final_default_seed42"]).max())))
    pd.DataFrame(rows).to_csv(f"{OUT}/liver_case_study.csv", index=False)
    pd.DataFrame(det).to_csv(f"{OUT}/determinism_liver.csv", index=False)


def mtx_signature(ref):
    counts = mmread(f"{D}/liver_ref_{ref}_counts.mtx").tocsr()  # genes x cells
    genes = [l.strip() for l in open(f"{D}/liver_ref_{ref}_genes.txt")]
    cts = np.array([l.strip() for l in open(f"{D}/liver_ref_{ref}_celltypes.txt")])
    X = np.vstack([np.asarray(counts[:, cts == t].mean(axis=1)).ravel() for t in bs.CELL_TYPES])
    return X, genes


def invivo_signature():
    s = pd.read_csv(f"{INP}/liver_inVivo_signature.csv.gz", index_col=0)
    return s[bs.CELL_TYPES].to_numpy().T, list(s.index)


def stability():
    refs = {"exVivo": mtx_signature("exVivo"), "inVivo": invivo_signature(),
            "nuclei": mtx_signature("nuclei")}
    out = []
    for name, kw in STAB.items():
        props = {r: {} for r in refs}
        for slide in bs.SLIDES:
            Y, sg, coords = bs.load_visium_slide(slide)
            for r, (X, rg) in refs.items():
                common = sorted(set(rg) & set(sg))
                ri = {g: i for i, g in enumerate(rg)}
                si = {g: i for i, g in enumerate(sg)}
                os.environ["FD_CONTEXT"] = f"liver_stability|{slide}|{r}|{name}"
                m = FlashDeconv(preprocess="log_cpm", n_hvg=2000, sketch_dim=512,
                                lambda_spatial="auto", **kw)
                props[r][slide] = m.fit_transform(Y[:, [si[g] for g in common]],
                                                  X[:, [ri[g] for g in common]], coords)
        for slide in bs.SLIDES:
            for i, r1 in enumerate(refs):
                for r2 in list(refs)[i + 1:]:
                    p1, p2 = props[r1][slide], props[r2][slide]
                    j = np.mean([jensenshannon(p1[k], p2[k]) ** 2 for k in range(p1.shape[0])])
                    out.append(dict(config=name, slide=slide, ref1=r1, ref2=r2, jsd=j))
        print(name, "stability JSD", np.mean([o["jsd"] for o in out if o["config"] == name]), flush=True)
    pd.DataFrame(out).to_csv(f"{OUT}/liver_stability.csv", index=False)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("case", "all"):
        case_study()
    if what in ("stability", "all"):
        stability()
