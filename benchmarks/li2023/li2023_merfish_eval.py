"""FlashDeconv on the Li et al. 2023 (Nat Commun 14:1548) MERFISH multi-resolution
benchmark: Moffitt 2018 hypothalamic preoptic MERFISH (animal 1, 12 Bregma
sections, 135 genes, 6 cell classes) binned at 100 / 50 / 20 um. Reference is the
one shipped by the benchmark (Zeisel-type somatosensory-cortex scRNA-seq,
1,691 cells, 6 classes) -> an external, cross-region reference.

Inputs are the exact files used by the benchmark (Zenodo 10184476, MERFISH.zip,
exported to CSV by li2023_export_merfish.R). FlashDeconv runs with package
defaults only. Competitor numbers are taken from the paper's Source Data
(Supp. Fig. 10: per-section median JSD; Fig. 2 / Supp. Fig. 1: pooled 100 um).

Usage: python li2023_merfish_eval.py [--res 100 50 20]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from li2023_metrics import li_metrics, li_metrics_by_group

DATA = Path("/Users/apple/Research/FlashDeconv/data/external_benchmarks/li2023_merfish")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def load_reference() -> ad.AnnData:
    sc = pd.read_csv(DATA / "raw/raw_somatosensory_sc_exp.txt", sep="\t", index_col=0)
    labels = pd.read_csv(DATA / "raw/somatosensory_sc_labels.txt", header=None)[0].to_numpy()
    assert len(labels) == sc.shape[1]
    return ad.AnnData(
        sc.T.to_numpy(np.float32),
        obs=pd.DataFrame({"cell_type": pd.Categorical(labels)}, index=sc.columns),
        var=pd.DataFrame(index=sc.index),
    )


def load_st(res: int):
    counts = pd.read_csv(DATA / f"csv/merfish_{res}_counts.csv", index_col=0)
    gt = pd.read_csv(DATA / f"csv/merfish_{res}_gt.csv", index_col=0)
    meta = pd.read_csv(DATA / f"csv/merfish_{res}_meta.csv", index_col=0)
    a = ad.AnnData(counts.to_numpy(np.float32), obs=meta.loc[counts.index],
                   var=pd.DataFrame(index=counts.columns))
    a.obsm["spatial"] = meta.loc[counts.index, ["x", "y"]].to_numpy(float)
    return a, gt


def published_per_section() -> dict[int, pd.DataFrame]:
    """Supp. Fig. 10 Source Data: median JSD per method x 12 sections."""
    x = pd.read_excel(DATA / "raw/li_MOESM6_ESM.xlsx", sheet_name="Supp. Fig 10", header=None)
    out = {}
    for res, start in [(100, 3), (50, 25), (20, 47)]:
        t = x.iloc[start:start + 18, :13].set_index(0).astype(float)
        t.index.name = "method"
        out[res] = t
    return out


def published_supp_dataset1(res: int) -> pd.DataFrame:
    """Supplementary Dataset 1: per-cell-type RMSE and per-cell-type JSD for all
    18 methods, pooled over the 12 sections. total_RMSE is recovered exactly as
    sqrt(mean_k RMSE_k^2). Unlike the Fig. 2 / Supp. Fig. 9 source data (whose
    50 um column duplicates the 100 um one), these files are distinct per
    resolution."""
    d = DATA / "raw/JSD_RMSE_individual_celltype"
    rm = pd.read_csv(d / f"MERFISH_{res}_RMSE.csv", index_col=0)
    js = pd.read_csv(d / f"MERFISH_{res}_JSD.csv", index_col=0)
    out = pd.DataFrame({
        "total_RMSE": np.sqrt((rm ** 2).mean(0)),
        "mean_typeJSD": js.mean(0),
    })
    out.index.name = "method"
    return out


def published_pooled_100() -> pd.DataFrame:
    x = pd.read_excel(DATA / "raw/li_MOESM6_ESM.xlsx", sheet_name="Supp. Fig 1", header=None)
    t = x.iloc[6:24, [0, 4, 5]].set_index(0).astype(float)
    t.columns = ["JSD", "total_RMSE"]
    t.index.name = "method"
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs="+", default=[100])
    args = ap.parse_args()

    import flashdeconv as fd

    ref = load_reference()
    pub_sec = published_per_section()
    summary = []
    for res in args.res:
        st, gt = load_st(res)
        shared = st.var_names.intersection(ref.var_names)
        print(f"[{res} um] {st.n_obs} spots, {st.n_vars} genes, {len(shared)} shared with reference")
        t0 = time.time()
        fd.tl.deconvolve(st, ref, cell_type_key="cell_type")  # package defaults
        dt = time.time() - t0
        pred = st.obsm["flashdeconv"].copy()
        pred.index = st.obs_names
        pred.to_csv(OUT / f"merfish_{res}_flashdeconv_pred.csv")

        pooled = li_metrics(pred, gt)
        by_sec = li_metrics_by_group(pred, gt, st.obs["bregma"])
        by_sec.to_csv(OUT / f"merfish_{res}_flashdeconv_by_section.csv")
        print(f"  runtime {dt:.1f}s | pooled JSD {pooled['JSD']:.4f}  "
              f"total_RMSE {pooled['total_RMSE']:.4f}  PCC {pooled['PCC']:.4f}")

        # Per-section comparison (Supp. Fig. 10). CAUTION: the exact definition of
        # these per-section values could not be verified (for several methods the
        # section mean is far from the pooled median JSD), so treat as secondary.
        comp = pub_sec[res].mean(1).rename("mean_section_JSD").to_frame()
        comp.loc["FlashDeconv", "mean_section_JSD"] = by_sec["JSD"].mean()
        comp = comp.sort_values("mean_section_JSD")
        comp["rank"] = np.arange(1, len(comp) + 1)
        comp.to_csv(OUT / f"merfish_{res}_section_jsd_vs_published.csv")
        print(comp.round(4).to_string())

        s1 = published_supp_dataset1(res)
        s1.loc["FlashDeconv"] = [pooled["total_RMSE"], pooled["mean_typeJSD"]]
        s1["rank_RMSE"] = s1["total_RMSE"].rank().astype(int)
        s1["rank_typeJSD"] = s1["mean_typeJSD"].rank().astype(int)
        s1 = s1.sort_values("total_RMSE")
        s1.to_csv(OUT / f"merfish_{res}_suppdata1_vs_published.csv")
        print(s1.round(4).to_string())

        if res == 100:
            p = published_pooled_100()
            p.loc["FlashDeconv"] = [pooled["JSD"], pooled["total_RMSE"]]
            p = p.sort_values("JSD")
            p.to_csv(OUT / "merfish_100_pooled_vs_published.csv")
            print(p.round(4).to_string())
        summary.append({"res_um": res, "runtime_s": dt, **pooled})
    pd.DataFrame(summary).to_csv(OUT / "merfish_flashdeconv_summary.csv", index=False)


if __name__ == "__main__":
    main()
