"""Sanity check: our metric re-implementation must reproduce the published
Li et al. 2023 numbers for the 16 method outputs that the authors released
(seqFISH+, 10,000 genes, 71 spots). Reference values: Source Data, Figure 2.
Optionally also runs FlashDeconv (package defaults) on the same input.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from li2023_metrics import li_metrics

ROOT = Path("/Users/apple/Research/FlashDeconv/data/external_benchmarks/li2023_seqfish/raw")
PRED = ROOT / "published_predictions"
SRC = Path("/Users/apple/Research/FlashDeconv/data/external_benchmarks/li2023_merfish/raw/li_MOESM6_ESM.xlsx")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def published_table() -> pd.DataFrame:
    x = pd.read_excel(SRC, sheet_name="Figure 2", header=None)
    t = x.iloc[6:24, [0, 1, 2]].dropna(subset=[0])
    t.columns = ["method", "JSD_pub", "total_RMSE_pub"]
    return t.set_index("method").astype(float)


def load_gt() -> pd.DataFrame:
    gt = pd.read_csv(ROOT / "seqFISH/Out_cell_ratio_1x.csv", index_col=0)
    return gt.dropna()


def run_flashdeconv(gt: pd.DataFrame) -> pd.DataFrame:
    import flashdeconv as fd

    st = pd.read_csv(ROOT / "seqFISH/Out_gene_expressions_10000genes.csv", index_col=0)
    loc = pd.read_csv(ROOT / "seqFISH/Out_rect_locations.csv", index_col=0)
    if st.shape[0] != loc.shape[0]:
        st = st.T  # genes x spots on disk
    st.index = st.index.astype(str)
    loc.index = loc.index.astype(str)
    sc = pd.read_csv(ROOT / "seqFISH/raw_somatosensory_sc_exp.txt", sep="\t", index_col=0)
    labels = pd.read_csv(ROOT / "seqFISH/somatosensory_sc_labels.txt", header=None)[0].to_numpy()
    a_st = ad.AnnData(st.to_numpy(float), obs=pd.DataFrame(index=st.index),
                      var=pd.DataFrame(index=st.columns))
    a_st.obsm["spatial"] = loc.loc[st.index].iloc[:, :2].to_numpy(float)
    a_ref = ad.AnnData(sc.T.to_numpy(float), obs=pd.DataFrame({"cell_type": labels}, index=sc.columns),
                       var=pd.DataFrame(index=sc.index))
    fd.tl.deconvolve(a_st, a_ref, cell_type_key="cell_type")
    return a_st.obsm["flashdeconv"].copy()


def main():
    gt = load_gt()
    pub = published_table()
    rows = {}
    for f in sorted(PRED.glob("*.csv")):
        pred = pd.read_csv(f, index_col=0)
        rows[f.stem] = li_metrics(pred, gt, by_position=True)
    ours = pd.DataFrame(rows).T[["JSD", "total_RMSE", "PCC"]]
    cmp = ours.join(pub, how="left")
    cmp["dJSD"] = cmp["JSD"] - cmp["JSD_pub"]
    cmp["dRMSE"] = cmp["total_RMSE"] - cmp["total_RMSE_pub"]
    print(cmp.round(4).to_string())
    print("max |dJSD| =", np.nanmax(np.abs(cmp["dJSD"])),
          " max |dRMSE| =", np.nanmax(np.abs(cmp["dRMSE"])))
    cmp.to_csv(OUT / "seqfish10000_metric_check.csv")

    try:
        fdp = run_flashdeconv(gt)
        m = li_metrics(fdp, gt)
        print("FlashDeconv (defaults) seqFISH+ 10000:", {k: round(v, 4) for k, v in m.items()})
        fdp.to_csv(OUT / "seqfish10000_flashdeconv_pred.csv")
    except Exception as e:  # keep the metric check usable even if the run fails
        print("FlashDeconv run failed:", repr(e))


if __name__ == "__main__":
    main()
