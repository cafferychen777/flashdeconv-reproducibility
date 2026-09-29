"""S1 step 6: extract a compact per-bin window (truth + FlashDeconv 10^6-bin predictions) for Fig. 3.

Usage: python s06_spatial_window.py DATA_DIR PROPS_CSV_GZ OUT_CSV_GZ
  DATA_DIR      S1 data dir (gt_bins.parquet, gt_cellfrac.npy, types.txt)
  PROPS_CSV_GZ  props/flashdeconv_default_selfref_1000000.csv.gz
Writes the window bins (local x/y in um, cell-fraction truth 'gt_<type>', FlashDeconv 'fd_<type>')
and prints dominant-compartment agreement for the window and for all 10^6 bins.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SLICE = "20211027_WT_ile_slice_2"      # SPF ileum; median-accuracy section, full crypt-villus axis
WIN_X = (350.0, 1850.0)                # um, local slice coordinates (1.5 mm)
WIN_Y = (1720.0, 2340.0)               # um (0.62 mm): lower wall, muscularis to villus tips
N = 1_000_000

COMP = {
    "Enterocyte": ["Enterocyte"], "Stem/TA": ["Stem/TA"],
    "Secretory": ["Goblet", "Paneth", "Tuft", "EEC"],
    "Immune": ["B cell", "Plasma cell", "T/ILC/NK", "Myeloid"],
    "Fibroblast": ["Fibroblast"], "Smooth muscle/ICC": ["Smooth muscle", "ICC"],
    "Vascular": ["Endothelial", "Pericyte"], "Neural": ["Enteric neuron", "Enteric glia"],
    "Mesothelium": ["Mesothelium"],
}


def main():
    data, props, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    types = [t for t in (data / "types.txt").read_text().splitlines() if t]
    bins = pd.read_parquet(data / "gt_bins.parquet").iloc[:N]
    gt = np.asarray(np.load(data / "gt_cellfrac.npy", mmap_mode="r")[:N])
    fd = pd.read_csv(props, index_col=0).reindex(columns=types)
    assert (fd.index == bins.index).all()
    fd = fd.to_numpy()

    M = np.zeros((len(types), len(COMP)))
    for j, ts in enumerate(COMP.values()):
        for t in ts:
            M[types.index(t), j] = 1
    agree_all = ((gt @ M).argmax(1) == (fd @ M).argmax(1)).mean()

    i = np.where(bins.slice.values == SLICE)[0]
    xl = bins.x_local.values[i] - bins.x_local.values[i].min()
    yl = bins.y_local.values[i] - bins.y_local.values[i].min()
    w = (xl >= WIN_X[0]) & (xl < WIN_X[1]) & (yl >= WIN_Y[0]) & (yl < WIN_Y[1])
    i, xl, yl = i[w], xl[w] - WIN_X[0], yl[w] - WIN_Y[0]
    df = pd.DataFrame({"bin": bins.index[i], "x": xl, "y": yl, "umi": bins.umi.values[i],
                       "n_cells": bins.n_cells.values[i]})
    for k, t in enumerate(types):
        df[f"gt_{t}"] = gt[i, k]
    for k, t in enumerate(types):
        df[f"fd_{t}"] = fd[i, k]
    df.to_csv(out, index=False, float_format="%.4g", compression="gzip")
    agree_win = ((gt[i] @ M).argmax(1) == (fd[i] @ M).argmax(1)).mean()
    print(f"slice {SLICE}; window {WIN_X} x {WIN_Y} um; {len(i)} bins")
    print(f"dominant-compartment agreement: window {agree_win:.4f}; all 1e6 bins {agree_all:.4f}")


if __name__ == "__main__":
    main()
