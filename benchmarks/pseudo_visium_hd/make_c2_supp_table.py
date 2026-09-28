"""Build the Supplementary pseudo-Visium HD (C2) table from upstream metric files.

Xenium CRC (patient P1, 38 cell types) aggregated into square bins of
2/4/8/16/32 um. No model is refit here: the table is a selection/relabelling of
rows from results/rerun_final/benchmarks/c2/c2_final_summary_long.csv, which is
written by validation/rerun_final/benchmarks/build_part_summary_arseven.py from
the rescored metrics (rescore_standard.py -> c2_standard_metrics_final*.csv).

This reproduces build_supp_table() in validation/figures/supp/supp_pseudo_vhd.py
(same selection logic), without drawing the figure.

Rows kept
---------
* FlashDeconv            : mode final_default_auto (package defaults, automatic lambda)
* FlashDeconv (lambda=0) : mode final_default_l0   (same run without spatial smoothing)
* RCTD (doublet / full)  : official spacexr, UMI_min = 100 (package default)
* NNLS, Marker scoring   : default
TACCO (reserve run) is not included.

Eval sets: all predicted bins, and bins shared with RCTD doublet / full at
UMI_min = 100. coverage = fraction of all bins at that size scored by the method;
n_bins_total = number of bins scored by FlashDeconv on the "all" set.

Usage:
    python make_c2_supp_table.py [--out PATH]
Default output is the canonical c2_supp_table.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Original project location (override with --src / --out).
RESULTS = Path("/Users/apple/Research/FlashDeconv/results")
C2DIR = RESULTS / "rerun_final" / "benchmarks" / "c2"
SRC = C2DIR / "c2_final_summary_long.csv"
OUT = C2DIR / "c2_supp_table.csv"

SIZES = [2, 4, 8, 16, 32]

# Display name -> row selector on the long summary table.
METHODS = {
    "FlashDeconv": lambda d: (d.method == "FlashDeconv") & (d["mode"] == "final_default_auto"),
    "FlashDeconv (λ = 0)": lambda d: (d.method == "FlashDeconv") & (d["mode"] == "final_default_l0"),
    "RCTD (doublet)": lambda d: (d.method == "RCTD") & (d["mode"] == "doublet") & (d.umi_min == 100),
    "RCTD (full)": lambda d: (d.method == "RCTD") & (d["mode"] == "full") & (d.umi_min == 100),
    "NNLS": lambda d: d.method == "NNLS",
    "Marker scoring": lambda d: d.method == "MarkerScoring",
}
EVAL_SETS = {
    "all": "all predicted bins",
    "common_doublet_umi100": "common bins with RCTD doublet",
    "common_full_umi100": "common bins with RCTD full",
}


def build_supp_table(src: Path = SRC) -> pd.DataFrame:
    d = pd.read_csv(src)
    # Total bins per size = bins scored by FlashDeconv (full coverage) on "all".
    n_total = (d[(d.method == "FlashDeconv") & (d["mode"] == "final_default_auto")
                 & (d.eval_set == "all")].set_index("resolution_um").n_bins)
    rows = []
    for size in SIZES:
        for name, sel in METHODS.items():
            for es, es_label in EVAL_SETS.items():
                t = d[sel(d) & (d.resolution_um == size) & (d.eval_set == es)]
                if t.empty:
                    continue  # RCTD doublet has no row on RCTD-full common bins
                r = t.iloc[0]
                rows.append({
                    "bin_size_um": size, "method": name, "eval_set": es_label,
                    "n_bins": int(r.n_bins),
                    "coverage": r.coverage,
                    "pearson_flat": r.pearson_flat, "pearson_type_mean": r.type_r,
                    "rmse": r.rmse, "jsd": r.jsd, "ap_flat": r.ap_flat,
                    "ap_type_mean": r.ap_type_mean,
                })
    tab = pd.DataFrame(rows)
    tab.insert(1, "n_bins_total", tab.bin_size_um.map(n_total).astype(int))
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    tab = build_supp_table(a.src)
    tab.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(tab)} rows)")


if __name__ == "__main__":
    main()
