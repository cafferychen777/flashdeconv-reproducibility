"""Hallmark (MSigDB 2020, via Enrichr library) over-representation of each set's top-20
unexplained genes: stage-1 regions vs null blocks and label-permutation sets (all patients).
Universe: HPA gene symbols (protein-coding; close to the Flex/Visium HD probe set)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

R = Path("/Users/apple/Research/FlashDeconv/results/b2_crc_refcheck")
HPA = R / "hpa_single_cell_type_ncpm.csv.gz"  # HPA v25 rna_single_cell_type.tsv, pivoted
universe = set(pd.read_csv(HPA, index_col=0, nrows=1).columns)
hm = {}
for line in open(R / "msigdb_hallmark_2020_enrichr.txt"):
    f = line.rstrip("\n").split("\t")
    hm[f[0]] = set(g for g in f[2:] if g) & universe
N = len(universe)
rows = []
for sid in ["P1_CRC", "P2_CRC", "P5_CRC"]:
    for fn in ["regions.csv", "null_regions.csv"]:
        d = pd.read_csv(R / "stage1" / sid / fn, keep_default_na=False)
        for _, r in d.iterrows():
            top = [g for g in r["top_genes"].split(";") if g in universe][:20]
            n = len(top)
            ps, ov = [], []
            for name, s in hm.items():
                k = len(set(top) & s)
                ps.append(hypergeom.sf(k - 1, N, len(s), n) if k else 1.0)
                ov.append(k)
            q = multipletests(ps, method="fdr_bh")[1]
            j = int(np.argmin(q))
            names = list(hm)
            rows.append({"sample": sid, "set": r["set"], "kind": r["set"].split("_")[0],
                         "n_bins": r["n_bins"], "best_hallmark": names[j], "overlap": ov[j],
                         "q": q[j], "hit_genes": ";".join(sorted(set(top) & hm[names[j]])),
                         "significant": bool(q[j] < 0.05 and ov[j] >= 3)})
df = pd.DataFrame(rows)
df.to_csv(R / "hallmark_enrichment.csv", index=False)
print(df[df.kind.isin(["region", "all"])].to_string())
print(df.groupby(["sample", "kind"]).significant.agg(["mean", "sum", "count"]))
print(df[(df.kind != "region") & df.significant][["sample", "set", "best_hallmark", "overlap", "q", "hit_genes"]].to_string())
