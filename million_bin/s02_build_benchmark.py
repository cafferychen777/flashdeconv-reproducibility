"""S1 step 2: build the million-bin MERFISH-derived deconvolution benchmark with ground truth.

Input : merfish_all_rawcounts.h5ad (s01; 2,060,051 cells x 1,815 genes, raw counts + metadata)
        vhd_si_8um_total_umi.npy (per-bin total UMI of the 10x Visium HD Mouse Small Intestine
        8 um filtered bins; 351,817 bins, median 325)
Output (in OUT):
  st/st_<N>.h5ad      nested spatially contiguous bin sets (CSR counts, obsm['spatial'] in um)
  gt_bins.parquet     per-bin metadata for every bin in the pool (order = nesting order)
  gt_cellfrac.npy     per-bin cell-count fractions  (n_pool x n_types, float32)
  gt_txfrac.npy       per-bin transcript-weighted fractions (raw MERFISH counts of member cells)
  types.txt           harmonised coarse cell types (column order of the GT matrices)
  selfref/ref.h5ad, ref_meta.csv   self-reference from held-out MERFISH slices
  bin_scaling.csv, depth_summary.csv, build_summary.json

Construction
  * Harmonised 18-type coarse taxonomy (COARSE below) applied to the authors' 78 cell types.
  * Held-out slices (one SPF slice per gut region, the one with the median cell count) supply the
    self-reference; all other slices are binned.
  * Cells are assigned to 8 um square bins by centroid (floor((xy - slice_min) / 8)); bin counts
    are the sums of member-cell raw counts. GT = fraction of member cells per type (cell count)
    and fraction of member-cell transcripts per type (transcript-weighted, before thinning).
  * Depth: per-bin target = rank-matched quantile of the Visium HD 8 um total-UMI distribution,
    capped at the bin's MERFISH total (thinning can only remove counts); counts are binomially
    thinned with p = target / total (seed 0).
  * Pool order (defines nesting): SPF ileum, SPF cecum, SPF proximal colon, SPF distal colon,
    GF ileum, GF cecum, GF proximal colon, GF distal colon; within region by dataset/slice; within
    slice raster order (y, then x). Scale N = first N bins of the pool, so 1e4 c 1e5 c ... and
    each set is spatially contiguous. Slices are laid out side by side (1 mm gaps).
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

BIN = 8.0
SCALES = [10_000, 100_000, 1_000_000]
REF_CAP = 2000

COARSE = {
    "Stem/TA": ["Stem"],
    "Enterocyte": ["Enterocyte (bottom)", "Enterocyte (mid)", "Enterocyte (top)",
                   "Enterocyte (Scnn1g+)", "M"],
    "Goblet": ["Goblet (bottom)", "Goblet (top)"],
    "Paneth": ["Paneth"],
    "Tuft": ["Tuft"],
    "EEC": ["L", "N", "EC (top)", "EC (bottom)", "Delta", "Progenitor (Arx+)",
            "Progenitor (Dll1+)", "Progenitor (Lmx1a+)"],
    "B cell": ["B (cycling)", "B (follicular)"],
    "Plasma cell": ["B (plasma)"],
    "T/ILC/NK": ["T (Cd4+)", "T (Cd4+, cycling)", "T (Cd8+)", "T (Cd8+, cycling)", "T (memory)",
                 "Treg", "ILC2", "ILC3", "NK"],
    "Myeloid": ["Macrophage (Itgax+)", "Macrophage (Lyve1+)", "Macrophage (Mrc1+)",
                "Macrophage (cycling)", "Monocyte", "cDC1", "cDC2", "cDC (Ccl22+)",
                "cDC (Il22ra2+)", "Mast"],
    "Fibroblast": ["FB1a (lamina propria, top)", "FB1b (lamina propria, bottom)",
                   "FB2a (submucosa, Grem1+)", "FB2b (submucosa, Emilin2+)",
                   "FB3 (muscularis externa)", "Telocyte (bottom)", "Telocyte (mid)",
                   "Telocyte (top)", "FRC", "FDC"],
    "Smooth muscle": ["SMC (Npy2r+)", "SMC (Tgfb3_hi)", "SMC (circular)", "SMC (lamina propria)",
                      "SMC (longitudinal)", "SMC (muscularis mucosae)"],
    "Pericyte": ["Pericyte"],
    "Endothelial": ["Endothelial (arterial)", "Endothelial (capillary)", "Endothelial (lymphatic)",
                    "Endothelial (venous)"],
    "Enteric neuron": ["Neuron1 (excitatory motor)", "Neuron2a (inhibitory motor, a)",
                       "Neuron2b (inhibitory motor, b)", "Neuron3a (interneuron, a)",
                       "Neuron3b (interneuron, b)", "Neuron4a (sensory, a)", "Neuron4b (sensory, b)",
                       "Neuron4c (sensory, c)", "Neuron5a (secretomotor, a)",
                       "Neuron5b (secretomotor, b)"],
    "Enteric glia": ["Glia1 (Slc18a2+)", "Glia2 (Gfra3+)"],
    "ICC": ["ICC (deep muscular plexus)", "ICC (intramuscular)", "ICC (myenteric plexus)",
            "ICC (submucosa)"],
    "Mesothelium": ["Mesothelium"],
}
TYPES = list(COARSE)
POOL_ORDER = [("WT", "ile"), ("WT", "ce"), ("WT", "pcol"), ("WT", "dcol"),
              ("GF", "ile"), ("GF", "ce"), ("GF", "pcol"), ("GF", "dcol")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--vhd-umi", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "st").mkdir(parents=True, exist_ok=True)
    (out / "selfref").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    m = ad.read_h5ad(a.counts)
    X = m.X.tocsr().astype(np.int32)
    o = m.obs
    fine2coarse = {f: c for c, fs in COARSE.items() for f in fs}
    missing = set(o.cell_type.astype(str)) - set(fine2coarse)
    assert not missing, missing
    ct = o.cell_type.astype(str).map(fine2coarse).to_numpy()
    ct_idx = pd.Categorical(ct, categories=TYPES).codes.astype(np.int64)
    xy = o[["x [μm]", "y [μm]"]].to_numpy()
    sl = o.slice_full_name.astype(str).to_numpy()
    mic = o.microbiome.astype(str).to_numpy()
    reg = o.region.astype(str).to_numpy()
    ds = o.dataset_name.astype(str).to_numpy()
    tx = np.asarray(X.sum(1)).ravel()
    print("cells", X.shape, "median transcripts/cell", np.median(tx), flush=True)

    # --- held-out slices for the self-reference (one SPF slice per region, median cell count)
    sl_tab = (pd.DataFrame({"slice": sl, "mic": mic, "reg": reg, "ds": ds})
              .groupby("slice").agg(mic=("mic", "first"), reg=("reg", "first"), ds=("ds", "first"),
                                    n=("mic", "size")).reset_index())
    held = []
    for r in ["ile", "ce", "pcol", "dcol"]:
        s = sl_tab[(sl_tab.mic == "WT") & (sl_tab.reg == r)].sort_values(["n", "slice"])
        held.append(s.iloc[len(s) // 2]["slice"])
    print("held-out reference slices:", held, flush=True)

    # --- self-reference
    ref_mask = np.isin(sl, held)
    ridx = np.flatnonzero(ref_mask)
    keep = []
    for t in range(len(TYPES)):
        ii = ridx[ct_idx[ridx] == t]
        keep.extend(rng.choice(ii, REF_CAP, replace=False) if len(ii) > REF_CAP else ii)
    keep = np.sort(np.asarray(keep))
    ref = ad.AnnData(X=X[keep].astype(np.float32),
                     obs=pd.DataFrame({"cell_type": ct[keep], "patient": sl[keep], "region": reg[keep]},
                                      index=[f"c{i}" for i in keep]),
                     var=pd.DataFrame(index=m.var_names.astype(str)))
    ref.write_h5ad(out / "selfref" / "ref.h5ad")
    pd.DataFrame({"barcode": ref.obs_names, "cell_type": ref.obs.cell_type.to_numpy(),
                  "patient": ref.obs.patient.to_numpy()}).to_csv(out / "selfref" / "ref_meta.csv", index=False)
    comp = ref.obs.groupby(["cell_type"]).size()
    print("self-reference composition:\n" + comp.to_string(), flush=True)
    comp.rename("n_cells").to_csv(out / "selfref" / "ref_composition.csv")

    # --- bin scaling check (all non-held-out slices)
    rows = []
    for s in sl_tab.slice:
        if s in held:
            continue
        c = np.flatnonzero(sl == s)
        p = xy[c] - xy[c].min(0)
        row = {"slice": s, "n_cells": len(c)}
        for b in [4, 8, 16, 32]:
            k = np.floor(p / b).astype(np.int64)
            row[f"bins_{b}um"] = len(np.unique(k[:, 0] * 10**7 + k[:, 1]))
        rows.append(row)
    scal = pd.DataFrame(rows)
    scal.to_csv(out / "bin_scaling.csv", index=False)
    tot = scal.drop(columns="slice").sum()
    print("bin scaling (sum over binned slices):\n" + tot.to_string(), flush=True)

    # --- bins in pool order
    sl_tab["ord"] = [POOL_ORDER.index((mm, rr)) for mm, rr in zip(sl_tab.mic, sl_tab.reg)]
    order = sl_tab[~sl_tab.slice.isin(held)].sort_values(["ord", "ds", "slice"]).slice.tolist()
    cell_bin = np.full(X.shape[0], -1, dtype=np.int64)
    meta = []
    nb = 0
    x_off = 0.0
    for s in order:
        c = np.flatnonzero(sl == s)
        mn = xy[c].min(0)
        p = xy[c] - mn
        k = np.floor(p / BIN).astype(np.int64)
        key = k[:, 1] * 10**7 + k[:, 0]  # raster: y then x
        uk, inv = np.unique(key, return_inverse=True)
        cell_bin[c] = nb + inv
        bx = (uk % 10**7) * BIN + BIN / 2
        by = (uk // 10**7) * BIN + BIN / 2
        meta.append(pd.DataFrame({
            "slice": s, "microbiome": mic[c[0]], "region": reg[c[0]], "dataset": ds[c[0]],
            "x_local": bx, "y_local": by, "x": bx + x_off, "y": by}))
        nb += len(uk)
        x_off += p[:, 0].max() + 1000.0
    bins = pd.concat(meta, ignore_index=True)
    bins.index = [f"b{i:07d}" for i in range(nb)]
    print("pool bins:", nb, flush=True)

    inb = np.flatnonzero(cell_bin >= 0)
    B = sparse.csr_matrix((np.ones(len(inb), np.int32), (cell_bin[inb], inb)), shape=(nb, X.shape[0]))
    Y = (B @ X).tocsr()
    T = sparse.csr_matrix((np.ones(X.shape[0], np.float32), (np.arange(X.shape[0]), ct_idx)),
                          shape=(X.shape[0], len(TYPES)))
    ncell = (B @ T).toarray()
    ntx = (B @ T.multiply(tx[:, None]).tocsr()).toarray()
    bins["n_cells"] = ncell.sum(1).astype(int)
    bins["raw_umi"] = np.asarray(Y.sum(1)).ravel().astype(int)
    gt_c = (ncell / ncell.sum(1, keepdims=True)).astype(np.float32)
    gt_t = (ntx / np.maximum(ntx.sum(1, keepdims=True), 1)).astype(np.float32)
    zero_tx = ntx.sum(1) == 0
    gt_t[zero_tx] = gt_c[zero_tx]

    # --- depth: rank-matched Visium HD quantiles, capped at available counts; binomial thinning
    vhd = np.sort(np.load(a.vhd_umi).astype(np.float64))
    raw = bins.raw_umi.to_numpy().astype(np.float64)
    u = (pd.Series(raw + rng.uniform(0, 1e-3, nb)).rank().to_numpy() - 0.5) / nb
    target = np.quantile(vhd, u)
    target = np.minimum(target, raw)
    pthin = np.where(raw > 0, target / np.maximum(raw, 1), 1.0)
    rowp = np.repeat(pthin, np.diff(Y.indptr))
    Y.data = rng.binomial(Y.data, rowp).astype(np.int32)
    Y.eliminate_zeros()
    bins["umi"] = np.asarray(Y.sum(1)).ravel().astype(int)
    bins["thin_p"] = pthin.astype(np.float32)
    q = [5, 25, 50, 75, 95]
    dep = pd.DataFrame({
        "quantity": ["merfish_raw_bin_counts", "visium_hd_8um_total_umi", "benchmark_bin_umi"],
        **{f"q{p}": [np.percentile(raw, p), np.percentile(vhd, p), np.percentile(bins.umi, p)] for p in q},
        "mean": [raw.mean(), vhd.mean(), bins.umi.mean()]})
    dep.to_csv(out / "depth_summary.csv", index=False)
    print(dep.to_string(), "\nfraction of bins thinned:", float((pthin < 1).mean()), flush=True)

    # --- write GT and nested sets
    bins.to_parquet(out / "gt_bins.parquet")
    np.save(out / "gt_cellfrac.npy", gt_c)
    np.save(out / "gt_txfrac.npy", gt_t)
    (out / "types.txt").write_text("\n".join(TYPES) + "\n")
    n_ile = int(((bins.microbiome == "WT") & (bins.region == "ile")).sum())
    scales = sorted(set([s for s in SCALES if s <= nb] + [n_ile]))
    var = pd.DataFrame(index=m.var_names.astype(str))
    for n in scales:
        st = ad.AnnData(X=Y[:n].astype(np.float32), obs=bins.iloc[:n][["slice", "region", "microbiome"]].copy(),
                        var=var)
        st.obsm["spatial"] = bins.iloc[:n][["x", "y"]].to_numpy()
        st.write_h5ad(out / "st" / f"st_{n}.h5ad")
        print("wrote st", n, flush=True)

    gfrac = pd.Series(gt_c[:scales[-1]].mean(0), index=TYPES)
    summ = {
        "n_cells_total": int(X.shape[0]), "n_genes": int(X.shape[1]), "held_out_slices": held,
        "n_ref_cells": int(ref.n_obs), "n_pool_bins": int(nb), "n_ileum_spf_bins": n_ile,
        "scales": scales, "n_types": len(TYPES),
        "pool_composition_1e6_first": bins.iloc[:scales[-1]].groupby(["microbiome", "region"]).size()
        .rename("n").reset_index().to_dict("records"),
        "mean_cells_per_bin": float(bins.n_cells.mean()),
        "global_cellfrac_largest_scale": gfrac.round(5).to_dict(),
        "fraction_bins_thinned": float((pthin < 1).mean()),
    }
    json.dump(summ, open(out / "build_summary.json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
