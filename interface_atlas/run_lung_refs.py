"""B1 follow-up (exploratory): lung sections with two references + reference diagnostic.

For LUNG_X1 / LUNG_X5K:
  - fit FlashDeconv (defaults) with the Lee et al. 2026 LUAD scRNA reference (author_cell_type_level_2) and save
    fd_<SEC>_luad.npz in the same format as run_fd.py;
  - refit with the Flex FFPE reference (as in run_fd.py) and run the reference diagnostic
    (reference_fit_scores + unexplained_genes on pooled flags); same diagnostic for the LUAD fit.
"""
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scanpy as sc
from scipy import sparse

import flashdeconv as fd
from flashdeconv import FlashDeconv, reference_fit_scores, unexplained_genes
from flashdeconv.io import prepare_data
from lineages import MARKERS_EPI, MARKERS_DIAG, LINEAGES, lung_lineage

S = Path("/scratch/user/cafferychen777")
OUT = S / "b1_pilot" / "results"
HD = {"LUNG_X1": "lung_postxenium/hd_post_xenium_v1_exp1", "LUNG_X5K": "lung_postxenium/hd_post_xenium_prime5k_exp2"}

_LUAD = {"Epithelial": "Epithelial", "Treg": "Treg", "Plasma": "Plasma", "Plasmablast": "Plasma", "Fibroblast": "Fibroblast",
         "Pericyte": "Pericyte/SMC", "Endothelial": "Endothelial", "NK": "NK", "Mast": "Mast", "Neutrophil": "Neutrophil",
         "pDC": "DC", "cDC2.FCER1A": "DC", "cDC.LAMP3": "DC", "cDC1.XCR1": "DC", "Mon.CD14": "Macrophage/Mono",
         "Mon.CD16": "Macrophage/Mono", "B.SELL": "B", "B.CD38": "B"}


def luad_lineage(n):
    if n in _LUAD:
        return _LUAD[n]
    if n.startswith("T.CD8"):
        return "CD8 T"
    if n.startswith("T.CD4"):
        return "CD4 T"
    if n.startswith("Mac."):
        return "Macrophage/Mono"
    return "Other"  # NKT, Tgd, ILC, Myeloid.Prolif, Lymphoid.Prolif


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_st(sec):
    d = S / "b1_pilot/data" / HD[sec] / "binned_outputs/square_008um"
    st = sc.read_10x_h5(d / "filtered_feature_bc_matrix.h5")
    st.var_names_make_unique()
    pos = pq.read_table(d / "spatial/tissue_positions.parquet").to_pandas().set_index("barcode")
    st.obsm["spatial"] = pos.loc[st.obs_names, ["array_col", "array_row"]].to_numpy(float) * 8.0
    st.X = sparse.csr_matrix(st.X, dtype=np.float32)
    st = st[np.asarray(st.X.sum(1)).ravel() > 0].copy()
    return st


def load_ref(kind):
    if kind == "luad":
        ref = ad.read_h5ad(S / "b1_pilot/data/lung_reference/luad_histological_subtypes_117k.h5ad")
        ref.var_names = ref.var["feature_name"].astype(str).values
        ref.obs["fine"] = ref.obs["author_cell_type_level_2"].astype(str)
        lin_of = luad_lineage
    else:
        ref = ad.read_h5ad(S / "b1_pilot/data/lung_reference/lung_cancer_ffpe_flex_4pt_36k.h5ad")
        ref.var_names = ref.var["feature_name"].astype(str).values
        ref.obs["fine"] = ref.obs["Harmonised_Level4"].astype(str)
        lin_of = lung_lineage
    ref.var_names_make_unique()
    ref.X = sparse.csr_matrix(ref.X, dtype=np.float32)
    return ref, lin_of


def fit(st, ref):
    Y, X, coords, names, genes = prepare_data(st, ref, cell_type_key="fine")
    m = FlashDeconv(random_state=0)
    t0 = time.perf_counter()
    P = m.fit_transform(Y, X, coords, cell_type_names=names)
    return m, np.asarray(P, dtype=np.float32), list(names), np.asarray(genes), time.perf_counter() - t0


def diagnose(m, genes, P, names, lin_of, tag):
    sc_ = reference_fit_scores(m)
    fl = sc_["flag_pooled"]
    ug = unexplained_genes(m, fl, gene_names=genes)
    top = pd.DataFrame({k: ug[k][:40] for k in ["gene", "score", "observed", "expected"]})
    lin = np.array([lin_of(n) for n in names])
    Lp = {l: float(P[fl][:, lin == l].sum(1).mean()) if fl.any() else np.nan for l in LINEAGES}
    La = {l: float(P[:, lin == l].sum(1).mean()) for l in LINEAGES}
    res = dict(tag=tag, frac_flag=float(sc_["flag"].mean()), frac_flag_pooled=float(fl.mean()),
               median_score=float(np.median(sc_["score"])), p95_score=float(np.percentile(sc_["score"], 95)),
               lineage_in_flagged=Lp, lineage_all=La, top_genes=top.gene.tolist()[:25])
    top.to_csv(OUT / f"refcheck_topgenes_{tag}.csv", index=False)
    np.savez_compressed(OUT / f"refcheck_scores_{tag}.npz", score=sc_["score"].astype(np.float32),
                        score_pooled=sc_["score_pooled"].astype(np.float32), flag_pooled=fl)
    log(f"refcheck {tag}: flagged {res['frac_flag']:.3f}, pooled {res['frac_flag_pooled']:.3f}; top genes {res['top_genes'][:20]}")
    log("   lineage mean in flagged vs all: " + json.dumps({l: (round(Lp[l], 3), round(La[l], 3)) for l in LINEAGES}))
    return res


def save(sec, tag, st, P, names, lin_of, t):
    lin = np.array([lin_of(n) for n in names])
    L = np.stack([P[:, lin == l].sum(1) for l in LINEAGES], 1)
    vn = {g: i for i, g in enumerate(st.var_names)}
    mg = list(dict.fromkeys(g for g in MARKERS_EPI + MARKERS_DIAG + ["IGKC", "IGHG1", "IGHA1", "JCHAIN", "MZB1", "XBP1"] if g in vn))
    M = st.X[:, [vn[g] for g in mg]].toarray().astype(np.float32)
    np.savez_compressed(OUT / f"fd_{sec}_{tag}.npz", coords_um=st.obsm["spatial"].astype(np.float32), fine=P.astype(np.float16),
                        fine_names=np.array(names), lineage=L.astype(np.float16), lineage_names=np.array(LINEAGES),
                        marker_counts=M, marker_names=np.array(mg), total_counts=np.asarray(st.X.sum(1)).ravel().astype(np.float32),
                        t_deconvolve_s=t, t_load_s=0.0, n_bins=st.n_obs, n_genes=st.n_vars, n_iterations=-1, converged=True,
                        lambda_spatial=np.nan, fd_version=fd.__version__)
    log(f"{sec} {tag}: fit {t:.1f}s; mean lineage " + json.dumps({l: round(float(v), 4) for l, v in zip(LINEAGES, L.mean(0))}))


def main(sec):
    st = load_st(sec)
    out = {}
    for kind in ["luad", "flex"]:
        ref, lin_of = load_ref(kind)
        log(f"{sec}: ST {st.shape}; ref {kind} {ref.shape}, {ref.obs['fine'].nunique()} types")
        m, P, names, genes, t = fit(st, ref)
        save(sec, kind, st, P, names, lin_of, t)
        out[kind] = diagnose(m, genes, P, names, lin_of, f"{sec}_{kind}")
        # IG share of UMIs in HD (plasma-overestimate check)
        vn = {g: i for i, g in enumerate(st.var_names)}
        ig = [g for g in st.var_names if g.startswith(("IGK", "IGL", "IGH")) or g in ("JCHAIN", "MZB1")]
        out[kind]["ig_umi_frac"] = float(st.X[:, [vn[g] for g in ig]].sum() / st.X.sum())
        del m
    json.dump(out, open(OUT / f"refcheck_summary_{sec}.json", "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
