"""Dataset loaders and fits for the depth-calibration study (package defaults unless noted)."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path("/Users/apple/Research/FlashDeconv")
CONV = ROOT / "validation/benchmark_data/converted"
CACHE = ROOT / "results/reference_diagnostic/cache"
BINS = ROOT / "results/reference_diagnostic/c2_bins"
PATTERNS = [2, 4, 7, 8]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def fit(Y, X, coords, names, **kw):
    from flashdeconv import FlashDeconv
    G = Y.shape[1]
    if G < 2000:
        kw.setdefault("sketch_dim", min(512, G))
        kw.setdefault("n_hvg", min(2000, G))
    m = FlashDeconv(verbose=False, **kw)
    m.fit(Y, X, coords, cell_type_names=np.array(names))
    return m


def spotless(ds, pat):
    """Returns Yc, Xfull, coords, cell types, true proportions T (or None if missing)."""
    from scipy.io import mmread
    pre = CONV / f"silver_{ds}_{pat}"
    if not Path(f"{pre}_counts.mtx").exists():
        return None
    d = np.load(CACHE / f"sig_{ds}.npz", allow_pickle=True)
    Xref, rtypes, rgenes = d["X"], list(d["types"]), list(d["genes"])
    rmap = {norm(t): i for i, t in enumerate(rtypes)}
    Y = mmread(f"{pre}_counts.mtx").T.tocsr().astype(np.float64)
    genes = pd.read_csv(f"{pre}_genes.txt", header=None)[0].astype(str).values
    props = pd.read_csv(f"{pre}_proportions.csv", index_col=0).select_dtypes(include=[np.number])
    n = Y.shape[0]
    g = int(np.ceil(np.sqrt(n)))
    coords = np.array([[i % g, i // g] for i in range(n)], dtype=float)
    cts = [c for c in props.columns if norm(c) in rmap]
    idx = [rmap[norm(c)] for c in cts]
    rg = set(rgenes)
    common = [x for x in genes if x in rg]
    Yc = Y[:, pd.Index(genes).get_indexer(common)].tocsr()
    Xfull = Xref[np.ix_(idx, pd.Index(rgenes).get_indexer(common))]
    return Yc, Xfull, coords, cts, props[cts].to_numpy()


def c2(res):
    d = np.load(BINS / f"bins_{res}um.npz", allow_pickle=True)
    Y = sparse.csr_matrix((d["Y_data"], d["Y_indices"], d["Y_indptr"]),
                          shape=tuple(d["Y_shape"])).astype(np.float64)
    s = np.load(BINS / "signature.npz", allow_pickle=True)
    keep = np.asarray(Y.sum(1)).ravel() > 0
    return (Y[keep], s["X_sig"].astype(np.float64), d["centers"][keep],
            [str(x) for x in d["cell_types"]], d["gt_props"][keep])


INTESTINE_REFS = {
    "haber": ROOT / "validation/visium_hd_data/haber_intestine_matched.h5ad",
    "composite": ROOT / "validation/intestine_reference_v2/results/reference/composite_reference.h5ad",
}


def intestine(ref_name):
    """Visium HD mouse small intestine 8 um; returns Y, X, coords, types, genes, regions dict."""
    import scanpy as sc
    sys.path.insert(0, str(ROOT / "validation/tuft_investigation"))
    from common import load_raw
    from flashdeconv.io.loader import prepare_data
    ad = load_raw("008um")
    mask = pd.read_csv(ROOT / "validation/intestine_pg_robustness/results/mask_8um.csv.gz",
                       index_col=0).reindex(ad.obs_names)
    regions = {"follicle": mask.follicle_buf.to_numpy(bool), "muscle": mask.muscle.to_numpy(bool),
               "epi_low": mask.epi_low.to_numpy(bool),
               "mask_full": mask.mask_full.to_numpy(bool),
               "epithelium": ~mask.mask_full.to_numpy(bool)}
    ref = sc.read_h5ad(INTESTINE_REFS[ref_name])
    Y, X, crd, ct, genes = prepare_data(ad, ref, cell_type_key="celltype1")
    return sparse.csr_matrix(Y).astype(np.float64), X, crd, list(ct), list(genes), regions, \
        np.asarray(ad.obsm["spatial"])
