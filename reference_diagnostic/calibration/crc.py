"""CRC Visium HD 8 um (P1/P2/P5) + matched Chromium Flex reference (Level2), as in B2 stage 1."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

PROJ = Path("/scratch/user/cafferychen777/FlashDeconv")
ST_DIR = PROJ / "analysis/crc_cohort_results"
REF_H5 = PROJ / "data/visium_hd_crc_cohort/scRNA_ref/HumanColonCancer_Flex_filtered_feature_bc_matrix.h5"
REF_META = PROJ / "data/visium_hd_crc_cohort/metadata/SingleCell_MetaData.csv.gz"


def load_reference():
    import scanpy as sc
    ref = sc.read_10x_h5(REF_H5)
    meta = pd.read_csv(REF_META, compression="gzip").set_index("Barcode")
    common = ref.obs_names.intersection(meta.index)
    ref = ref[common].copy()
    ref = ref[(meta.loc[ref.obs_names, "QCFilter"] == "Keep").to_numpy()].copy()
    ref.obs["Level2"] = meta.loc[ref.obs_names, "Level2"].values
    return ref


def fit_sample(sid):
    import scanpy as sc
    import flashdeconv
    from flashdeconv import FlashDeconv
    from flashdeconv.io import prepare_data
    st = sc.read_h5ad(ST_DIR / f"{sid}_deconv.h5ad")
    st.obs = st.obs[[]]
    st.obsm.pop("flashdeconv", None)
    st.uns.pop("flashdeconv_params", None)
    Xraw = st.X.tocsr() if sparse.issparse(st.X) else sparse.csr_matrix(st.X)
    umi_total = np.asarray(Xraw.sum(1)).ravel()
    ref = load_reference()
    Y, Xsig, crd, ctn, genes = prepare_data(st, ref, cell_type_key="Level2")
    del ref
    t0 = time.perf_counter()
    m = FlashDeconv()
    m.fit(Y, Xsig, crd, cell_type_names=np.asarray([str(t) for t in ctn]))
    t_fit = time.perf_counter() - t0
    info = {"sample": sid, "n_bins": int(Y.shape[0]), "fit_s": t_fit,
            "n_iterations": m.info_.get("n_iterations"), "version": flashdeconv.__version__,
            "pkg": flashdeconv.__file__}
    extra = {"barcode": np.asarray(st.obs_names).astype(str), "umi_total": umi_total,
             "x": crd[:, 0].astype(np.float32), "y": crd[:, 1].astype(np.float32),
             "proportions": m.proportions_.astype(np.float32),
             "types": np.asarray([str(t) for t in ctn])}
    return m, info, extra
