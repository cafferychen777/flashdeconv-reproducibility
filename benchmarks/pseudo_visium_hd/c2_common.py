"""Shared paths and I/O helpers for benchmark C2 (pseudo-Visium HD, official RCTD + TACCO).

All binning, downsampling, reference construction and metrics are imported from
validation/xenium_pseudo_visiumhd_benchmark.py so numbers stay comparable with the
existing supplementary table (tab:pseudo_vhd).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

PROJ = Path(os.environ.get("FLASHDECONV_PROJECT_ROOT",
                           Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "validation"))

OUT = Path(os.environ.get("C2_OUT", PROJ / "results" / "pseudo_vhd_c2"))
BIN_DIR = OUT / "bins"
PRED_DIR = OUT / "preds"
EVAL_DIR = OUT / "eval"
RESOLUTIONS = [2, 4, 8, 16, 32]


def load_bins(res):
    """Return dict with Y (csr, bins x genes), gt_props, centers, cell_types, genes."""
    d = np.load(BIN_DIR / f"bins_{res}um.npz", allow_pickle=True)
    Y = sparse.csr_matrix((d["Y_data"], d["Y_indices"], d["Y_indptr"]),
                          shape=tuple(d["Y_shape"]))
    return {
        "Y": Y,
        "gt_props": d["gt_props"],
        "centers": d["centers"],
        "cell_types": [str(x) for x in d["cell_types"]],
        "genes": [str(x) for x in d["genes"]],
        "stats": json.loads(str(d["stats"])),
    }


def load_signature():
    d = np.load(BIN_DIR / "signature.npz", allow_pickle=True)
    return d["X_sig"], [str(x) for x in d["cell_types"]], [str(x) for x in d["genes"]]


def load_reference_cells():
    """All annotated (non-Unassigned) Xenium cells: csr counts (cells x genes), labels."""
    d = np.load(BIN_DIR / "ref_cells_all.npz", allow_pickle=True)
    X = sparse.csr_matrix((d["X_data"], d["X_indices"], d["X_indptr"]),
                          shape=tuple(d["X_shape"]))
    return X, [str(x) for x in d["labels"]], [str(x) for x in d["genes"]]


def pred_path(res, method, mode, umi_min="NA"):
    return PRED_DIR / f"{res}um" / f"{method}__{mode}__umi{umi_min}.npz"


def save_pred(res, method, mode, props, covered, fit_seconds, version,
              notes="", umi_min="NA", extra=None):
    p = pred_path(res, method, mode, umi_min)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = {"method": method, "mode": mode, "umi_min": umi_min,
            "fit_seconds": float(fit_seconds), "version": version,
            "notes": notes, **(extra or {})}
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, props=props.astype(np.float32),
                        covered=covered.astype(bool), meta=json.dumps(meta))
    os.replace(tmp, p)
    return p
