"""Check that v0.2.0 (gene_weighting='expected') is seed-independent on the
Xenium CRC pseudo-VHD 8 um bins, which justifies running the 20-seed
ensemble of the benchmark with a single seed."""
import numpy as np
from xenium_pseudo_visiumhd_benchmark import (
    load_xenium, create_bins, build_reference_from_xenium,
    run_flashdeconv_raw, VHD_TARGET_MEDIAN_UMI,
)

adata_xen = load_xenium()
cell_types = sorted([ct for ct in adata_xen.obs["Level2"].unique() if ct != "Unassigned"])
X_sig, idx, _ = build_reference_from_xenium(adata_xen, cell_types)
Y, gt, _, centers, _, _ = create_bins(
    adata_xen, 8, target_median_umi=VHD_TARGET_MEDIAN_UMI[8], rng=np.random.default_rng(42))
Y = Y[:, idx]
preds = [run_flashdeconv_raw(Y, X_sig, centers, cell_types, random_state=s) for s in (0, 1, 42)]
for s, p in zip((1, 42), preds[1:]):
    print(f"seed 0 vs {s}: max |diff| = {np.abs(p - preds[0]).max():.3e}")
