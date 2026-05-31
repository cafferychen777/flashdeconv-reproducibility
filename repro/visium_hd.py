"""Visium HD, resolution-horizon, and tuft niche helpers."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


def load_visium_hd_data(data_dir: str | Path, bin_size: str = "016um"):
    """Load 10x Visium HD binned data at a specific bin size."""
    import pyarrow.parquet as pq
    import scanpy as sc

    data_path = Path(data_dir) / "Visium_HD_Mouse_Small_Intestine_binned_outputs"
    bin_path = data_path / f"square_{bin_size}"
    adata = sc.read_10x_h5(bin_path / "filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()

    positions_path = bin_path / "spatial" / "tissue_positions.parquet"
    if not positions_path.exists():
        raise FileNotFoundError(f"Positions file not found: {positions_path}")
    positions = pq.read_table(positions_path).to_pandas().set_index("barcode")
    common_barcodes = adata.obs_names.intersection(positions.index)
    adata = adata[common_barcodes].copy()
    adata.obsm["spatial"] = positions.loc[
        common_barcodes, ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    ].values
    return adata


def load_reference_data(data_dir: str | Path):
    """Load Haber intestine reference data and the cell type column name."""
    from repro.signatures import load_intestine_reference

    return load_intestine_reference(data_dir)


def create_custom_bins(adata, target_bin_size: int, original_bin_size: int = 16):
    """Aggregate Visium HD bins to a larger target bin size."""
    import scanpy as sc

    scale_factor = target_bin_size // original_bin_size
    if scale_factor == 1:
        return adata.copy()

    coords = adata.obsm["spatial"]
    tree = cKDTree(coords[: min(10000, len(coords))])
    distances, _ = tree.query(coords[: min(10000, len(coords))], k=2)
    pixel_spacing = np.median(distances[:, 1])
    grid_size_pixels = pixel_spacing * scale_factor

    min_x, min_y = coords.min(axis=0)
    grid_x = ((coords[:, 0] - min_x) / grid_size_pixels).astype(int)
    grid_y = ((coords[:, 1] - min_y) / grid_size_pixels).astype(int)
    grid_ids = grid_x.astype(str) + "_" + grid_y.astype(str)
    unique_grids = np.unique(grid_ids)

    grid_to_idx = {g: i for i, g in enumerate(unique_grids)}
    row_indices = [grid_to_idx[g] for g in grid_ids]
    col_indices = np.arange(adata.n_obs)
    agg_matrix = sparse.csr_matrix(
        (np.ones(adata.n_obs), (row_indices, col_indices)),
        shape=(len(unique_grids), adata.n_obs),
    )

    X = agg_matrix @ (adata.X if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X))
    new_coords = np.zeros((len(unique_grids), 2))
    for i, grid_id in enumerate(unique_grids):
        new_coords[i] = coords[grid_ids == grid_id].mean(axis=0)

    adata_new = sc.AnnData(X=X, var=adata.var.copy())
    adata_new.obs_names = pd.Index(unique_grids)
    adata_new.obsm["spatial"] = new_coords
    adata_new.uns["bin_size"] = target_bin_size
    return adata_new


def compute_morans_i(values, coords, k: int = 10) -> float:
    """Compute lightweight Moran's I spatial autocorrelation."""
    values = np.asarray(values)
    coords = np.asarray(coords)
    if len(values) < 3:
        return np.nan
    k = min(k, len(values) - 1)
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)
    y = values - values.mean()
    lag = np.array([y[indices[i, 1:]].mean() for i in range(len(values))])
    return (len(values) / (k * len(values))) * np.sum(y * lag) / (np.sum(y**2) + 1e-10)


def compute_morans_i_permutation(values, coords, k: int = 10, n_perm: int = 999, random_state: int = 42):
    """Compute Moran's I with a permutation null model."""
    rng = np.random.default_rng(random_state)
    values = np.asarray(values)
    coords = np.asarray(coords)
    n = len(values)
    k = min(k, n - 1)
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)

    W = np.zeros((n, n))
    for i in range(n):
        W[i, indices[i, 1:]] = 1
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    y = values - values.mean()
    denominator = np.sum(y**2)
    if denominator == 0:
        return 0, 1, 0, 0

    I = (n / W.sum()) * (np.sum(W * np.outer(y, y)) / denominator)
    I_perm = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        I_perm.append((n / W.sum()) * (np.sum(W * np.outer(y_perm, y_perm)) / denominator))
    I_perm = np.asarray(I_perm)
    p_value = (np.sum(I_perm >= I) + 1) / (n_perm + 1)
    return I, p_value, float(I_perm.mean()), float(I_perm.std())


def run_multiscale_analysis(data_dir: str | Path, output_dir: str | Path, bin_sizes: list[int]):
    """Run FlashDeconv at multiple Visium HD spatial scales."""
    from flashdeconv import FlashDeconv
    from flashdeconv.io.loader import prepare_data

    results = {}
    cell_types = None
    adata_ref, ct_col = load_reference_data(data_dir)
    adata_base = load_visium_hd_data(data_dir, "016um")
    base_bin_size = 16

    model = FlashDeconv(
        sketch_dim=512,
        lambda_spatial=5000.0,
        n_hvg=2000,
        n_markers_per_type=50,
        k_neighbors=6,
        max_iter=200,
        tol=1e-4,
        verbose=False,
        random_state=42,
    )

    for bin_size in bin_sizes:
        if bin_size == base_bin_size:
            adata_st = adata_base.copy()
        elif bin_size < base_bin_size:
            continue
        else:
            adata_st = create_custom_bins(adata_base, bin_size, base_bin_size)

        try:
            Y, X, coords, cell_type_names, _ = prepare_data(adata_st, adata_ref, cell_type_key=ct_col)
            start = time.time()
            proportions_raw = model.fit_transform(Y, X, coords)
            proportions = pd.DataFrame(proportions_raw, columns=cell_type_names)
            if cell_types is None:
                cell_types = list(proportions.columns)
            results[bin_size] = {
                "proportions": proportions,
                "n_spots": Y.shape[0],
                "time": time.time() - start,
                "coords": coords.copy(),
            }
        except Exception as exc:
            print(f"{bin_size}um failed: {exc}")
            results[bin_size] = None
        gc.collect()
    return results, cell_types


def compute_resolution_metrics(results: dict, cell_types: list[str] | None = None) -> pd.DataFrame:
    """Compute resolution-horizon metrics from multiscale results."""
    rows = []
    for bin_size, res in results.items():
        if res is None:
            continue
        props = res["proportions"]
        coords = res["coords"]
        for cell_type in props.columns:
            p = np.clip(props[cell_type].values, 1e-10, 1)
            rows.append(
                {
                    "bin_size": bin_size,
                    "cell_type": cell_type,
                    "cv": np.std(p) / (np.mean(p) + 1e-10),
                    "max_prop": np.max(p),
                    "pct_detectable": (p > 0.01).mean() * 100,
                    "pct_high": (p > 0.1).mean() * 100,
                    "morans_i": compute_morans_i(p, coords, k=min(10, len(p) - 1)),
                    "mean_prop": np.mean(p),
                }
            )
    return pd.DataFrame(rows)


def flatten_multiscale_proportions(results: dict) -> pd.DataFrame:
    """Flatten multiscale proportions and coordinates to one CSV-ready table."""
    rows = []
    for bin_size, res in results.items():
        if res is None:
            continue
        props = res["proportions"]
        coords = res["coords"]
        for i in range(len(props)):
            row = {"bin_size": bin_size, "coord_x": coords[i, 0], "coord_y": coords[i, 1]}
            row.update(props.iloc[i].to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_colocalization(
    props_df: pd.DataFrame,
    tuft_col: str = "brush cell",
    stem_col: str = "epithelial fate stem cell",
    tuft_threshold: float = 0.1,
) -> pd.DataFrame:
    """Analyze enrichment of cell types in tuft hotspots."""
    if tuft_col not in props_df.columns:
        return pd.DataFrame(columns=["cell_type", "enrichment", "mean_in_hotspot", "mean_elsewhere"])
    high_tuft = props_df[tuft_col] > tuft_threshold
    ct_cols = [c for c in props_df.columns if c not in {"bin_size", "coord_x", "coord_y", "spot_id", tuft_col}]
    rows = []
    for ct in ct_cols:
        hot_mean = props_df.loc[high_tuft, ct].mean()
        cold_mean = props_df.loc[~high_tuft, ct].mean()
        rows.append(
            {
                "cell_type": ct,
                "enrichment": hot_mean / (cold_mean + 1e-10),
                "mean_in_hotspot": hot_mean,
                "mean_elsewhere": cold_mean,
            }
        )
    return pd.DataFrame(rows).sort_values("enrichment", ascending=False)


def run_tuft_validation(props_df: pd.DataFrame, output_dir: str | Path | None = None, random_state: int = 42) -> dict:
    """Run Moran's I validation for tuft cell spatial clustering."""
    props = props_df[props_df["bin_size"] == props_df["bin_size"].min()].copy() if "bin_size" in props_df else props_df
    tuft_col = "brush cell" if "brush cell" in props.columns else None
    if tuft_col is None:
        return {}
    coords = props[["coord_x", "coord_y"]].values
    tuft_vals = props[tuft_col].values
    n_sample = min(5000, len(tuft_vals))
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(tuft_vals), n_sample, replace=False)
    I, p_val, I_rand_mean, I_rand_std = compute_morans_i_permutation(
        tuft_vals[sample_idx],
        coords[sample_idx],
        k=10,
        random_state=random_state,
    )
    result = {
        "morans_i": I,
        "morans_pvalue": p_val,
        "morans_random_mean": I_rand_mean,
        "morans_random_std": I_rand_std,
    }
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(Path(output_dir) / "tuft_validation_results.csv", index=False)
    return result

