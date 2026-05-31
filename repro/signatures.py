"""Reference signature, leverage-score, cortex, and mechanism helpers."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import pearsonr


def compute_leverage_scores(X: np.ndarray) -> np.ndarray:
    """Compute gene leverage scores from a cell-type-by-gene matrix."""
    X = np.asarray(X)
    X_centered = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
    return np.sum(vt**2, axis=0)


def _mean_expression(matrix, mask) -> np.ndarray:
    if issparse(matrix):
        return matrix[mask].toarray().mean(axis=0)
    return np.asarray(matrix[mask]).mean(axis=0)


def build_signature_matrix(
    adata,
    ct_col: str,
    normalize: str | None = "cpm",
    sort_cell_types: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build an average expression signature matrix from an AnnData object."""
    cell_types = list(adata.obs[ct_col].unique())
    if sort_cell_types:
        cell_types = sorted(cell_types)

    X = np.zeros((len(cell_types), adata.n_vars), dtype=float)
    labels = adata.obs[ct_col].values
    for i, ct in enumerate(cell_types):
        X[i] = _mean_expression(adata.X, labels == ct)

    if normalize in {"cpm", "log_cpm"}:
        X = X / (X.sum(axis=1, keepdims=True) + 1e-10) * 1e4
    if normalize == "log_cpm":
        X = np.log1p(X)

    return X, cell_types


def compute_marker_visibility(
    X: np.ndarray,
    cell_types: list[str],
    gene_names: list[str],
    n_markers: int = 30,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compare leverage and variance ranks for each cell type's markers."""
    leverage = compute_leverage_scores(X)
    gene_var = np.var(X, axis=0)
    leverage_rank = np.argsort(np.argsort(-leverage))
    variance_rank = np.argsort(np.argsort(-gene_var))

    n_genes = len(gene_names)
    results = []
    for ct_idx, ct_name in enumerate(cell_types):
        ct_expr = X[ct_idx]
        other_mask = np.ones(len(cell_types), dtype=bool)
        other_mask[ct_idx] = False
        other_expr = X[other_mask].mean(axis=0)
        fc = np.log2((ct_expr + 1) / (other_expr + 1))
        marker_idx = np.argsort(-fc)[:n_markers]
        marker_genes = [gene_names[i] for i in marker_idx]

        marker_leverage_pct = np.mean(leverage_rank[marker_idx]) / n_genes * 100
        marker_variance_pct = np.mean(variance_rank[marker_idx]) / n_genes * 100
        hvg_blindness = marker_variance_pct - marker_leverage_pct

        results.append(
            {
                "cell_type": ct_name,
                "marker_leverage_pct": marker_leverage_pct,
                "marker_variance_pct": marker_variance_pct,
                "hvg_blindness": hvg_blindness,
                "top_marker": marker_genes[0],
                "top_marker_fc": fc[marker_idx[0]],
                "top_5_markers": ", ".join(marker_genes[:5]),
            }
        )

    df = pd.DataFrame(results).sort_values("hvg_blindness", ascending=False)
    if verbose:
        print("\nCell types ranked by HVG blindness:")
        for _, row in df.head(10).iterrows():
            print(f"  {row['cell_type']:<30} {row['hvg_blindness']:>6.1f}%")
    return df


def compute_hvg_blindness(adata_ref, ct_col: str, n_markers: int = 30) -> pd.DataFrame:
    """Compute HVG blindness directly from an annotated reference AnnData."""
    X, cell_types = build_signature_matrix(adata_ref, ct_col, normalize="cpm")
    return compute_marker_visibility(X, cell_types, adata_ref.var_names.tolist(), n_markers)


def find_hidden_genes(X: np.ndarray, gene_names: list[str], top_n: int = 50) -> pd.DataFrame:
    """Find high-leverage, low-variance genes."""
    leverage = compute_leverage_scores(X)
    gene_var = np.var(X, axis=0)
    n_genes = len(gene_names)
    leverage_pct = np.argsort(np.argsort(-leverage)) / n_genes * 100
    variance_pct = np.argsort(np.argsort(-gene_var)) / n_genes * 100
    hidden_score = variance_pct - leverage_pct
    hidden_idx = np.argsort(-hidden_score)[:top_n]
    return pd.DataFrame(
        {
            "gene": [gene_names[i] for i in hidden_idx],
            "leverage": leverage[hidden_idx],
            "variance": gene_var[hidden_idx],
            "leverage_pct": leverage_pct[hidden_idx],
            "variance_pct": variance_pct[hidden_idx],
            "hidden_score": hidden_score[hidden_idx],
        }
    )


def load_intestine_reference(data_dir: str | Path):
    """Load Haber intestine reference and return (adata, cell_type_column)."""
    data_dir = Path(data_dir)
    for ref_name in ["haber_intestine_matched.h5ad", "haber_intestine_reference.h5ad", "haber_processed.h5ad"]:
        ref_path = data_dir / ref_name
        if ref_path.exists():
            import scanpy as sc

            ref = sc.read_h5ad(ref_path)
            ref.var_names_make_unique()
            ct_col = "celltype1" if "celltype1" in ref.obs else "cell_type"
            return ref, ct_col
    raise FileNotFoundError(
        f"No Haber reference found in {data_dir}. Expected haber_intestine_matched.h5ad, "
        "haber_intestine_reference.h5ad, or haber_processed.h5ad."
    )


def load_mouse_brain_reference(data_dir: str | Path):
    """Load Cell2location mouse brain scRNA-seq reference with annotations."""
    data_path = Path(data_dir) / "mouse_brain"
    ref_path = data_path / "scrna_reference.h5ad"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Missing Cell2location reference: {ref_path}\n"
            "Run:\n  bash scripts/download_cell2location_data.sh ./data/mouse_brain"
        )

    import scanpy as sc

    sc_adata = sc.read_h5ad(ref_path)

    anno_path = data_path / "cell_annotation.csv"
    if anno_path.exists():
        anno = pd.read_csv(anno_path, index_col=0)
        if "annotation_1" not in anno:
            raise KeyError(f"{anno_path} is missing required column 'annotation_1'.")
        common_cells = sc_adata.obs_names.intersection(anno.index)
        sc_adata = sc_adata[common_cells].copy()
        sc_adata.obs["cell_type"] = anno.loc[sc_adata.obs_names, "annotation_1"].values
    elif "annotation_1" in sc_adata.obs:
        sc_adata = sc_adata[~sc_adata.obs["annotation_1"].isna()].copy()
        sc_adata.obs["cell_type"] = sc_adata.obs["annotation_1"].astype(str).values
    else:
        raise FileNotFoundError(
            f"Missing {anno_path}, and scrna_reference.h5ad has no obs['annotation_1'] column."
        )
    return sc_adata


def load_cortex_paired_data(data_dir: str | Path):
    """Load Cell2location paired mouse brain Visium and scRNA-seq data."""
    data_path = Path(data_dir) / "mouse_brain"
    sc_adata = load_mouse_brain_reference(data_dir)

    if "SYMBOL" not in sc_adata.var:
        raise KeyError(
            "scrna_reference.h5ad is missing var['SYMBOL']; "
            "use scripts/download_cell2location_data.sh."
        )

    symbols = sc_adata.var["SYMBOL"].astype(str).values
    has_symbol = (symbols != "") & (symbols != "nan") & (symbols != "None")
    sc_adata = sc_adata[:, has_symbol].copy()

    name_counts: Counter[str] = Counter()
    unique_names = []
    for name in symbols[has_symbol]:
        unique_names.append(f"{name}-{name_counts[name]}" if name_counts[name] else name)
        name_counts[name] += 1
    sc_adata.var_names = pd.Index(unique_names)

    candidate_st_paths = [
        data_path / "C2L" / "ST" / "48",
        data_path / "mouse_brain_visium_wo_cloupe_data" / "rawdata" / "ST8059048",
    ]
    st_path = next((path for path in candidate_st_paths if path.exists()), candidate_st_paths[0])
    matrix_path = st_path / "ST8059048_filtered_feature_bc_matrix.h5"
    if not matrix_path.exists():
        matrix_path = st_path / "filtered_feature_bc_matrix.h5"
    positions_path = st_path / "spatial" / "tissue_positions_list.csv"
    missing = [path for path in [matrix_path, positions_path] if not path.exists()]
    if missing:
        expected = "\n  - ".join(str(path) for path in candidate_st_paths)
        missing_text = "\n  - ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing Cell2location spatial inputs:\n"
            f"  - {missing_text}\n"
            "Expected one of these ST8059048 layouts:\n"
            f"  - {expected}\n"
            "Run:\n  bash scripts/download_cell2location_data.sh ./data/mouse_brain"
        )

    import scanpy as sc

    sp_adata = sc.read_10x_h5(matrix_path)
    sp_adata.var_names_make_unique()

    coords_df = pd.read_csv(
        positions_path,
        header=None,
        index_col=0,
    )
    coords_df.columns = ["in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    in_tissue = coords_df[coords_df["in_tissue"] == 1].index
    common_spots = sp_adata.obs_names.intersection(in_tissue)
    sp_adata = sp_adata[common_spots].copy()
    sp_adata.obsm["spatial"] = coords_df.loc[sp_adata.obs_names, ["pxl_row", "pxl_col"]].values

    common_genes = sc_adata.var_names.intersection(sp_adata.var_names)
    if len(common_genes) == 0:
        raise ValueError("No shared genes found between Cell2location scRNA and Visium inputs.")
    sc_adata = sc_adata[:, common_genes].copy()
    sp_adata = sp_adata[:, common_genes].copy()
    return sc_adata, sp_adata, common_genes


def create_reference_signatures(sc_adata) -> tuple[np.ndarray, list[str]]:
    """Create cortex reference signatures preserving observed cell type order."""
    return build_signature_matrix(sc_adata, "cell_type", normalize=None, sort_cell_types=False)


def run_cortex_flashdeconv(sp_adata, X_ref: np.ndarray, cell_types: list[str]):
    """Run FlashDeconv on cortex Visium data."""
    from flashdeconv import FlashDeconv

    Y = sp_adata.X.toarray() if issparse(sp_adata.X) else np.asarray(sp_adata.X)
    coords = sp_adata.obsm["spatial"]

    t0 = time.time()
    model = FlashDeconv(
        sketch_dim=512,
        lambda_spatial=5000.0,
        rho_sparsity=0.01,
        preprocess="log_cpm",
        n_hvg=2000,
        max_iter=100,
        verbose=True,
        random_state=42,
    )
    proportions = model.fit_transform(Y, X_ref, coords)
    elapsed = time.time() - t0
    prop_df = pd.DataFrame(proportions, index=sp_adata.obs_names, columns=cell_types)
    return prop_df, elapsed


def evaluate_with_markers(sp_adata, prop_df: pd.DataFrame, sc_adata):
    """Evaluate predictions using marker gene correlation."""
    import scanpy as sc

    sc_adata_eval = sc_adata.copy()
    sc.pp.normalize_total(sc_adata_eval, target_sum=1e4)
    sc.pp.log1p(sc_adata_eval)

    try:
        sc.tl.rank_genes_groups(sc_adata_eval, "cell_type", method="wilcoxon", n_genes=20)
    except Exception as exc:
        print(f"Marker finding failed: {exc}")
        return None

    results = []
    var_names = list(sp_adata.var_names)
    for ct in prop_df.columns:
        try:
            markers = sc_adata_eval.uns["rank_genes_groups"]["names"][ct][:10]
            markers = [g for g in markers if g in sp_adata.var_names]
            if len(markers) < 2:
                continue

            marker_idx = [var_names.index(g) for g in markers]
            if issparse(sp_adata.X):
                marker_expr = sp_adata.X[:, marker_idx].toarray().mean(axis=1)
            else:
                marker_expr = sp_adata.X[:, marker_idx].mean(axis=1)
            r, p = pearsonr(marker_expr.flatten(), prop_df[ct].values.flatten())
            results.append({"cell_type": ct, "correlation": r, "p_value": p, "n_markers": len(markers)})
        except Exception:
            continue

    if not results:
        return None
    return pd.DataFrame(results).sort_values("correlation", ascending=False)


def save_cortex_outputs(output_dir: str | Path, prop_df: pd.DataFrame, sp_adata, cell_types: list[str], results_df=None) -> None:
    """Save cortex CSV and NPZ outputs consumed by Figure 5."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prop_df.to_csv(output_dir / "level2_v3_proportions.csv")
    if results_df is not None:
        results_df.to_csv(output_dir / "level2_v3_correlations.csv", index=False)
    np.savez(
        output_dir / "level2_v3_data.npz",
        proportions=prop_df.values,
        coordinates=sp_adata.obsm["spatial"],
        spot_names=np.array(prop_df.index.tolist(), dtype=object),
        cell_types=np.array(cell_types, dtype=object),
    )


def experiment1_abundance_invariance(
    X: np.ndarray,
    cell_types: list[str],
    gene_names: list[str],
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Test how leverage and variance marker ranks change with abundance."""
    dominant_idx = int(np.argmax(X.sum(axis=1)))
    dom_expr = X[dominant_idx]
    other_expr = np.delete(X, dominant_idx, axis=0).mean(axis=0)
    marker_genes_idx = np.argsort(-(dom_expr - other_expr))[:50]

    rows = []
    for pct in [100, 80, 60, 40, 20, 10, 5]:
        X_sim = X.copy()
        X_sim[dominant_idx] *= pct / 100
        gene_var = np.var(X_sim, axis=0)
        leverage = compute_leverage_scores(X_sim)
        var_rank = np.argsort(np.argsort(-gene_var)) + 1
        lev_rank = np.argsort(np.argsort(-leverage)) + 1
        rows.append(
            {
                "abundance_dominant_pct": pct,
                "avg_var_rank_dominant": np.mean(var_rank[marker_genes_idx]),
                "avg_lev_rank_dominant": np.mean(lev_rank[marker_genes_idx]),
                "dominant_type": cell_types[dominant_idx],
            }
        )

    df = pd.DataFrame(rows)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(output_dir) / "experiment1_abundance_invariance.csv", index=False)
    return df


def experiment2_gene_quadrant(
    X: np.ndarray,
    gene_names: list[str],
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Categorize genes by variance and leverage score quadrants."""
    gene_var = np.var(X, axis=0)
    leverage = compute_leverage_scores(X)
    log_var = np.log10(gene_var + 1e-10)
    log_lev = np.log10(leverage + 1e-10)
    var_thresh = np.median(log_var)
    lev_thresh = np.median(log_lev)

    quadrants = []
    for gene, v, lev in zip(gene_names, log_var, log_lev):
        if v < var_thresh and lev > lev_thresh:
            q = "Low Var / High Lev"
        elif v > var_thresh and lev < lev_thresh:
            q = "High Var / Low Lev"
        elif v > var_thresh and lev > lev_thresh:
            q = "High Var / High Lev"
        else:
            q = "Low Var / Low Lev"
        quadrants.append({"gene": gene, "log_variance": v, "log_leverage": lev, "quadrant": q})

    df = pd.DataFrame(quadrants)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "experiment2_gene_quadrant_all.csv", index=False)
        pd.DataFrame({"symbol": df.loc[df["quadrant"] == "Low Var / High Lev", "gene"]}).to_csv(
            output_dir / "experiment2_gold_genes_symbols.csv",
            index=False,
        )
        pd.DataFrame({"symbol": df.loc[df["quadrant"] == "High Var / Low Lev", "gene"]}).to_csv(
            output_dir / "experiment2_noise_genes_symbols.csv",
            index=False,
        )
    return df
