"""Reusable plotting helpers for paper figures and notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _configure_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica"],
            "font.size": 8,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "pdf.fonttype": 42,
        }
    )
    return plt


def load_leverage_figure_data(results_dir: str | Path) -> dict:
    """Load intermediate data for Figure 2."""
    results_path = Path(results_dir) / "leverage_deep_dive"
    abundance_file = results_path / "experiment1_abundance_invariance.csv"
    if not abundance_file.exists():
        raise FileNotFoundError(f"Missing {abundance_file}. Run analysis/leverage_deep_dive.py first.")

    data = {"abundance": pd.read_csv(abundance_file)}
    quadrant_file = results_path / "experiment2_gene_quadrant_all.csv"
    if quadrant_file.exists():
        data["quadrant"] = pd.read_csv(quadrant_file)
    go_file = results_path / "enrichr_gold/GO_Biological_Process_2021.mouse.enrichr.reports.txt"
    if go_file.exists():
        data["go_gold"] = pd.read_csv(go_file, sep="\t")
    for key, name in [("gold_genes", "experiment2_gold_genes_symbols.csv"), ("noise_genes", "experiment2_noise_genes_symbols.csv")]:
        path = results_path / name
        if path.exists():
            data[key] = pd.read_csv(path)["symbol"].tolist()
    return data


def create_leverage_figure(data: dict, output_dir: str | Path, visium_path=None, prefix: str = "figure2_leverage_mechanism"):
    """Create Figure 2 from leverage analysis outputs."""
    plt = _configure_matplotlib()
    from matplotlib.gridspec import GridSpec

    colors = {
        "variance": "#3498db",
        "leverage": "#e74c3c",
        "gold": "#27ae60",
        "noise": "#e74c3c",
        "high_high": "#3498db",
        "low_low": "#ecf0f1",
    }
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2], hspace=0.25, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    df = data["abundance"]
    ax.plot(df["abundance_dominant_pct"], df["avg_var_rank_dominant"], "o-", color=colors["variance"], label="Variance")
    ax.plot(df["abundance_dominant_pct"], df["avg_lev_rank_dominant"], "s-", color=colors["leverage"], label="Leverage")
    ax.set_xlabel("Dominant type abundance (%)")
    ax.set_ylabel("Marker gene rank")
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.legend(frameon=False)
    ax.text(-0.15, 1.05, "a", transform=ax.transAxes, fontsize=12, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    qdf = data.get("quadrant")
    if qdf is not None:
        qdf = qdf.copy()
        mapped = {
            "Low Var / High Lev": ("Low Var / High Lev (GOLD)", colors["gold"]),
            "High Var / Low Lev": ("High Var / Low Lev (NOISE)", colors["noise"]),
            "High Var / High Lev": ("High Var / High Lev", colors["high_high"]),
            "Low Var / Low Lev": ("Low Var / Low Lev", colors["low_low"]),
        }
        for raw, (label, color) in mapped.items():
            mask = qdf["quadrant"] == raw
            if mask.any():
                ax.scatter(qdf.loc[mask, "log_variance"], qdf.loc[mask, "log_leverage"], s=3, c=color, alpha=0.5, label=f"{label} ({mask.sum()})", rasterized=True)
        ax.axvline(qdf["log_variance"].median(), color="gray", linestyle="--", linewidth=0.5)
        ax.axhline(qdf["log_leverage"].median(), color="gray", linestyle="--", linewidth=0.5)
        ax.legend(frameon=False, fontsize=6, markerscale=2)
    ax.set_xlabel("log10(Variance)")
    ax.set_ylabel("log10(Leverage score)")
    ax.text(-0.08, 1.05, "b", transform=ax.transAxes, fontsize=12, fontweight="bold")

    ax = fig.add_subplot(gs[1, 0])
    if "go_gold" in data:
        go_df = data["go_gold"].nsmallest(8, "Adjusted P-value").copy()
        go_df["Term_clean"] = go_df["Term"].str.split("(GO:", regex=False).str[0].str.strip()
        y_pos = np.arange(len(go_df))
        ax.barh(y_pos, -np.log10(go_df["Adjusted P-value"]), color=colors["gold"], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(go_df["Term_clean"], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=0.8)
    else:
        ax.text(0.5, 0.5, "GO data not available", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("-log10(Adjusted P-value)")
    ax.text(-0.02, 1.05, "c", transform=ax.transAxes, fontsize=12, fontweight="bold")

    ax = fig.add_subplot(gs[1, 1])
    ax.text(0.5, 0.5, "Spatial visualization\n(requires Visium data)", ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    ax.text(-0.02, 1.02, "d", transform=ax.transAxes, fontsize=12, fontweight="bold")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{prefix}.png", dpi=300, facecolor="white")
    fig.savefig(output / f"{prefix}.pdf", facecolor="white")
    plt.close(fig)
    return output / f"{prefix}.png"


def load_cortex_figure_data(results_dir: str | Path):
    """Load cortex NPZ data for Figure 5."""
    path = Path(results_dir) / "level2_v3_data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run analysis/cortex_deconvolution.py first.")
    return np.load(path, allow_pickle=True)


def create_cortex_lamination_figure(data, output_dir: str | Path, prefix: str = "figure5_cortex_lamination"):
    """Create Figure 5 cortex lamination panel."""
    plt = _configure_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.gridspec import GridSpec
    from scipy.ndimage import gaussian_filter1d

    props = data["proportions"]
    coords = data["coordinates"]
    cell_types = list(data["cell_types"])

    def get_idx(col):
        return cell_types.index(col) if col in cell_types else None

    layer_spatial = [("L2/3", "Ext_L23", "#2166ac"), ("L2-5", "Ext_L25", "#4393c3"), ("L5-6", "Ext_L56", "#f4a582"), ("L6", "Ext_L6", "#d6604d")]
    layer_profile = layer_spatial + [("L6b", "Ext_L6B", "#b2182b")]
    context_maps = [("Thalamus", "Ext_Thal_1", "Oranges"), ("Hippocampus CA1", "Ext_Hpc_CA1", "Greens"), ("Hippocampus DG", "Ext_Hpc_DG2", "Purples"), ("Oligodendrocytes", "Oligo_2", "YlOrBr"), ("Microglia", "Micro", "BuPu")]

    layer_indices = [get_idx(col) for col in ["Ext_L23", "Ext_L25", "Ext_L56", "Ext_L6", "Ext_L6B"] if get_idx(col) is not None]
    layer_data = np.column_stack([props[:, idx] for idx in layer_indices])
    total_cortical = layer_data.sum(axis=1)
    cortex_mask = total_cortical > 0.05
    x_center = np.median(coords[cortex_mask, 1]) if cortex_mask.any() else np.median(coords[:, 1])
    near_line = np.abs(coords[:, 1] - x_center) < 150
    if near_line.sum() < 3:
        near_line = np.ones(len(coords), dtype=bool)
    line_coords = coords[near_line]
    line_props = props[near_line]
    sort_idx = np.argsort(line_coords[:, 0])
    y_sorted = line_coords[sort_idx, 0]
    y_normalized = (y_sorted - y_sorted.min()) / (y_sorted.max() - y_sorted.min() + 1e-10) * 100

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 5, figure=fig, height_ratios=[1.0, 0.75, 1.0], hspace=0.25, wspace=0.15)

    for i, (name, col, color) in enumerate(layer_spatial):
        ax = fig.add_subplot(gs[0, i])
        idx = get_idx(col)
        if idx is None:
            ax.axis("off")
            continue
        values = props[:, idx]
        cmap = LinearSegmentedColormap.from_list("custom", ["#f7f7f7", color])
        sc = ax.scatter(coords[:, 1], -coords[:, 0], c=values, cmap=cmap, s=8, vmin=0, vmax=max(np.percentile(values, 98), 0.01), rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.01, aspect=15)
        ax.axis("off")
        ax.set_aspect("equal")
        ax.text(-0.02, 1.02, chr(97 + i), transform=ax.transAxes, fontsize=12, fontweight="bold")

    ax = fig.add_subplot(gs[0, 4])
    depth_weights = np.array([0, 0.25, 0.5, 0.75, 1.0])[: len(layer_indices)]
    depth_index = np.sum(layer_data * depth_weights, axis=1) / (total_cortical + 1e-10)
    depth_index_masked = np.where(total_cortical > 0.02, depth_index, np.nan)
    sc = ax.scatter(coords[:, 1], -coords[:, 0], c=depth_index_masked, cmap="RdYlBu_r", s=8, vmin=0, vmax=1, rasterized=True)
    plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.01, aspect=15)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.text(-0.02, 1.02, "e", transform=ax.transAxes, fontsize=12, fontweight="bold")

    ax = fig.add_subplot(gs[1, 0:3])
    for name, col, color in layer_profile:
        idx = get_idx(col)
        if idx is not None:
            ax.plot(y_normalized, gaussian_filter1d(line_props[sort_idx, idx], sigma=3), label=name, color=color, linewidth=2)
    ax.set_xlabel("Cortical Depth (% from surface)")
    ax.set_ylabel("Proportion")
    ax.legend(loc="upper right", fontsize=7)
    ax.text(-0.05, 1.02, "f", transform=ax.transAxes, fontsize=12, fontweight="bold")

    ax = fig.add_subplot(gs[1, 3:5])
    stack_data, labels, colors = [], [], []
    for name, col, color in layer_profile:
        idx = get_idx(col)
        if idx is not None:
            stack_data.append(gaussian_filter1d(line_props[sort_idx, idx], sigma=3))
            labels.append(name)
            colors.append(color)
    if stack_data:
        ax.stackplot(y_normalized, np.asarray(stack_data), labels=labels, colors=colors, alpha=0.45)
    ax.set_xlabel("Cortical Depth (% from surface)")
    ax.set_ylabel("Cumulative Proportion")
    ax.legend(loc="upper right", fontsize=7)
    ax.text(-0.05, 1.02, "g", transform=ax.transAxes, fontsize=12, fontweight="bold")

    for i, (_, col, cmap_name) in enumerate(context_maps):
        ax = fig.add_subplot(gs[2, i])
        idx = get_idx(col)
        if idx is None:
            ax.axis("off")
            continue
        values = props[:, idx]
        sc = ax.scatter(coords[:, 1], -coords[:, 0], c=values, cmap=cmap_name, s=8, vmin=0, vmax=max(np.percentile(values, 98), 0.01), rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.01, aspect=15)
        ax.axis("off")
        ax.set_aspect("equal")
        ax.text(-0.02, 1.02, chr(104 + i), transform=ax.transAxes, fontsize=12, fontweight="bold")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{prefix}.png", dpi=300, facecolor="white")
    fig.savefig(output / f"{prefix}.pdf", dpi=300, facecolor="white")
    plt.close(fig)
    return output / f"{prefix}.png"


def load_tuft_figure_data(results_dir: str | Path):
    """Load inputs for Figure 7."""
    path = Path(results_dir)
    props_path = path / "multiscale_proportions.csv"
    visibility_path = path / "cell_type_visibility.csv"
    if not props_path.exists():
        raise FileNotFoundError(f"Missing {props_path}. Run analysis/resolution_horizon_analysis.py first.")
    if not visibility_path.exists():
        raise FileNotFoundError(f"Missing {visibility_path}. Run analysis/tuft_stem_discovery.py first.")
    return pd.read_csv(props_path), pd.read_csv(visibility_path)


def create_tuft_discovery_figure(
    props: pd.DataFrame,
    visibility: pd.DataFrame,
    output_dir: str | Path,
    coloc_df: pd.DataFrame | None = None,
    prefix: str = "figure7_tuft_discovery",
):
    """Create Figure 7 tuft/stem discovery panel."""
    plt = _configure_matplotlib()
    from matplotlib.gridspec import GridSpec
    from repro.visium_hd import analyze_colocalization

    subset = props[props["bin_size"] == props["bin_size"].min()].copy() if "bin_size" in props else props.copy()
    subset = subset.reset_index(drop=True)
    coloc_df = coloc_df if coloc_df is not None else analyze_colocalization(subset)

    fig = plt.figure(figsize=(7, 6))
    gs = GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.35)
    tuft_color, stem_color, other_color = "#e41a1c", "#377eb8", "#999999"

    ax = fig.add_subplot(gs[0, 0])
    vis_sorted = visibility.sort_values("hvg_blindness", ascending=True)
    y_pos = np.arange(len(vis_sorted))
    names = [("Tuft" if "brush" in ct.lower() else ct.replace(" cell", "").replace("epithelial fate ", "")[:14]) for ct in vis_sorted["cell_type"]]
    colors = [tuft_color if "brush" in ct.lower() else other_color for ct in vis_sorted["cell_type"]]
    ax.barh(y_pos, vis_sorted["hvg_blindness"], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("HVG Blindness (%)")
    ax.text(-0.2, 1.02, "a", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    tuft_col = "brush cell" if "brush cell" in subset.columns else None
    if tuft_col:
        sc = ax.scatter(subset["coord_x"], subset["coord_y"], c=subset[tuft_col], s=0.3, cmap="Reds", vmin=0, vmax=0.3, rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.5, aspect=15, pad=0.02).set_label("Tuft\nproportion")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(-0.05, 1.02, "b", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[0, 2])
    stem_col = next((c for c in ["epithelial fate stem cell", "stem cell", "Stem"] if c in subset.columns), None)
    if stem_col:
        sc = ax.scatter(subset["coord_x"], subset["coord_y"], c=subset[stem_col], s=0.3, cmap="Blues", vmin=0, vmax=0.5, rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.5, aspect=15, pad=0.02).set_label("Stem\nproportion")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(-0.05, 1.02, "c", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[1, 0])
    if tuft_col and "bin_size" in props.columns:
        sizes = sorted(props["bin_size"].unique())
        max_props = [props.loc[props["bin_size"] == size, tuft_col].max() * 100 for size in sizes]
        ax.plot(sizes, max_props, "o-", color=tuft_color)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Resolution (um)")
    ax.set_ylabel("Max Tuft Proportion (%)")
    ax.text(-0.2, 1.02, "d", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[1, 1])
    top_coloc = coloc_df.head(6).sort_values("enrichment", ascending=True)
    if not top_coloc.empty:
        colors = [stem_color if "stem" in ct.lower() else other_color for ct in top_coloc["cell_type"]]
        ax.barh(range(len(top_coloc)), top_coloc["enrichment"], color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_coloc)))
        ax.set_yticklabels([ct.replace(" cell", "").replace("epithelial fate ", "")[:15] for ct in top_coloc["cell_type"]])
        ax.axvline(1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Fold enrichment")
    ax.text(-0.2, 1.02, "e", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[1, 2])
    if tuft_col and stem_col:
        tuft_vals = subset[tuft_col].values
        stem_vals = subset[stem_col].values
        categories = np.zeros(len(subset), dtype=int)
        categories[(tuft_vals > 0.05) & ~(stem_vals > 0.10)] = 1
        categories[~(tuft_vals > 0.05) & (stem_vals > 0.10)] = 2
        categories[(tuft_vals > 0.05) & (stem_vals > 0.10)] = 3
        center = subset.loc[categories == 3, ["coord_x", "coord_y"]].median() if (categories == 3).any() else subset[["coord_x", "coord_y"]].median()
        zoom = subset[(subset["coord_x"].between(center["coord_x"] - 400, center["coord_x"] + 400)) & (subset["coord_y"].between(center["coord_y"] - 400, center["coord_y"] + 400))]
        zoom_cats = categories[zoom.index.to_numpy()]
        for cat, color, label, size in [(0, "#E5E5E5", None, 1), (2, "#0072B2", "Stem-high", 4), (1, "#D55E00", "Tuft-high", 6), (3, "#CC79A7", "Co-localized", 8)]:
            mask = zoom_cats == cat
            if mask.any():
                ax.scatter(zoom.loc[mask, "coord_x"], zoom.loc[mask, "coord_y"], c=color, s=size, alpha=0.8, label=label, rasterized=True)
        ax.legend(loc="upper left", fontsize=6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(-0.05, 1.02, "f", transform=ax.transAxes, fontsize=10, fontweight="bold")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{prefix}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{prefix}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output / f"{prefix}.png"


def create_resolution_figure(results: dict, metrics_df: pd.DataFrame, output_dir: str | Path, prefix: str = "resolution_horizon_figure"):
    """Create a compact resolution-horizon summary figure."""
    plt = _configure_matplotlib()
    fig = plt.figure(figsize=(7, 6))
    valid_bins = sorted([bs for bs, res in results.items() if res is not None])
    if not valid_bins:
        raise ValueError("No valid multiscale results to plot")

    rare_ct = next((ct for ct in ["brush cell", "Tuft", "enteroendocrine cell"] if ct in results[valid_bins[0]]["proportions"].columns), None)
    if rare_ct:
        for i, bin_size in enumerate(valid_bins[:4]):
            ax = fig.add_subplot(2, len(valid_bins[:4]), i + 1)
            res = results[bin_size]
            ax.scatter(res["coords"][:, 0], res["coords"][:, 1], c=res["proportions"][rare_ct], cmap="Reds", s=max(0.5, 5 - i), vmin=0, vmax=0.3, rasterized=True)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{bin_size}um")

    ax = fig.add_subplot(2, 2, 3)
    for ct, color in [("brush cell", "#e74c3c"), ("enteroendocrine cell", "#3498db")]:
        subset = metrics_df[metrics_df["cell_type"] == ct].sort_values("bin_size")
        if len(subset) > 1:
            ax.plot(subset["bin_size"], subset["max_prop"] * 100, "o-", label=ct.replace(" cell", ""), color=color)
    ax.set_xlabel("Resolution (um)")
    ax.set_ylabel("Max proportion (%)")
    ax.set_xscale("log", base=2)
    ax.legend(frameon=False, fontsize=7)

    ax = fig.add_subplot(2, 2, 4)
    for ct in metrics_df.groupby("cell_type")["mean_prop"].mean().nlargest(4).index:
        subset = metrics_df[metrics_df["cell_type"] == ct].sort_values("bin_size")
        if len(subset) > 1:
            ax.plot(subset["bin_size"], subset["morans_i"], "o-", label=ct.replace(" cell", "")[:15])
    ax.set_xlabel("Resolution (um)")
    ax.set_ylabel("Moran's I")
    ax.set_xscale("log", base=2)
    ax.legend(frameon=False, fontsize=6)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    return output / f"{prefix}.png"
