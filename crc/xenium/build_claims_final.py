"""Xenium CRC validation claims: manuscript / ORIG (archive) / V020 / FINAL.

ORIG and V020 strings are taken from results/rerun_v020/crc/xenium/claims_summary_old_vs_v020.csv;
FINAL values are recomputed here from results/rerun_final/crc/{xenium,demo} with the same
definitions (verified by recomputing the V020 column from the v020 outputs: --check).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path("/Users/apple/Research/FlashDeconv/results")
V020_DIR = B / "rerun_v020/crc"
FIN_DIR = B / "rerun_final/crc"


def f3(x):
    return f"{x:.3f}"


def compute(R, P1_sub, tag):
    """R: xenium results dir; P1_sub: visium_hd_p1/props_* dir. Returns {claim: value string}."""
    c = {}
    vb = pd.read_csv(R / "virtual_binning/virtual_binning_metrics.csv")
    v = vb.pivot_table(index="metric", columns="bin_size_um", values="value")
    sizes = sorted(v.columns)
    lin = v.loc["mean_per_lineage_r"]
    c["VB lineage r (mean) min over 8-128um"] = f"{lin.min():.3f} ({int(lin.idxmin())}um)"
    c["VB global r 8/16/32/64/128um"] = "/".join(f3(v.loc["global_pearson_r", s]) for s in sizes)
    c["VB global r 32um"] = f3(v.loc["global_pearson_r", 32])
    c["VB per-type r 32um Goblet/Enterocyte/Endothelial"] = "/".join(
        f3(v.loc[f"per_type_r__{t}", 32]) for t in ["Goblet", "Enterocyte", "Endothelial"])
    c["VB per-type r 32um Tumor III/IV/V"] = "/".join(
        f3(v.loc[f"per_type_r__{t}", 32]) for t in ["Tumor III", "Tumor IV", "Tumor V"])
    c["VB median cells/bin 8um"] = f"{v.loc['median_cells_per_bin', 8]:.0f}"
    if P1_sub is not None and (P1_sub / "global_proportion_comparison.csv").exists():
        g = pd.read_csv(P1_sub / "global_proportion_comparison.csv")
        ct = g[~g.cell_type.str.startswith("[LINEAGE]")]
        ln = g[g.cell_type.str.startswith("[LINEAGE]")]
        r = lambda a, b: np.corrcoef(a, b)[0, 1]
        c["Global 38-type r FD vs Xenium"] = f3(r(ct.flashdeconv_prop, ct.xenium_prop))
        c["Global 38-type r RCTD vs Xenium"] = f3(r(ct.rctd_prop, ct.xenium_prop))
        c["Global lineage r FD / RCTD"] = f"{f3(r(ln.flashdeconv_prop, ln.xenium_prop))} / {f3(r(ln.rctd_prop, ln.xenium_prop))}"
        p = pd.read_csv(P1_sub / "pathologist_concordance.csv").set_index(["category", "lineage"])
        c["Patho Neoplasm Tumor/Stromal/Immune mean"] = "/".join(
            f"{100 * p.loc[('Neoplasm', L), 'mean_proportion']:.1f}" for L in ["Tumor", "Stromal", "Immune"]) + "%"
        c["Patho Vessel Tumor mean"] = f"{100 * p.loc[('Vessel', 'Tumor'), 'mean_proportion']:.1f}%"
        c["Patho fold vs tissue mean (Neoplasm Tumor / Vessel Tumor)"] = (
            f"{p.loc[('Neoplasm', 'Tumor'), 'enrichment']:.2f}x / {p.loc[('Vessel', 'Tumor'), 'enrichment']:.2f}x")
    s = pd.read_csv(R / "pseudo_vhd/benchmark_summary.csv")
    pt = pd.read_csv(R / "pseudo_vhd/benchmark_per_type.csv")
    s4 = s[s.bin_size_um == 4].set_index("method")
    order = ["FlashDeconv_auto", "NNLS", "MarkerScoring"]
    c["pVHD 4um AUPRC FD / NNLS / marker"] = "/".join(f3(s4.loc[m, "global_auprc"]) for m in order)
    p4 = pt[pt.bin_size_um == 4].set_index(["method", "cell_type"])
    for t in ["mRegDC", "Neutrophil"]:
        c[f"pVHD 4um {t} AUPRC FD/NNLS/marker"] = "/".join(f3(p4.loc[(m, t), "auprc"]) for m in order)
    d_r = s4.loc["FlashDeconv_auto", "global_r"] - s4.loc["FlashDeconv_lambda0", "global_r"]
    d_a = s4.loc["FlashDeconv_auto", "global_auprc"] - s4.loc["FlashDeconv_lambda0", "global_auprc"]
    c["pVHD 4um lambda auto-0 dPearson / dAUPRC"] = f"{d_r:+.3f} / {d_a:+.3f}"
    la = pd.read_csv(R / "lambda_ablation/xenium_crc_lambda_ablation_summary.csv").set_index(["bin_size_um", "condition"])
    for b in [8, 16]:
        z, a = la.loc[(b, "no_spatial")], la.loc[(b, "auto")]
        c[f"Lap Xenium {b}um lambda0 vs auto r/RMSE"] = (
            f"{z.overall_pearson_r:.3f}/{z.overall_rmse:.4f} vs {a.overall_pearson_r:.3f}/{a.overall_rmse:.4f}")
    return c


def main():
    old = pd.read_csv(V020_DIR / "xenium/claims_summary_old_vs_v020.csv")
    old = old[~old.claim.str.startswith("UQ")].reset_index(drop=True)
    if "--check" in sys.argv:
        chk = compute(V020_DIR / "xenium", V020_DIR / "xenium/visium_hd_p1/props_new", "V020")
        for _, r in old.iterrows():
            print(f"{r.claim:60s} | table: {r.new_v020:40s} | recomputed: {chk.get(r.claim)}")
        return
    fin = compute(FIN_DIR / "xenium", FIN_DIR / "xenium/visium_hd_p1/props_final", "FINAL")
    rows = []
    for _, r in old.iterrows():
        rows.append(dict(claim=r.claim, manuscript=r.manuscript, ORIG=r.old_recomputed_from_archive,
                         V020=r.new_v020, FINAL=fin.get(r.claim, "NA"), holds=r.holds,
                         note=("pending: FINAL P1 Visium HD proportions not yet available"
                               if r.claim not in fin else
                               "FINAL == V020 at reported precision" if fin[r.claim] == r.new_v020
                               else "FINAL differs from V020 in last digit(s)")))
    # Visium HD 8um Laplacian ablation (demo) and Fig. 3b zoom predictions
    lap = pd.read_csv(FIN_DIR / "demo/laplacian_old_v020_final.csv").set_index("cell_type")
    for ct, rr in lap.iterrows():
        rows.append(dict(claim=f"VHD 8um Laplacian Moran's I gain % {ct} (lambda0 -> auto)", manuscript="",
                         ORIG=f"{rr.rel_gain_pct_old:+.2f} ({rr.I_lambda0_old:.3f}->{rr.I_auto_old:.3f})",
                         V020=f"{rr.rel_gain_pct_v020:+.2f} ({rr.I_lambda0_v020:.3f}->{rr.I_auto_v020:.3f})",
                         FINAL=f"{rr.rel_gain_pct_final:+.2f} ({rr.I_lambda0_final:.3f}->{rr.I_auto_final:.3f})",
                         holds="", note="demo max_iter 200 -> package default 1000; fits converge in 49 iterations"))
    z = pd.read_csv(FIN_DIR / "demo/zoom_old_v020_final_summary.csv").set_index("region")
    for reg in ["all_bins", "zoom_window"]:
        q = z.loc[reg]
        rows.append(dict(claim=f"Zoom 8um pseudo-bins ({reg}) flat r / RMSE / dominant-lineage acc vs Xenium GT",
                         manuscript="",
                         ORIG=f"{q.old_vs_gt_flat_r:.3f} / {q.old_vs_gt_rmse:.4f} / {q.old_vs_gt_dom_lineage_acc:.3f}",
                         V020=f"{q.v020_vs_gt_flat_r:.3f} / {q.v020_vs_gt_rmse:.4f} / {q.v020_vs_gt_dom_lineage_acc:.3f}",
                         FINAL=f"{q.final_vs_gt_flat_r:.3f} / {q.final_vs_gt_rmse:.4f} / {q.final_vs_gt_dom_lineage_acc:.3f}",
                         holds="", note=f"final vs v020 flat r {q.v020_vs_final_flat_r:.6f}"))
    out = pd.DataFrame(rows)
    out.to_csv(FIN_DIR / "xenium/claims_summary_final.csv", index=False)
    # headline numbers for the part summary
    t = out.set_index("claim")
    ps = []
    for claim, bench, metric, rank, note in [
        ("VB lineage r (mean) min over 8-128um", "Xenium virtual binning", "min mean lineage r (8-128 um)", "", ""),
        ("VB global r 32um", "Xenium virtual binning", "global r 32 um (38 types)", "", ""),
        ("Global 38-type r FD vs Xenium", "Xenium vs Visium HD P1", "global 38-type proportion r FD", "1 of 2 (RCTD -0.020)", ""),
        ("Global lineage r FD / RCTD", "Xenium vs Visium HD P1", "global lineage r FD / RCTD", "1 of 2", ""),
        ("Patho fold vs tissue mean (Neoplasm Tumor / Vessel Tumor)", "Pathologist concordance P1", "tumor fold Neoplasm / Vessel", "", ""),
        ("pVHD 4um AUPRC FD / NNLS / marker", "Pseudo-Visium HD 4 um", "AUPRC FD / NNLS / marker", "1 of 3", "max_iter/tol at package defaults (orig 500/1e-6)"),
        ("pVHD 4um mRegDC AUPRC FD/NNLS/marker", "Pseudo-Visium HD 4 um", "mRegDC AUPRC FD / NNLS / marker", "1 of 3", ""),
        ("pVHD 4um Neutrophil AUPRC FD/NNLS/marker", "Pseudo-Visium HD 4 um", "Neutrophil AUPRC FD / NNLS / marker", "1 of 3", ""),
        ("pVHD 4um lambda auto-0 dPearson / dAUPRC", "Pseudo-Visium HD 4 um", "lambda auto minus 0: dPearson / dAUPRC", "", ""),
        ("Lap Xenium 8um lambda0 vs auto r/RMSE", "Xenium CRC lambda ablation", "8 um r/RMSE lambda0 vs auto", "", ""),
        ("Lap Xenium 16um lambda0 vs auto r/RMSE", "Xenium CRC lambda ablation", "16 um r/RMSE lambda0 vs auto", "", ""),
    ]:
        ps.append(dict(benchmark=bench, metric=metric, manuscript_value=t.loc[claim, "manuscript"],
                       v020_value=t.loc[claim, "V020"], final_value=t.loc[claim, "FINAL"],
                       rank=rank, p_value="", notes=note))
    pd.DataFrame(ps).to_csv(FIN_DIR / "xenium/part_summary_xenium.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(out[["claim", "manuscript", "ORIG", "V020", "FINAL"]].to_string())


if __name__ == "__main__":
    main()
