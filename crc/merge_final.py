"""Merge per-patient CRC FINAL outputs into one claim table (ORIG / V020 / FINAL).
Copy of validation/rerun_v020/crc/merge_v020.py: claims are computed for ORIG and
FINAL from the final-run tables; the V020 column is taken from
results/rerun_v020/crc/claims_old_vs_new.csv (same code, same claim names).

Usage: python merge_final.py <results_dir>
Writes claims_final.csv, type_means_final.csv, fit_diagnostics_final.csv.
The first block reuses validation/crc_seed_stability/merge.py verbatim in logic.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/rerun_final/crc")
V020_CLAIMS = Path("/Users/apple/Research/FlashDeconv/results/rerun_v020/crc/claims_old_vs_new.csv")
SAMPLES = ["P1_CRC", "P2_CRC", "P5_CRC"]
FITS = ["ORIG", "FINAL"]
IMMUNE8 = ["Neutrophil", "Macrophage", "mRegDC", "Mast", "CD8 T cell", "CD4 T cell", "Plasma", "Mature B"]


def load(name):
    return pd.concat([pd.read_csv(R / s / f"{name}.csv") for s in SAMPLES], ignore_index=True)


def rng3(v, fmt="{:.1f}"):
    return "/".join(fmt.format(x) for x in v)


def rng(v, fmt="{:.1f}"):
    v = np.asarray(v, dtype=float)
    return f"{fmt.format(np.nanmin(v))}-{fmt.format(np.nanmax(v))}"


cmp_, cmpt = load("compare_summary"), load("compare_types")
knn, agg, mk, lr = load("knn_enrichment"), load("aggregates"), load("markers"), load("lr")
rc, lm, bd, tm = load("rctd"), load("lineage_markers"), load("boundary"), load("type_means")
fd = load("fit_diagnostics")
cls = load("rctd_class_fractions")

rows = {}
for f in FITS:
    c = {}
    h = rc[rc.fit == f].set_index("sample").loc[SAMPLES]
    c["hotspot bins P1/P2/P5 (5295/8305/3227)"] = rng3(h.n_hotspot, "{:.0f}")
    c["hotspot bins total (16,827)"] = int(h.n_hotspot.sum())
    n_bins = tm[(tm.fit == f)].groupby("sample").size()  # placeholder for ordering
    c["RCTD Neutrophil singlet % (2.3)"] = round(100 * h.neut_singlet_label2.sum() / h.n_hotspot.sum(), 2)
    c["RCTD withheld % (reject+NA; 61)"] = round(100 * (h.reject.sum() + h.NA.sum()) / h.n_hotspot.sum(), 1)
    # self-enrichment
    k = knn[(knn.fit == f) & (knn.threshold == 0.10) & (knn.focal_type == knn.neighbor_type)]
    ns = k[k.focal_type == "Neutrophil"].set_index("sample").loc[SAMPLES].enrichment_ratio
    c["Neutrophil self-enrichment x (16.6/22.8/56.2)"] = rng3(ns)
    c["Neutrophil self-enrichment log2 (4.1/4.5/5.8)"] = rng3(np.log2(ns))
    top = k.loc[k.groupby("sample").enrichment_ratio.idxmax()].set_index("sample").loc[SAMPLES]
    c["top self-enriched of all 38 types per patient"] = "/".join(
        f"{t} ({v:.1f}x)" for t, v in zip(top.focal_type, top.enrichment_ratio))
    ranks = []
    for s in SAMPLES:
        ks = k[k["sample"] == s].sort_values("enrichment_ratio", ascending=False).reset_index(drop=True)
        ranks.append(int(ks.index[ks.focal_type == "Neutrophil"][0]) + 1)
    c["Neutrophil self-enrichment rank among types (1=top)"] = "/".join(map(str, ranks)) + \
        f" of {k.groupby('sample').size().loc[SAMPLES].astype(str).str.cat(sep='/')}"
    ki = k[k.focal_type.isin(IMMUNE8)]
    topi = ki.loc[ki.groupby("sample").enrichment_ratio.idxmax()].set_index("sample").loc[SAMPLES]
    c["top self-enriched among 8 immune types"] = "/".join(topi.focal_type)
    kn = knn[(knn.fit == f) & (knn.focal_type == "Neutrophil")]
    for thr in [0.10, 0.02]:
        for nt in ["mRegDC", "Macrophage", "Mast", "Endothelial"]:
            v = kn[(kn.threshold == thr) & (kn.neighbor_type == nt)].set_index("sample").loc[SAMPLES]
            c[f"kNN Neut(>{thr:g})->{nt} x"] = rng3(v.enrichment_ratio)
    for nt, old in [("Goblet", "0.03-0.1"), ("Enterocyte", "0.2-0.3")]:
        v = kn[(kn.threshold == 0.10) & (kn.neighbor_type == nt)].set_index("sample").loc[SAMPLES]
        c[f"kNN Neut->{nt} x (SN9 {old})"] = rng3(v.enrichment_ratio, "{:.2f}")
    tumc = [t for t in kn.neighbor_type.unique() if t.startswith("Tumor ")]
    v = kn[(kn.threshold == 0.10) & kn.neighbor_type.isin(tumc)]
    c["kNN Neut->Tumor subtypes x range (SN9 0.2-0.7)"] = rng(v.enrichment_ratio, "{:.2f}")
    # aggregates
    a = agg[agg.fit == f]
    c["aggregates total (72)"] = len(a)
    c["stromal-resident / tumor-proximal (25/47)"] = (
        f"{(a.location == 'stromal-resident').sum()}/{(a.location == 'tumor-proximal').sum()}")
    c["aggregates per patient SR/TP"] = " ".join(
        f"{s[:2]}:{((a['sample'] == s) & (a.location == 'stromal-resident')).sum()}/"
        f"{((a['sample'] == s) & (a.location == 'tumor-proximal')).sum()}" for s in SAMPLES)
    old_sr = {"Macrophage": "+0.8..+1.7", "mRegDC": "+1.4..+2.1", "Mast": "+0.7..+1.9",
              "Endothelial": "+0.5..+0.6", "CD8 T cell": "+0.4..+0.8"}
    for ct in ["mRegDC", "Macrophage", "Mast", "Endothelial", "CD8 T cell"]:
        col = f"niche_log2_{ct}"
        sr = a[a.location == "stromal-resident"].groupby("sample")[col].median().reindex(SAMPLES)
        tp = a[a.location == "tumor-proximal"].groupby("sample")[col].median().reindex(SAMPLES)
        c[f"SR median log2 {ct} P1/P2/P5 (SN9 {old_sr[ct]})"] = rng3(sr, "{:+.2f}")
        c[f"TP median log2 {ct} P1/P2/P5"] = rng3(tp, "{:+.2f}")
        c[f"pooled median log2 {ct} SR | TP"] = (
            f"{a[a.location == 'stromal-resident'][col].median():+.2f} | "
            f"{a[a.location == 'tumor-proximal'][col].median():+.2f}")
    srv = a[a.location == "stromal-resident"]["niche_log2_mRegDC"].dropna()
    tpv = a[a.location == "tumor-proximal"]["niche_log2_mRegDC"].dropna()
    c["Wilcoxon SR mRegDC log2>0 p"] = (f"{stats.wilcoxon(srv, alternative='greater').pvalue:.1e}"
                                         if len(srv) > 1 else "NA")
    c["MWU SR>TP mRegDC p"] = (f"{stats.mannwhitneyu(srv, tpv, alternative='greater').pvalue:.1e}"
                                if len(srv) and len(tpv) else "NA")
    for ct in ["Macrophage", "CD8 T cell", "Mast"]:
        s_ = a[a.location == "stromal-resident"][f"niche_log2_{ct}"].dropna()
        t_ = a[a.location == "tumor-proximal"][f"niche_log2_{ct}"].dropna()
        c[f"MWU SR>TP {ct} p"] = f"{stats.mannwhitneyu(s_, t_, alternative='greater').pvalue:.1e}"
    # markers
    m = mk[(mk.fit == f) & (mk.marker_class == "neutrophil")]
    c["neutrophil marker FC range (11-63)"] = f"{m.fold_change.min():.0f}-{m.fold_change.max():.0f}"
    for g, old in [("S100A8", "16-59"), ("S100A9", "13-48"), ("FCGR3B", "11-23"), ("CSF3R", "12-63")]:
        c[f"{g} FC ({old})"] = rng(m[m.gene == g].fold_change, "{:.0f}")
    c["CXCR1/CXCR2 FC (16-47)"] = rng(m[m.gene.isin(["CXCR1", "CXCR2"])].fold_change, "{:.0f}")
    neg = mk[(mk.fit == f) & (mk.marker_class == "negative")]
    c["negative-control FC range (<=1 expected)"] = rng(neg.fold_change, "{:.2f}")
    amp = (m.fold_change_high_umi > m.fold_change).mean()
    c["frac marker x patient amplified at high UMI (all)"] = round(amp, 2)
    c["hotspot mean UMI (79-197) | background (209-494)"] = (
        f"{rng(m.groupby('sample').mean_umi_hotspot.first(), '{:.0f}')} | "
        f"{rng(m.groupby('sample').mean_umi_background.first(), '{:.0f}')}")
    # LR
    l = lr[(lr.fit == f) & (lr.gene == "LAMP3")].set_index("sample").loc[SAMPLES]
    c["LAMP3 neighborhood fold (1.40; 1.13-1.67)"] = (
        f"{l.fold_neighborhood_vs_bg.median():.2f} ({rng3(l.fold_neighborhood_vs_bg, '{:.2f}')})")
    c["LAMP3 MWU p per patient"] = rng3(l.p_neigh_vs_bg, "{:.1e}")
    for g, old in [("IDO1", 1.68), ("PECAM1", 1.46), ("CD163", 1.34)]:
        v = lr[(lr.fit == f) & (lr.gene == g)].set_index("sample").loc[SAMPLES]
        c[f"{g} neighborhood fold median ({old})"] = (
            f"{v.fold_neighborhood_vs_bg.median():.2f} (p max {v.p_neigh_vs_bg.max():.1e})")
    s89 = lr[(lr.fit == f) & lr.gene.isin(["S100A8", "S100A9"])]
    c["S100A8/9 fold within hotspots (25-84)"] = rng(s89.fold_hotspot_vs_bg, "{:.0f}")
    # lineage markers
    L = lm[lm.fit == f]
    nd = L[L.category.str.startswith("RCTD_")].groupby(["sample", "category"]).n_bins.first().sum()
    c["disputed immune/stromal bins (71,769)"] = int(nd)
    piv = L.groupby(["gene", "category"]).mean_expression.mean().unstack()
    ok, tot, ok_rsfi = 0, 0, 0
    for g, r in piv.iterrows():
        if not np.isnan(r.get("RCTD_Immune_FD_Stromal", np.nan)):
            tot += 1
            ok += abs(r.RCTD_Immune_FD_Stromal - r.Agreed_Stromal) < abs(r.RCTD_Immune_FD_Stromal - r.Agreed_Immune)
        if not np.isnan(r.get("RCTD_Stromal_FD_Immune", np.nan)):
            tot += 1
            hit = abs(r.RCTD_Stromal_FD_Immune - r.Agreed_Immune) < abs(r.RCTD_Stromal_FD_Immune - r.Agreed_Stromal)
            ok += hit
            ok_rsfi += hit
    c["marker verdicts FD correct (19/22)"] = f"{ok}/{tot} ({100 * ok / tot:.1f}%)"
    c["RS->FI verdicts FD correct (11/11)"] = f"{ok_rsfi}/{len(piv)}"
    # boundary
    B = bd[bd.fit == f].groupby(["distance_min_um", "cell_type"]).mean_proportion.mean().unstack()
    tum = B[[t for t in B.columns if t.startswith("Tumor ")]].sum(1)
    c["tumor % at -25um / +75um (94/14)"] = f"{100 * tum.loc[-50]:.0f}/{100 * tum.loc[50]:.0f}"
    c["Plasma max % (16) [band]"] = f"{100 * B['Plasma'].max():.1f} [{B['Plasma'].idxmax()}]"
    c["CAF peak % (14) [band; claim 50-100um]"] = f"{100 * B['CAF'].max():.1f} [{B['CAF'].idxmax()}]"
    c["CAF % at 50-100um"] = round(100 * B.loc[50, "CAF"], 1)
    c["Macrophage max % (2)"] = round(100 * B["Macrophage"].max(), 1)
    c["CD8 % at 200-500um (0.8)"] = round(100 * B.loc[200, "CD8 T cell"], 2)
    imm = [t for t in B.columns if t in ("Macrophage", "Proliferating Macrophages", "CD8 T cell", "CD4 T cell",
                                          "NK", "Plasma", "Mature B", "Memory B", "Neutrophil", "Mast", "pDC",
                                          "mRegDC", "cDC I", "Proliferating Immune II")]
    ti = B[imm].sum(1)
    c["total immune % by band (-200,-100,-50,0,50,100,200)"] = "/".join(
        f"{100 * ti.loc[b]:.1f}" for b in [-200, -100, -50, 0, 50, 100, 200] if b in ti.index)
    rows[f] = c

tab = pd.DataFrame(rows)
# method-independent / fit-level rows
v = fd[(fd.fit == "FINAL") & (fd.resolution_um == 8)].set_index("sample").loc[SAMPLES]
extra = {
    "V020 fit_transform seconds P1/P2/P5": rng3(v.fit_seconds),
    "V020 total seconds (153) / bins per s (10,400)": f"{v.fit_seconds.sum():.1f} / {v.n_bins.sum() / v.fit_seconds.sum():.0f}",
    "V020 bins total (1,595,565)": int(v.n_bins.sum()),
    "V020 converged / iterations / lambda": " ".join(f"{a}/{b}/{c_:.2f}" for a, b, c_ in
                                                      zip(v.converged, v.n_iterations, v.lambda_used)),
    "V020 deterministic refit identical": "/".join(map(str, v.refit_identical)),
    "V020 host / cpus": f"{v.host.iloc[0]} / {v.cpus.iloc[0]} ({v.get('cpu_model', pd.Series(['?'])).iloc[0]})",
}
cp = cmp_[(cmp_.fit == "FINAL") & (cmp_.ref == "ORIG")].set_index("sample").loc[SAMPLES]
ct = cmpt[(cmpt.fit == "FINAL") & (cmpt.ref == "ORIG")]
extra["V020 vs ORIG hotspot Jaccard"] = rng3(cp.hotspot_jaccard, "{:.2f}")
extra["V020 vs ORIG median per-bin JSD"] = rng3(cp.median_jsd, "{:.3f}")
extra["V020 vs ORIG dominant-type agreement"] = rng3(cp.dominant_agreement, "{:.2f}")
extra["V020 vs ORIG per-bin r Neutrophil"] = rng3(ct[ct.cell_type == "Neutrophil"].set_index("sample").loc[SAMPLES].pearson, "{:.2f}")
extra["V020 vs ORIG per-bin r mRegDC"] = rng3(ct[ct.cell_type == "mRegDC"].set_index("sample").loc[SAMPLES].pearson, "{:.2f}")
extra["V020 vs ORIG median per-type per-bin r"] = rng3(ct.groupby("sample").pearson.median().loc[SAMPLES], "{:.2f}")
cf = cls.pivot_table(index="sample", columns="index", values="fraction").loc[SAMPLES]
extra["RCTD singlet % per patient (46-59)"] = rng3(100 * cf.get("singlet", np.nan), "{:.1f}")
extra["RCTD reject % per patient (5-7)"] = rng3(100 * cf.get("reject", np.nan), "{:.1f}")
tab = pd.concat([tab, pd.DataFrame({"ORIG": {k: "" for k in extra}, "FINAL": extra})])

# multi-resolution niche
mr_new = pd.concat([pd.read_csv(R / s / "multires_enrichment.csv") for s in SAMPLES
                    if (R / s / "multires_enrichment.csv").exists()], ignore_index=True)
old_mr = Path("/Users/apple/Research/FlashDeconv/analysis/crc_cohort_results/neutrophil_multiresolution_enrichment.csv")
if len(mr_new):
    mr_old = pd.read_csv(old_mr)
    for nt in ["Neutrophil", "Macrophage", "mRegDC", "Mast", "Endothelial"]:
        for lab, df in [("ORIG", mr_old), ("FINAL", mr_new)]:
            s = []
            for smp in SAMPLES:
                d = df[df["sample"] == smp].set_index("resolution_um")
                s.append(smp[:2] + ":" + ",".join(
                    f"{d.loc[r, f'ratio_{nt}']:.1f}" if r in d.index and pd.notna(d.loc[r].get(f"ratio_{nt}", np.nan))
                    else "NA" for r in [8, 16, 32, 64]))
            tab.loc[f"multires Neut-> {nt} x (8/16/32/64um)", lab] = " ".join(s)
    for lab, df in [("ORIG", mr_old), ("FINAL", mr_new)]:
        tab.loc["multires n_hot (8/16/32/64um)", lab] = " ".join(
            smp[:2] + ":" + ",".join(str(int(x)) for x in df[df["sample"] == smp].sort_values("resolution_um").n_hot)
            for smp in SAMPLES)
        tab.loc["multires n_bins (8/16/32/64um)", lab] = " ".join(
            smp[:2] + ":" + ",".join(str(int(x)) for x in df[df["sample"] == smp].sort_values("resolution_um")
                                     [("n_total" if "n_total" in df else "n_bins")]) for smp in SAMPLES)

# rename fit-level rows "V020 ..." -> "fit ..." so V020 and FINAL values share one row
tab.index = [i.replace("V020 vs ORIG", "fit vs ORIG").replace("V020 ", "fit ") if i.startswith("V020") else i
             for i in tab.index]
old = pd.read_csv(V020_CLAIMS, index_col=0)
old.index = [i.replace("V020 vs ORIG", "fit vs ORIG").replace("V020 ", "fit ") if i.startswith("V020") else i
             for i in old.index]
tab["V020"] = old["V020"].reindex(tab.index)
# FINAL vs V020 agreement rows
cp2 = cmp_[(cmp_.fit == "FINAL") & (cmp_.ref == "V020")].set_index("sample").loc[SAMPLES]
ct2 = cmpt[(cmpt.fit == "FINAL") & (cmpt.ref == "V020")]
tab.loc["FINAL vs V020 hotspot Jaccard", "FINAL"] = rng3(cp2.hotspot_jaccard, "{:.2f}")
tab.loc["FINAL vs V020 median per-bin JSD", "FINAL"] = rng3(cp2.median_jsd, "{:.3f}")
tab.loc["FINAL vs V020 dominant-type agreement", "FINAL"] = rng3(cp2.dominant_agreement, "{:.2f}")
tab.loc["FINAL vs V020 per-bin r Neutrophil", "FINAL"] = rng3(
    ct2[ct2.cell_type == "Neutrophil"].set_index("sample").loc[SAMPLES].pearson, "{:.2f}")
tab.loc["FINAL vs V020 median per-type per-bin r", "FINAL"] = rng3(ct2.groupby("sample").pearson.median().loc[SAMPLES], "{:.2f}")
mres = fd[(fd.fit == "FINAL") & (fd.resolution_um > 8)]
tab.loc["fit converged / iterations multires 16/32/64um", "FINAL"] = " ".join(
    f"{s[:2]}:" + ",".join(f"{a}/{b}" for a, b in zip(g.sort_values("resolution_um").converged,
                                                        g.sort_values("resolution_um").n_iterations))
    for s, g in mres.groupby("sample"))
tab["note"] = ""
for i in tab.index:
    if i.startswith("multires"):
        tab.loc[i, "note"] = ("definition changed: ORIG archived multires bins were built on a pixel grid scaled by a "
                              "subsample median-NN distance (2.24 bins), i.e. effective 36/72/143 um instead of 16/32/64 um; "
                              "V020 and FINAL use array_row/array_col integer division (identical to Space Ranger square_016um)")
    elif i.startswith("fit ") or i.startswith("FINAL vs"):
        tab.loc[i, "note"] = "fit-level diagnostic (V020: max_iter=100; FINAL: max_iter=1000, early stop tol=1e-4)"
tab = tab[["ORIG", "V020", "FINAL", "note"]]
tab.index.name = "claim"
tab.to_csv(R / "claims_final.csv")
fd.to_csv(R / "fit_diagnostics_final.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.max_rows", 300)
print(tab.to_string())

# per-type means old vs new
p = tm.pivot_table(index=["sample", "cell_type"], columns="fit", values="mean_proportion")
vt = pd.read_csv("/Users/apple/Research/FlashDeconv/results/rerun_v020/crc/type_means_old_vs_new.csv").set_index(["sample", "cell_type"])
p["V020"] = vt["V020"].reindex(p.index)
p = p[["ORIG", "V020", "FINAL"]]
p["diff_pp"] = 100 * (p.FINAL - p.ORIG)
p["diff_pp_vs_V020"] = 100 * (p.FINAL - p.V020)
p.to_csv(R / "type_means_final.csv")
print("\nPer-type mean proportion change (pp):")
print(p.diff_pp.describe().round(3).to_string())
print(p.reindex(p.diff_pp.abs().sort_values(ascending=False).index).head(10).round(4).to_string())
