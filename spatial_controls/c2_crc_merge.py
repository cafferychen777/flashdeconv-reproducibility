"""Control 2a merge: manuscript CRC claims for the default fit (FINAL, results/rerun_final/crc)
and the lambda_spatial=0 refit (LAM0, results/controls_editor/crc). The claim block is copied
verbatim from validation/rerun_final/crc/merge_final.py (only FITS and the loader differ)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RF = Path("/Users/apple/Research/FlashDeconv/results/rerun_final/crc")
RC = Path("/Users/apple/Research/FlashDeconv/results/controls_editor/crc")
SAMPLES = ["P1_CRC", "P2_CRC", "P5_CRC"]
FITS = ["FINAL", "LAM0"]
IMMUNE8 = ["Neutrophil", "Macrophage", "mRegDC", "Mast", "CD8 T cell", "CD4 T cell", "Plasma", "Mature B"]


def load(name):
    a = pd.concat([pd.read_csv(RF / s / f"{name}.csv") for s in SAMPLES], ignore_index=True)
    b = pd.concat([pd.read_csv(RC / s / f"{name}.csv") for s in SAMPLES], ignore_index=True)
    return pd.concat([a[a.fit == "FINAL"], b], ignore_index=True)


def rng3(v, fmt="{:.1f}"):
    return "/".join(fmt.format(x) for x in v)


def rng(v, fmt="{:.1f}"):
    v = np.asarray(v, dtype=float)
    return f"{fmt.format(np.nanmin(v))}-{fmt.format(np.nanmax(v))}"


knn, agg, mk, lr = load("knn_enrichment"), load("aggregates"), load("markers"), load("lr")
rc, lm, bd, tm = load("rctd"), load("lineage_markers"), load("boundary"), load("type_means")
rows = {}
for f in FITS:
    c = {}
    h = rc[rc.fit == f].set_index("sample").loc[SAMPLES]
    c["hotspot bins P1/P2/P5 (5295/8305/3227)"] = rng3(h.n_hotspot, "{:.0f}")
    c["hotspot bins total (16,827)"] = int(h.n_hotspot.sum())
    n_bins = tm[(tm.fit == f)].groupby("sample").size()  # placeholder for ordering
    # RCTD singlet call = DeconvolutionLabel1 (Label2 is the runner-up type); NA = bin not scored by RCTD
    n_h = h.n_hotspot.sum()
    c["RCTD Neutrophil singlet % of hotspot bins"] = round(100 * h.neut_singlet_label1.sum() / n_h, 1)
    c["RCTD Neutrophil % of singlet-called hotspot bins"] = round(100 * h.neut_singlet_label1.sum() / h.singlet.sum(), 1)
    c["RCTD Neutrophil % of singlet-called hotspot bins P1/P2/P5"] = rng3(100 * h.neut_singlet_label1 / h.singlet)
    c["RCTD Neutrophil-containing doublet % of hotspot bins"] = round(100 * h.neut_doublet.sum() / n_h, 1)
    c["RCTD singlet / doublet / reject / no call (NA) % of hotspot bins"] = "/".join(
        f"{100 * h[k].sum() / n_h:.1f}" for k in ["singlet", "doublet", "reject", "NA"])
    c["RCTD withheld % (reject+NA; 61)"] = round(100 * (h.reject.sum() + h.NA.sum()) / n_h, 1)
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
cs = pd.concat([pd.read_csv(RC / s / "compare_summary.csv") for s in SAMPLES]).set_index("sample").loc[SAMPLES]
ct = pd.concat([pd.read_csv(RC / s / "compare_types.csv") for s in SAMPLES])
tab.loc["LAM0 vs FINAL hotspot Jaccard", "LAM0"] = rng3(cs.hotspot_jaccard, "{:.2f}")
tab.loc["LAM0 vs FINAL median per-bin JSD", "LAM0"] = rng3(cs.median_jsd, "{:.3f}")
tab.loc["LAM0 vs FINAL dominant-type agreement", "LAM0"] = rng3(cs.dominant_agreement, "{:.2f}")
tab.loc["LAM0 vs FINAL per-bin r Neutrophil", "LAM0"] = rng3(
    ct[ct.cell_type == "Neutrophil"].set_index("sample").loc[SAMPLES].pearson, "{:.2f}")
tab.loc["LAM0 vs FINAL median per-type per-bin r", "LAM0"] = rng3(ct.groupby("sample").pearson.median().loc[SAMPLES], "{:.2f}")
ac = pd.concat([pd.read_csv(RC / s / "autocorr.csv") for s in SAMPLES])
for f in FITS:
    tab.loc["median kNN-6 autocorrelation over 38 types P1/P2/P5", f] = rng3(
        ac[ac.fit == f].groupby("sample").knn6_autocorr.median().loc[SAMPLES], "{:.2f}")
    tab.loc["Neutrophil kNN-6 autocorrelation P1/P2/P5", f] = rng3(
        ac[(ac.fit == f) & (ac.cell_type == "Neutrophil")].set_index("sample").loc[SAMPLES].knn6_autocorr, "{:.2f}")
fdg = pd.concat([pd.read_csv(RC / s / "fit_diagnostics.csv") for s in SAMPLES])
tab.loc["LAM0 fit converged/iter/lambda", "LAM0"] = " ".join(
    f"{a}/{b}/{c:.2f}" for a, b, c in zip(fdg.converged, fdg.n_iterations, fdg.lambda_used))
tab.index.name = "claim"
tab.to_csv(RC.parent / "c2_crc_claims_default_vs_lam0.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90); pd.set_option("display.max_rows", 300)
print(tab.to_string())
