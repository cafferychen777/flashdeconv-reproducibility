"""
TCGA-COAD survival analysis using cBioPortal API.

Fetches gene expression and clinical data for specific genes
via cBioPortal REST API, then performs survival analysis.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json
import urllib.request

warnings.filterwarnings("ignore")

BASE = Path("/scratch/user/cafferychen777/FlashDeconv/analysis/cross_cohort_evidence")
DATA = BASE / "data" / "tcga"
RESULTS = BASE / "tcga_survival" / "results"
DATA.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

STUDY_ID = "coadread_tcga_pan_can_atlas_2018"
PROFILE_ID = f"{STUDY_ID}_rna_seq_v2_mrna"

GENES = [
    "S100A8", "S100A9", "LAMP3", "CCR7", "CD274", "IDO1",
    "FCGR3B", "CSF3R", "CXCR1", "CXCR2",
    "CD68", "CD163", "PECAM1", "KIT",
    "AGER", "TLR4",
]

NEUTROPHIL_SIG = ["S100A8", "S100A9", "FCGR3B", "CSF3R", "CXCR1", "CXCR2"]
MREGDC_SIG = ["LAMP3", "CCR7", "CD274", "IDO1"]


def cbio_api(endpoint, params=None):
    """Call cBioPortal API."""
    base_url = "https://www.cbioportal.org/api"
    url = f"{base_url}/{endpoint}"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    return json.loads(resp.read())


def cbio_post(endpoint, data):
    """POST to cBioPortal API."""
    base_url = "https://www.cbioportal.org/api"
    url = f"{base_url}/{endpoint}"
    payload = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    return json.loads(resp.read())


def fetch_expression():
    """Fetch gene expression from cBioPortal."""
    cache_file = DATA / "cbio_expression.json"
    if cache_file.exists():
        print("  Loading cached expression data...")
        with open(cache_file) as f:
            return json.load(f)

    print(f"  Fetching expression for {len(GENES)} genes from cBioPortal...")

    # Get available profiles
    profiles = cbio_api(f"studies/{STUDY_ID}/molecular-profiles")
    mrna_profiles = [p for p in profiles if "mrna" in p["molecularProfileId"].lower()
                     or "rna_seq" in p["molecularProfileId"].lower()]
    print(f"  Available mRNA profiles: {[p['molecularProfileId'] for p in mrna_profiles]}")

    # Use the first mRNA profile
    if mrna_profiles:
        profile_id = mrna_profiles[0]["molecularProfileId"]
    else:
        profile_id = PROFILE_ID
    print(f"  Using profile: {profile_id}")

    # Get all sample IDs
    samples = cbio_api(f"studies/{STUDY_ID}/samples")
    sample_ids = [s["sampleId"] for s in samples]
    print(f"  Total samples: {len(sample_ids)}")

    # Fetch gene expression
    # cBioPortal gene query uses Hugo symbols
    gene_str = ",".join(GENES)

    try:
        # Try fetching via the molecular data endpoint
        expr_data = cbio_post(
            f"molecular-profiles/{profile_id}/molecular-data/fetch",
            {
                "sampleIds": sample_ids[:500],
                "entrezGeneIds": [],
                "sampleListId": f"{STUDY_ID}_all",
            }
        )
    except Exception as e1:
        print(f"  Fetch method 1 failed: {e1}")
        # Alternative: fetch gene by gene
        expr_data = []
        for gene in GENES:
            try:
                gene_data = cbio_api(
                    f"molecular-profiles/{profile_id}/molecular-data",
                    {"sampleListId": f"{STUDY_ID}_all",
                     "entrezGeneId": gene})
                expr_data.extend(gene_data)
            except Exception as e2:
                print(f"    Gene {gene} failed: {e2}")

    # Cache
    with open(cache_file, "w") as f:
        json.dump(expr_data, f)
    print(f"  Got {len(expr_data)} data points")

    return expr_data


def fetch_clinical():
    """Fetch clinical/survival data from cBioPortal."""
    cache_file = DATA / "cbio_clinical_survival.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    print("  Fetching clinical data...")
    clinical = cbio_api(
        f"studies/{STUDY_ID}/clinical-data",
        {"clinicalDataType": "PATIENT", "projection": "DETAILED"})
    print(f"  Got {len(clinical)} clinical records")

    with open(cache_file, "w") as f:
        json.dump(clinical, f)

    return clinical


def build_matrices(expr_data, clinical_data):
    """Build expression and survival matrices."""
    # Expression matrix: sample x gene
    expr_records = {}
    for record in expr_data:
        sample = record.get("sampleId", "")
        gene = record.get("hugoGeneSymbol", "")
        value = record.get("value", None)
        if sample and gene and value is not None:
            if sample not in expr_records:
                expr_records[sample] = {}
            expr_records[sample][gene] = value

    expr_df = pd.DataFrame.from_dict(expr_records, orient="index")
    print(f"  Expression matrix: {expr_df.shape}")

    # Clinical: extract survival columns
    clin_records = {}
    for record in clinical_data:
        patient = record.get("patientId", "")
        attr = record.get("clinicalAttributeId", "")
        value = record.get("value", "")
        if patient not in clin_records:
            clin_records[patient] = {}
        clin_records[patient][attr] = value

    clin_df = pd.DataFrame.from_dict(clin_records, orient="index")
    print(f"  Clinical matrix: {clin_df.shape}")
    print(f"  Clinical columns: {sorted(clin_df.columns.tolist())[:20]}")

    return expr_df, clin_df


def compute_signatures(expr_df):
    """Compute signature scores."""
    scores = pd.DataFrame(index=expr_df.index)

    for name, genes in [("neutrophil", NEUTROPHIL_SIG), ("mregdc", MREGDC_SIG)]:
        available = [g for g in genes if g in expr_df.columns]
        if available:
            sub = expr_df[available]
            z = (sub - sub.mean()) / (sub.std() + 1e-10)
            scores[name] = z.mean(axis=1)
            print(f"  {name} signature: {len(available)}/{len(genes)} genes")

    # Individual genes
    for gene in GENES:
        if gene in expr_df.columns:
            scores[gene] = expr_df[gene]

    return scores


def survival_analysis(scores, clin_df):
    """Kaplan-Meier analysis."""
    # Find survival columns
    os_col = None
    event_col = None
    for c in clin_df.columns:
        if c in ["OS_MONTHS", "OS_STATUS", "OVERALL_SURVIVAL_MONTHS",
                  "OVERALL_SURVIVAL_STATUS"]:
            if "MONTH" in c or "TIME" in c:
                os_col = c
            elif "STATUS" in c:
                event_col = c

    if os_col is None or event_col is None:
        print("  Cannot find OS columns")
        print(f"  Available: {sorted(clin_df.columns.tolist())}")
        return None

    print(f"  Survival columns: time={os_col}, event={event_col}")

    # Map patient IDs to sample IDs
    # cBioPortal: sample IDs are usually patient-01, patient-06, etc.
    scores["patient"] = scores.index.str.rsplit("-", n=1).str[0]
    clin_df["patient"] = clin_df.index

    merged = scores.merge(clin_df[["patient", os_col, event_col]],
                          on="patient", how="inner")
    merged[os_col] = pd.to_numeric(merged[os_col], errors="coerce")

    # Convert status
    if merged[event_col].dtype == object:
        merged["event"] = merged[event_col].str.contains(
            "DECEASED|dead|1", case=False, na=False).astype(int)
    else:
        merged["event"] = merged[event_col].astype(int)

    merged = merged.dropna(subset=[os_col])
    print(f"  Merged: {len(merged)} patients with survival data")

    results = []
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        has_lifelines = True
    except ImportError:
        has_lifelines = False
        print("  lifelines not available, using manual log-rank approximation")

    for score_name in ["neutrophil", "mregdc", "S100A8", "S100A9", "LAMP3",
                        "CCR7", "CD274"]:
        if score_name not in merged.columns:
            continue
        col = merged[score_name].dropna()
        if len(col) < 20:
            continue

        median_val = col.median()
        high = merged[merged[score_name] >= median_val]
        low = merged[merged[score_name] < median_val]

        res = {
            "score": score_name,
            "n_total": len(merged),
            "n_high": len(high),
            "n_low": len(low),
        }

        if has_lifelines:
            lr = logrank_test(
                high[os_col], low[os_col],
                high["event"], low["event"])
            res["logrank_p"] = float(lr.p_value)
            res["logrank_stat"] = float(lr.test_statistic)

            kmf_h = KaplanMeierFitter()
            kmf_h.fit(high[os_col], high["event"])
            kmf_l = KaplanMeierFitter()
            kmf_l.fit(low[os_col], low["event"])
            res["median_surv_high"] = float(kmf_h.median_survival_time_)
            res["median_surv_low"] = float(kmf_l.median_survival_time_)
        else:
            rate_h = high["event"].sum() / (high[os_col].sum() + 1)
            rate_l = low["event"].sum() / (low[os_col].sum() + 1)
            res["hazard_rate_high"] = float(rate_h)
            res["hazard_rate_low"] = float(rate_l)
            res["hazard_ratio"] = float(rate_h / (rate_l + 1e-10))

        results.append(res)
        if "logrank_p" in res:
            print(f"  {score_name}: p={res['logrank_p']:.4f}, "
                  f"median_high={res['median_surv_high']:.0f}, "
                  f"median_low={res['median_surv_low']:.0f}")
        elif "hazard_ratio" in res:
            print(f"  {score_name}: HR={res['hazard_ratio']:.2f}")

    return pd.DataFrame(results)


def correlation_analysis(scores):
    """Correlate neutrophil and mRegDC signatures."""
    results = {}
    pairs = [
        ("neutrophil", "mregdc"),
        ("S100A8", "LAMP3"),
        ("S100A9", "LAMP3"),
        ("S100A8", "CCR7"),
        ("S100A9", "CD274"),
    ]
    print("\n=== Signature correlations ===")
    for g1, g2 in pairs:
        if g1 in scores.columns and g2 in scores.columns:
            x = scores[g1].dropna()
            y = scores[g2].dropna()
            common = x.index.intersection(y.index)
            if len(common) >= 10:
                r, p = stats.spearmanr(x[common], y[common])
                results[f"{g1}_vs_{g2}"] = {"rho": float(r), "p": float(p), "n": len(common)}
                print(f"  {g1} vs {g2}: rho={r:.3f}, p={p:.2e} (n={len(common)})")
    return results


def main():
    print("=" * 60)
    print("TCGA-COAD Survival Analysis (via cBioPortal API)")
    print("=" * 60)

    expr_data = fetch_expression()
    clinical_data = fetch_clinical()

    expr_df, clin_df = build_matrices(expr_data, clinical_data)

    scores = compute_signatures(expr_df)
    scores.to_csv(RESULTS / "signature_scores.csv")

    # Correlations
    corr = correlation_analysis(scores)
    with open(RESULTS / "correlation_results.json", "w") as f:
        json.dump(corr, f, indent=2)

    # Survival
    surv_results = survival_analysis(scores, clin_df)
    if surv_results is not None and len(surv_results) > 0:
        surv_results.to_csv(RESULTS / "survival_results.csv", index=False)
        print(f"\nSaved: {RESULTS / 'survival_results.csv'}")

    print("\n✓ TCGA analysis complete.")


if __name__ == "__main__":
    main()
