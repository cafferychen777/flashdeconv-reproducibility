"""S1 step 3: external scRNA-seq composite mouse small-intestine reference for S1 (reference (a)).

Identical construction to validation/intestine_reference_v2/a1_build_reference.py (same public
sources, same seeds, same k-means + ordered-marker-rule annotation of the non-epithelial cells,
3,000-cell cap per type), except that genes are restricted to those present in every source and in
the 1,815-gene MERFISH panel (instead of the Visium HD probe set), and labels are harmonised to the
S1 coarse taxonomy (HARMONISE below). The S1 types ICC and Mesothelium have no counterpart in these
sources; they are absent from this reference (see the S1 evaluation for how this is handled).

Usage: python s03_build_extref.py <merfish_counts.h5ad> <out_dir>
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io, sparse
from sklearn.cluster import KMeans

import sys
HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "intestine_reference_v2" / "data_ref"
ROOT = HERE.parents[1]
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
CAP = 3000
rng = np.random.default_rng(0)


MARKERS = ["Ptprc", "Cd79a", "Ms4a1", "Aicda", "Mki67", "Jchain", "Xbp1", "Cd3e", "Il7r", "Nkg7",
           "Lyz2", "C1qa", "Cpa3", "Itgax", "Epcam", "Pdgfra", "Col1a1", "Dcn", "Acta2", "Myh11",
           "Des", "Rgs5", "Pdgfrb", "Pecam1", "Cdh5", "Lyve1", "Plp1", "Sox10", "S100b", "Elavl4",
           "Snap25", "Phox2b"]


def RULES_IMMUNE(m):
    if m.Epcam >= 0.8 or m.Col1a1 >= 1.0:
        return "DROP"
    if (m.Pecam1 >= 1.0 or m.Cdh5 >= 1.0) and m.Ptprc < 0.5:
        return "endothelial"
    if m.Jchain >= 5.0:
        return "plasma cell"
    if m.Cd79a >= 2.0 and m.Aicda >= 0.8:
        return "GC B cell"
    if m.Cd79a >= 2.0:
        return "B cell"
    if m.Lyz2 >= 1.5 or m.C1qa >= 1.0 or m.Cpa3 >= 1.5:
        return "myeloid"
    if m.Cd3e >= 1.0 or m.Il7r >= 1.0 or m.Nkg7 >= 1.5:
        return "T/ILC/NK"
    return "DROP"


def RULES_STROMA(m):
    if m.Ptprc >= 0.3 or m.Epcam >= 0.5:
        return "DROP"
    if m.Pecam1 >= 1.0 or m.Cdh5 >= 1.0:
        return "endothelial"
    if m.Myh11 >= 2.0 and m.Pdgfra < 1.0:
        return "smooth muscle"
    if m.Rgs5 >= 2.5 and m.Pdgfrb >= 1.0 and m.Dcn < 2.0:
        return "pericyte"
    if m.Pdgfra >= 0.5 and (m.Dcn >= 3.0 or m.Col1a1 >= 2.0):
        return "fibroblast"
    return "DROP"


def RULES_ENS(m):
    if m.Elavl4 >= 1.5 and m.Snap25 >= 1.5 and m.n_genes >= 2000:
        return "enteric neuron"
    if (m.Plp1 >= 1.0 or m.Sox10 >= 0.8) and m.Elavl4 < 0.5:
        return "enteric glia"
    return "DROP"


def read_mtx(prefix, genes_file):
    m = io.mmread(str(RAW / f"{prefix}_matrix.mtx.gz")).T.tocsr()
    g = pd.read_csv(RAW / genes_file, sep="\t", header=None)
    b = pd.read_csv(RAW / f"{prefix}_barcodes.tsv.gz", sep="\t", header=None)[0].to_numpy()
    ad = sc.AnnData(X=m.astype(np.float32))
    ad.var_names = g[1].astype(str).to_numpy()
    ad.obs_names = [f"{prefix}_{x}" for x in b]
    ad.var_names_make_unique()
    return ad


def annotate(ad, name, rules, k, min_genes=300):
    """k-means on PCA; label clusters by ordered marker rules on cluster-mean log(CP10k+1)."""
    sc.pp.filter_cells(ad, min_genes=min_genes)
    w = ad.copy()
    sc.pp.normalize_total(w, target_sum=1e4)
    sc.pp.log1p(w)
    sc.pp.highly_variable_genes(w, n_top_genes=2000, flavor="seurat")
    sc.pp.pca(w, n_comps=30, mask_var="highly_variable")
    lab = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(w.obsm["X_pca"])
    gg = [g for g in MARKERS if g in w.var_names]
    df = pd.DataFrame(w[:, gg].X.toarray(), columns=gg)
    for g in set(MARKERS) - set(gg):
        df[g] = 0.0
    df["n_genes"] = ad.obs["n_genes"].to_numpy()
    cm = df.groupby(lab).mean()
    cm.insert(0, "n", pd.Series(lab).value_counts().sort_index())
    cm["label"] = [rules(r) for _, r in cm.iterrows()]
    cm.round(2).to_csv(OUT / f"annotation_{name}.csv")
    print(f"--- {name}\n" + cm.groupby("label").n.sum().to_string(), flush=True)
    ad.obs["celltype1"] = cm.label.reindex(lab).to_numpy()
    ad.obs["source"] = name
    return ad[ad.obs.celltype1 != "DROP"].copy()


# ---- MERFISH panel genes
import anndata  # noqa: E402

vgenes = set(anndata.read_h5ad(sys.argv[1], backed="r").var_names.astype(str))
print("MERFISH panel genes:", len(vgenes))

parts = []
# ---- Haber epithelium
hab = sc.read_h5ad(ROOT / "examples/haber_intestine_reference.h5ad")
hab.obs["source"] = "Haber2017"
parts.append(hab[:, ~hab.var_names.duplicated()].copy())

# ---- Xu 2019 immune (control libraries only)
m = io.mmread(str(RAW / "GSE124880_PP_LP_mm10_count_matrix.mtx.gz")).tocsr()
genes = pd.read_csv(RAW / "GSE124880_PP_LP_mm10_count_gene.tsv.gz", header=None)[0].astype(str)
bcs = pd.read_csv(RAW / "GSE124880_PP_LP_mm10_count_barcode.tsv.gz", header=None)[0].astype(str)
if m.shape[0] == len(genes):
    m = m.T.tocsr()
xu = sc.AnnData(X=m.astype(np.float32))
xu.var_names = genes.to_numpy()
xu.obs_names = bcs.to_numpy()
xu.var_names_make_unique()
lib = pd.Series(xu.obs_names.str.replace(r"_[ACGT]+$", "", regex=True))
# GEO sample characteristics: 'C*' / ctrl / Control / Ctrl libraries = wild type; 'A*' / Allergy = OVA
ctrl = (lib.str.contains(r"_C[DIJ]$") | lib.str.contains(r"_(?:ctrl|Control(?:_\d)?|Ctrl_\d|Ctrl_IgDLow|Ctrl_nonTB)$")).to_numpy()
print("Xu libraries kept (control):", sorted(set(lib[ctrl])), flush=True)
xu = xu[ctrl].copy()
parts.append(annotate(xu, "Xu2019_immune", RULES_IMMUNE, k=40))

# ---- Paerregaard 2023 SI stroma
st = [read_mtx("GSM5469261_SI_01", "GSM5469261_SI_01_genes.tsv.gz"),
      read_mtx("GSM5469262_SI_02", "GSM5469262_SI_02_features.tsv.gz")]
common = st[0].var_names.intersection(st[1].var_names)
stro = sc.concat([s[:, common] for s in st])
parts.append(annotate(stro, "Paerregaard2023_stroma", RULES_STROMA, k=25))

# ---- Morarach 2021 myenteric neurons (P21) + glia captured in the same library
ens = read_mtx("GSM4504450_P21_1", "GSM4504450_P21_1_features.tsv.gz")
parts.append(annotate(ens, "Morarach2021_ENS", RULES_ENS, k=8))

# ---- Combine
# The GEO matrix of Xu 2019 (GSE124880) lists only 19,221 genes (genes undetected in that study are
# not deposited); the ~820 MERFISH-panel genes it lacks are mostly neural/olfactory/hormone
# receptors. They are zero-filled for the immune cells rather than dropped from the panel.
shared = set(vgenes)
for p in parts:
    if p.obs.source.iloc[0] != "Xu2019_immune":
        shared &= set(p.var_names)
shared = sorted(shared)
for i, p in enumerate(parts):
    if p.obs.source.iloc[0] == "Xu2019_immune":
        have = [g for g in shared if g in set(p.var_names)]
        print(f"Xu2019_immune: {len(have)} of {len(shared)} genes deposited; "
              f"{len(shared) - len(have)} zero-filled", flush=True)
        Xp = sparse.csr_matrix(p[:, have].X)
        col = {g: j for j, g in enumerate(shared)}
        Z = sparse.csr_matrix((Xp.data, np.array([col[have[k]] for k in Xp.indices]), Xp.indptr),
                              shape=(p.n_obs, len(shared)))
        parts[i] = sc.AnnData(X=Z, obs=p.obs.copy(), var=pd.DataFrame(index=shared))
print("genes shared by the MERFISH panel and all sources (Xu 2019 zero-filled):", len(shared), flush=True)
ref = sc.concat([p[:, shared] for p in parts], join="inner")
ref.obs = ref.obs[["celltype1", "source"]].copy()
keep = []
for ct, idx in ref.obs.groupby("celltype1").indices.items():
    keep.extend(rng.choice(idx, CAP, replace=False) if len(idx) > CAP else idx)
ref = ref[np.sort(keep)].copy()
ref.X = sparse.csr_matrix(ref.X)
ref.obs["celltype1"] = ref.obs.celltype1.astype(str)
comp = ref.obs.groupby(["celltype1", "source"]).size().rename("n_cells").reset_index()
comp.to_csv(OUT / "reference_composition.csv", index=False)
print(comp.to_string(index=False))
HARMONISE = {
    "epithelial fate stem cell": "Stem/TA", "transit amplifying cell": "Stem/TA",
    "enterocyte progenitor": "Enterocyte", "immature enterocyte": "Enterocyte", "enterocyte": "Enterocyte",
    "goblet cell": "Goblet", "immature goblet cell": "Goblet", "paneth cell": "Paneth",
    "brush cell": "Tuft", "enteroendocrine cell": "EEC", "B cell": "B cell", "GC B cell": "B cell",
    "plasma cell": "Plasma cell", "T/ILC/NK": "T/ILC/NK", "myeloid": "Myeloid",
    "fibroblast": "Fibroblast", "smooth muscle": "Smooth muscle", "pericyte": "Pericyte",
    "endothelial": "Endothelial", "enteric neuron": "Enteric neuron", "enteric glia": "Enteric glia",
}
ref.obs["cell_type"] = ref.obs.celltype1.map(HARMONISE).astype(str)
assert not ref.obs.cell_type.isin(["nan"]).any()
ref.obs["patient"] = ref.obs.source.astype(str)
ref.obs_names = [f"r{i}" for i in range(ref.n_obs)]
ref.obs = ref.obs[["cell_type", "celltype1", "patient"]]
ref.write_h5ad(OUT / "ref.h5ad")
import pandas as _pd  # noqa: E402
_pd.DataFrame({"barcode": ref.obs_names, "cell_type": ref.obs.cell_type.to_numpy(),
               "patient": ref.obs.patient.to_numpy()}).to_csv(OUT / "ref_meta.csv", index=False)
ref.obs.groupby(["cell_type", "celltype1", "patient"], observed=True).size().rename("n_cells").to_csv(
    OUT / "ref_composition_harmonised.csv")
