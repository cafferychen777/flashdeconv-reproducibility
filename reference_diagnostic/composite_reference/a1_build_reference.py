"""Build a composite mouse small-intestine reference (raw counts) for FlashDeconv.

Sources (all public, verified accessions):
  epithelium : Haber et al. Nature 2017 (GSE92332), 10 homeostatic epithelial types (celltype1),
               already harmonised in examples/haber_intestine_reference.h5ad.
  immune     : Xu et al. Immunity 2019 (GSE124880), 10x 3' v2, Peyer's patch + lamina propria,
               wild-type (control) libraries only (Food-Allergy/OVA libraries excluded).
  mesenchyme : Paerregaard et al. Nat Commun 2023 (GSE180735, SI replicates 01/02), sorted
               CD45-CD31-EpCAM- lamina-propria stromal cells (muscularis externa removed).
  endothelium: endothelial clusters present in the Paerregaard stromal and Xu immune libraries
               (blood + lymphatic lamina-propria ECs). Kalucka et al. Cell 2020 (E-MTAB-8077) SI ECs
               were evaluated but not used: the deposited matrix holds only 6,787 genes.
  ENS        : Morarach et al. Nat Neurosci 2021 (GSE149524, P21_1), myenteric neurons (Baf53b-Cre
               sort) plus the glial cluster captured in the same library.

GEO deposits carry no cell labels for these datasets (the author labels for Xu 2019 sit behind a
Single Cell Portal login), so non-epithelial cells are labelled here: per dataset, log-normalise,
PCA (30 PCs), k-means, then explicit, ordered marker rules on cluster-mean log(CP10k+1) (RULES_*
below); clusters matching no rule or a contaminant rule are dropped. Coarse taxonomy:
  B cell, GC B cell, plasma cell, T/ILC/NK, myeloid, fibroblast, smooth muscle, pericyte,
  endothelial, enteric neuron, enteric glia   (+ the 10 Haber epithelial types, names unchanged).
Cells per type capped at 3,000 (random, seed 0). Genes restricted to those present in every source
and in the Visium HD Mouse Small Intestine (probe set v2) feature list.
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io, sparse
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
RAW = HERE / "data_ref"
ROOT = HERE.parents[1]
OUT = HERE / "results" / "reference"
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


# ---- Visium HD feature list
import h5py  # noqa: E402

vhd = ROOT / "validation/visium_hd_data/Visium_HD_Mouse_Small_Intestine_binned_outputs"
with h5py.File(vhd / "square_016um/filtered_feature_bc_matrix.h5") as f:
    vgenes = set(x.decode() for x in f["matrix/features/name"][:])
print("Visium HD features:", len(vgenes))

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
shared = set(vgenes)
for p in parts:
    shared &= set(p.var_names)
shared = sorted(shared)
print("genes shared by all sources and Visium HD:", len(shared), flush=True)
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
ref.write_h5ad(OUT / "composite_reference.h5ad", compression="gzip")

# Marker sanity table: mean CP10k of key markers per type
norm = ref.copy()
sc.pp.normalize_total(norm, target_sum=1e4)
mk = [g for g in ["Cd79a", "Ms4a1", "Aicda", "Jchain", "Cd3e", "Lyz2", "Pdgfra", "Myh11", "Rgs5",
                  "Pecam1", "Elavl4", "Epcam", "Lyz1", "Muc2", "Pou2f3", "Trpm5", "Lrmp", "Olfm4",
                  "Lgr5", "Chga"] if g in norm.var_names]
tab = pd.DataFrame(norm[:, mk].X.toarray(), columns=mk).groupby(norm.obs.celltype1.to_numpy()).mean()
tab.round(1).to_csv(OUT / "reference_marker_cp10k.csv")
print(tab.round(1).to_string())
