"""Shared lineage label set and label mappings for the B1 pilot."""
import numpy as np

LINEAGES = ["Epithelial", "Fibroblast", "Pericyte/SMC", "Endothelial", "CD4 T", "CD8 T", "Treg", "NK", "B",
            "Plasma", "Macrophage/Mono", "DC", "Neutrophil", "Mast", "Other"]

MARKERS_EPI = ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT20", "CDH1"]
# Epithelial genes shared by Visium HD and the Xenium CRC panel (which lacks EPCAM/KRT8/18/19/20/CDH1)
MARKERS_EPI_CRC_XEN = ["CEACAM5", "CEACAM6", "PERP"]
# Generic epithelial candidates; per Xenium section the boundary uses those present in the Xenium panel (and the same genes in HD)
MARKERS_EPI_GENERIC = MARKERS_EPI + ["KRT7", "ELF3", "CLDN4"] + MARKERS_EPI_CRC_XEN
MARKERS_DIAG = ["KRT7", "ELF3", "CLDN4", "CEACAM5", "CEACAM6", "PERP", "CD3E", "CD3D", "MS4A1", "CD68", "C1QC", "COL1A1", "PECAM1", "CD8A", "FOXP3", "NKG7", "JCHAIN",
                "LAMP3", "S100A8", "CPA3", "ACTA2", "RGS5", "CD163"]

_CRC = {
    "Tumor I": "Epithelial", "Tumor II": "Epithelial", "Tumor III": "Epithelial", "Tumor IV": "Epithelial",
    "Tumor V": "Epithelial", "Enterocyte": "Epithelial", "Goblet": "Epithelial", "Tuft": "Epithelial",
    "Epithelial": "Epithelial", "Neuroendocrine": "Epithelial",
    "CAF": "Fibroblast", "Myofibroblast": "Fibroblast", "Fibroblast": "Fibroblast",
    "Proliferating Fibroblast": "Fibroblast", "Vascular Fibroblast": "Fibroblast",
    "Smooth Muscle": "Pericyte/SMC", "SM Stress Response": "Pericyte/SMC", "vSM": "Pericyte/SMC",
    "Pericytes": "Pericyte/SMC", "Unknown III (SM)": "Pericyte/SMC",
    "Endothelial": "Endothelial", "Lymphatic Endothelial": "Endothelial",
    "CD4 T cell": "CD4 T", "CD8 T cell": "CD8 T", "NK": "NK", "Mature B": "B", "Memory B": "B", "Plasma": "Plasma",
    "Macrophage": "Macrophage/Mono", "Proliferating Macrophages": "Macrophage/Mono",
    "cDC I": "DC", "mRegDC": "DC", "pDC": "DC", "Neutrophil": "Neutrophil", "Mast": "Mast",
    "Enteric Glial": "Other", "Adipocyte": "Other", "Proliferating Immune II": "Other",
}

_SPATCH = {
    "Epithelial": "Epithelial", "Hepatocyte": "Epithelial", "Fibroblast": "Fibroblast", "SMC": "Pericyte/SMC",
    "Endothelial": "Endothelial", "CD4T": "CD4 T", "CD8T": "CD8 T", "Treg": "Treg", "NK": "NK", "B": "B",
    "Plasma": "Plasma", "Macrophage": "Macrophage/Mono", "Monocyte": "Macrophage/Mono", "Kupffer": "Macrophage/Mono",
    "mregDC": "DC", "cDC1": "DC", "cDC2": "DC", "pDC": "DC", "DC": "DC", "Neutrophil": "Neutrophil", "Mast": "Mast",
    "Tprolif": "Other",
}


def crc_lineage(name):
    return _CRC.get(name, "Other")


def spatch_lineage(name):
    return _SPATCH.get(name, "Other")


def spatch_fine_labels(obs):
    lab = obs["major_annotation"].astype(str).to_numpy().copy()
    minor = obs["minor_annotation"].astype(str).to_numpy()
    lab[minor == "CD4T_FOXP3"] = "Treg"
    lab[minor == "NK_NCAM1"] = "NK"
    return lab


CODEX_MAP = {"Epithelial": "Epithelial", "Fibroblast": "Fibroblast", "Endothelial": "Endothelial", "CD4T": "CD4 T",
             "CD8T": "CD8 T", "Treg": "Treg", "NK": "NK", "B": "B", "Macrophage": "Macrophage/Mono", "T": "Other"}


def codex_lineage(labels):
    return np.array([CODEX_MAP.get(str(x), "Other") for x in labels])


_LUNG_FLEX = {
    "Macrophage": "Macrophage/Mono", "Monocyte": "Macrophage/Mono", "T_reg": "Treg", "T_CTL": "CD8 T",
    "T_CD8_exhausted": "CD8 T", "T_CD4": "CD4 T", "T_CXCL13": "CD4 T", "TNK_dividing": "Other", "NK": "NK",
    "B_cell": "B", "Fibroblast": "Fibroblast", "Muscle_smooth": "Pericyte/SMC", "Pericyte": "Pericyte/SMC",
    "Endothelia_vascular": "Endothelial", "Epi_lung": "Epithelial", "Mast_cell": "Mast", "DC_1": "DC", "DC_2": "DC",
    "DC_activated": "DC", "DC_pc": "DC", "Granulocyte": "Neutrophil",
}


def lung_lineage(name):
    if name.startswith("Tu_"):
        return "Epithelial"
    if name.startswith("B_plasma"):
        return "Plasma"
    return _LUNG_FLEX.get(name, "Other")
