"""B1 pilot: boundary, signed-distance bands, gradients, concordance and stop rule (see results/b1_pilot/PROTOCOL.md)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects

from lineages import LINEAGES, MARKERS_EPI, MARKERS_EPI_CRC_XEN

S = Path("/scratch/user/cafferychen777")
RES = S / "b1_pilot" / "results"
PX = 8.0
SIGMA = 1.5
EDGES = np.arange(-200, 301, 25, dtype=float)
MID = (EDGES[:-1] + EDGES[1:]) / 2
NB = len(MID)
BLOCK = 250.0
NBOOT = 200
MIN_FRAC = 0.005
PRIMARY = ["CRC_P1", "CRC_P2", "CRC_P5", "SPATCH_COAD", "SPATCH_OV"]
SECONDARY = ["SPATCH_HCC"]
# Exploratory extension (no stop rule): 10x lung post-Xenium (same section) and 10x ovarian FF (adjacent Xenium)
EXT = ["LUNG_X1", "LUNG_X5K", "OV10X", "OV10X_v1"]
FD_FILE = {"OV10X_v1": "OV10X"}
CODEX_COMPARE = ["Fibroblast", "Endothelial", "CD4 T", "CD8 T", "Treg", "NK", "B", "Macrophage/Mono"]
DIAG = {"CD3E+CD3D": ["CD3E", "CD3D"], "MS4A1": ["MS4A1"], "CD68": ["CD68"], "COL1A1": ["COL1A1"], "PECAM1": ["PECAM1"]}
DIAG_TARGET = {"CD3E+CD3D": ["CD4 T", "CD8 T", "Treg"], "MS4A1": ["B"], "CD68": ["Macrophage/Mono"],
               "COL1A1": ["Fibroblast"], "PECAM1": ["Endothelial"]}


def boundary(coords, m, t):
    """Signed distance (um) of each unit to the epithelial-marker boundary; returns (sd, grid dict)."""
    x0, y0 = coords.min(0)
    ij = np.floor((coords - [x0, y0]) / PX).astype(int)
    shp = tuple(ij.max(0) + 1)
    M = np.zeros(shp); T = np.zeros(shp); occ = np.zeros(shp, bool)
    np.add.at(M, (ij[:, 0], ij[:, 1]), m)
    np.add.at(T, (ij[:, 0], ij[:, 1]), t)
    occ[ij[:, 0], ij[:, 1]] = True
    Ms, Ts = ndimage.gaussian_filter(M, SIGMA), ndimage.gaussian_filter(T, SIGMA)
    valid = Ts > 1e-3 * Ts[occ].mean()
    score = np.zeros(shp)
    score[valid] = Ms[valid] / Ts[valid]
    thr = threshold_otsu(score[occ])
    mask = (score > thr) & valid
    mask = remove_small_objects(mask, 50)
    mask = remove_small_holes(mask, 50)
    d_out = ndimage.distance_transform_edt(~mask)
    d_in = ndimage.distance_transform_edt(mask)
    sd_grid = np.where(mask, -(d_in - 0.5), d_out - 0.5) * PX
    sd = sd_grid[ij[:, 0], ij[:, 1]]
    return sd, dict(mask=mask, occ=occ, thr=float(thr), frac_epi_px=float(mask[occ].mean()), origin=(float(x0), float(y0)))


def band_stats(sd, coords, V):
    """V: (n_units, k) values. Returns curve (NB,k), ci_lo, ci_hi (NB,k), n per band."""
    inb = (sd >= EDGES[0]) & (sd < EDGES[-1])
    b = np.digitize(sd[inb], EDGES) - 1
    c = coords[inb]; V = V[inb].astype(np.float64)
    blk = np.floor(c / BLOCK).astype(np.int64)
    _, blk_id = np.unique(blk[:, 0] * 100000 + blk[:, 1], return_inverse=True)
    nblk = blk_id.max() + 1
    key = blk_id * NB + b
    N = np.bincount(key, minlength=nblk * NB).reshape(nblk, NB)
    Ssum = np.stack([np.bincount(key, weights=V[:, j], minlength=nblk * NB).reshape(nblk, NB) for j in range(V.shape[1])], -1)
    curve = Ssum.sum(0) / np.maximum(N.sum(0), 1)[:, None]
    rng = np.random.default_rng(0)
    boots = np.empty((NBOOT, NB, V.shape[1]))
    for r in range(NBOOT):
        w = np.bincount(rng.integers(0, nblk, nblk), minlength=nblk).astype(float)
        boots[r] = np.einsum("b,bnk->nk", w, Ssum) / np.maximum(w @ N, 1)[:, None]
    return curve, np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0), N.sum(0)


def pearson(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def analyse(sec, hd_epi=None, tag=None):
    tag = tag or sec
    f = np.load(RES / f"fd_{FD_FILE.get(sec, sec)}.npz", allow_pickle=True)
    base = sec.rsplit("_", 1)[0] if sec.endswith(("_luad", "_flex")) else sec
    o = np.load(RES / f"orth_{base}.npz", allow_pickle=True)
    mods = {}
    # --- Visium HD / FlashDeconv
    mn = list(f["marker_names"]); Mc = f["marker_counts"]
    if hd_epi is None:
        hd_epi = MARKERS_EPI_CRC_XEN if sec.startswith("CRC") else (list(o["epi_markers"]) if base in EXT else MARKERS_EPI)
    epi_idx = [mn.index(g) for g in hd_epi if g in mn]
    print(sec, tag, "HD boundary genes:", [mn[i] for i in epi_idx], "orth:", list(o["epi_markers"]), flush=True)
    coords = f["coords_um"].astype(float)
    sd_hd, g_hd = boundary(coords, Mc[:, epi_idx].sum(1), f["total_counts"])
    L = f["lineage"].astype(np.float32)
    cpm = Mc / np.maximum(f["total_counts"], 1)[:, None] * 1e4
    D = np.stack([cpm[:, [mn.index(g) for g in gs if g in mn]].sum(1) for gs in DIAG.values()], 1)
    curve, lo, hi, n = band_stats(sd_hd, coords, np.hstack([L, D]))
    k = len(LINEAGES)
    mods["FD"] = dict(curve=curve[:, :k], lo=lo[:, :k], hi=hi[:, :k], n=n)
    mods["HDmarker"] = dict(curve=curve[:, k:], lo=lo[:, k:], hi=hi[:, k:], n=n)
    # --- orthogonal
    oc = o["coords_um"].astype(float)
    sd_o, g_o = boundary(oc, o["epi_counts"], o["total"])
    lin = o["lineage"].astype(str)
    keep = lin != "Unassigned"
    Oh = np.stack([(lin[keep] == l).astype(np.float32) for l in LINEAGES], 1)
    oc_curve, olo, ohi, on = band_stats(sd_o[keep], oc[keep], Oh)
    modality = str(o["modality"])
    mods["ORTH"] = dict(curve=oc_curve, lo=olo, hi=ohi, n=on)
    # --- concordance
    FDc, ORc = mods["FD"]["curve"], mods["ORTH"]["curve"]
    li = {l: i for i, l in enumerate(LINEAGES)}
    if modality == "CODEX":
        comp = CODEX_COMPARE
        fd_curve = {l: FDc[:, li[l]] for l in comp}
        fd_curve["Fibroblast"] = FDc[:, li["Fibroblast"]] + FDc[:, li["Pericyte/SMC"]]
    else:
        comp = [l for l in LINEAGES if l not in ("Epithelial", "Other") and not (l == "Treg" and sec.startswith("CRC"))]
        fd_curve = {l: FDc[:, li[l]] for l in comp}
    rows = []
    for l in comp + ["Epithelial"]:
        fc = fd_curve[l] if l in fd_curve else FDc[:, li[l]]
        orc = ORc[:, li[l]]
        rows.append(dict(section=tag, modality=modality, lineage=l, r=pearson(fc, orc), orth_mean=float(orc.mean()),
                         fd_mean=float(fc.mean()), included=bool(l != "Epithelial" and orc.mean() >= MIN_FRAC)))
    conc = pd.DataFrame(rows)
    inc = conc[conc.included]
    med = float(inc.r.median()) if len(inc) >= 3 else np.nan
    # specificity: mismatched pairs among included
    incl = list(inc.lineage)
    mm = [pearson(fd_curve[a], ORc[:, li[b]]) for a in incl for b in incl if a != b]
    # marker diagnostic
    drows = []
    for j, (dn, tg) in enumerate(DIAG.items()):
        orc = ORc[:, [li[t] for t in DIAG_TARGET[dn]]].sum(1)
        fdt = sum(fd_curve[t] if t in fd_curve else FDc[:, li[t]] for t in DIAG_TARGET[dn])
        drows.append(dict(section=tag, marker=dn, target="+".join(DIAG_TARGET[dn]),
                          r_marker_vs_orth=pearson(mods["HDmarker"]["curve"][:, j], orc),
                          r_fd_vs_orth=pearson(fdt, orc), r_fd_vs_marker=pearson(fdt, mods["HDmarker"]["curve"][:, j])))
    summ = dict(section=tag, hd_boundary_genes=[mn[i] for i in epi_idx], modality=modality, n_hd_bins=int(len(sd_hd)), n_orth_cells=int(keep.sum()),
                hd_frac_epi_px=g_hd["frac_epi_px"], orth_frac_epi_px=g_o["frac_epi_px"],
                n_included=int(len(inc)), included=incl, median_r=med, passes=bool(med > 0.7) if np.isfinite(med) else False,
                mean_matched_r=float(inc.r.mean()) if len(inc) else np.nan,
                mean_mismatched_r=float(np.nanmean(mm)) if mm else np.nan,
                epithelial_r=float(conc.loc[conc.lineage == "Epithelial", "r"].iloc[0]),
                t_deconvolve_s=float(f["t_deconvolve_s"]), fd_n_iter=int(f["n_iterations"]),
                fd_converged=bool(f["converged"]))
    # long-form curves
    crows = []
    for mod, dct in mods.items():
        names = list(DIAG) if mod == "HDmarker" else LINEAGES
        for bi in range(NB):
            for j, nm in enumerate(names):
                crows.append(dict(section=tag, source=mod if mod != "ORTH" else modality, band_mid=MID[bi], name=nm,
                                  value=dct["curve"][bi, j], lo=dct["lo"][bi, j], hi=dct["hi"][bi, j], n_units=int(dct["n"][bi])))
    np.savez_compressed(RES / f"grid_{tag}.npz", hd_mask=g_hd["mask"], hd_occ=g_hd["occ"], orth_mask=g_o["mask"], orth_occ=g_o["occ"])
    return summ, conc, pd.DataFrame(drows), pd.DataFrame(crows)


def main(secs):
    out = [analyse(s) for s in secs]
    out += [analyse(s, hd_epi=MARKERS_EPI, tag=s + "_sensHD-EPCAMKRT") for s in secs if s.startswith("CRC")]
    summ = pd.DataFrame([o[0] for o in out])
    conc = pd.concat([o[1] for o in out]); diag = pd.concat([o[2] for o in out]); curves = pd.concat([o[3] for o in out])
    summ.to_csv(RES / "section_summary.csv", index=False)
    conc.to_csv(RES / "concordance_per_lineage.csv", index=False)
    diag.to_csv(RES / "marker_diagnostic.csv", index=False)
    curves.to_csv(RES / "gradient_curves.csv.gz", index=False)
    prim = summ[summ.section.isin(PRIMARY)]
    stop = dict(rule="median r > 0.7 in >= 4 of 5 primary sections", primary=PRIMARY,
                median_r={r.section: r.median_r for r in prim.itertuples()},
                n_pass=int(prim.passes.sum()), n_primary=int(len(prim)), go=bool(prim.passes.sum() >= 4),
                secondary={r.section: r.median_r for r in summ[~summ.section.isin(PRIMARY)].itertuples()})
    json.dump(stop, open(RES / "stop_rule.json", "w"), indent=2)
    pd.set_option("display.width", 250)
    print(summ.drop(columns=["included", "hd_boundary_genes"]).to_string())
    print(conc.to_string())
    print(diag.to_string())
    print(json.dumps(stop, indent=2))


def main_ext(secs, suffix="_ext"):
    out = [analyse(s) for s in secs]
    summ = pd.DataFrame([o[0] for o in out])
    summ.to_csv(RES / f"section_summary{suffix}.csv", index=False)
    pd.concat([o[1] for o in out]).to_csv(RES / f"concordance_per_lineage{suffix}.csv", index=False)
    pd.concat([o[2] for o in out]).to_csv(RES / f"marker_diagnostic{suffix}.csv", index=False)
    pd.concat([o[3] for o in out]).to_csv(RES / f"gradient_curves{suffix}.csv.gz", index=False)
    pd.set_option("display.width", 250)
    print(summ.drop(columns=["included"]).to_string())
    print(pd.concat([o[1] for o in out]).to_string())
    print(pd.concat([o[2] for o in out]).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ext":
        main_ext(sys.argv[2:] or EXT)
    elif len(sys.argv) > 1 and sys.argv[1] == "--lungref":
        main_ext(["LUNG_X1_flex", "LUNG_X1_luad", "LUNG_X5K_flex", "LUNG_X5K_luad"], suffix="_lungref")
    else:
        main(sys.argv[1:] if len(sys.argv) > 1 else PRIMARY + SECONDARY)
