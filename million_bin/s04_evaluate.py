"""S1 step 4: evaluate every completed run against the MERFISH ground truth.

Usage: python s04_evaluate.py <data_dir> <results_dir>

Predictions: <results_dir>/props/<method>_<mode>_<ref>_<scale>.csv.gz (bin x type, one row per
fitted bin). Rows missing from a prediction file count as not covered.

Variants evaluated
  flashdeconv / card / cell2location : proportions as returned.
  rctd full                        : row-normalised full-mode weights.
  rctd doublet (primary)           : doublet-mode call - singlet -> first type = 1; doublet_certain /
                                     doublet_uncertain -> weights_doublet on first/second type;
                                     'reject' bins are not covered.
  rctd doublet-allw                : the full-fit weights returned by the doublet-mode run.
  rctd doublet_umi20               : sensitivity arm (UMI_min=20), doublet-mode call.

Ground truth: cell-count fractions (primary) and transcript-weighted fractions. For the external
reference (16 types; no ICC / Mesothelium), GT is restricted to the 16 shared types and
renormalised; bins whose cells are all ICC/Mesothelium are dropped from evaluation.

Metrics (identical for all methods; computed on bins x types, no joint-zero removal):
  flat_pearson, mean_type_pearson (constant prediction -> r = 0), rmse, jsd (mean per-bin
  Jensen-Shannon divergence, log2, exact agreement = 0), flat_ap (presence = GT > 0, score =
  predicted fraction), mean_type_ap; the same per-type means for rare types (< 5% mean cell
  fraction over the 1e6 set) and common types. Bin sets: 'own' (each run's covered bins) and
  'common' (bins covered by every primary run at that reference and scale). Coverage = covered /
  evaluable bins.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score

DATA = Path(sys.argv[1])
RES = Path(sys.argv[2])
PRIMARY = {("flashdeconv", "default"), ("rctd", "doublet"), ("rctd", "full"), ("card", "default"),
           ("cell2location", "fullbatch"), ("cell2location", "minibatch")}
LABEL = {"flashdeconv:default": "FlashDeconv", "rctd:doublet": "RCTD (doublet)",
         "rctd:doublet-allw": "RCTD (doublet, full-fit weights)", "rctd:full": "RCTD (full)",
         "rctd:doublet_umi20": "RCTD (doublet, UMI_min=20)", "card:default": "CARD", "card:default_min20": "CARD (minCountGene=20)",
         "cell2location:fullbatch": "Cell2location", "cell2location:minibatch": "Cell2location"}
DEPTH_EDGES = [0, 50, 100, 200, 400, np.inf]
DEPTH_LABELS = ["<50", "50-100", "100-200", "200-400", ">=400"]


def key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


types = [t for t in (DATA / "types.txt").read_text().splitlines() if t]
bins = pd.read_parquet(DATA / "gt_bins.parquet")
gt_all = {"cell": np.load(DATA / "gt_cellfrac.npy"), "tx": np.load(DATA / "gt_txfrac.npy")}
bin_pos = pd.Series(np.arange(len(bins)), index=bins.index)
n1e6 = min(1_000_000, len(bins))
glob = gt_all["cell"][:n1e6].mean(0)
RARE = {t for t, g in zip(types, glob) if g < 0.05}
ext_types = [t for t in types if t not in ("ICC", "Mesothelium")]
REF_TYPES = {"selfref": types, "extref": ext_types}


def read_pred(path, method, mode, ttypes):
    df = pd.read_csv(path)
    df = df.set_index(df.columns[0])
    kmap = {key(t): t for t in ttypes}
    variants = {}
    if method == "rctd" and mode.startswith("doublet"):
        wcols = [c for c in df.columns if key(c) in kmap]
        allw = df[wcols].rename(columns=lambda c: kmap[key(c)]).reindex(columns=ttypes).fillna(0.0)
        call = pd.DataFrame(0.0, index=df.index, columns=ttypes)
        ok = df.spot_class != "reject"
        sing = df.spot_class == "singlet"
        ft = df.first_type.map(lambda s: kmap.get(key(s)))
        stp = df.second_type.map(lambda s: kmap.get(key(s)))
        cidx = {t: i for i, t in enumerate(ttypes)}
        M = call.to_numpy()
        r = np.arange(len(df))
        f_i = ft.map(cidx).to_numpy()
        s_i = stp.map(cidx).to_numpy()
        wf, ws = df.w_first.to_numpy(float), df.w_second.to_numpy(float)
        dbl = (ok & ~sing).to_numpy()
        m = sing.to_numpy() & ~pd.isna(f_i)
        M[r[m], f_i[m].astype(int)] = 1.0
        m2 = dbl & ~pd.isna(f_i) & ~pd.isna(s_i)
        tot = np.maximum(wf + ws, 1e-12)
        M[r[m2], f_i[m2].astype(int)] += wf[m2] / tot[m2]
        M[r[m2], s_i[m2].astype(int)] += ws[m2] / tot[m2]
        call = pd.DataFrame(M, index=df.index, columns=ttypes)[ok.to_numpy()]
        variants[mode] = call
        if mode == "doublet":
            variants["doublet-allw"] = allw
    else:
        cols = [c for c in df.columns if key(c) in kmap]
        p = df[cols].rename(columns=lambda c: kmap[key(c)]).reindex(columns=ttypes).fillna(0.0)
        variants[mode] = p
    out = {}
    for v, p in variants.items():
        a = p.to_numpy(float)
        a = np.clip(a, 0, None)
        s = a.sum(1, keepdims=True)
        good = s[:, 0] > 0
        out[v] = pd.DataFrame(a[good] / s[good], index=p.index[good], columns=ttypes)
    return out


def jsd_rows(P, G):
    M = 0.5 * (P + G)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(P > 0, P * np.log2(P / M), 0.0).sum(1)
        b = np.where(G > 0, G * np.log2(G / M), 0.0).sum(1)
    return 0.5 * (a + b)


def pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def metrics(P, G, ttypes):
    out = {"n_bins": P.shape[0]}
    out["flat_pearson"] = pearson(P.ravel(), G.ravel())
    out["rmse"] = float(np.sqrt(((P - G) ** 2).mean()))
    out["jsd"] = float(jsd_rows(P, G).mean())
    pres = (G > 0)
    out["flat_ap"] = float(average_precision_score(pres.ravel(), P.ravel())) if pres.any() else np.nan
    per = []
    for j, t in enumerate(ttypes):
        g, p = G[:, j], P[:, j]
        r = pearson(p, g) if g.std() > 0 else np.nan
        ap = average_precision_score(g > 0, p) if (g > 0).any() else np.nan
        per.append({"cell_type": t, "pearson": r, "ap": ap, "rmse": float(np.sqrt(((p - g) ** 2).mean())),
                    "gt_mean": float(g.mean()), "pred_mean": float(p.mean()),
                    "n_present": int((g > 0).sum()), "rare": t in RARE})
    per = pd.DataFrame(per)
    out["mean_type_pearson"] = float(per.pearson.mean())
    out["mean_type_ap"] = float(per.ap.mean())
    for lab, sub in [("rare", per[per.rare]), ("common", per[~per.rare])]:
        out[f"{lab}_type_pearson"] = float(sub.pearson.mean())
        out[f"{lab}_type_ap"] = float(sub.ap.mean())
    return out, per


def gt_for(ref, idx, which):
    G = gt_all[which][idx]
    tt = REF_TYPES[ref]
    if ref == "extref":
        cols = [types.index(t) for t in tt]
        G = G[:, cols]
        s = G.sum(1, keepdims=True)
        keep = s[:, 0] > 0
        G = np.where(keep[:, None], G / np.maximum(s, 1e-12), 0.0)
        return G, keep
    return G, np.ones(len(G), bool)


# ------------------------------------------------------------------ load runs
rt = pd.concat([pd.read_csv(RES / "s1_runtime.csv")]
               + [pd.read_csv(f) for f in sorted((RES / "c2l_aces").glob("*runtime*.csv"))], ignore_index=True)
rt = rt.drop_duplicates(["method", "mode", "scale"], keep="last")
runs = []
# Cell2location runs made on the ACES cluster (same runner, same file naming) are merged here.
for f in sorted(list((RES / "props").glob("*.csv.gz")) + list((RES / "c2l_aces").rglob("*.csv.gz"))):
    m = re.match(r"(flashdeconv|rctd|card|cell2location)_(.+)_(selfref|extref)_(\d+)\.csv\.gz", f.name)
    if not m:
        continue
    method, mode, ref, scale = m.group(1), m.group(2), m.group(3), int(m.group(4))
    st = rt[(rt.method == method) & (rt["mode"] == f"{ref}-{mode}") & (rt.scale == scale)]
    if len(st) and st.status.iloc[-1] != "OK":
        continue
    runs.append((method, mode, ref, scale, f))
print("runs with predictions:", len(runs), flush=True)

summary, pertype, depth, paired = [], [], [], []
for ref in ["selfref", "extref"]:
    for scale in sorted({r[3] for r in runs if r[2] == ref}):
        ttypes = REF_TYPES[ref]
        idx_all = np.arange(scale)
        preds = {}
        for method, mode, rref, sc_, f in runs:
            if rref != ref or sc_ != scale:
                continue
            for v, p in read_pred(f, method, mode, ttypes).items():
                preds[(method, v)] = p
        if not preds:
            continue
        G_cell, evaluable = gt_for(ref, idx_all, "cell")
        ev_names = bins.index[:scale][evaluable]
        covered = {k: p.index.intersection(ev_names) for k, p in preds.items()}
        prim = [k for k in preds if k in PRIMARY]
        common = ev_names
        for k in prim:
            common = common.intersection(covered[k])
        print(f"{ref} {scale}: evaluable={len(ev_names)} common={len(common)} runs={list(preds)}", flush=True)
        for (method, v), p in preds.items():
            lab = LABEL.get(f"{method}:{v}", f"{method}:{v}")
            for binset, names in [("own", covered[(method, v)]), ("common", common.intersection(covered[(method, v)]))]:
                if len(names) == 0:
                    continue
                pos = bin_pos[names].to_numpy()
                P = p.loc[names].to_numpy()
                for which in ["cell", "tx"]:
                    G, _ = gt_for(ref, pos, which)
                    mt, per = metrics(P, G, ttypes)
                    row = dict(ref=ref, scale=scale, method=method, variant=v, label=lab, gt=which,
                               binset=binset, coverage=len(covered[(method, v)]) / len(ev_names),
                               n_evaluable=len(ev_names), **mt)
                    summary.append(row)
                    if binset == "common":
                        per.insert(0, "gt", which)
                        for c, val in [("label", lab), ("variant", v), ("method", method), ("scale", scale),
                                       ("ref", ref)]:
                            per.insert(0, c, val)
                        pertype.append(per)
                # depth strata on own bins (cell GT)
                if binset == "own":
                    G, _ = gt_for(ref, pos, "cell")
                    umi = bins.umi.to_numpy()[pos]
                    j = jsd_rows(P, G)
                    cat = pd.cut(umi, DEPTH_EDGES, labels=DEPTH_LABELS, right=False)
                    all_umi = bins.umi.to_numpy()[bin_pos[ev_names].to_numpy()]
                    all_cat = pd.cut(all_umi, DEPTH_EDGES, labels=DEPTH_LABELS, right=False)
                    for lvl in DEPTH_LABELS:
                        mk = np.asarray(cat == lvl)
                        n_all = int((all_cat == lvl).sum())
                        depth.append(dict(ref=ref, scale=scale, method=method, variant=v, label=lab,
                                          umi_bin=lvl, n_bins=int(mk.sum()), n_evaluable=n_all,
                                          coverage=mk.sum() / max(n_all, 1),
                                          jsd=float(j[mk].mean()) if mk.any() else np.nan,
                                          flat_pearson=pearson(P[mk].ravel(), G[mk].ravel()) if mk.any() else np.nan))
        # paired comparisons on common bins: FlashDeconv vs each other variant
        fd = ("flashdeconv", "default")
        if fd in preds and len(common):
            pos = bin_pos[common].to_numpy()
            G, _ = gt_for(ref, pos, "cell")
            Pf = preds[fd].loc[common].to_numpy()
            jf = jsd_rows(Pf, G)
            sl = bins.slice.to_numpy()[pos]
            rng = np.random.default_rng(0)
            usl = np.unique(sl)
            for k, p in preds.items():
                if k == fd:
                    continue
                cm = np.asarray(common.isin(p.index))  # hash lookup; non-primary variants may miss a few common bins
                Po = p.loc[common[cm]].to_numpy()
                jo = jsd_rows(Po, G[cm])
                d = jf[cm] - jo
                try:
                    pw = wilcoxon(d[d != 0]).pvalue if (d != 0).sum() > 10 else np.nan
                except ValueError:
                    pw = np.nan
                # slice-block bootstrap of the flattened-Pearson difference
                grp = {s: np.flatnonzero(sl == s) for s in usl}
                boots = []
                if len(usl) > 1:
                    for _ in range(200):
                        ii = np.concatenate([grp[s] for s in rng.choice(usl, len(usl))])
                        ii = ii[cm[ii]]
                        jj = np.cumsum(cm) - 1
                        boots.append(pearson(Pf[ii].ravel(), G[ii].ravel())
                                     - pearson(Po[jj[ii]].ravel(), G[ii].ravel()))
                dr = pearson(Pf[cm].ravel(), G[cm].ravel()) - pearson(Po.ravel(), G[cm].ravel())
                paired.append(dict(ref=ref, scale=scale, other=LABEL.get(f"{k[0]}:{k[1]}", k),
                                   n_common=int(cm.sum()), mean_jsd_fd=jf[cm].mean(), mean_jsd_other=jo.mean(),
                                   mean_delta_jsd=d.mean(), frac_bins_fd_better=float((d < 0).mean()),
                                   frac_bins_tied=float((d == 0).mean()), wilcoxon_p=pw,
                                   delta_flat_pearson=dr,
                                   delta_flat_pearson_ci_lo=np.percentile(boots, 2.5) if boots else np.nan,
                                   delta_flat_pearson_ci_hi=np.percentile(boots, 97.5) if boots else np.nan,
                                   n_slices=len(usl)))

# nested analysis: 1e6 self-reference runs evaluated on the 1e4 / 1e5 bins they contain
nested = []
for method, mode, ref, scale, f in runs:
    if ref != "selfref" or scale < 1_000_000:
        continue
    for v, p in read_pred(f, method, mode, types).items():
        for sub in [10_000, 100_000]:
            names = p.index.intersection(bins.index[:sub])
            if not len(names):
                continue
            pos = bin_pos[names].to_numpy()
            G, _ = gt_for(ref, pos, "cell")
            mt, _ = metrics(p.loc[names].to_numpy(), G, types)
            nested.append(dict(ref=ref, run_scale=scale, eval_subset=sub, method=method, variant=v,
                               label=LABEL.get(f"{method}:{v}", v), coverage=len(names) / sub, **mt))

pd.DataFrame(summary).to_csv(RES / "s1_summary.csv", index=False)
pd.concat(pertype).to_csv(RES / "s1_per_type.csv", index=False) if pertype else None
pd.DataFrame(depth).to_csv(RES / "s1_depth.csv", index=False)
pd.DataFrame(paired).to_csv(RES / "s1_paired.csv", index=False)
pd.DataFrame(nested).to_csv(RES / "s1_nested_1e6_on_subsets.csv", index=False)
print("rare types:", sorted(RARE))
print("done")
