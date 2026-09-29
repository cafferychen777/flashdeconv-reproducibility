"""Shared metrics and spatial block bootstrap for the penalty x sparsity analysis.

See PROTOCOL.md. dX = X(auto) - X(lambda0); negative dJSD means the penalty helps.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score

EDGES = [0, 50, 100, 200, 400, np.inf]
LABELS = ["<50", "50-100", "100-200", "200-400", ">=400"]
BLOCK_UM = 200.0
N_BOOT = 1000
N_BOOT_AP = 200
MIN_BINS = 200


def normalize(P):
    P = np.clip(np.asarray(P, dtype=np.float64), 0, None)
    s = P.sum(1, keepdims=True)
    out = np.full_like(P, 1.0 / P.shape[1])
    ok = s[:, 0] > 0
    out[ok] = P[ok] / s[ok]
    return out


def jsd_rows(P, G):
    """Per-row Jensen-Shannon divergence, base 2 (0 = exact agreement, max 1)."""
    M = 0.5 * (P + G)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(P > 0, P * np.log2(P / M), 0.0).sum(1)
        b = np.where(G > 0, G * np.log2(G / M), 0.0).sum(1)
    return np.clip(0.5 * (a + b), 0, None)


def rmse_rows(P, G):
    return np.sqrt(((P - G) ** 2).mean(1))


def stratum(umi):
    return pd.cut(np.asarray(umi, float), EDGES, right=False, labels=LABELS)


def block_ids(xy, section=None, size=BLOCK_UM):
    k = np.floor(np.asarray(xy, float) / size).astype(np.int64)
    key = k[:, 0] * 1_000_003 + k[:, 1]
    if section is not None:
        key = pd.factorize(pd.Series(section).astype(str))[0].astype(np.int64) * 10**13 + key
    return pd.factorize(key)[0]


def boot_counts(nblk, rng, n_boot):
    """(n_boot, nblk) multiplicities of a block bootstrap (blocks drawn with replacement)."""
    idx = rng.integers(0, nblk, size=(n_boot, nblk))
    C = np.zeros((n_boot, nblk), dtype=np.float32)
    for b in range(n_boot):
        C[b] = np.bincount(idx[b], minlength=nblk)
    return C


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan
    return float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))


def pearson_from_stats(S):
    """S[..., 6] = n, sx, sy, sxx, syy, sxy -> Pearson r."""
    n, sx, sy, sxx, syy, sxy = [S[..., i] for i in range(6)]
    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt(vx * vy)


def _pstats(P, G, blk, nblk):
    """Per-block sufficient statistics for flattened Pearson over bins x types."""
    K = P.shape[1]
    cols = [np.full(len(P), K, float), P.sum(1), G.sum(1), (P * P).sum(1), (G * G).sum(1),
            (P * G).sum(1)]
    return np.stack([np.bincount(blk, weights=c, minlength=nblk) for c in cols], axis=1)


def rare_ap(P, G, rare_idx, thr):
    aps = []
    for j in rare_idx:
        t = (G[:, j] > thr).astype(int)
        if 0 < t.sum() < len(t):
            aps.append(average_precision_score(t, P[:, j]))
    return float(np.mean(aps)) if aps else np.nan


def analyze(dataset, arm, Pa, P0, G, umi, xy, section=None, presence_thr=0.0, rare_idx=None,
            seed=0, n_boot=N_BOOT, n_boot_ap=N_BOOT_AP, lambda_auto=np.nan, extra=None):
    """Stratified auto-vs-lambda0 comparison with a spatial block bootstrap.

    Returns (summary rows, per-bin frame, trend row). Blocks are shared across strata so that
    the low-minus-high stratum difference has a paired bootstrap.
    """
    rng = np.random.default_rng(seed)
    Pa, P0, G = normalize(Pa), normalize(P0), np.asarray(G, np.float64)
    G = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
    ja, j0 = jsd_rows(Pa, G), jsd_rows(P0, G)
    ra, r0 = rmse_rows(Pa, G), rmse_rows(P0, G)
    dj, dr = ja - j0, ra - r0
    st = stratum(umi)
    blk = block_ids(xy, section)
    nblk = blk.max() + 1
    if rare_idx is None:
        rare_idx = np.flatnonzero(G.mean(0) < 0.05)
    C = boot_counts(nblk, rng, n_boot)
    per_bin = pd.DataFrame({"umi": np.asarray(umi, float), "stratum": st, "jsd_auto": ja,
                            "jsd_l0": j0, "rmse_auto": ra, "rmse_l0": r0, "block": blk})
    if section is not None:
        per_bin["section"] = np.asarray(section).astype(str)
    rows, boot_means = [], {}
    for lab in LABELS + ["all"]:
        m = np.ones(len(dj), bool) if lab == "all" else np.asarray(st == lab)
        n = int(m.sum())
        row = dict(dataset=dataset, arm=arm, stratum=lab, n_bins=n, lambda_auto=lambda_auto,
                   **(extra or {}))
        if n == 0:
            rows.append(row)
            continue
        cnt = np.bincount(blk[m], minlength=nblk).astype(float)
        sj = np.bincount(blk[m], weights=dj[m], minlength=nblk)
        sr = np.bincount(blk[m], weights=dr[m], minlength=nblk)
        with np.errstate(invalid="ignore", divide="ignore"):
            bj = (C @ sj) / (C @ cnt)
            br = (C @ sr) / (C @ cnt)
        boot_means[lab] = bj
        Sa = _pstats(Pa[m], G[m], blk[m], nblk)
        S0 = _pstats(P0[m], G[m], blk[m], nblk)
        pa, p0 = pearson_from_stats(Sa.sum(0)), pearson_from_stats(S0.sum(0))
        bp = pearson_from_stats(C @ Sa) - pearson_from_stats(C @ S0)
        try:
            wp = float(wilcoxon(dj[m][dj[m] != 0]).pvalue) if (dj[m] != 0).sum() > 10 else np.nan
        except ValueError:
            wp = np.nan
        row.update(
            median_umi=float(np.median(np.asarray(umi)[m])), n_blocks=int((cnt > 0).sum()),
            jsd_auto=float(ja[m].mean()), jsd_l0=float(j0[m].mean()),
            d_jsd=float(dj[m].mean()), d_jsd_lo=ci(bj)[0], d_jsd_hi=ci(bj)[1],
            d_jsd_median=float(np.median(dj[m])),
            frac_improved=float((dj[m] < -1e-12).mean()), frac_worse=float((dj[m] > 1e-12).mean()),
            wilcoxon_p=wp,
            rmse_auto=float(ra[m].mean()), rmse_l0=float(r0[m].mean()),
            d_rmse=float(dr[m].mean()), d_rmse_lo=ci(br)[0], d_rmse_hi=ci(br)[1],
            pearson_auto=float(pa), pearson_l0=float(p0), d_pearson=float(pa - p0),
            d_pearson_lo=ci(bp)[0], d_pearson_hi=ci(bp)[1],
        )
        if n >= MIN_BINS and len(rare_idx):
            apa = rare_ap(Pa[m], G[m], rare_idx, presence_thr)
            ap0 = rare_ap(P0[m], G[m], rare_idx, presence_thr)
            row.update(rare_aupr_auto=apa, rare_aupr_l0=ap0, d_rare_aupr=apa - ap0)
            if n_boot_ap:
                bins_of = pd.Series(np.flatnonzero(m)).groupby(blk[m]).apply(np.asarray)
                keys = bins_of.index.to_numpy()
                bd = []
                for _ in range(n_boot_ap):
                    ii = np.concatenate(bins_of.to_numpy()[rng.integers(0, len(keys), len(keys))])
                    bd.append(rare_ap(Pa[ii], G[ii], rare_idx, presence_thr)
                              - rare_ap(P0[ii], G[ii], rare_idx, presence_thr))
                row.update(d_rare_aupr_lo=ci(bd)[0], d_rare_aupr_hi=ci(bd)[1])
        if section is not None:
            sec = per_bin.loc[m].groupby("section").agg(n=("jsd_auto", "size"),
                                                         ja=("jsd_auto", "mean"),
                                                         j0=("jsd_l0", "mean"))
            sec = sec[sec.n >= 50]
            row["n_sections"] = len(sec)
            row["sections_improved"] = int((sec.ja < sec.j0).sum())
            if len(sec) >= 5:
                row["section_wilcoxon_p"] = float(wilcoxon(sec.ja - sec.j0).pvalue)
        rows.append(row)
    # trend: lowest vs highest populated stratum (>= MIN_BINS bins)
    pop = [l for l in LABELS if l in boot_means and
           next(r for r in rows if r["stratum"] == l)["n_bins"] >= MIN_BINS]
    trend = dict(dataset=dataset, arm=arm, **(extra or {}))
    if len(pop) >= 2:
        lo, hi = pop[0], pop[-1]
        d = boot_means[lo] - boot_means[hi]
        rlo = next(r for r in rows if r["stratum"] == lo)
        rhi = next(r for r in rows if r["stratum"] == hi)
        from scipy.stats import spearmanr
        sub = [next(r for r in rows if r["stratum"] == l) for l in pop]
        rho = spearmanr([r["median_umi"] for r in sub], [r["d_jsd"] for r in sub])
        trend.update(low_stratum=lo, high_stratum=hi, d_jsd_low=rlo["d_jsd"],
                     d_jsd_high=rhi["d_jsd"], low_minus_high=rlo["d_jsd"] - rhi["d_jsd"],
                     lmh_lo=ci(d)[0], lmh_hi=ci(d)[1],
                     lmh_boot_p=float(2 * min((d >= 0).mean(), (d <= 0).mean())),
                     spearman_umi_vs_djsd=float(rho.correlation), n_strata=len(pop))
    return rows, per_bin, trend
