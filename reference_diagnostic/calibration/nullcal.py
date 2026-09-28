"""Depth-dependence of the reference-fit score and candidate null calibrations (see PROTOCOL.md).

Building blocks shared by the diagnosis and the validation scripts:
  core(...)        observed per-bin deviance dev_i = n_i (l_i + H_i) and n_i V_i (package kernels)
  simulate(...)    the same quantities for counts drawn from the fitted mixture at each bin's depth
  calibrate(...)   candidate null calibrations of the raw score
"""
import numpy as np
from numba import njit, prange
from scipy import sparse

from flashdeconv.core.refcheck import _entropy_var, _mixture_loglik

Z95 = 1.6448536269514722


def prep(model):
    gidx = model.gene_idx_
    Y = sparse.csr_matrix(model._Y_raw[:, gidx], dtype=np.float64)
    Y.sort_indices()
    X = np.asarray(model._X_raw[:, gidx], dtype=np.float64)
    Xbar = np.ascontiguousarray(X / np.maximum(X.sum(1, keepdims=True), 1e-300))
    P0 = np.ascontiguousarray(model.proportions_, dtype=np.float64)
    return Y, Xbar, P0


def core(Y, Xbar, P0, eta=0.01, n_em=10, tol=1e-4):
    ell, n, P = _mixture_loglik(Y.indptr.astype(np.int64), Y.indices.astype(np.int64), Y.data,
                                Xbar, P0, float(eta), int(n_em), float(tol))
    H, V = _entropy_var(P, Xbar, float(eta))
    dev = n * (ell + H)
    return {"dev": dev, "nV": n * V, "n": n, "P": P}


@njit(inline="always")
def _splitmix(state):
    state = state + np.uint64(0x9E3779B97F4A7C15)
    z = state
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return state, z


@njit(inline="always")
def _unif(st):
    st, z = _splitmix(st)
    return st, (z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(parallel=True, cache=True)
def _sim_kernel(Pgen, cumX, Xbar, P0, n_int, eta, n_em, em_tol, seed):
    """Per bin: draw y* ~ Multinomial(n_i, (1-eta) Pgen_i Xbar + eta/G) (type first, then gene),
    run the package EM from P0_i on y*, return per-UMI log-lik and EM weights."""
    N, K = Pgen.shape
    G = Xbar.shape[1]
    ell = np.zeros(N)
    Pout = P0.copy()
    fl = eta / G
    for i in prange(N):
        n = n_int[i]
        if n <= 0:
            continue
        st = np.uint64(seed) * np.uint64(1000003) + np.uint64(i) * np.uint64(0x632BE59BD9B4E019)
        cp = np.empty(K)
        acc = 0.0
        for k in range(K):
            acc += Pgen[i, k]
            cp[k] = acc
        cnt = np.zeros(G)
        touched = np.empty(n, dtype=np.int64)
        nt = 0
        for _ in range(n):
            st, u = _unif(st)
            if u < eta:
                st, u2 = _unif(st)
                g = min(int(u2 * G), G - 1)
            else:
                st, u2 = _unif(st)
                u2 *= acc
                k = 0
                while k < K - 1 and cp[k] < u2:
                    k += 1
                st, u3 = _unif(st)
                lo, hi = 0, G - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if cumX[k, mid] < u3:
                        lo = mid + 1
                    else:
                        hi = mid
                g = lo
            if cnt[g] == 0.0:
                touched[nt] = g
                nt += 1
            cnt[g] += 1.0
        p = Pout[i]
        s = 0.0
        for k in range(K):
            if p[k] < 0.0:
                p[k] = 0.0
            s += p[k]
        if s <= 0.0:
            for k in range(K):
                p[k] = 1.0 / K
        else:
            for k in range(K):
                p[k] /= s
        a = np.zeros(K)
        for _ in range(n_em):
            for k in range(K):
                a[k] = 0.0
            for t in range(nt):
                g = touched[t]
                q = 0.0
                for k in range(K):
                    q += p[k] * Xbar[k, g]
                q = (1.0 - eta) * q + fl
                f = cnt[g] * (1.0 - eta) / q
                for k in range(K):
                    a[k] += f * p[k] * Xbar[k, g]
            tot = 0.0
            for k in range(K):
                tot += a[k]
            if tot <= 0.0:
                break
            dmax = 0.0
            for k in range(K):
                nv = a[k] / tot
                d = abs(nv - p[k])
                if d > dmax:
                    dmax = d
                p[k] = nv
            if dmax < em_tol:
                break
        ll = 0.0
        for t in range(nt):
            g = touched[t]
            q = 0.0
            for k in range(K):
                q += p[k] * Xbar[k, g]
            q = (1.0 - eta) * q + fl
            ll += cnt[g] * np.log(q)
        ell[i] = ll / n
    return ell, Pout


def simulate(Xbar, P0, Pgen, n, eta=0.01, n_em=10, tol=1e-4, seed=0):
    """Deviance and n V for one multinomial replicate per bin drawn from the fitted mixture Pgen."""
    cumX = np.cumsum(Xbar, axis=1)
    cumX /= cumX[:, -1:]
    n_int = np.rint(n).astype(np.int64)
    ell, P = _sim_kernel(np.ascontiguousarray(Pgen), np.ascontiguousarray(cumX), Xbar, P0,
                         n_int, float(eta), int(n_em), float(tol), int(seed))
    H, V = _entropy_var(P, Xbar, float(eta))
    nn = n_int.astype(np.float64)
    return {"dev": nn * (ell + H), "nV": nn * V, "n": nn, "P": P}


def zscore(dev, nV, n, A=None):
    if A is None:
        z = -dev / np.sqrt(np.maximum(nV, 1e-12))
        z[n <= 0] = 0.0
        return z, n
    return -(A @ dev) / np.sqrt(np.maximum(A @ nV, 1e-12)), A @ n


def pool_matrix(model):
    A = getattr(model, "adjacency_", None)
    if A is None:
        return None
    A = sparse.csr_matrix(A, dtype=np.float64)
    A = (A + sparse.identity(A.shape[0], format="csr")).tocsr()
    A.data[:] = 1.0
    return A


# ----------------------------------------------------------------------------------------------
# Calibrations
# ----------------------------------------------------------------------------------------------


def global_null(z, lo_q=0.16):
    mu = float(np.median(z))
    sd = max(mu - float(np.quantile(z, lo_q)), 1e-9)
    return (z - mu) / sd


def depth_strata(logn, n_strata=20, min_per=200):
    """Quantile strata of log depth; returns labels, number of strata, stratum centres."""
    ns = int(max(1, min(n_strata, len(logn) // min_per)))
    edges = np.unique(np.quantile(logn, np.linspace(0, 1, ns + 1)))
    if len(edges) < 2:
        return np.zeros(len(logn), dtype=np.int64), 1
    lab = np.clip(np.searchsorted(edges, logn, side="right") - 1, 0, len(edges) - 2)
    return lab, len(edges) - 1


def depth_null(z, depth, n_strata=20, lo_q=0.16, min_per=200):
    """Candidate (a): centre (median) and left-half scale (median - q16) estimated within quantile
    strata of log depth, linearly interpolated in log depth between stratum centres."""
    logn = np.log(np.maximum(depth, 1.0))
    lab, ns = depth_strata(logn, n_strata, min_per)
    ctr, mu, sd = np.empty(ns), np.empty(ns), np.empty(ns)
    for s in range(ns):
        v = z[lab == s]
        ctr[s] = np.median(logn[lab == s])
        mu[s] = np.median(v)
        sd[s] = max(mu[s] - np.quantile(v, lo_q), 1e-9)
    o = np.argsort(ctr)
    ctr, mu, sd = ctr[o], mu[o], sd[o]
    return (z - np.interp(logn, ctr, mu)) / np.interp(logn, ctr, sd), (np.exp(ctr), mu, sd)


def sim_moments(zs, depth, n_strata=20, min_per=200):
    """Median and symmetric robust scale ((q84 - q16)/2) of simulated null scores per depth
    stratum, interpolated in log depth."""
    logn = np.log(np.maximum(depth, 1.0))
    lab, ns = depth_strata(logn, n_strata, min_per)
    ctr, mu, sd = np.empty(ns), np.empty(ns), np.empty(ns)
    for s in range(ns):
        v = zs[lab == s]
        ctr[s] = np.median(logn[lab == s])
        q16, q50, q84 = np.quantile(v, [0.16, 0.5, 0.84])
        mu[s], sd[s] = q50, max((q84 - q16) / 2, 1e-9)
    o = np.argsort(ctr)
    return ctr[o], mu[o], sd[o]


def apply_moments(z, depth, mom):
    ctr, mu, sd = mom
    logn = np.log(np.maximum(depth, 1.0))
    return (z - np.interp(logn, ctr, mu)) / np.interp(logn, ctr, sd)


def summarize_by_depth(z, depth, n_dec=10, lo_q=0.16, thr=Z95):
    """Per depth decile: median depth, median/q16/q84 of z, left scale, flag rate (z > thr)."""
    q = np.quantile(depth, np.linspace(0, 1, n_dec + 1))
    lab = np.clip(np.searchsorted(q, depth, side="right") - 1, 0, n_dec - 1)
    rows = []
    for d in range(n_dec):
        v = z[lab == d]
        if len(v) == 0:
            continue
        q16, q50, q84 = np.quantile(v, [lo_q, 0.5, 0.84])
        rows.append({"decile": d, "n_bins": len(v), "median_depth": float(np.median(depth[lab == d])),
                     "median": q50, "q16": q16, "q84": q84, "q01": np.quantile(v, 0.01),
                     "q99": np.quantile(v, 0.99), "left_scale": q50 - q16,
                     "flag_rate": float((v > thr).mean())})
    return rows


def re_null(dev, nV, n, n_strata=20, lo_q=0.16, min_per=200):
    """Candidate (d): random-effects null in per-UMI units (see PROTOCOL.md)."""
    ok = n > 0
    u = np.zeros_like(dev)
    v = np.ones_like(dev)
    u[ok] = -dev[ok] / n[ok]
    v[ok] = nV[ok] / n[ok] ** 2
    logn = np.log(np.maximum(n, 1.0))
    lab, ns = depth_strata(logn[ok], n_strata, min_per)
    uo, vo, lo = u[ok], v[ok], logn[ok]
    c, m, t2 = np.empty(ns), np.empty(ns), np.empty(ns)
    for s in range(ns):
        w = lab == s
        c[s] = np.median(lo[w])
        m[s] = np.median(uo[w])
        sl = m[s] - np.quantile(uo[w], lo_q)
        t2[s] = max(sl ** 2 - np.median(vo[w]), 1e-6)
    if ns >= 2:
        b1, b0 = np.polyfit(c, m, 1)
        g1, g0 = np.polyfit(c, np.log(t2), 1)
    else:
        b1, b0, g1, g0 = 0.0, m[0], 0.0, np.log(t2[0])
    mu = b0 + b1 * logn
    tau2 = np.exp(g0 + g1 * logn)
    sc = (u - mu) / np.sqrt(tau2 + v)
    sc[~ok] = 0.0
    return global_null(sc), (b0, b1, g0, g1)


def candidates(obs10, obs0, sim10, A=None):
    """All candidate calibrated scores from observed (EM 10 / EM 0) and simulated (EM 10) arrays.
    Each input is a dict with dev, nV, n. Returns {name: score}."""
    out = {}
    for tag, o in (("10", obs10), ("0", obs0)):
        if o is None:
            continue
        z, depth = zscore(o["dev"], o["nV"], o["n"], A)
        if A is None:
            dev, nV, n = o["dev"], o["nV"], o["n"]
        else:
            dev, nV, n = A @ o["dev"], A @ o["nV"], A @ o["n"]
        out["G" + tag] = global_null(z)
        out["D" + tag] = depth_null(z, depth)[0]
        out["R" + tag] = re_null(dev, nV, n)[0]
        out["C" + tag] = central_null(z)
        zk = z[n > 0] if A is None else z
        mu_c, sd_c = central_moments(zk, drop=1.0)
        med = float(np.median(zk))
        sd_g = max(med - float(np.quantile(zk, 0.16)), 1e-9)
        out["C1_" + tag] = (z - mu_c) / sd_c
        out["H" + tag] = (z - mu_c) / sd_c if sd_c < sd_g else (z - med) / sd_g
        out["RC" + tag] = re_central_null(dev, nV, n)[0]
        if tag == "10":
            out["DC10"] = depth_central_null(z, depth)
        if tag == "10" and sim10 is not None:
            zs, _ = zscore(sim10["dev"], sim10["nV"], sim10["n"], A)
            mom = sim_moments(zs, depth)
            out["S10"] = global_null(apply_moments(z, depth, mom))
            out["S10_modelonly"] = apply_moments(z, depth, mom)
    return out


def central_moments(z, drop=0.5):
    """Efron-style central matching: mode and curvature of the log density near its peak.
    Histogram on the central 99.8% of z, Gaussian-smoothed; quadratic fit of log counts over the
    contiguous window around the mode where the smoothed density is >= exp(-drop) x its peak
    (drop = 0.5 -> about +-1 null SD). Returns (mu0, sigma0)."""
    from scipy.ndimage import gaussian_filter1d
    z = np.asarray(z, dtype=np.float64)
    N = len(z)
    lo, hi = np.quantile(z, [0.001, 0.999])
    if not hi > lo:
        return float(np.median(z)), 1.0
    nb = int(np.clip(N // 100, 30, 400))
    h, e = np.histogram(z, bins=nb, range=(lo, hi))
    c = 0.5 * (e[1:] + e[:-1])
    w = e[1] - e[0]
    # smoothing bandwidth: ~ 1/10 of a robust spread, at least 1 bin
    iqr = np.subtract(*np.quantile(z, [0.75, 0.25]))
    bw = max(0.1 * iqr / 1.349, w)
    hs = gaussian_filter1d(h.astype(np.float64), bw / w, mode="constant")
    i0 = int(np.argmax(hs))
    thr = hs[i0] * np.exp(-drop)
    a = i0
    while a > 0 and hs[a - 1] >= thr:
        a -= 1
    b = i0
    while b < nb - 1 and hs[b + 1] >= thr:
        b += 1
    if b - a < 4:
        a, b = max(0, i0 - 3), min(nb - 1, i0 + 3)
    x, y = c[a:b + 1], np.log(np.maximum(hs[a:b + 1], 1e-12))
    d2, d1, _ = np.polyfit(x, y, 2, w=np.sqrt(np.maximum(hs[a:b + 1], 1e-12)))
    if d2 >= 0:
        return float(np.median(z)), max(float(np.median(z) - np.quantile(z, 0.16)), 1e-9)
    s2 = -1.0 / (2.0 * d2) - bw ** 2
    mu = -d1 / (2.0 * d2)
    return float(mu), float(np.sqrt(max(s2, 1e-12)))


def central_null(z):
    mu, sd = central_moments(z)
    return (z - mu) / sd


def depth_central_null(z, depth, n_strata=10, min_per=2000):
    logn = np.log(np.maximum(depth, 1.0))
    lab, ns = depth_strata(logn, n_strata, min_per)
    ctr, mu, sd = np.empty(ns), np.empty(ns), np.empty(ns)
    for s in range(ns):
        ctr[s] = np.median(logn[lab == s])
        mu[s], sd[s] = central_moments(z[lab == s])
    o = np.argsort(ctr)
    ctr, mu, sd = ctr[o], mu[o], sd[o]
    return (z - np.interp(logn, ctr, mu)) / np.interp(logn, ctr, sd)


def re_central_null(dev, nV, n, n_bisect=40):
    """Post hoc candidate (e): random-effects null in per-UMI units with GLOBAL location mu and
    between-bin heterogeneity tau, both estimated by central matching (robust to both tails):
        u_i = -dev_i / n_i,  v_i = V_i / n_i,  s_i = (u_i - mu) / sqrt(tau^2 + v_i).
    tau is chosen so that the central-matching SD of s equals 1 (bisection on log tau)."""
    ok = n > 0
    u = np.zeros_like(dev)
    v = np.ones_like(dev)
    u[ok] = -dev[ok] / n[ok]
    v[ok] = nV[ok] / n[ok] ** 2
    uo, vo = u[ok], v[ok]
    mu, su = central_moments(uo)

    def sd_of(t):
        return central_moments((uo - mu) / np.sqrt(t * t + vo))

    lo, hi = 0.0, max(4.0 * su, 1e-6)
    m0, s0 = sd_of(lo)
    if s0 <= 1.0:
        tau = 0.0
    else:
        while sd_of(hi)[1] > 1.0:
            hi *= 2.0
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            if sd_of(mid)[1] > 1.0:
                lo = mid
            else:
                hi = mid
        tau = 0.5 * (lo + hi)
    s = np.zeros_like(dev)
    s[ok] = (uo - mu) / np.sqrt(tau * tau + vo)
    ms, ss = central_moments(s[ok])
    s[ok] = (s[ok] - ms) / ss
    return s, (mu, tau)
