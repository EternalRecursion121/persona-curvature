"""FACTOR ANALYSIS over the 134-trait Qwen3.5 sweep Gram matrix.

Adapted from sweep100/analyse_fa.py.  The MATHS (principal axis factoring with
SMC starts, Horn's parallel analysis, varimax x2 + direct oblimin rotation,
Tucker congruence, and the verify() self-test) is IDENTICAL, copied verbatim.
Only the IO head and the labels differ:

  - Input Gram:  results/gram_sweep.npz  (keys G 134x134, names, norms, scale,
    n_modules) instead of gram.npy / gram_names.json.
  - Labels:      traits_primary.json (100 Goldberg markers, 20 per factor) +
    traits_secondary.json (Lexicon traits; 34 of the 40 are in this sweep).
    Names are Capitalised; joined to the npz's lowercase slugs by case-folding.
  - Lexicon traits are INCLUDED in the factoring (they are real variables in
    the Gram) but EXCLUDED from every congruence target: their rows are 0 in
    all five Goldberg marker targets and in the Eval target.
  - No reseed controls and no pca.json exist for this sweep, so the
    effective-dimensionality estimate, the reseed noise floor, and the PCA
    cross-reference of sweep100 are not reproduced; the kernel PCA eigenvalues
    are computed directly from the double-centred Gram instead.

Usage:  ~/cartovenv/bin/python analyse_fa_qwen35.py
Reads:  results/gram_sweep.npz, traits_primary.json, traits_secondary.json
Writes: results/fa_qwen35.json, results/fa_qwen35.md
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.join(HERE, "results")
GRAM_NPZ = os.environ.get("PC_GRAM_NPZ", os.path.join(RDIR, "gram_sweep.npz"))
TAG = os.environ.get("PC_FA_TAG", "")   # suffix for output files, e.g. "_personas"
FACTORS = ["Extraversion", "Agreeableness", "Conscientiousness",
           "EmotionalStability", "Intellect"]
FSHORT = {"Extraversion": "E", "Agreeableness": "A", "Conscientiousness": "C",
          "EmotionalStability": "ES", "Intellect": "I", "Lexicon": "L"}
TCOLS = [FSHORT[f] for f in FACTORS] + ["Eval"]

PAF_TOL = 1e-7          # convergence on max |change in communality|
PAF_MAXIT = 5000
HEYWOOD_CAP = 0.999
RIDGE = 1e-3            # for SMC on the (singular) ipsatised matrix
PA_REPS = 500
PA_GRID = [150, 300, 1000, 1528, 5809, 20000]
PA_REFERENCE_N = 1528   # sweep100's lower effective-dimensionality estimate
SEED = 0


# ===========================================================================
# 1.  MATHS PRIMITIVES  (verbatim from sweep100/analyse_fa.py)
# ===========================================================================

def smc(R, ridge=0.0):
    """Squared multiple correlations: SMC_i = 1 - 1/(R^-1)_ii.

    `ridge` regularises a singular R (needed after ipsatisation, which removes
    exactly one dimension and makes R rank p-1, so every SMC would be 1).
    """
    Rr = R + ridge * np.eye(len(R))
    if ridge:
        d = np.sqrt(np.diag(Rr))
        Rr = Rr / np.outer(d, d)
    return 1.0 - 1.0 / np.diag(np.linalg.inv(Rr))


def paf(R, k, h2_init=None, ridge=0.0, tol=PAF_TOL, maxit=PAF_MAXIT,
        cap=HEYWOOD_CAP):
    """Principal axis factoring, iterated to convergence.

    Start: communalities = SMC.  Each step replaces the diagonal of R by the
    current communalities, eigen-decomposes the REDUCED matrix, keeps the k
    leading eigenpairs as loadings L = U_k diag(sqrt(lambda_k)), and re-estimates
    communalities as the row sums of L^2.  Converged when the largest change in
    any communality is < tol.
    """
    h2 = smc(R, ridge) if h2_init is None else np.asarray(h2_init, float).copy()
    n_hey = 0
    for it in range(1, maxit + 1):
        Rr = R.copy()
        np.fill_diagonal(Rr, h2)
        w, V = np.linalg.eigh(Rr)
        w, V = w[::-1], V[:, ::-1]
        L = V[:, :k] * np.sqrt(np.clip(w[:k], 0.0, None))
        h2n = (L ** 2).sum(1)
        n_hey = int((h2n > cap).sum())
        h2n = np.minimum(h2n, cap)
        dmax = float(np.abs(h2n - h2).max())
        h2 = h2n
        if dmax < tol:
            break
    return {"loadings": L, "communalities": h2, "iterations": it,
            "final_delta": dmax, "reduced_eigenvalues": w,
            "n_heywood": n_hey, "converged": bool(dmax < tol)}


# ------------------------------------------------- varimax, implementation 1
def varimax_pairwise(L, maxit=2000, tol=1e-9):
    """Kaiser (1958) cyclic pairwise varimax: for each pair of factors rotate by
    the angle that maximises the varimax criterion in that plane (closed form).
    """
    L = np.array(L, float)
    p, k = L.shape
    T = np.eye(k)
    for it in range(1, maxit + 1):
        mx = 0.0
        for a in range(k - 1):
            for b in range(a + 1, k):
                x, y = L[:, a], L[:, b]
                u, v = x ** 2 - y ** 2, 2 * x * y
                A, B = u.sum(), v.sum()
                C, D = (u ** 2 - v ** 2).sum(), 2 * (u * v).sum()
                num, den = D - 2 * A * B / p, C - (A ** 2 - B ** 2) / p
                if abs(num) < 1e-300 and abs(den) < 1e-300:
                    continue
                phi = 0.25 * np.arctan2(num, den)
                if abs(phi) < 1e-15:
                    continue
                c, s = np.cos(phi), np.sin(phi)
                L[:, a], L[:, b] = c * x + s * y, -s * x + c * y
                Ta, Tb = T[:, a].copy(), T[:, b].copy()
                T[:, a], T[:, b] = c * Ta + s * Tb, -s * Ta + c * Tb
                mx = max(mx, abs(phi))
        if mx < tol:
            break
    return L, T, it


# ------------------------------------------------- varimax, implementation 2
def _vgQ_varimax(L):
    L2 = L ** 2
    QL = L2 - L2.mean(axis=0, keepdims=True)
    return -(QL ** 2).sum() / 4.0, -L * QL


def varimax_gpa(A, maxit=2000, tol=1e-10):
    """Jennrich (2001) gradient-projection varimax. Algorithmically independent
    of varimax_pairwise; the two are cross-checked against each other."""
    A = np.array(A, float)
    k = A.shape[1]
    T = np.eye(k)
    al = 1.0
    Lr = A @ T
    f, Gq = _vgQ_varimax(Lr)
    G = A.T @ Gq
    for it in range(1, maxit + 1):
        M = T.T @ G
        Gp = G - T @ ((M + M.T) / 2.0)
        s = float(np.sqrt((Gp ** 2).sum()))
        if s < tol:
            break
        al *= 2.0
        for _ in range(60):
            U, _, Vt = np.linalg.svd(T - al * Gp, full_matrices=False)
            Tt = U @ Vt
            Lr = A @ Tt
            ft, Gqt = _vgQ_varimax(Lr)
            if ft < f - 0.5 * s * s * al:
                break
            al /= 2.0
        T, f, Gq = Tt, ft, Gqt
        G = A.T @ Gq
    return A @ T, T, it


def varimax_criterion(L):
    p = L.shape[0]
    L2 = L ** 2
    return float((L2 ** 2).sum() - (L2.sum(0) ** 2).sum() / p)


# ------------------------------------------------- direct oblimin (oblique)
def _vgQ_oblimin(L, gamma):
    p, k = L.shape
    L2 = L ** 2
    N = np.ones((k, k)) - np.eye(k)
    X = L2 @ N if gamma == 0.0 else (np.eye(p) - (gamma / p) * np.ones((p, p))) @ L2 @ N
    return float((L2 * X).sum() / 4.0), L * X


def oblimin(A, gamma=0.0, maxit=5000, tol=1e-9):
    """Direct oblimin by the Jennrich & Bentler (2002) oblique gradient
    projection algorithm.  gamma=0 is direct quartimin, the usual default.

    T is k x k with unit-length columns; the PATTERN matrix is L = A (T^-1)^T
    and the factor correlation matrix is Phi = T^T T.  When T is orthogonal this
    reduces exactly to an orthogonal rotation with Phi = I.
    """
    A = np.array(A, float)
    k = A.shape[1]
    T = np.eye(k)
    al = 1.0
    Ti = np.linalg.inv(T)
    Lr = A @ Ti.T
    f, Gq = _vgQ_oblimin(Lr, gamma)
    G = -(Lr.T @ Gq @ Ti).T
    s = np.inf
    for it in range(1, maxit + 1):
        Gp = G - T * (T * G).sum(axis=0, keepdims=True)
        s = float(np.sqrt((Gp ** 2).sum()))
        if s < tol:
            break
        al *= 2.0
        ok = False
        for _ in range(60):
            X = T - al * Gp
            X = X / np.sqrt((X ** 2).sum(axis=0, keepdims=True))
            try:
                Tix = np.linalg.inv(X)
            except np.linalg.LinAlgError:
                al /= 2.0
                continue
            Lx = A @ Tix.T
            fx, Gqx = _vgQ_oblimin(Lx, gamma)
            if fx < f - 0.5 * s * s * al:
                ok = True
                break
            al /= 2.0
        if not ok:
            break
        T, f, Gq, Ti, Lr = X, fx, Gqx, Tix, Lx
        G = -(Lr.T @ Gq @ Ti).T
    return {"pattern": Lr, "T": T, "Phi": T.T @ T, "iterations": it,
            "criterion": f, "gradient_norm": s}


def tucker(x, y):
    """Tucker's congruence coefficient: phi = <x,y> / sqrt(<x,x><y,y>)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(x @ y / np.sqrt((x @ x) * (y @ y)))


def kaiser_normalise(L, h2):
    s = np.sqrt(h2)
    return L / s[:, None], s


# ===========================================================================
# 2.  VERIFICATION  (verbatim from sweep100/analyse_fa.py)
# ===========================================================================

def verify():
    v = {}
    rng = np.random.default_rng(7)

    # --- eigen-decomposition -------------------------------------------------
    # (a) a 2x2 with an exactly known spectrum: [[2,1],[1,2]] -> 3 and 1,
    #     eigenvectors (1,1)/sqrt2 and (1,-1)/sqrt2.
    w2, V2 = np.linalg.eigh(np.array([[2.0, 1.0], [1.0, 2.0]]))
    # (b) residual + orthonormality on a random 100x100 symmetric matrix
    Ar = rng.normal(size=(100, 100))
    Ar = Ar + Ar.T
    wr, Vr = np.linalg.eigh(Ar)
    res = float(np.abs(Ar @ Vr - Vr * wr).max())
    orth = float(np.abs(Vr.T @ Vr - np.eye(100)).max())
    recon = float(np.abs((Vr * wr) @ Vr.T - Ar).max())
    v["eigh"] = {
        "hand_case_matrix": [[2, 1], [1, 2]],
        "eigenvalues": sorted(w2.tolist(), reverse=True),
        "expected_eigenvalues": [3.0, 1.0],
        "eigenvalue_err": float(abs(sorted(w2)[::-1][0] - 3.0) + abs(sorted(w2)[::-1][1] - 1.0)),
        "random_100x100_max_residual_AV_minus_VL": res,
        "random_100x100_max_orthonormality_err": orth,
        "random_100x100_max_reconstruction_err": recon,
        "ok": bool(abs(sorted(w2)[::-1][0] - 3.0) < 1e-12 and res < 1e-10
                   and orth < 1e-12 and recon < 1e-10)}

    # --- SMC vs an explicit OLS regression -----------------------------------
    X = rng.normal(size=(400, 6))
    X[:, 1] += 0.8 * X[:, 0]
    X[:, 2] += 0.5 * X[:, 0] - 0.4 * X[:, 1]
    Xc = (X - X.mean(0)) / X.std(0)          # ddof=0 throughout, so that
    Rx = Xc.T @ Xc / len(Xc)                 # Rx is exactly the correlation matrix
    smc_inv = smc(Rx)
    smc_ols = []
    for i in range(6):
        y = Xc[:, i]
        Z = np.delete(Xc, i, axis=1)
        Z = np.column_stack([np.ones(len(Z)), Z])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        r2 = 1 - ((y - Z @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        smc_ols.append(float(r2))
    v["smc"] = {"from_inverse": smc_inv.tolist(), "from_ols_regression": smc_ols,
                "max_abs_diff": float(np.abs(smc_inv - np.array(smc_ols)).max()),
                "ok": bool(np.abs(smc_inv - np.array(smc_ols)).max() < 1e-10)}

    # --- varimax: recovery of a KNOWN simple structure ------------------------
    L0 = np.zeros((8, 2))
    L0[:4, 0] = [0.8, 0.7, 0.9, 0.6]
    L0[4:, 1] = [0.75, 0.85, 0.65, 0.70]
    th = 0.6
    Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = L0 @ Rot                      # destroy simple structure by 0.6 rad

    def canon(L):
        L = L[:, np.argsort(-(L ** 2).sum(0))].copy()
        for j in range(L.shape[1]):
            if L[np.argmax(np.abs(L[:, j])), j] < 0:
                L[:, j] *= -1
        return L

    Lp, _, ip = varimax_pairwise(A)
    Lg, _, ig = varimax_gpa(A)
    e_p = float(np.abs(canon(Lp) - canon(L0)).max())
    e_g = float(np.abs(canon(Lg) - canon(L0)).max())

    # --- varimax: two independent algorithms agree on random loadings --------
    agree, crits = [], []
    for _ in range(5):
        Ar2 = rng.normal(size=(20, 4))
        L1, _, _ = varimax_pairwise(Ar2)
        L2, _, _ = varimax_gpa(Ar2)
        agree.append(float(np.abs(canon(L1) - canon(L2)).max()))
        crits.append([varimax_criterion(L1), varimax_criterion(L2)])
        # rotation must preserve communalities exactly
        agree.append(float(np.abs((L1 ** 2).sum(1) - (Ar2 ** 2).sum(1)).max()))
    v["varimax"] = {
        "known_simple_structure": {
            "true_loadings": L0.tolist(),
            "input_rotated_by_rad": th,
            "max_abs_err_pairwise": e_p, "max_abs_err_gpa": e_g,
            "iterations_pairwise": ip, "iterations_gpa": ig},
        "two_algorithms_max_disagreement": max(agree),
        "criteria_pairwise_vs_gpa": crits,
        "ok": bool(e_p < 1e-10 and e_g < 1e-6 and max(agree) < 1e-5)}

    # --- oblimin: recovery of a KNOWN oblique structure -----------------------
    k, p = 3, 15
    Lo0 = np.zeros((p, k))
    for j in range(k):
        Lo0[j * 5:(j + 1) * 5, j] = [0.8, 0.7, 0.75, 0.65, 0.85]
    Phi0 = np.array([[1.0, 0.5, 0.4], [0.5, 1.0, 0.6], [0.4, 0.6, 1.0]])
    Rstar = Lo0 @ Phi0 @ Lo0.T
    w, V = np.linalg.eigh(Rstar)
    w, V = w[::-1], V[:, ::-1]
    Aun = V[:, :k] * np.sqrt(np.clip(w[:k], 0, None))
    ob = oblimin(Aun, 0.0)
    Lr, Phi = ob["pattern"], ob["Phi"]
    # match recovered factors to the true ones by |congruence|
    perm, sgn = [], []
    for j in range(k):
        c = [tucker(Lr[:, i], Lo0[:, j]) for i in range(k)]
        i = int(np.argmax(np.abs(c)))
        perm.append(i)
        sgn.append(1.0 if c[i] > 0 else -1.0)
    Lm = Lr[:, perm] * np.array(sgn)
    Pm = (Phi[np.ix_(perm, perm)] * np.outer(sgn, sgn))
    v["oblimin"] = {
        "true_pattern": Lo0.tolist(), "true_Phi": Phi0.tolist(),
        "recovered_pattern_matched": np.round(Lm, 6).tolist(),
        "recovered_Phi_matched": np.round(Pm, 6).tolist(),
        "max_abs_err_pattern": float(np.abs(Lm - Lo0).max()),
        "max_abs_err_Phi": float(np.abs(Pm - Phi0).max()),
        "unrotated_reproduces_Rstar": float(np.abs(Aun @ Aun.T - Rstar).max()),
        "rotation_preserves_L_Phi_Lt": float(np.abs(Lr @ Phi @ Lr.T - Rstar).max()),
        "iterations": ob["iterations"], "gradient_norm": ob["gradient_norm"],
        "ok": bool(np.abs(Lm - Lo0).max() < 1e-6 and np.abs(Pm - Phi0).max() < 1e-6
                   and np.abs(Lr @ Phi @ Lr.T - Rstar).max() < 1e-10)}

    # --- Tucker congruence against a hand-computed case ----------------------
    # x = (1,2,3), y = (1,0,-1):  <x,y> = 1 + 0 - 3 = -2
    # <x,x> = 14, <y,y> = 2  ->  phi = -2/sqrt(28) = -0.3779644730092272
    got = tucker([1, 2, 3], [1, 0, -1])
    exp = -2.0 / np.sqrt(28.0)
    # second case: identical vectors -> 1; sign-flipped -> -1; scale invariance
    v["tucker"] = {
        "hand_case": {"x": [1, 2, 3], "y": [1, 0, -1],
                      "numerator": -2.0, "xx": 14.0, "yy": 2.0,
                      "expected": exp, "got": got, "err": abs(got - exp)},
        "identical": tucker([3, -1, 2], [3, -1, 2]),
        "sign_flipped": tucker([3, -1, 2], [-3, 1, -2]),
        "scale_invariance_err": abs(tucker([1, 2, 3], [1, 0, -1])
                                    - tucker([10, 20, 30], [-0.5, 0, 0.5]) * -1),
        "ok": bool(abs(got - exp) < 1e-15
                   and abs(tucker([3, -1, 2], [3, -1, 2]) - 1) < 1e-15
                   and abs(tucker([3, -1, 2], [-3, 1, -2]) + 1) < 1e-15)}

    # --- PAF: recovers a known factor model ----------------------------------
    Lt = np.zeros((12, 2))
    Lt[:6, 0] = 0.7
    Lt[6:, 1] = 0.6
    Rt = Lt @ Lt.T
    np.fill_diagonal(Rt, 1.0)
    r = paf(Rt, 2)
    Lf = canon(r["loadings"])
    v["paf"] = {"true_loadings_col_sums": Lt.sum(0).tolist(),
                "recovered_communalities": np.round(r["communalities"], 6).tolist(),
                "expected_communalities": (Lt ** 2).sum(1).tolist(),
                "max_abs_err_communality":
                    float(np.abs(r["communalities"] - (Lt ** 2).sum(1)).max()),
                "iterations": r["iterations"],
                "offdiag_reproduction_err":
                    float(np.abs((Lf @ Lf.T - Lt @ Lt.T)[~np.eye(12, dtype=bool)]).max()),
                "ok": bool(np.abs(r["communalities"] - (Lt ** 2).sum(1)).max() < 1e-6)}

    v["all_ok"] = bool(all(v[k2]["ok"] for k2 in
                           ["eigh", "smc", "varimax", "oblimin", "tucker", "paf"]))
    return v


# ===========================================================================
# 3.  PARALLEL ANALYSIS  (verbatim from sweep100/analyse_fa.py)
# ===========================================================================

def parallel_analysis(R_obs, n_obs_grid, p, reps=PA_REPS, ipsatise=False,
                      ridge=0.0, seed=SEED):
    """Horn's parallel analysis: eigenvalues of correlation matrices of random
    data with the same number of variables (p) and N observations, compared
    rank-for-rank against the observed eigenvalues.  Retain k = the number of
    LEADING eigenvalues that exceed the null (first-crossing rule).

    Two variants: on the raw correlation matrix (Horn's original, PCA-based) and
    on the SMC-reduced matrix (the variant appropriate to PAF).  If the observed
    matrix was ipsatised, the null data is ipsatised the same way.
    """
    def red(M):
        Mr = M.copy()
        np.fill_diagonal(Mr, smc(M, ridge))
        return np.linalg.eigvalsh(Mr)[::-1]

    obs_un = np.linalg.eigvalsh(R_obs)[::-1]
    obs_red = red(R_obs)

    def first_cross(obs, null):
        k = 0
        for i in range(len(obs)):
            if obs[i] > null[i]:
                k = i + 1
            else:
                break
        return k

    rng = np.random.default_rng(seed)
    grid = []
    for N in n_obs_grid:
        U, D = [], []
        for _ in range(reps):
            X = rng.standard_normal((N, p))
            if ipsatise:
                X = X - X.mean(1, keepdims=True)
            X = X - X.mean(0)
            X = X / X.std(0)
            Rr = X.T @ X / (N - 1)
            U.append(np.linalg.eigvalsh(Rr)[::-1])
            D.append(red(Rr))
        U, D = np.array(U), np.array(D)
        u95, r95 = np.percentile(U, 95, 0), np.percentile(D, 95, 0)
        grid.append({"N": N, "reps": reps,
                     "k_unreduced_95pct": first_cross(obs_un, u95),
                     "k_unreduced_mean": first_cross(obs_un, U.mean(0)),
                     "k_reduced_95pct": first_cross(obs_red, r95),
                     "k_reduced_mean": first_cross(obs_red, D.mean(0)),
                     "null_unreduced_95pct": u95[:12].tolist(),
                     "null_reduced_95pct": r95[:12].tolist()})
    return {"observed_unreduced": obs_un.tolist(),
            "observed_reduced": obs_red.tolist(), "grid": grid}


# ===========================================================================
# 4.  SOLUTION ASSEMBLY  (verbatim from sweep100/analyse_fa.py)
# ===========================================================================

def order_and_orient(L, targets, Phi=None):
    """Order factors by descending SS loading; orient each so its congruence
    with its own best-matching Goldberg target is POSITIVE (factor sign is
    arbitrary in FA, so this is a labelling convention only)."""
    k = L.shape[1]
    order = list(np.argsort(-(L ** 2).sum(0)))
    L = L[:, order]
    sgn = np.ones(k)
    for j in range(k):
        c = [tucker(L[:, j], targets[:, a]) for a in range(5)]
        a = int(np.argmax(np.abs(c)))
        if c[a] < 0:
            sgn[j] = -1.0
    L = L * sgn
    if Phi is not None:
        Phi = Phi[np.ix_(order, order)] * np.outer(sgn, sgn)
    return L, Phi, order, sgn


def solution(R, k, targets, ridge=0.0, label=""):
    ex = paf(R, k, ridge=ridge)
    L0, h2 = ex["loadings"], ex["communalities"]
    Ln, s = kaiser_normalise(L0, h2)

    Lv, Tv, itv = varimax_pairwise(Ln)
    Lv = Lv * s[:, None]
    Lv_g, _, _ = varimax_gpa(Ln)
    vm_agree = float(np.abs(np.sort(np.abs(Lv / s[:, None]), axis=None)
                            - np.sort(np.abs(Lv_g), axis=None)).max())

    ob = oblimin(Ln, 0.0)
    Lo = ob["pattern"] * s[:, None]
    Phi = ob["Phi"]

    Lu, _, _, _ = order_and_orient(L0.copy(), targets)
    Lv, _, _, _ = order_and_orient(Lv, targets)
    Lo, Phi, _, _ = order_and_orient(Lo, targets, Phi)

    out = {"k": k, "label": label,
           "paf_iterations": ex["iterations"], "paf_final_delta": ex["final_delta"],
           "paf_converged": ex["converged"], "n_heywood": ex["n_heywood"],
           "reduced_eigenvalues": ex["reduced_eigenvalues"][:20].tolist(),
           "communalities": h2.tolist(),
           "varimax_iterations": itv,
           "varimax_two_algorithm_max_disagreement": vm_agree,
           "oblimin_iterations": ob["iterations"],
           "oblimin_gradient_norm": ob["gradient_norm"],
           "oblimin_criterion": ob["criterion"],
           "Phi": Phi.tolist(),
           "reproduction_err_L_Phi_Lt_vs_unrotated":
               float(np.abs(Lo @ Phi @ Lo.T - L0 @ L0.T).max()),
           "loadings": {"unrotated": Lu.tolist(), "varimax": Lv.tolist(),
                        "oblimin": Lo.tolist()},
           "ss_loadings": {"unrotated": (Lu ** 2).sum(0).tolist(),
                           "varimax": (Lv ** 2).sum(0).tolist(),
                           "oblimin": (Lo ** 2).sum(0).tolist()}}
    for rot, M in [("unrotated", Lu), ("varimax", Lv), ("oblimin", Lo)]:
        out["congruence_" + rot] = [
            [tucker(M[:, j], targets[:, a]) for a in range(6)] for j in range(k)]
    return out


# ===========================================================================
# 5.  MAIN  (qwen35 IO head; the sweep100 analysis flow minus the pieces this
#     sweep has no data for: entry-mean check, reseed controls, pca.json)
# ===========================================================================

def slugify(t):
    return str(t).lower().replace(" ", "_").replace("-", "_")


def load_data():
    """Gram + labels, joined by slug, in npz order."""
    d = np.load(GRAM_NPZ, allow_pickle=True)
    G = np.asarray(d["G"], dtype=np.float64)
    names = [str(n) for n in d["names"]]
    norms = np.asarray(d["norms"], dtype=np.float64)
    assert G.shape == (len(names), len(names)) and len(names) in (100, 134), G.shape
    assert np.allclose(np.sqrt(np.diag(G)), norms, rtol=1e-8), \
        "npz norms disagree with the Gram diagonal"

    prim = json.load(open(os.path.join(HERE, "traits_primary.json")))
    seco = json.load(open(os.path.join(HERE, "traits_secondary.json")))
    assert len(prim) == 100 and all(t["factor"] in FACTORS for t in prim)
    assert all(t["factor"] == "Lexicon" for t in seco)
    by_slug = {slugify(t["trait"]): t for t in prim + seco}
    assert len(by_slug) == len(prim) + len(seco), "slug collision"
    missing = [n for n in names if n not in by_slug]
    assert not missing, f"npz traits without a label: {missing}"

    traits = [by_slug[n] for n in names]
    n_prim = sum(1 for t in traits if t["factor"] != "Lexicon")
    assert n_prim == 100, n_prim
    meta = {"scale_alpha_over_r": float(d["scale"]),
            "n_modules": int(d["n_modules"]),
            "n_traits": len(names), "n_primary": n_prim,
            "n_lexicon": len(names) - n_prim,
            "lexicon_absent_from_sweep":
                sorted(s for s in by_slug if s not in set(names))}
    return G, names, traits, meta


def main():
    os.makedirs(RDIR, exist_ok=True)
    out = {}

    print("verifying maths ...", file=sys.stderr)
    out["verification"] = verify()
    assert out["verification"]["all_ok"], out["verification"]
    print("  all verification checks PASSED", file=sys.stderr)

    # ---- data -------------------------------------------------------------
    G, names, traits, meta = load_data()
    p = len(names)
    label = np.array([t["trait"] for t in traits])
    factor = np.array([t["factor"] for t in traits])
    keyed = np.array([t["keyed"] for t in traits])
    pole = np.where(keyed == "+", 1.0, -1.0)
    is_marker = factor != "Lexicon"
    out["setup"] = meta

    # ---- targets ----------------------------------------------------------
    # Lexicon traits are variables in the factoring but NOT part of any
    # congruence target: their target rows are 0 everywhere, including Eval.
    targets = np.zeros((p, 6))
    for a, F in enumerate(FACTORS):
        targets[:, a] = np.where(factor == F, pole, 0.0)
    targets[:, 5] = np.where(is_marker, pole, 0.0)   # general evaluative target
    TT = np.array([[tucker(targets[:, i], targets[:, j]) for j in range(6)]
                   for i in range(6)])
    out["targets"] = {
        "definition": "target_F[i] = +1 if trait i is a positively-keyed marker "
                      "of Goldberg factor F, -1 if negatively-keyed marker of F, "
                      "0 if it belongs to another factor OR is a Lexicon trait. "
                      "'Eval' is the general evaluative target: +1/-1 by keying "
                      "for every PRIMARY trait, 0 for Lexicon traits.",
        "labels": TCOLS, "target_target_congruence": TT.tolist(),
        "n_markers_per_factor": {F: int((factor == F).sum()) for F in FACTORS},
        "note": "The 34 Lexicon traits load freely in the factoring but score 0 "
                "in every target, so congruence is judged on the 100 Goldberg "
                "markers only. Each Goldberg target still has congruence "
                "20/sqrt(20*100) = 0.447 with the Eval target BY CONSTRUCTION."}

    # ---- the correlation matrix -------------------------------------------
    d = np.sqrt(np.diag(G))
    R = G / np.outer(d, d)                                   # uncentred
    H = np.eye(p) - np.ones((p, p)) / p
    Gc = 0.5 * ((H @ G @ H) + (H @ G @ H).T)
    dc = np.sqrt(np.diag(Gc))
    Rc = Gc / np.outer(dc, dc)                               # ipsatised
    off = ~np.eye(p, dtype=bool)
    iu = np.triu_indices(p, 1)

    ev_u = np.linalg.eigvalsh(R)[::-1]
    ev_c = np.linalg.eigvalsh(Rc)[::-1]
    out["correlation_matrix"] = {
        "centring_note":
            "No entry-level mean check here (it needs the raw adapter files, "
            "which live on the Modal volume, not this box); in sweep100 the "
            "entry-level correction was < 1e-6 relative, and these adapters "
            "have the same architecture of construction. Across-trait centring "
            "(ipsatisation) is reported alongside the raw cosine matrix, as in "
            "sweep100.",
        "uncentred_offdiag": {"mean": float(R[off].mean()), "sd": float(R[off].std()),
                              "min": float(R[off].min()), "max": float(R[off].max())},
        "centred_offdiag": {"mean": float(Rc[off].mean()), "sd": float(Rc[off].std()),
                            "min": float(Rc[off].min()), "max": float(Rc[off].max())},
        "corr_between_the_two_offdiag": float(np.corrcoef(R[iu], Rc[iu])[0, 1]),
        "uncentred_eigenvalues": ev_u.tolist(),
        "centred_eigenvalues": ev_c.tolist(),
        "uncentred_condition_number": float(ev_u[0] / ev_u[-1]),
        "centred_is_singular": True,
        "centred_singularity_note":
            "Ipsatisation removes exactly one dimension, so R_centred has rank "
            "p-1 and its smallest eigenvalue is 0. SMC = 1 - 1/(R^-1)_ii is then "
            f"identically 1, so the centred run starts PAF from ridge-SMC "
            f"(ridge={RIDGE})."}

    # ---- kernel PCA of the double-centred Gram (replaces pca.json) --------
    lc = np.clip(np.linalg.eigvalsh(Gc)[::-1], 0.0, None)
    out["pca_from_gram"] = {
        "what": "eigenvalues of the double-centred Gram (kernel PCA); computed "
                "inline because this sweep has no pca.json. No reseed controls "
                "exist for this sweep, so there is NO noise-floor estimate.",
        "centered_eigenvalues": lc[:20].tolist(),
        "centered_var_pct": (100.0 * lc / lc.sum())[:12].tolist()}

    # ---- SMCs -------------------------------------------------------------
    smc_u = smc(R)
    smc_c = smc(Rc, RIDGE)
    out["smc"] = {"uncentred": {"min": float(smc_u.min()), "mean": float(smc_u.mean()),
                                "max": float(smc_u.max())},
                  "centred_ridge": {"ridge": RIDGE, "min": float(smc_c.min()),
                                    "mean": float(smc_c.mean()), "max": float(smc_c.max())},
                  "per_trait_uncentred": {label[i]: float(smc_u[i]) for i in range(p)}}

    # ---- start-value sensitivity for the centred run ----------------------
    maxabs = np.array([np.max(np.abs(np.delete(Rc[i], i))) for i in range(p)])
    starts = {"ridge_smc": smc_c, "constant_0.5": np.full(p, 0.5), "max_abs_r": maxabs}
    sens = {}
    for nm, h0 in starts.items():
        rr = paf(Rc, 5, h2_init=h0)
        sens[nm] = {"iterations": rr["iterations"],
                    "sum_communalities": float(rr["communalities"].sum())}
    out["paf_start_sensitivity_centred_k5"] = sens

    # ---- number of factors ------------------------------------------------
    print("parallel analysis (uncentred) ...", file=sys.stderr)
    pa_u = parallel_analysis(R, PA_GRID, p, ipsatise=False, ridge=0.0)
    print("parallel analysis (centred) ...", file=sys.stderr)
    pa_c = parallel_analysis(Rc, PA_GRID, p, ipsatise=True, ridge=RIDGE)

    ref = next(g for g in pa_c["grid"] if g["N"] == PA_REFERENCE_N)
    K_PA = ref["k_unreduced_95pct"]
    out["n_factors"] = {
        "kaiser_uncentred_eig_gt_1": int((ev_u > 1).sum()),
        "kaiser_centred_eig_gt_1": int((ev_c > 1).sum()),
        "reduced_eig_gt_1_uncentred": int((np.array(pa_u["observed_reduced"]) > 1).sum()),
        "parallel_analysis_uncentred": pa_u,
        "parallel_analysis_centred": pa_c,
        "reference_N": PA_REFERENCE_N,
        "chosen": K_PA,
        "chosen_rationale":
            f"k = Horn's original (unreduced, 95th-percentile) parallel analysis "
            f"on the ipsatised matrix at N={PA_REFERENCE_N}. This sweep has no "
            f"reseed controls, so the effective dimensionality of weight space "
            f"cannot be re-estimated here; N={PA_REFERENCE_N} is carried over "
            f"from sweep100's lower estimate (m in [1528, 5809]). The full grid "
            f"is reported so the sensitivity of k to N is visible. Solutions "
            f"are also extracted at exactly 5 because 5 is the hypothesis."}

    # ---- solutions ----------------------------------------------------------
    sols = {}
    for tag, M, ridge in [("centred", Rc, RIDGE), ("uncentred", R, 0.0)]:
        for k in sorted({K_PA, 5}):
            if k < 1:
                # A null arm can retain zero factors (parallel analysis crossing
                # at rank 0); PAF/rotation are undefined for k = 0. The count
                # itself stays in n_factors.chosen.
                continue
            key = f"{tag}_k{k}"
            print(f"solution {key} ...", file=sys.stderr)
            sols[key] = solution(M, k, targets, ridge=ridge, label=key)
    out["solutions"] = sols
    out["trait_order"] = label.tolist()
    out["trait_slug"] = list(names)
    out["trait_factor"] = factor.tolist()
    out["trait_keyed"] = keyed.tolist()

    # ---- communality / uniqueness (no reseed noise floor available) --------
    uni = {}
    for key, s in sols.items():
        h2 = np.array(s["communalities"])
        uni[key] = {
            "communality": {"min": float(h2.min()), "mean": float(h2.mean()),
                            "max": float(h2.max())},
            "uniqueness": {"min": float(1 - h2.max()), "mean": float(1 - h2.mean()),
                           "max": float(1 - h2.min())}}
    out["uniqueness"] = {
        "per_solution": uni,
        "note": "No reseed controls exist in this sweep, so the test-retest "
                "reliability / noise-floor comparison of sweep100 section 10 "
                "cannot be reproduced."}

    # ---- steering block: what steer134_on_modal.py consumes -----------------
    steer_key = f"centred_k{K_PA}" if K_PA >= 1 else "centred_k5"
    Ls = np.array(sols[steer_key]["loadings"]["oblimin"])
    Cs = np.array(sols[steer_key]["congruence_oblimin"])
    best = [TCOLS[int(np.argmax(np.abs(Cs[j, :5])))] for j in range(Ls.shape[1])]
    out["steering"] = {
        "solution": steer_key,
        "chosen_k_from_parallel_analysis": int(K_PA),
        "fallback_note": None if K_PA >= 1 else
            "parallel analysis retained 0 factors on this matrix, so there is no "
            "centred_k0 solution; the k=5 solution is reported here instead and "
            "nothing should be steered from it.",
        "slug_order": list(names),
        "k": int(Ls.shape[1]),
        "oblimin_loadings": {names[i]: Ls[i].tolist() for i in range(p)},
        "best_goldberg_per_factor": best,
        "best_goldberg_congruence": [
            float(np.abs(Cs[j, :5]).max()) for j in range(Ls.shape[1])],
        "note": "fa<j> steering direction = column j-1 of these oblimin "
                "loadings, mean-centred and normalised to unit norm in the "
                "double-centred Gram metric (see steer134_on_modal.py "
                "fa_coeffs)."}

    per_trait = {}
    s5 = sols["centred_k5"]
    for i in range(p):
        per_trait[label[i]] = {
            "factor": factor[i], "keyed": keyed[i],
            "smc": float(smc_u[i]),
            "communality_centred_k5": s5["communalities"][i],
            "uniqueness_centred_k5": 1 - s5["communalities"][i],
            "oblimin_loadings_centred_k5": s5["loadings"]["oblimin"][i]}
    out["per_trait"] = per_trait

    json.dump(out, open(f"{RDIR}/fa_qwen35{TAG}.json", "w"), indent=1)
    write_md(out, label, factor, keyed)
    print(f"wrote results/fa_qwen35{TAG}.json and results/fa_qwen35{TAG}.md", file=sys.stderr)
    return out


# ===========================================================================
# 6.  MARKDOWN
# ===========================================================================

def scree(vals, n=15, width=52):
    mx = max(vals[:n])
    L = []
    for i, v in enumerate(vals[:n]):
        bar = "#" * max(int(round(width * v / mx)), 0)
        L.append(f"  {i+1:2d} {v:8.3f} |{bar}")
    return "\n".join(L)


def write_md(o, label, factor, keyed):
    L = []
    W = L.append
    S = o["solutions"]
    K = o["n_factors"]["chosen"]
    keys = sorted(S)
    p = o["setup"]["n_traits"]

    W(f"# Factor analysis of the {p}-trait Qwen3.5 sweep Gram\n")
    W(f"Adapted from sweep100's `analyse_fa.py` with identical maths. "
      f"{o['setup']['n_primary']} Goldberg markers (20 per factor) + "
      f"{o['setup']['n_lexicon']} Lexicon traits; the Lexicon traits are "
      f"factored but excluded from every congruence target. This sweep has no "
      f"reseed controls and no pca.json, so the effective-dimensionality "
      f"estimate and the noise-floor sections of sweep100 are not reproduced; "
      f"the kernel-PCA eigenvalues are computed from the double-centred Gram "
      f"inline.\n")

    # ---- verification
    v = o["verification"]
    W("\n## 1. Verification\n")
    W("| check | result |")
    W("|---|---|")
    W(f"| eigh hand case [[2,1],[1,2]] | err {v['eigh']['eigenvalue_err']:.1e} |")
    W(f"| SMC vs explicit OLS | max abs diff {v['smc']['max_abs_diff']:.1e} |")
    W(f"| varimax recovers known structure | pairwise "
      f"{v['varimax']['known_simple_structure']['max_abs_err_pairwise']:.1e}, "
      f"GPA {v['varimax']['known_simple_structure']['max_abs_err_gpa']:.1e} |")
    W(f"| two varimax algorithms agree | "
      f"{v['varimax']['two_algorithms_max_disagreement']:.1e} |")
    W(f"| oblimin recovers known oblique structure | pattern "
      f"{v['oblimin']['max_abs_err_pattern']:.1e}, Phi "
      f"{v['oblimin']['max_abs_err_Phi']:.1e} |")
    W(f"| Tucker hand case | err {v['tucker']['hand_case']['err']:.1e} |")
    W(f"| PAF recovers known model | "
      f"{v['paf']['max_abs_err_communality']:.1e} |")
    W(f"\nAll checks pass (`all_ok = {v['all_ok']}`).\n")

    # ---- correlation matrix
    cm = o["correlation_matrix"]
    W("\n## 2. The correlation matrix\n")
    W("| | off-diagonal mean | sd | min | max | eigenvalues 1-6 |")
    W("|---|---|---|---|---|---|")
    W(f"| raw cosine (uncentred) | **{cm['uncentred_offdiag']['mean']:+.4f}** | "
      f"{cm['uncentred_offdiag']['sd']:.4f} | {cm['uncentred_offdiag']['min']:+.3f} | "
      f"{cm['uncentred_offdiag']['max']:+.3f} | " +
      ", ".join(f"{x:.2f}" for x in cm["uncentred_eigenvalues"][:6]) + " |")
    W(f"| ipsatised (grand mean removed) | **{cm['centred_offdiag']['mean']:+.4f}** | "
      f"{cm['centred_offdiag']['sd']:.4f} | {cm['centred_offdiag']['min']:+.3f} | "
      f"{cm['centred_offdiag']['max']:+.3f} | " +
      ", ".join(f"{x:.2f}" for x in cm["centred_eigenvalues"][:6]) + " |")
    W(f"\nOff-diagonal agreement between the two: r = "
      f"{cm['corr_between_the_two_offdiag']:.4f}. "
      f"Uncentred condition number {cm['uncentred_condition_number']:.1f}. "
      f"{cm['centred_singularity_note']}\n")
    W("PAF start-value sensitivity on the centred matrix (k=5):\n")
    W("| start | iterations | sum of communalities |")
    W("|---|---|---|")
    for nm, s in o["paf_start_sensitivity_centred_k5"].items():
        W(f"| {nm} | {s['iterations']} | {s['sum_communalities']:.6f} |")

    # ---- number of factors
    nf = o["n_factors"]
    pau, pac = nf["parallel_analysis_uncentred"], nf["parallel_analysis_centred"]
    W("\n## 3. How many factors?\n")
    W("Reduced (SMC-diagonal) eigenvalues, centred:\n")
    W("```")
    W("rank  eigenvalue (reduced, centred)")
    W(scree(pac["observed_reduced"]))
    W("```")
    W(f"\n- Kaiser (unreduced eig > 1): {nf['kaiser_uncentred_eig_gt_1']} "
      f"uncentred, {nf['kaiser_centred_eig_gt_1']} centred.")
    W(f"- Reduced eigenvalues > 1: {nf['reduced_eig_gt_1_uncentred']} (uncentred).\n")
    W("Horn's parallel analysis (95th percentile, first-crossing, "
      f"{PA_REPS} reps per cell). N grid carried over from sweep100; this "
      "sweep has no reseed controls to re-estimate the effective "
      "dimensionality, so read the grid as a sensitivity table.\n")
    W("| N | uncentred: k (Horn, unreduced) | uncentred: k (SMC-reduced) | "
      "centred: k (Horn, unreduced) | centred: k (SMC-reduced) |")
    W("|---|---|---|---|---|")
    for a, b in zip(pau["grid"], pac["grid"]):
        star = " <-" if a["N"] == nf["reference_N"] else ""
        W(f"| {a['N']}{star} | **{a['k_unreduced_95pct']}** | {a['k_reduced_95pct']} | "
          f"**{b['k_unreduced_95pct']}** | {b['k_reduced_95pct']} |")
    W(f"\n**Chosen k = {K}** ({nf['chosen_rationale']})\n")

    # ---- solutions
    W("\n## 4. Principal axis factoring\n")
    W("| solution | PAF iters | final delta | Heywood | varimax iters | "
      "oblimin iters | oblimin |grad| | `L Phi L'` vs unrotated |")
    W("|---|---|---|---|---|---|---|---|")
    for key in keys:
        s = S[key]
        W(f"| {key} | {s['paf_iterations']} | {s['paf_final_delta']:.1e} | "
          f"{s['n_heywood']} | {s['varimax_iterations']} | {s['oblimin_iterations']} | "
          f"{s['oblimin_gradient_norm']:.1e} | "
          f"{s['reproduction_err_L_Phi_Lt_vs_unrotated']:.1e} |")

    # ---- oblimin Phi
    W("\n## 5. The oblimin factor correlation matrix\n")
    for key in keys:
        s = S[key]
        k = s["k"]
        Phi = np.array(s["Phi"])
        W(f"\n**{key}** -- Phi (direct oblimin, gamma=0):\n")
        W("| | " + " | ".join(f"F{j+1}" for j in range(k)) + " |")
        W("|---|" + "---|" * k)
        for i in range(k):
            W(f"| **F{i+1}** | " + " | ".join(f"{Phi[i,j]:+.3f}" for j in range(k)) + " |")
        offv = Phi[~np.eye(k, dtype=bool)]
        W(f"\nLargest |correlation| {np.abs(offv).max():.3f}; mean |correlation| "
          f"{np.abs(offv).mean():.3f}.")

    # ---- congruence
    tg = o["targets"]
    W("\n## 6. Tucker congruence with the Goldberg marker targets\n")
    W(f"**Target definition.** {tg['definition']}\n")
    W(f"{tg['note']}\n")
    for key in keys:
        s = S[key]
        k = s["k"]
        W(f"\n### {key}\n")
        for rot in ["unrotated", "varimax", "oblimin"]:
            C = np.array(s["congruence_" + rot])
            W(f"\n**{rot}**:\n")
            W("| factor | SS loading | " + " | ".join(TCOLS) + " | best Goldberg |")
            W("|---|---|" + "---|" * 7)
            for j in range(k):
                b = int(np.argmax(np.abs(C[j, :5])))
                cells = []
                for a in range(6):
                    x = C[j, a]
                    cells.append(f"**{x:+.3f}**" if abs(x) >= 0.85 and a < 5
                                 else f"{x:+.3f}")
                W(f"| F{j+1} | {s['ss_loadings'][rot][j]:.2f} | " + " | ".join(cells) +
                  f" | {TCOLS[b]} {abs(C[j,b]):.2f} |")

    W("\n### Which Goldberg factors clear the thresholds?\n")
    for key in keys:
        s = S[key]
        C = np.abs(np.array(s["congruence_oblimin"]))[:, :5]
        best = C.max(0)
        line = ", ".join(f"{TCOLS[a]} {best[a]:.3f}" for a in range(5))
        fair = [TCOLS[a] for a in range(5) if best[a] >= 0.85]
        eqv = [TCOLS[a] for a in range(5) if best[a] >= 0.95]
        W(f"- **{key}, oblimin** -- best congruence per Goldberg factor: {line}. "
          f"Clearing 0.85 (fair): {fair or '**none**'}. Clearing 0.95 "
          f"(equivalent): {eqv or '**none**'}.")

    # ---- top loadings
    W(f"\n## 7. Top-10 loading traits per factor\n")
    W("Ranked by |loading|. Factor codes: E Extraversion, A Agreeableness, "
      "C Conscientiousness, ES EmotionalStability, I Intellect, L Lexicon.\n")
    lab = np.array(label)
    fac = np.array(factor)
    kyd = np.array(keyed)
    for key in keys:
        s = S[key]
        k = s["k"]
        M = np.array(s["loadings"]["oblimin"])
        C = np.array(s["congruence_oblimin"])
        W(f"\n### {key}, OBLIMIN pattern\n")
        for j in range(k):
            b = int(np.argmax(np.abs(C[j, :5])))
            W(f"\n**F{j+1}** (SS {s['ss_loadings']['oblimin'][j]:.2f}; best Goldberg "
              f"match {TCOLS[b]}, phi = {C[j,b]:+.3f}; Eval phi = {C[j,5]:+.3f})\n")
            W("| rank | trait | loading | factor | keyed |")
            W("|---|---|---|---|---|")
            for rk, i in enumerate(np.argsort(-np.abs(M[:, j]))[:10]):
                W(f"| {rk+1} | {lab[i]} | {M[i,j]:+.3f} | {FSHORT[fac[i]]} | {kyd[i]} |")

    # ---- evaluative factor
    W("\n## 8. Does an evaluative factor survive rotation?\n")
    W("| solution | rotation | max Eval congruence over factors | which factor |")
    W("|---|---|---|---|")
    for key in keys:
        s = S[key]
        for rot in ["unrotated", "varimax", "oblimin"]:
            C = np.array(s["congruence_" + rot])
            j = int(np.argmax(np.abs(C[:, 5])))
            W(f"| {key} | {rot} | **{abs(C[j,5]):.3f}** | F{j+1} |")
    W("\n(Recall the 0.447 construction baseline: a perfectly clean Goldberg "
      "factor already scores 0.447 on the Eval column.)\n")

    # ---- communalities
    un = o["uniqueness"]["per_solution"]
    W("\n## 9. Communality and uniqueness\n")
    W(f"{o['uniqueness']['note']}\n")
    W("| solution | mean h^2 | mean u^2 | min h^2 | max h^2 |")
    W("|---|---|---|---|---|")
    for key in keys:
        u = un[key]
        W(f"| {key} | {u['communality']['mean']:.3f} | {u['uniqueness']['mean']:.3f} | "
          f"{u['communality']['min']:.3f} | {u['communality']['max']:.3f} |")

    st = o["steering"]
    W("\n## 10. Steering hand-off\n")
    W(f"`steering.oblimin_loadings` holds the {st['k']}-column oblimin pattern "
      f"of **{st['solution']}** keyed by npz slug; steer134_on_modal.py's "
      f"fa_coeffs turns column j into the fa{{j}} direction (mean-centred, "
      f"unit norm in the double-centred Gram metric). Best Goldberg match per "
      f"column: " + ", ".join(
          f"fa{j+1}={b} ({c:.2f})" for j, (b, c) in enumerate(
              zip(st["best_goldberg_per_factor"],
                  st["best_goldberg_congruence"]))) + ".\n")

    open(f"{RDIR}/fa_qwen35{TAG}.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
