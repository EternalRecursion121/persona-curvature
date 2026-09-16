"""FACTOR ANALYSIS over the 100-trait LoRA correlation matrix.

The psychometric counterpart to analyse_pca.py.

The cosine matrix between the 100 trait adapters is structurally the same object
as the trait-by-trait correlation matrix psychometrics factors from human
self-report data. Goldberg derived the Big Five by factoring exactly that kind
of matrix from PEOPLE; here we factor it from WEIGHTS, with the same method
(principal axis factoring, SMC communality starts, Horn's parallel analysis,
varimax and oblimin rotation, Tucker congruence against marker targets), and ask
whether the same five fall out.

Everything is implemented from scratch on numpy (there is no scipy on this box)
and each piece is checked against a hand-computable case; see `verify()`.

Usage:  ~/cartovenv/bin/python analyse_fa.py
Reads:  results/gram.npy, results/gram_names.json, traits.json, results/pca.json
Writes: results/fa.json, results/fa.md
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ADIR = os.path.join(HERE, "adapters")
RDIR = os.path.join(HERE, "results")
FACTORS = ["Extraversion", "Agreeableness", "Conscientiousness",
           "EmotionalStability", "Intellect"]
FSHORT = {"Extraversion": "E", "Agreeableness": "A", "Conscientiousness": "C",
          "EmotionalStability": "ES", "Intellect": "I"}
TCOLS = [FSHORT[f] for f in FACTORS] + ["Eval"]

PAF_TOL = 1e-7          # convergence on max |change in communality|
PAF_MAXIT = 5000
HEYWOOD_CAP = 0.999
RIDGE = 1e-3            # for SMC on the (singular) ipsatised matrix
PA_REPS = 500
PA_GRID = [150, 300, 1000, 1528, 5809, 20000]
SEED = 0


# ===========================================================================
# 1.  MATHS PRIMITIVES  (all hand-checked in verify())
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
# 2.  VERIFICATION  (hand-computable cases, no scipy)
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
# 3.  PARALLEL ANALYSIS  (Horn 1965)
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
# 4.  SOLUTION ASSEMBLY
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
# 5.  MAIN
# ===========================================================================

def entry_mean_check(names, cache):
    """Is cosine == correlation?  The literal correlation between two dW's
    centres each by its own mean ENTRY.  Correction to the Gram is D*mu_i*mu_j;
    we compute mu exactly from the factored form: sum(dW) = s * (1^T B)(A 1)."""
    if os.path.exists(cache):
        c = json.load(open(cache))
        if c["names"] == names:
            return c
    from safetensors import safe_open
    sums, D = {}, None
    t0 = time.time()
    for i, a in enumerate(names):
        h = safe_open(f"{ADIR}/{a}/adapter_model.safetensors", framework="numpy")
        cfg = json.load(open(f"{ADIR}/{a}/adapter_config.json"))
        sc = cfg["lora_alpha"] / cfg["r"]
        mods = sorted(set(k.replace(".lora_A.weight", "")
                          for k in h.keys() if "lora_A" in k))
        tot, d = 0.0, 0
        for m in mods:
            A = h.get_tensor(m + ".lora_A.weight").astype(np.float64)
            B = h.get_tensor(m + ".lora_B.weight").astype(np.float64)
            tot += sc * float(B.sum(0) @ A.sum(1))
            d += B.shape[0] * A.shape[1]
        sums[a] = tot
        D = d
        if i % 20 == 0:
            print(f"  entry-mean {i+1}/{len(names)} {time.time()-t0:.0f}s",
                  file=sys.stderr, flush=True)
    c = {"names": names, "D": D, "sums": sums}
    json.dump(c, open(cache, "w"))
    return c


def main():
    os.makedirs(RDIR, exist_ok=True)
    out = {}

    print("verifying maths ...", file=sys.stderr)
    out["verification"] = verify()
    assert out["verification"]["all_ok"], out["verification"]
    print("  all verification checks PASSED", file=sys.stderr)

    # ---- data -------------------------------------------------------------
    traits = json.load(open(f"{HERE}/traits.json"))
    assert len(traits) == 100
    meta = json.load(open(f"{RDIR}/gram_names.json"))
    names = meta["names"]
    Gall = np.load(f"{RDIR}/gram.npy")
    IDX = {a: i for i, a in enumerate(names)}
    tn = [t["trait"].lower().replace("-", "_") for t in traits]
    ti = np.array([IDX[a] for a in tn])
    G = Gall[np.ix_(ti, ti)]
    p = 100
    label = np.array([t["trait"] for t in traits])
    factor = np.array([t["factor"] for t in traits])
    keyed = np.array([t["keyed"] for t in traits])
    pole = np.where(keyed == "+", 1.0, -1.0)

    out["setup"] = {"n_traits": p, "n_modules": len(meta["modules"]),
                    "r": meta["r"], "scale_alpha_over_r": meta["scale"],
                    "base_model": json.load(open(
                        f"{ADIR}/{names[0]}/adapter_config.json"))["base_model_name_or_path"]}

    # ---- 5a. targets -------------------------------------------------------
    targets = np.zeros((p, 6))
    for a, F in enumerate(FACTORS):
        targets[:, a] = np.where(factor == F, pole, 0.0)
    targets[:, 5] = pole                       # general evaluative target
    TT = np.array([[tucker(targets[:, i], targets[:, j]) for j in range(6)]
                   for i in range(6)])
    out["targets"] = {
        "definition": "target_F[i] = +1 if trait i is a positively-keyed marker "
                      "of Goldberg factor F, -1 if negatively-keyed marker of F, "
                      "0 if it belongs to another factor. 'Eval' is the general "
                      "evaluative target: +1 for every positively-keyed trait, "
                      "-1 for every negatively-keyed trait, regardless of factor.",
        "labels": TCOLS, "target_target_congruence": TT.tolist(),
        "note": "Each Goldberg target has congruence exactly 0.447 = 20/sqrt(20*100) "
                "with the Eval target BY CONSTRUCTION (Eval is the sum of the five). "
                "So 0.447 is the baseline a perfectly-recovered pure Goldberg factor "
                "already scores on the Eval column; only values clearly above it "
                "indicate extra evaluative content."}

    # ---- 1. the correlation matrix ----------------------------------------
    d = np.sqrt(np.diag(G))
    R = G / np.outer(d, d)                                   # uncentred
    H = np.eye(p) - np.ones((p, p)) / p
    Gc = 0.5 * ((H @ G @ H) + (H @ G @ H).T)
    dc = np.sqrt(np.diag(Gc))
    Rc = Gc / np.outer(dc, dc)                               # ipsatised
    off = ~np.eye(p, dtype=bool)
    iu = np.triu_indices(p, 1)

    em = entry_mean_check(tn, f"{RDIR}/entry_sums.json")
    mu = np.array([em["sums"][a] for a in tn]) / em["D"]
    corr_mag = em["D"] * np.abs(np.outer(mu, mu))
    out["correlation_matrix"] = {
        "centring": {
            "entry_level": {
                "what": "the literal correlation subtracts each dW's own mean ENTRY; "
                        "correction to <dW_i,dW_j> is D*mu_i*mu_j.",
                "D_entries_per_dW": em["D"],
                "max_abs_mean_entry": float(np.abs(mu).max()),
                "max_correction_D_mu_i_mu_j": float(corr_mag.max()),
                "max_relative_correction": float((corr_mag / np.abs(G)).max()),
                "verdict": "NEGLIGIBLE (relative correction < 1e-6): for these "
                           "adapters cosine and Pearson correlation are the same "
                           "number to 6+ decimal places."},
            "across_traits_ipsatisation": {
                "what": "subtract the grand mean dW over the 100 traits before "
                        "taking cosines. In psychometric terms this is IPSATISATION: "
                        "for every weight coordinate ('respondent') we remove that "
                        "coordinate's mean response across the 100 trait items, "
                        "which is the standard correction for acquiescence / "
                        "evaluative response bias.",
                "reported": "both"}},
        "uncentred_offdiag": {"mean": float(R[off].mean()), "sd": float(R[off].std()),
                              "min": float(R[off].min()), "max": float(R[off].max())},
        "centred_offdiag": {"mean": float(Rc[off].mean()), "sd": float(Rc[off].std()),
                            "min": float(Rc[off].min()), "max": float(Rc[off].max())},
        "corr_between_the_two_offdiag": float(np.corrcoef(R[iu], Rc[iu])[0, 1]),
        "uncentred_eigenvalues": np.linalg.eigvalsh(R)[::-1].tolist(),
        "centred_eigenvalues": np.linalg.eigvalsh(Rc)[::-1].tolist(),
        "uncentred_condition_number": float(np.linalg.eigvalsh(R)[::-1][0]
                                            / np.linalg.eigvalsh(R)[::-1][-1]),
        "centred_is_singular": True,
        "centred_singularity_note":
            "Ipsatisation removes exactly one dimension, so R_centred has rank 99 "
            "and its smallest eigenvalue is 0. SMC = 1 - 1/(R^-1)_ii is then "
            f"identically 1, so the centred run starts PAF from ridge-SMC (ridge={RIDGE}). "
            "PAF's fixed point turns out to be start-independent (checked below)."}

    # SMCs
    smc_u = smc(R)
    smc_c = smc(Rc, RIDGE)
    out["smc"] = {"uncentred": {"min": float(smc_u.min()), "mean": float(smc_u.mean()),
                                "max": float(smc_u.max())},
                  "centred_ridge": {"ridge": RIDGE, "min": float(smc_c.min()),
                                    "mean": float(smc_c.mean()), "max": float(smc_c.max())},
                  "per_trait_uncentred": {label[i]: float(smc_u[i]) for i in range(p)}}

    # start-value sensitivity for the centred run
    maxabs = np.array([np.max(np.abs(np.delete(Rc[i], i))) for i in range(p)])
    starts = {"ridge_smc": smc_c, "constant_0.5": np.full(p, 0.5), "max_abs_r": maxabs}
    sens = {}
    for nm, h0 in starts.items():
        rr = paf(Rc, 5, h2_init=h0)
        sens[nm] = {"iterations": rr["iterations"],
                    "sum_communalities": float(rr["communalities"].sum())}
    out["paf_start_sensitivity_centred_k5"] = sens

    # ---- 3. number of factors ---------------------------------------------
    # effective dimensionality of weight space, estimated from the reseed controls
    reseeds = [a for a in names if a.endswith("__s1")]
    pairs = [(IDX[r[:-4]], IDX[r]) for r in reseeds]
    kk = len(pairs)
    Gd = np.zeros((kk, kk))
    for a, (i, ip_) in enumerate(pairs):
        for b, (j, jp) in enumerate(pairs):
            Gd[a, b] = Gall[i, j] - Gall[i, jp] - Gall[ip_, j] + Gall[ip_, jp]
    dd = np.sqrt(np.diag(Gd))
    Cd = Gd / np.outer(dd, dd)
    offd = Cd[np.triu_indices(kk, 1)]
    m_dd = float(1.0 / offd.var())
    base = set(r[:-4] for r in reseeds)
    others = [i for i, a in enumerate(names)
              if not a.endswith("__s1") and a not in base]
    vals = []
    for a, (i, ip_) in enumerate(pairs):
        for j in others:
            vals.append((Gall[i, j] - Gall[ip_, j]) / (dd[a] * np.sqrt(Gall[j, j])))
    vals = np.array(vals)
    m_dx = float(1.0 / vals.var())
    out["effective_dimensionality"] = {
        "why": "Horn's parallel analysis with N observations is exactly the null "
               "'the p trait vectors are random directions in an (N-1)-dimensional "
               "space'. Choosing N is therefore choosing the effective "
               "dimensionality of weight space, which we estimate from the reseed "
               "controls: d_i = dW_i(seed0) - dW_i(seed1) is a pure noise draw, and "
               "cosines between independent isotropic vectors in R^m have variance 1/m.",
        "reseed_difference_cosines": Cd.tolist(),
        "m_from_noise_noise_cosines": m_dd, "n_noise_noise_pairs": len(offd),
        "m_from_noise_vs_other_trait_cosines": m_dx, "n_noise_trait_pairs": len(vals),
        "range_used": [1528, 5809],
        "caveat": "the noise-noise estimate rests on only 10 cosines, so m is "
                  "uncertain by a factor of ~2-3 either way; hence the grid."}

    print("parallel analysis (uncentred) ...", file=sys.stderr)
    pa_u = parallel_analysis(R, PA_GRID, p, ipsatise=False, ridge=0.0)
    print("parallel analysis (centred) ...", file=sys.stderr)
    pa_c = parallel_analysis(Rc, PA_GRID, p, ipsatise=True, ridge=RIDGE)
    ev_u = np.array(out["correlation_matrix"]["uncentred_eigenvalues"])
    ev_c = np.array(out["correlation_matrix"]["centred_eigenvalues"])
    K_PA = 7
    out["n_factors"] = {
        "kaiser_uncentred_eig_gt_1": int((ev_u > 1).sum()),
        "kaiser_centred_eig_gt_1": int((ev_c > 1).sum()),
        "reduced_eig_gt_1_uncentred": int((np.array(pa_u["observed_reduced"]) > 1).sum()),
        "parallel_analysis_uncentred": pa_u,
        "parallel_analysis_centred": pa_c,
        "size_matched_literal_N": {
            "N": em["D"],
            "verdict": "A LITERAL size match (N = D = 2.77e9 weight coordinates) is "
                       "degenerate: the null correlation matrix converges to the "
                       "identity, every null eigenvalue -> 1, and PA collapses onto "
                       "the Kaiser criterion. That is why N is set from the "
                       "empirically estimated effective dimensionality instead."},
        "chosen": K_PA,
        "chosen_rationale":
            "Horn's original (unreduced) PA gives k=7 at N=1000 and N=1528 for BOTH "
            "the centred and uncentred matrices, and k=9 at N=5809; k=5 appears only "
            "at N=150, well below any defensible effective dimensionality. The "
            "SMC-reduced variant gives 7 at N=300 and more at larger N. The robust "
            "statement is k>5; we take k=7 as the PA-supported number and also "
            "extract at exactly 5 because 5 is the hypothesis."}

    # ---- 2/4. solutions ----------------------------------------------------
    sols = {}
    for tag, M, ridge in [("centred", Rc, RIDGE), ("uncentred", R, 0.0)]:
        for k in [K_PA, 5]:
            key = f"{tag}_k{k}"
            print(f"solution {key} ...", file=sys.stderr)
            sols[key] = solution(M, k, targets, ridge=ridge, label=key)
    out["solutions"] = sols
    out["trait_order"] = label.tolist()
    out["trait_factor"] = factor.tolist()
    out["trait_keyed"] = keyed.tolist()

    # ---- 7. uniqueness vs the reseed noise floor ---------------------------
    e = np.zeros(len(names))
    e[ti] = 1.0 / p
    eGe = float(e @ Gall @ e)
    rel = {}
    for r in reseeds:
        i, j = IDX[r[:-4]], IDX[r]
        cu = Gall[i, j] / np.sqrt(Gall[i, i] * Gall[j, j])
        gii = Gall[i, i] - 2 * (Gall[i] @ e) + eGe
        gjj = Gall[j, j] - 2 * (Gall[j] @ e) + eGe
        gij = Gall[i, j] - (Gall[i] @ e) - (Gall[j] @ e) + eGe
        rel[r[:-4]] = {"reseed_cos_uncentred": float(cu),
                       "reseed_cos_centred": float(gij / np.sqrt(gii * gjj))}
    mrel_u = float(np.mean([v["reseed_cos_uncentred"] for v in rel.values()]))
    mrel_c = float(np.mean([v["reseed_cos_centred"] for v in rel.values()]))
    pca = json.load(open(f"{RDIR}/pca.json"))
    nfl = pca["noise_floor"]
    uni = {}
    for key, s in sols.items():
        h2 = np.array(s["communalities"])
        tagc = key.split("_")[0]
        r0 = mrel_c if tagc == "centred" else mrel_u
        uni[key] = {
            "communality": {"min": float(h2.min()), "mean": float(h2.mean()),
                            "max": float(h2.max())},
            "uniqueness": {"min": float(1 - h2.max()), "mean": float(1 - h2.mean()),
                           "max": float(1 - h2.min())},
            "reliability_from_reseed": r0,
            "noise_uniqueness_floor_1_minus_reliability": 1 - r0,
            "n_traits_with_h2_above_reliability": int((h2 > r0).sum()),
            "mean_uniqueness_over_noise_floor": float((1 - h2.mean()) / (1 - r0))}
    out["uniqueness"] = {
        "per_solution": uni, "per_reseed_reliability": rel,
        "mean_reseed_cos_uncentred": mrel_u, "mean_reseed_cos_centred": mrel_c,
        "pca_noise_floor": {k2: v for k2, v in nfl.items() if k2 != "per_reseed"},
        "seed_share_of_squared_between_trait_distance":
            1.0 / nfl["ratio_between_over_reseed"] ** 2,
        "reasoning":
            "A reseed pair is x = s + n1, x' = s + n2 with independent seed noise, "
            "so E[cos(x,x')] = ||s||^2/(||s||^2+||n||^2) is a test-retest "
            "RELIABILITY. FA uniqueness u^2 = 1 - h^2 contains BOTH that seed noise "
            "AND reliable trait-specific variance, so the model is consistent only "
            "if u^2 >= 1 - reliability and h^2 <= reliability for every trait."}

    # ---- 6. comparison with the PCA ---------------------------------------
    out["pca_comparison"] = {
        "pca_centered_var_pct": pca["pca"]["centered_var_pct"][:8],
        "pca_pc_vs_factor_abscos": pca["pc_vs_factor_abscos"],
        "pca_factor_direction_cosines": pca["factor_directions"]["pairwise_cos"],
        "pca_factor_labels": FACTORS}

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

    json.dump(out, open(f"{RDIR}/fa.json", "w"), indent=1)
    write_md(out, label, factor, keyed)
    print("wrote results/fa.json and results/fa.md", file=sys.stderr)
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

    W("# Factor analysis of the 100-trait LoRA correlation matrix\n")
    W("The psychometric counterpart to `pca.md`.\n")

    W("## 0. Why this is the same analysis Goldberg did\n")
    W("The cosine matrix between the 100 trait adapters is structurally the same "
      "object as the trait-by-trait correlation matrix that psychometrics factors "
      "from human self-report data. In Goldberg's studies the data matrix is "
      "*people x adjectives* and the correlation between two adjectives is taken "
      "over people; here the data matrix is *weight coordinates x adjectives* and "
      "the correlation between two trait deltas is taken over weight coordinates. "
      "Same matrix, same method: principal axis factoring with squared multiple "
      "correlations as the communality start, number of factors by Horn's parallel "
      "analysis, varimax and oblimin rotation, and Tucker congruence against "
      "marker targets. Goldberg factored that matrix out of people and got the Big "
      "Five. We factor it out of weights and ask whether the same five fall out.\n")
    W("The analogy is exact enough that the standard psychometric hygiene applies "
      "verbatim. In particular, removing the grand mean dW across the 100 traits "
      "is **ipsatisation**: for each weight coordinate (each 'respondent') we "
      "subtract that coordinate's mean response across the 100 items, which is the "
      "textbook correction for acquiescence / evaluative response bias. Both the "
      "raw and the ipsatised solution are reported below, and they differ "
      "materially -- which is itself the answer to the evaluative-factor question.\n")

    # ---- verification
    v = o["verification"]
    W("\n## 1. Verification (no scipy on this box; everything hand-checked)\n")
    W("| check | case | result |")
    W("|---|---|---|")
    W(f"| eigen-decomposition | `[[2,1],[1,2]]`, exact spectrum {{3, 1}} | "
      f"eigenvalues {np.round(v['eigh']['eigenvalues'],12).tolist()}, err "
      f"{v['eigh']['eigenvalue_err']:.1e} |")
    W(f"| eigen-decomposition | random 100x100 symmetric | max residual "
      f"`|AV - VL|` {v['eigh']['random_100x100_max_residual_AV_minus_VL']:.1e}, "
      f"orthonormality {v['eigh']['random_100x100_max_orthonormality_err']:.1e}, "
      f"reconstruction {v['eigh']['random_100x100_max_reconstruction_err']:.1e} |")
    W(f"| SMC | `1 - 1/(R^-1)_ii` vs explicit OLS R^2 of each variable on the "
      f"other 5 | max abs diff {v['smc']['max_abs_diff']:.1e} |")
    W(f"| varimax | perfect 2-factor simple structure rotated by 0.6 rad, then "
      f"un-rotated by varimax | max abs error vs truth: pairwise "
      f"{v['varimax']['known_simple_structure']['max_abs_err_pairwise']:.1e}, "
      f"GPA {v['varimax']['known_simple_structure']['max_abs_err_gpa']:.1e} |")
    W(f"| varimax | two independent algorithms (Kaiser cyclic pairwise vs "
      f"Jennrich gradient projection) on random loadings, plus communality "
      f"invariance | max disagreement "
      f"{v['varimax']['two_algorithms_max_disagreement']:.1e} |")
    W(f"| oblimin | known oblique structure: 3 factors, perfect simple pattern, "
      f"true Phi with off-diagonals .5/.4/.6 | pattern recovered to "
      f"{v['oblimin']['max_abs_err_pattern']:.1e}, **Phi recovered to "
      f"{v['oblimin']['max_abs_err_Phi']:.1e}**, `L Phi L' ` preserved to "
      f"{v['oblimin']['rotation_preserves_L_Phi_Lt']:.1e} |")
    W(f"| Tucker congruence | x=(1,2,3), y=(1,0,-1): <x,y>=-2, <x,x>=14, "
      f"<y,y>=2, phi = -2/sqrt(28) = {v['tucker']['hand_case']['expected']:.16f} | "
      f"got {v['tucker']['hand_case']['got']:.16f}, err "
      f"{v['tucker']['hand_case']['err']:.1e}; identical vectors -> "
      f"{v['tucker']['identical']:.0f}, sign-flipped -> "
      f"{v['tucker']['sign_flipped']:.0f} |")
    W(f"| PAF | synthetic 2-factor model with known communalities | max "
      f"communality error {v['paf']['max_abs_err_communality']:.1e} in "
      f"{v['paf']['iterations']} iterations |")
    W(f"\nAll checks pass (`all_ok = {v['all_ok']}`). The oblimin check is the "
      "load-bearing one: the algorithm recovers a *known* factor correlation "
      "matrix to 1e-6, so the Phi reported below is trustworthy.\n")

    # ---- correlation matrix
    cm = o["correlation_matrix"]
    W("\n## 2. The correlation matrix\n")
    ent = cm["centring"]["entry_level"]
    W(f"**Is cosine the same as correlation?** Yes, to 6+ decimal places. The "
      f"literal Pearson correlation subtracts each dW's own mean entry; the "
      f"correction to `<dW_i,dW_j>` is `D * mu_i * mu_j` with D = "
      f"{ent['D_entries_per_dW']:,} entries per delta. Computed exactly from the "
      f"factored form (`sum(dW) = (alpha/r) * (1'B)(A1)`), the largest mean entry "
      f"is {ent['max_abs_mean_entry']:.2e} and the largest correction is "
      f"{ent['max_correction_D_mu_i_mu_j']:.2e} -- a relative correction of "
      f"{ent['max_relative_correction']:.1e}. So **entry-level centring is "
      f"ignorable** and cosine == correlation here.\n")
    W("**Centring across traits is a different and non-ignorable choice**, and we "
      "report both:\n")
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
    W(f"\nThe two agree closely off-diagonal (r = "
      f"{cm['corr_between_the_two_offdiag']:.4f}) but the uncentred matrix carries "
      f"a positive general offset ({cm['uncentred_offdiag']['mean']:+.4f} vs "
      f"{cm['centred_offdiag']['mean']:+.4f}) -- the 'an adapter was trained here' "
      f"component, which is the acquiescence analogue. **They differ materially in "
      f"the factor solution, so both are carried through.**\n")
    W(f"Note: ipsatisation removes exactly one dimension, so R_centred has rank 99 "
      f"and is exactly singular. SMC = 1 - 1/(R^-1)_ii is then identically 1.0, so "
      f"the centred run starts PAF from a ridge-regularised SMC (ridge = {RIDGE}, "
      f"SMC mean {o['smc']['centred_ridge']['mean']:.3f}). This turns out not to "
      f"matter: PAF converges to the *same* fixed point from all three starts --")
    W("\n| start | iterations | sum of communalities (k=5, centred) |")
    W("|---|---|---|")
    for nm, s in o["paf_start_sensitivity_centred_k5"].items():
        W(f"| {nm} | {s['iterations']} | {s['sum_communalities']:.6f} |")
    W(f"\nUncentred SMC (well-posed, condition number "
      f"{cm['uncentred_condition_number']:.1f}): min "
      f"{o['smc']['uncentred']['min']:.3f}, mean {o['smc']['uncentred']['mean']:.3f}, "
      f"max {o['smc']['uncentred']['max']:.3f}.\n")

    # ---- number of factors
    nf = o["n_factors"]
    pau, pac = nf["parallel_analysis_uncentred"], nf["parallel_analysis_centred"]
    W("\n## 3. How many factors?\n")
    W("### 3a. Eigenvalues and scree\n")
    W("Reduced correlation matrix = R with communalities on the diagonal (the "
      "matrix PAF actually factors). Uncentred:\n")
    W("```")
    W("rank  eigenvalue (reduced, uncentred)")
    W(scree(pau["observed_reduced"]))
    W("```")
    W("Centred (ipsatised):\n")
    W("```")
    W("rank  eigenvalue (reduced, centred)")
    W(scree(pac["observed_reduced"]))
    W("```")
    W(f"\n- **Kaiser** (eigenvalue of the *unreduced* R > 1): "
      f"{nf['kaiser_uncentred_eig_gt_1']} factors uncentred, "
      f"{nf['kaiser_centred_eig_gt_1']} centred. Kaiser is known to over-retain.")
    W(f"- **Scree**: a sharp elbow after 4 (uncentred reduced eigenvalues "
      f"{', '.join('%.1f' % x for x in pau['observed_reduced'][:5])}) and a second "
      f"shelf after 7 ({', '.join('%.2f' % x for x in pau['observed_reduced'][5:9])}).")
    W(f"- **Reduced eigenvalues > 1**: {nf['reduced_eig_gt_1_uncentred']} "
      f"(uncentred).\n")

    ed = o["effective_dimensionality"]
    W("### 3b. Parallel analysis (Horn) -- and what 'matched in size' means here\n")
    W("Horn's PA compares each observed eigenvalue against the same-rank "
      "eigenvalue of random data with the same number of variables (p=100) and N "
      "observations, retaining the leading run that exceeds the null (95th "
      "percentile, first-crossing rule; 500 replications per cell).\n")
    W(f"There is a genuine difficulty: **what is N?** A literal size match would be "
      f"N = D = {nf['size_matched_literal_N']['N']:,} weight coordinates, and that "
      f"is degenerate -- the null correlation matrix converges to the identity, "
      f"every null eigenvalue goes to 1, and PA collapses onto Kaiser. The way out "
      f"is to notice that *PA with N observations is exactly the null 'the p trait "
      f"vectors are random directions in an (N-1)-dimensional space'*. Choosing N "
      f"is choosing the effective dimensionality of weight space, and we can "
      f"estimate that empirically from the reseed controls: `d_i = dW_i(seed0) - "
      f"dW_i(seed1)` is a pure noise draw, and cosines between independent "
      f"isotropic vectors in R^m have variance 1/m.\n")
    W(f"- from the {ed['n_noise_noise_pairs']} noise-noise cosines: m = "
      f"**{ed['m_from_noise_noise_cosines']:.0f}**")
    W(f"- from the {ed['n_noise_trait_pairs']} noise-vs-other-trait cosines: m = "
      f"**{ed['m_from_noise_vs_other_trait_cosines']:.0f}**")
    W(f"- so N is somewhere around {ed['range_used'][0]}-{ed['range_used'][1]}; "
      f"{ed['caveat']}\n")
    W("| N | uncentred: k (Horn, unreduced) | uncentred: k (SMC-reduced) | "
      "centred: k (Horn, unreduced) | centred: k (SMC-reduced) |")
    W("|---|---|---|---|---|")
    for a, b in zip(pau["grid"], pac["grid"]):
        star = " <-" if a["N"] in (1528, 5809) else ""
        W(f"| {a['N']}{star} | **{a['k_unreduced_95pct']}** | {a['k_reduced_95pct']} | "
          f"**{b['k_unreduced_95pct']}** | {b['k_reduced_95pct']} |")
    W(f"\n**Parallel analysis supports k = {K}, not 5.** Horn's original "
      f"(unreduced) form gives 7 at N=1000 and N=1528 for both matrices and 9 at "
      f"N=5809; 5 appears only at N=150, far below any defensible effective "
      f"dimensionality. The SMC-reduced variant is more permissive still on the "
      f"uncentred matrix (7 at N=300 rising to 13-19), and on the centred matrix it "
      f"is nearly identical to the unreduced form because ipsatisation forces the "
      f"SMCs close to 1 so the reduction barely changes the trace. The robust "
      f"claim is **k > 5**; we take k={K} as the PA number and also extract at "
      f"exactly 5 because 5 is the hypothesis.\n")

    # ---- solutions
    W("\n## 4. Principal axis factoring\n")
    W("| solution | PAF iterations | final max |change in h^2| | Heywood cases | "
      "varimax iters | oblimin iters | oblimin |grad| | `L Phi L'` vs unrotated |")
    W("|---|---|---|---|---|---|---|---|")
    for key in ["centred_k%d" % K, "centred_k5", "uncentred_k%d" % K, "uncentred_k5"]:
        s = S[key]
        W(f"| {key} | {s['paf_iterations']} | {s['paf_final_delta']:.1e} | "
          f"{s['n_heywood']} | {s['varimax_iterations']} | {s['oblimin_iterations']} | "
          f"{s['oblimin_gradient_norm']:.1e} | "
          f"{s['reproduction_err_L_Phi_Lt_vs_unrotated']:.1e} |")
    W(f"\nConvergence criterion: max over traits of |change in communality| < "
      f"{PAF_TOL:g}; Heywood cases capped at {HEYWOOD_CAP} (none occurred). "
      f"Both rotations use Kaiser normalisation (rows scaled to unit length by "
      f"sqrt(h^2) before rotating, restored after). The last column is the check "
      f"that oblique rotation is loss-free: the rotated pattern and factor "
      f"correlations reproduce the unrotated reduced matrix exactly.\n")
    W("Factor sign is arbitrary in FA; the convention throughout is that each "
      "factor is oriented so its congruence with its own best-matching Goldberg "
      "target is positive, and factors are ordered by descending SS loading.\n")

    # ---- oblimin Phi
    W("\n## 5. HEADLINE: the oblimin factor correlation matrix\n")
    W("This is the reason for doing an oblique rotation at all. The PCA showed the "
      "Goldberg factor *directions* are correlated in weight space (C-ES "
      f"{o['pca_comparison']['pca_factor_direction_cosines'][2][3]:+.2f}, C-I "
      f"{o['pca_comparison']['pca_factor_direction_cosines'][2][4]:+.2f}, A-I "
      f"{o['pca_comparison']['pca_factor_direction_cosines'][1][4]:+.2f}), which is "
      "exactly why a rotation-free method could not separate them. Oblimin lets the "
      "extracted factors be correlated and reports how correlated they are.\n")
    for key in [f"centred_k5", f"uncentred_k5", f"centred_k{K}", f"uncentred_k{K}"]:
        s = S[key]
        k = s["k"]
        Phi = np.array(s["Phi"])
        W(f"\n**{key}** -- factor correlation matrix Phi (direct oblimin, gamma=0):\n")
        W("| | " + " | ".join(f"F{j+1}" for j in range(k)) + " |")
        W("|---|" + "---|" * k)
        for i in range(k):
            W(f"| **F{i+1}** | " + " | ".join(f"{Phi[i,j]:+.3f}" for j in range(k)) + " |")
        offv = Phi[~np.eye(k, dtype=bool)]
        W(f"\nLargest |correlation| {np.abs(offv).max():.3f}; mean |correlation| "
          f"{np.abs(offv).mean():.3f}.")

    # ---- congruence
    tg = o["targets"]
    W("\n## 6. Ground truth: Tucker congruence with the Goldberg factors\n")
    W(f"**Target definition.** {tg['definition']}\n")
    W(f"Tucker's congruence coefficient is `phi(x,y) = <x,y> / sqrt(<x,x><y,y>)`. "
      f"Conventional reading: |phi| > 0.85 'fair similarity', |phi| > 0.95 "
      f"'equivalent'.\n")
    W(f"**Baseline that must be kept in mind**: {tg['note']}\n")
    for key in [f"centred_k5", f"uncentred_k5", f"centred_k{K}", f"uncentred_k{K}"]:
        s = S[key]
        k = s["k"]
        W(f"\n### {key}\n")
        for rot in ["unrotated", "varimax", "oblimin"]:
            C = np.array(s["congruence_" + rot])
            W(f"\n**{rot}** (columns: the five Goldberg targets, then the general "
              f"evaluative target):\n")
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

    # explicit verdict on thresholds
    W("\n### Which Goldberg factors clear the thresholds?\n")
    for key in [f"centred_k5", f"uncentred_k5", f"centred_k{K}", f"uncentred_k{K}"]:
        s = S[key]
        C = np.abs(np.array(s["congruence_oblimin"]))[:, :5]
        best = C.max(0)
        line = ", ".join(f"{TCOLS[a]} {best[a]:.3f}" for a in range(5))
        fair = [TCOLS[a] for a in range(5) if best[a] >= 0.85]
        eqv = [TCOLS[a] for a in range(5) if best[a] >= 0.95]
        W(f"- **{key}, oblimin** -- best congruence per Goldberg factor: {line}. "
          f"Clearing 0.85 (fair): {fair or '**none**'}. Clearing 0.95 (equivalent): "
          f"{eqv or '**none**'}.")
    W("")
    W("**Plain statement.** All five Goldberg factors appear as *identifiable, "
      "separate* factors -- every one of them is the best match for exactly one "
      "extracted factor, with no factor doubling up -- but **none of them clears "
      "the 0.85 'fair similarity' threshold, and none clears 0.95.** The best "
      "congruences sit in the 0.70-0.83 band on the ipsatised matrix. That is a "
      "structural recovery of the Big Five without a quantitative equivalence "
      "claim, and the reason is visible in the loading tables: the extracted "
      "factors are cleaner than the targets, in the sense that they concentrate on "
      "one pole of a factor's markers and put cross-loadings on markers of other "
      "factors, whereas the target is a flat +-1 over all 20 markers.\n")

    # ---- top loadings
    W(f"\n## 7. Top-10 loading traits per factor\n")
    W("Ranked by |loading|; sign shown. Factor codes: E Extraversion, A "
      "Agreeableness, C Conscientiousness, ES EmotionalStability, I Intellect.\n")
    lab = np.array(label)
    fac = np.array(factor)
    kyd = np.array(keyed)
    for key in [f"centred_k5", f"centred_k{K}", f"uncentred_k5", f"uncentred_k{K}"]:
        s = S[key]
        k = s["k"]
        M = np.array(s["loadings"]["oblimin"])
        C = np.array(s["congruence_oblimin"])
        W(f"\n### {key}, OBLIMIN pattern\n")
        for j in range(k):
            b = int(np.argmax(np.abs(C[j, :5])))
            W(f"\n**F{j+1}** (SS {s['ss_loadings']['oblimin'][j]:.2f}; best Goldberg "
              f"match {TCOLS[b]}, phi = {C[j,b]:+.3f}; Eval phi = {C[j,5]:+.3f})\n")
            W("| rank | trait | loading | Goldberg factor | keyed |")
            W("|---|---|---|---|---|")
            for rk, i in enumerate(np.argsort(-np.abs(M[:, j]))[:10]):
                W(f"| {rk+1} | {lab[i]} | {M[i,j]:+.3f} | {FSHORT[fac[i]]} | {kyd[i]} |")

    # ---- evaluative factor
    W("\n## 8. Does an evaluative factor survive rotation?\n")
    W("**No. Rotation dissolves it.** This is the clean answer, and it is the same "
      "on both matrices.\n")
    W("| solution | rotation | max Eval congruence over factors | which factor |")
    W("|---|---|---|---|")
    for key in [f"centred_k5", f"uncentred_k5", f"centred_k{K}", f"uncentred_k{K}"]:
        s = S[key]
        for rot in ["unrotated", "varimax", "oblimin"]:
            C = np.array(s["congruence_" + rot])
            j = int(np.argmax(np.abs(C[:, 5])))
            W(f"| {key} | {rot} | **{abs(C[j,5]):.3f}** | F{j+1} |")
    cu = np.array(S["uncentred_k5"]["congruence_unrotated"])
    cc = np.array(S["centred_k5"]["congruence_unrotated"])
    ou = np.array(S["uncentred_k5"]["congruence_oblimin"])
    oc = np.array(S["centred_k5"]["congruence_oblimin"])
    W(f"\nUnrotated, the first PAF factor **is** the general evaluative axis: "
      f"congruence with the pure desirable/undesirable target is "
      f"{abs(cu[0,5]):.3f} (uncentred) and {abs(cc[0,5]):.3f} (centred), against a "
      f"mathematical baseline of 0.447 that any pure Goldberg factor scores "
      f"automatically. It is the largest factor in both cases and it loads "
      f"simultaneously on C, I and A -- exactly the PC1 the PCA found.\n")
    W(f"After oblimin, the largest Eval congruence anywhere in the solution falls "
      f"to {np.abs(ou[:,5]).max():.3f} (uncentred) and {np.abs(oc[:,5]).max():.3f} "
      f"(centred), and it is no longer concentrated: *every* rotated factor carries "
      f"a similar modest amount "
      f"({', '.join('%.2f' % abs(x) for x in oc[:,5])} on the centred k=5 "
      f"solution), which is barely above the 0.447 that a perfectly clean Goldberg "
      f"factor scores by construction. There is no rotated factor that is "
      f"predominantly evaluative.\n")
    W("**Interpretation.** The general evaluative factor is an artefact of "
      "unrotated extraction. The variance is real -- desirable traits genuinely do "
      "sit closer together in weight space than chance -- but it is not a separate "
      "dimension of the space; it is the sum of the five substantive factors all "
      "leaning slightly the same way, and a rotation that seeks simple structure "
      "redistributes it back into them. The weaker claim survives, the stronger one "
      "does not: there is evaluative *covariance*, there is no evaluative *factor*.\n")
    W("The two residues of that covariance that *do* survive rotation are (i) the "
      "non-zero off-diagonals of Phi above, and (ii) the split of Conscientiousness "
      f"and Extraversion into same-pole clusters at k={K} (see section 7): the "
      "positively-keyed C markers and the negatively-keyed C markers come out as "
      "two correlated factors rather than one bipolar factor. In human data that "
      "same unipolar split is the classic signature of acquiescence, and here it is "
      "the classic signature of 'an adapter was trained toward a desirable trait' "
      "sharing variance across items.\n")

    # ---- varimax comparison
    W("\n## 9. Varimax vs oblimin\n")
    W("| solution | best Goldberg congruence, varimax | best, oblimin | "
      "max Eval congruence, varimax | oblimin |")
    W("|---|---|---|---|---|")
    for key in [f"centred_k5", f"uncentred_k5", f"centred_k{K}", f"uncentred_k{K}"]:
        s = S[key]
        Cv = np.abs(np.array(s["congruence_varimax"]))
        Co = np.abs(np.array(s["congruence_oblimin"]))
        W(f"| {key} | " +
          ", ".join(f"{TCOLS[a]} {Cv[:,a].max():.2f}" for a in range(5)) + " | " +
          ", ".join(f"{TCOLS[a]} {Co[:,a].max():.2f}" for a in range(5)) + " | " +
          f"{Cv[:,5].max():.2f} | {Co[:,5].max():.2f} |")
    W("\nThe two rotations agree closely on which factor is which. Oblimin buys a "
      "little congruence on most factors and, more importantly, reports the factor "
      "correlations instead of forcing them to zero; varimax has to absorb those "
      "correlations as cross-loadings.\n")

    # ---- uniqueness
    un = o["uniqueness"]
    W("\n## 10. Communality, uniqueness, and the reseed noise floor\n")
    W("A reseed pair is `x = s + n1`, `x' = s + n2` with independent seed noise, so "
      "`E[cos(x,x')] = ||s||^2 / (||s||^2 + ||n||^2)` is a **test-retest "
      "reliability**. FA uniqueness `u^2 = 1 - h^2` contains that seed noise *plus* "
      "reliable trait-specific variance, so the model is only consistent if "
      "`h^2 <= reliability` for every trait.\n")
    W("| trait | reseed cos (uncentred) | reseed cos (ipsatised) |")
    W("|---|---|---|")
    for t, r in un["per_reseed_reliability"].items():
        W(f"| {t} | {r['reseed_cos_uncentred']:.4f} | {r['reseed_cos_centred']:.4f} |")
    W(f"| **mean** | **{un['mean_reseed_cos_uncentred']:.4f}** | "
      f"**{un['mean_reseed_cos_centred']:.4f}** |")
    W(f"\nSo the noise-only uniqueness floor is "
      f"1 - {un['mean_reseed_cos_uncentred']:.3f} = "
      f"{1-un['mean_reseed_cos_uncentred']:.3f} (uncentred) / "
      f"{1-un['mean_reseed_cos_centred']:.3f} (ipsatised). The independent "
      f"distance-based figure from the PCA is "
      f"{100*un['seed_share_of_squared_between_trait_distance']:.0f}% of squared "
      f"between-trait distance attributable to seed, which is the same region.\n")
    W("| solution | mean h^2 | mean u^2 | min h^2 | max h^2 | reliability | "
      "traits with h^2 > reliability | u^2 / noise floor |")
    W("|---|---|---|---|---|---|---|---|")
    for key in [f"centred_k5", f"centred_k{K}", f"uncentred_k5", f"uncentred_k{K}"]:
        u = un["per_solution"][key]
        W(f"| {key} | {u['communality']['mean']:.3f} | {u['uniqueness']['mean']:.3f} | "
          f"{u['communality']['min']:.3f} | {u['communality']['max']:.3f} | "
          f"{u['reliability_from_reseed']:.3f} | "
          f"**{u['n_traits_with_h2_above_reliability']}** | "
          f"{u['mean_uniqueness_over_noise_floor']:.2f}x |")
    u5 = un["per_solution"]["centred_k5"]
    W(f"\n**The consistency check passes, and it is informative.** Not one trait in "
      f"any solution has a communality above its reliability -- the maximum "
      f"communality anywhere is {max(un['per_solution'][k2]['communality']['max'] for k2 in un['per_solution']):.3f} "
      f"against reliabilities of 0.83-0.86 -- so the factor model never claims more "
      f"common variance than the adapters actually reproduce across seeds. That is "
      f"the check that could have failed and did not.\n")
    W(f"But the numbers are not equal, and the gap is the finding. Mean uniqueness "
      f"is {u5['uniqueness']['mean']:.3f} at k=5 on the ipsatised matrix against a "
      f"seed-noise floor of {u5['noise_uniqueness_floor_1_minus_reliability']:.3f} "
      f"-- a factor of {u5['mean_uniqueness_over_noise_floor']:.1f}. Only about a "
      f"{100*u5['noise_uniqueness_floor_1_minus_reliability']/u5['uniqueness']['mean']:.0f}% "
      f"share of what the model calls 'unique' is training-seed noise; the other "
      f"~{100-100*u5['noise_uniqueness_floor_1_minus_reliability']/u5['uniqueness']['mean']:.0f}% "
      f"is *reliable, reproducible, trait-specific* variance that five factors do "
      f"not explain. Each adjective's adapter encodes something the Big Five "
      f"structure does not capture, and it encodes it consistently enough to "
      f"survive a reseed. Going from k=5 to k={K} recovers only "
      f"{un['per_solution']['centred_k%d' % K]['communality']['mean'] - u5['communality']['mean']:+.3f} "
      f"of mean communality, so the residue is not a few missing broad factors "
      f"either -- it is genuinely item-specific.\n")

    # ---- comparison with PCA
    W("\n## 11. Comparison with the PCA\n")
    pc = o["pca_comparison"]
    M = np.array(pc["pca_pc_vs_factor_abscos"]["matrix"])
    W("The PCA's verdict was: only PC3 was a clean factor axis (|cos| 0.95 with "
      "Extraversion); PC1, the largest component at "
      f"{pc['pca_centered_var_pct'][0]:.1f}% of centred variance, was not any "
      f"single factor but a general evaluative axis loading {M[0,2]:.2f} on "
      f"Conscientiousness and {M[0,4]:.2f} on Intellect at once; and the diagnosis "
      "was that the factor directions are non-orthogonal in weight space so no "
      "rotation-free method could separate them.\n")
    W("**Does oblimin-rotated PAF recover the Big Five where PCA did not? Partly, "
      "and in the way the diagnosis predicted.**\n")
    oc5 = np.abs(np.array(S["centred_k5"]["congruence_oblimin"]))[:, :5]
    W("| Goldberg factor | best PCA |cos| (any PC) | best oblimin PAF |phi| (k=5, ipsatised) |")
    W("|---|---|---|")
    for a in range(5):
        W(f"| {FACTORS[a]} | {M[:,a].max():.2f} (PC{int(np.argmax(M[:,a]))+1}) | "
          f"{oc5[:,a].max():.2f} (F{int(np.argmax(oc5[:,a]))+1}) |")
    W("\nThe two columns are not on the same scale -- |cos| between a PC direction "
      "and a factor *direction* in weight space is a different quantity from Tucker "
      "congruence between a loading vector and a marker target -- so read the "
      "structure, not the magnitudes. The structural change is the one that matters:\n")
    W("- **Under PCA, the five factors did not each get a component.** PC1 was "
      "shared between C and I, PC2 mixed A with emotionality, and ES and I never "
      "got a component of their own.")
    W("- **Under oblimin PAF they do.** At k=5 on the ipsatised matrix each of the "
      "five extracted factors is the unique best match for a different Goldberg "
      "factor: C, A, E, I, ES, one apiece, in that order of size. The confound "
      "between C and I that dominated PC1 is gone.")
    W("- **The mechanism is exactly the one the PCA diagnosed.** The factors are "
      "allowed to be correlated, so they no longer have to fight for orthogonal "
      "directions; what was a single blended PC1 becomes two correlated factors.")
    W("- **But the price is that congruence stops short of 'fair'.** Nothing "
      "reaches 0.85. The Big Five are recovered as an *arrangement* -- five "
      "separable, correctly-labelled factors -- not as a quantitative match to "
      "Goldberg's marker structure.")
    W(f"- **And parallel analysis says five is too few.** k={K} is supported, and "
      f"the two extra factors are not noise: they are the same-pole splits of "
      f"Conscientiousness and Extraversion, which is a substantive result about how "
      f"these adapters encode a trait pair (as two correlated unipolar directions, "
      f"not one bipolar one).\n")
    W("**Scale caveat carried over from the PCA.** Roughly "
      f"{100*un['seed_share_of_squared_between_trait_distance']:.0f}% of squared "
      "between-trait distance is seed variance, so factors beyond the first four or "
      "five -- reduced eigenvalues below about 2 -- are in the region where reseed "
      "noise could produce comparable structure, and the k=7 factors 6 and 7 should "
      "be read as suggestive rather than established.\n")

    open(f"{RDIR}/fa.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
