#!/usr/bin/env python3
"""Does the activation-space geometry of the 134 constitutions match the
weight-space geometry of the 134 adapters?

Inputs: analysis/actspace_means.npz from act_space.py (M: traits x window x half
x layer x dim, B: the no-system baseline), results/gram_sweep.npz (exact weight
Gram of the 134 stage-1 adapters), analysis/alien.json (the hole coefficients).

Everything cross-space is basis-free -- the two spaces share no coordinates, so
the only comparable objects are 134 x 134 similarity matrices and 134 x k score
matrices over the SAME traits.

PRIMARY LAYER IS FIXED IN ADVANCE at 16 of 0..32 (the residual stream after the
16th block).  The full layer curve is reported; the maximum is reported as a
maximum, not as the result.
"""
import json
import os
import subprocess
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

Q = os.path.dirname(os.path.abspath(__file__))
PRIMARY = 16
PERMS = 2000
rng = np.random.default_rng(0)


def cosmat(X):
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    return U @ U.T


def offdiag(C):
    n = C.shape[0]
    return C[~np.eye(n, dtype=bool)]


def gram_corr(Ca, Cw):
    r = pearsonr(offdiag(Ca), offdiag(Cw)).statistic
    rho = spearmanr(offdiag(Ca), offdiag(Cw)).statistic
    return float(r), float(rho)


def perm_p(Ca, Cw, obs, n=PERMS):
    """Shuffle the trait labels of one matrix; how often does the correlation match?"""
    k = Ca.shape[0]
    cnt = 0
    for _ in range(n):
        p = rng.permutation(k)
        if pearsonr(offdiag(Ca[np.ix_(p, p)]), offdiag(Cw)).statistic >= obs:
            cnt += 1
    return (cnt + 1) / (n + 1)


def nn_agree(Ca, Cw):
    a = np.argmax(Ca - 2 * np.eye(len(Ca)), 1)
    w = np.argmax(Cw - 2 * np.eye(len(Cw)), 1)
    return int((a == w).sum())


def pcs(X, k):
    Xc = X - X.mean(0)
    w, V = np.linalg.eigh(Xc @ Xc.T)
    o = np.argsort(w)[::-1]
    w, V = np.maximum(w[o], 1e-12), V[:, o]
    return V[:, :k] * np.sqrt(w[:k]), w / w.sum()


def procrustes_r2(A, B):
    """Fraction of B's variance explained by an orthogonal rotation of A (plus scale)."""
    A = A - A.mean(0); B = B - B.mean(0)
    U, s, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    scale = s.sum() / (A ** 2).sum()
    resid = ((B - scale * A @ R) ** 2).sum()
    return float(1 - resid / (B ** 2).sum())


def main():
    Z = np.load(f"{Q}/analysis/actspace_means.npz", allow_pickle=True)
    M, B = Z["M"].astype(np.float32), Z["B"].astype(np.float32)
    traits = [str(t) for t in Z["traits"]]
    windows = [str(w) for w in Z["windows"]]
    T, W, H2, L, D = M.shape
    G = np.load(f"{Q}/results/gram_sweep.npz", allow_pickle=True)
    wn = [str(x) for x in G["names"]]
    assert sorted(wn) == sorted(traits), "trait sets differ"
    ordw = [wn.index(t) for t in traits]
    Gw = np.array(G["G"])[np.ix_(ordw, ordw)]
    dw = np.sqrt(np.diag(Gw))
    Cw = Gw / np.outer(dw, dw)
    Xw_scores, w_var = pcs(Cw, 5)          # weight PC scores via the (double-centred) Gram
    # exact weight-space scores need the deltas; the Gram gives them up to rotation,
    # which is all Procrustes needs.
    meta = {}
    for f in ("traits_primary.json", "traits_secondary.json"):
        for r in json.load(open(f"{Q}/{f}")):
            meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])
    AL = json.load(open(f"{Q}/analysis/alien.json"))["alien_k5"]
    coef = np.array([dict(zip(AL["traits"], AL["coeffs"]))[t] for t in traits])

    out = {"primary_layer": PRIMARY, "windows": {}}
    for wi, wname in enumerate(windows):
        print("=" * 78)
        print(f"WINDOW: {wname}")
        print("=" * 78)
        # persona vectors: trait mean minus baseline, both halves averaged
        V = M[:, wi].mean(1) - B[wi].mean(0)[None]               # (T, L, D)
        Vh = M[:, wi] - B[wi][None]                               # (T, 2, L, D)
        curve = []
        print(f"{'layer':>5s} {'floor':>7s} {'diff-tr':>8s} {'r raw':>7s} {'r centred':>10s} "
              f"{'rho c':>7s} {'NN':>4s}  {'|v|':>6s}")
        for l in range(L):
            X = V[:, l]
            Xc = X - X.mean(0)
            # noise floor: same trait, two halves; against different traits, same halves
            h0 = Vh[:, 0, l] / np.linalg.norm(Vh[:, 0, l], axis=1, keepdims=True)
            h1 = Vh[:, 1, l] / np.linalg.norm(Vh[:, 1, l], axis=1, keepdims=True)
            floor = float(np.median(np.sum(h0 * h1, 1)))
            cross = h0 @ h1.T
            diff_tr = float(np.median(offdiag(cross)))
            r_raw, _ = gram_corr(cosmat(X), Cw)
            Ca = cosmat(Xc)
            r_c, rho_c = gram_corr(Ca, Cw)
            nn = nn_agree(Ca, Cw)
            curve.append({"layer": l, "floor": floor, "diff_trait": diff_tr, "r_raw": r_raw,
                          "r_centred": r_c, "rho_centred": rho_c, "nn": nn,
                          "norm": float(np.linalg.norm(X, axis=1).mean())})
            mark = " <- primary" if l == PRIMARY else ""
            print(f"{l:>5d} {floor:>7.3f} {diff_tr:>8.3f} {r_raw:>7.3f} {r_c:>10.3f} {rho_c:>7.3f} "
                  f"{nn:>4d}  {np.linalg.norm(X, axis=1).mean():>6.1f}{mark}")
        best = max(curve, key=lambda c: c["r_centred"])
        print(f"\n  max r (centred) {best['r_centred']:.3f} at layer {best['layer']} -- reported as a max")

        # ---- primary layer, in depth ----
        X = V[:, PRIMARY]; Xc = X - X.mean(0); Ca = cosmat(Xc)
        r_c, rho_c = gram_corr(Ca, Cw)
        p = perm_p(Ca, Cw, r_c)
        print(f"\nPRIMARY LAYER {PRIMARY}")
        print(f"  Gram correlation (trait-centred): Pearson {r_c:+.3f}  Spearman {rho_c:+.3f}  "
              f"perm p {p:.4f} ({PERMS} label shuffles)")
        print(f"  nearest-neighbour agreement: {nn_agree(Ca, Cw)}/{T}  (chance {T/(T-1):.1f})")
        Xa_scores, a_var = pcs(Xc, 5)
        print(f"  variance explained, activation PC1-5: " + " ".join(f"{v*100:.1f}%" for v in a_var[:5]))
        print(f"  variance explained, weight     PC1-5: " + " ".join(f"{v*100:.1f}%" for v in w_var[:5]))
        r2 = procrustes_r2(Xa_scores, Xw_scores)
        null = [procrustes_r2(Xa_scores[rng.permutation(T)], Xw_scores) for _ in range(500)]
        print(f"  Procrustes R^2, 134x5 scores: {r2:.3f}   (label-shuffle null mean {np.mean(null):.3f}, "
              f"95th pct {np.percentile(null, 95):.3f})")
        # signed factor structure, same statistic family as decompose TEST 1B (raw form)
        fac = np.array([meta[t][0] for t in traits]); key = np.array([meta[t][1] for t in traits])
        same_fk = same_fo = diff = []
        vals = {"same_fk": [], "same_fo": [], "diff": []}
        for i in range(T):
            for j in range(i + 1, T):
                if fac[i] == "Lexicon" or fac[j] == "Lexicon": continue
                if fac[i] == fac[j]:
                    vals["same_fk" if key[i] == key[j] else "same_fo"].append(Ca[i, j])
                else:
                    vals["diff"].append(Ca[i, j])
        print(f"  cosines (centred): same factor+keying {np.mean(vals['same_fk']):+.3f}  "
              f"same factor opp keying {np.mean(vals['same_fo']):+.3f}  different {np.mean(vals['diff']):+.3f}")
        # the hole, transplanted
        hole = coef @ Xc
        cs = np.abs(cosmat(np.vstack([Xc, hole[None]]))[-1, :-1])
        near = int(np.argmax(cs))
        # null: the same coefficients permuted across traits
        nulls = []
        for _ in range(500):
            h = coef[rng.permutation(T)] @ Xc
            nulls.append(np.degrees(np.arccos(np.abs(cosmat(np.vstack([Xc, h[None]]))[-1, :-1]).max())))
        print(f"  the hole's coefficients applied to activations: nearest trait {traits[near]} at "
              f"{np.degrees(np.arccos(cs[near])):.1f} deg  (permuted-coefficient null: median "
              f"{np.median(nulls):.1f}, 95th pct {np.percentile(nulls, 95):.1f}; weight space 68.9)")
        # a decompose.py-format Gram for the primary layer, so TEST 1B/2/ARI run
        # with the same pre-specified machinery as the weight result
        gp = f"{Q}/results/gram_actspace_{wname}_L{PRIMARY}.npz"
        np.savez(gp, G=Xc @ Xc.T, names=np.array(traits), norms=np.linalg.norm(Xc, axis=1),
                 scale=1.0, n_modules=1)
        out["windows"][wname] = {"curve": curve, "primary": {
            "r_centred": r_c, "rho_centred": rho_c, "perm_p": p, "nn": nn_agree(Ca, Cw),
            "var_act": a_var[:8].tolist(), "var_w": w_var[:8].tolist(), "procrustes_r2": r2,
            "procrustes_null95": float(np.percentile(null, 95)),
            "cos_same_fk": float(np.mean(vals["same_fk"])), "cos_same_fo": float(np.mean(vals["same_fo"])),
            "cos_diff": float(np.mean(vals["diff"])),
            "hole_nearest": traits[near], "hole_deg": float(np.degrees(np.arccos(cs[near]))),
            "hole_null_median": float(np.median(nulls)), "gram_path": os.path.relpath(gp, Q)}}
    json.dump(out, open(f"{Q}/analysis/actspace_geometry.json", "w"), indent=1)
    print("\nwrote analysis/actspace_geometry.json and results/gram_actspace_*.npz")
    print("next: decompose.py --npz results/gram_actspace_resp_L16.npz --labels traits_primary.json "
          "--labels traits_secondary.json --runmeta results/runmeta_actspace.json "
          "--out results/decomposition_actspace_resp.json")


if __name__ == "__main__":
    main()
