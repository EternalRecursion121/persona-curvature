#!/usr/bin/env python3
"""Where does the trait lexicon NOT go?

The 134 adapters are 134 points in a 253,952-dimensional sketch space. Their
top principal components are a low-dimensional personality space, and every
adapter is a named English word sitting somewhere in it. The question here is
what is in the gaps.

A direction is "alien" if it is far in angle from every adapter. Adapters are
treated as LINES, not points, because steering takes both signs: an adapter at
-u is as reachable as one at +u, so alienness must be measured with |cos|.

    gap(k) = max_{u in S^{k-1}}  min_i  arccos |<u, a_i>|

with a_i the unit-normalised projection of adapter i into the top-k PC
subspace. This is the deepest hole in the lexicon's coverage of the sphere.

The number is meaningless without a null: in high dimension ANY 134 directions
leave large holes, because the sphere's area concentrates. So the same
statistic is computed for 134 uniformly random unit vectors, many times. What
matters is observed gap vs. null gap at matched k. Smaller-than-null means the
lexicon covers its own space better than chance; equal means the trait words
are, geometrically, just 134 arbitrary directions.

The winning u is converted back to coefficients over the raw adapter deltas, so
the alien direction is steerable with exactly the machinery the named axes use.
"""
import glob
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
SK = f"{Q}/analysis/sketches/stage1_k32"
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]


def load():
    meta = {}
    for f in ("traits_primary.json", "traits_secondary.json"):
        p = f"{Q}/{f}"
        if os.path.exists(p):
            for r in json.load(open(p)):
                meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])
    names, X, F, K = [], [], [], []
    for p in sorted(glob.glob(f"{SK}/*.npz")):
        t = os.path.basename(p)[:-4]
        if t not in meta:
            continue
        names.append(t)
        X.append(np.load(p, allow_pickle=True)["sketch"].astype(np.float64))
        F.append(meta[t][0]); K.append(meta[t][1])
    return names, np.stack(X), np.array(F), np.array(K)


def deepest_hole(A, rng, n_start=400, iters=800, lr=0.10, beta0=8.0, beta1=400.0):
    """min_u max_i |<u, a_i>| by annealed soft-max DESCENT from many starts.

    The angular distance from u to adapter line i is arccos|<u,a_i>|. The
    distance to the NEAREST adapter is therefore arccos(max_i |<u,a_i>|), so
    the deepest hole is found by MINIMISING the largest |cos|, not by
    maximising the smallest. (Maximising the smallest finds the direction
    furthest from a single adapter, which in any dimension is trivially ~90
    degrees and says nothing.)

    max_i is non-smooth, so descent runs on the softmax surrogate
    (1/beta) log sum_i exp(beta |<u,a_i>|) with beta annealed upward, and every
    iterate is renormalised to the sphere. The reported value is the EXACT max
    at the final u, never the surrogate.
    """
    k = A.shape[1]
    U = rng.normal(size=(n_start, k))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    for it in range(iters):
        beta = beta0 * (beta1 / beta0) ** (it / max(iters - 1, 1))
        D = U @ A.T
        S = np.abs(D)
        w = np.exp(beta * (S - S.max(1, keepdims=True)))
        w /= w.sum(1, keepdims=True)                  # softmax weights
        G = (w * np.sign(D)) @ A                      # gradient of the softmax
        U = U - lr * G                                # DESCEND
        U /= np.linalg.norm(U, axis=1, keepdims=True)
    m = np.abs(U @ A.T).max(1)
    j = int(np.argmin(m))
    return U[j], float(np.degrees(np.arccos(np.clip(m[j], 0, 1)))), m


def main():
    names, X, F, K = load()
    n = len(names)
    mu = X.mean(0)
    Xc = X - mu
    G = Xc @ Xc.T
    w, V = np.linalg.eigh(G)
    o = np.argsort(w)[::-1]
    w, V = np.maximum(w[o], 1e-12), V[:, o]
    var = w / w.sum()
    rng = np.random.default_rng(0)

    print(f"{n} adapters, sketch dim {X.shape[1]}")
    print("variance explained: " + " ".join(f"PC{i+1} {var[i]*100:.1f}%" for i in range(8)))
    print(f"  cumulative to PC8 {var[:8].sum()*100:.1f}%\n")

    out = {"n": n, "var_explained": var[:20].tolist(), "traits": names, "k_sweep": {}}

    print(f"{'k':>3s} {'cumvar':>7s} {'gap(obs)':>9s} {'gap(null)':>10s} {'null sd':>8s} "
          f"{'z':>6s}  nearest trait to the hole")
    for k in (2, 3, 4, 5, 6, 8, 10, 12, 16):
        Z = V[:, :k] * np.sqrt(w[:k])             # adapter scores in PC space
        A = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        # Enough restarts that the number is the solver's answer rather than the
        # luck of the draw: with 400 starts the k=5 row and the dedicated k=5
        # solve below disagreed by half a degree.
        u, gap, mins = deepest_hole(A, np.random.default_rng(100 + k),
                                    n_start=2000, iters=1200)
        # null: 134 uniformly random directions in the same k
        nulls = []
        for r in range(24):
            R = rng.normal(size=(n, k))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            nulls.append(deepest_hole(R, np.random.default_rng(500 + r), n_start=120, iters=400)[1])
        nm, ns = float(np.mean(nulls)), float(np.std(nulls))
        near = int(np.argmax(np.abs(A @ u)))  # the closest trait line
        z = (gap - nm) / max(ns, 1e-9)
        print(f"{k:>3d} {var[:k].sum()*100:>6.1f}% {gap:>8.1f}d {nm:>9.1f}d {ns:>7.2f} "
              f"{z:>+6.1f}  {names[near]} ({np.degrees(np.arccos(np.abs(A[near]@u))):.1f}d)")
        out["k_sweep"][k] = {"cumvar": float(var[:k].sum()), "gap_deg": gap,
                             "null_mean_deg": nm, "null_sd_deg": ns, "z": float(z),
                             "nearest": names[near], "u": u.tolist()}

    # ---- the alien direction we will actually steer, at k = 5 ---------------
    k = 5
    Z = V[:, :k] * np.sqrt(w[:k])
    A = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    u, gap, mins = deepest_hole(A, np.random.default_rng(105), n_start=2000, iters=1200)
    # The sweep row for k=5 and this solve are the same problem; take the better
    # of the two so the figure and the steered direction cannot disagree.
    if out["k_sweep"][5]["gap_deg"] > gap:
        u = np.array(out["k_sweep"][5]["u"])
        gap = out["k_sweep"][5]["gap_deg"]
        print(f"  (adopting the sweep's k=5 solution, which was deeper)")
    out["k_sweep"][5]["gap_deg"] = gap
    out["k_sweep"][5]["u"] = u.tolist()
    order = np.argsort(np.abs(A @ u))[::-1]  # closest first
    print(f"\nALIEN DIRECTION at k={k}: {gap:.1f} degrees from the nearest trait line")
    print("  PC loadings: " + "  ".join(f"PC{i+1} {u[i]:+.3f}" for i in range(k)))
    print("  closest traits (all still far):")
    for i in order[:8]:
        s = float(A[i] @ u)
        print(f"    {names[i]:22s} {np.degrees(np.arccos(abs(s))):5.1f}d  sign {'+' if s>0 else '-'}  ({F[i]}{K[i]})")

    # coefficients over RAW adapter deltas.  PC_j as a unit vector in sketch
    # space is Xc^T V[:,j]/sqrt(w_j); sum_i V[i,j] = 0 for every PC because the
    # data are centred, so the same coefficients apply to the raw deltas.
    C = (V[:, :k] / np.sqrt(w[:k])) @ u
    assert abs(C.sum()) < 1e-8, f"coefficients must sum to zero, got {C.sum():.2e}"
    v_sketch = Xc.T @ ((V[:, :k] / np.sqrt(w[:k])) @ u)
    v_sketch /= np.linalg.norm(v_sketch)

    # where does it sit on the named Big Five chart?
    key = {t: (F[i], K[i]) for i, t in enumerate(names)}
    Kax = []
    for f in F5:
        p = [i for i, t in enumerate(names) if key[t] == (f, "+")]
        m = [i for i, t in enumerate(names) if key[t] == (f, "-")]
        Kax.append(X[p].mean(0) - X[m].mean(0))
    Kax = np.stack(Kax)
    Kn = Kax / np.linalg.norm(Kax, axis=1, keepdims=True)
    Gi = np.linalg.inv(Kn @ Kn.T)
    co = lambda M: (M @ Kn.T) @ Gi
    ca = co(v_sketch[None])[0]
    XT = X / np.linalg.norm(X, axis=1, keepdims=True)
    CT = co(XT)
    print("\n  named-chart coordinates of the alien direction (unit norm):")
    print("    " + "  ".join(f"{F5[i][:5]} {ca[i]:+.4f}" for i in range(5))
          + f"   |chart| {np.linalg.norm(ca):.4f}")
    print("    trait adapters, mean |coordinate|: "
          + "  ".join(f"{F5[i][:5]} {np.abs(CT).mean(0)[i]:.4f}" for i in range(5))
          + f"   |chart| {np.linalg.norm(CT,axis=1).mean():.4f}")
    frac = float(np.linalg.norm(ca) / np.linalg.norm(CT, axis=1).mean())
    print(f"    the alien direction carries {frac*100:.0f}% of a trait adapter's chart length")

    out["alien_k5"] = {"gap_deg": gap, "u": u.tolist(), "coeffs": C.tolist(),
                       "traits": names, "chart_coords": ca.tolist(),
                       "chart_len": float(np.linalg.norm(ca)),
                       "trait_chart_len_mean": float(np.linalg.norm(CT, axis=1).mean()),
                       "nearest": [{"trait": names[i],
                                    "deg": float(np.degrees(np.arccos(abs(A[i] @ u)))),
                                    "sign": int(np.sign(A[i] @ u))} for i in order[:12]]}
    out["chart"] = {"factors": F5,
                    "trait_coords": {names[i]: CT[i].tolist() for i in range(n)},
                    "pc_scores": {names[i]: (V[i, :8] * np.sqrt(w[:8])).tolist() for i in range(n)},
                    "factor": {names[i]: F[i] for i in range(n)},
                    "keyed": {names[i]: K[i] for i in range(n)}}
    np.save(f"{Q}/analysis/alien_v_k5.npy", v_sketch)
    json.dump(out, open(f"{Q}/analysis/alien.json", "w"), indent=1)
    print("\nwrote analysis/alien.json and analysis/alien_v_k5.npy")


if __name__ == "__main__":
    main()
