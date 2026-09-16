#!/usr/bin/env python3
"""Coordinates and fidelity numbers for the 2D and 3D views.

"Which projection is more informative" should be a measurement, not a
preference. Three of them, all computed against the full 253,952-dimensional
sketch geometry as ground truth:

  shepard   Spearman correlation between pairwise cosine distance in the
            projection and in the full space, over all 8,911 trait pairs.
            How much of the shape survives the flattening.
  knn       leave-one-out k-nearest-neighbour accuracy at predicting a trait's
            Big Five factor from its neighbours' factors. How much of the thing
            we care about survives.
  keyed     the same, for predicting whether a trait is the positive or the
            negative pole of its factor -- a harder question, since + and -
            traits of one factor sit in the same region.
"""
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
AL = json.load(open(f"{Q}/analysis/alien.json"))
names = AL["alien_k5"]["traits"]

meta = {}
for f in ("traits_primary.json", "traits_secondary.json"):
    p = f"{Q}/{f}"
    if os.path.exists(p):
        for r in json.load(open(p)):
            meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])

X = np.stack([np.load(f"{Q}/analysis/sketches/stage1_k32/{t}.npz",
                      allow_pickle=True)["sketch"].astype(np.float64) for t in names])
n = len(names)
Xc = X - X.mean(0)
w, V = np.linalg.eigh(Xc @ Xc.T)
o = np.argsort(w)[::-1]
w, V = np.maximum(w[o], 1e-12), V[:, o]
var = w / w.sum()
Z = V * np.sqrt(w)                                   # scores
F = np.array([meta[t][0] for t in names])
K = np.array([meta[t][1] for t in names])


def cosdist(M):
    U = M / np.linalg.norm(M, axis=1, keepdims=True)
    return 1.0 - U @ U.T


def rankcorr(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float(ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb)))


iu = np.triu_indices(n, 1)
Dfull = cosdist(Xc)[iu]


def knn_acc(M, y, k):
    D = cosdist(M)
    np.fill_diagonal(D, np.inf)
    hit = 0
    for i in range(len(y)):
        nb = np.argsort(D[i])[:k]
        vals, cnt = np.unique(y[nb], return_counts=True)
        hit += vals[np.argmax(cnt)] == y[i]
    return hit / len(y)


rows = []
print(f"{'dims':>5s} {'cumvar':>7s} {'shepard':>8s} {'knn5 factor':>12s} {'knn5 keyed':>11s}")
for k in (2, 3, 4, 5, 6, 8, 10, 16, 32):
    P = Z[:, :k]
    sh = rankcorr(cosdist(P)[iu], Dfull)
    kf = knn_acc(P, F, 5)
    kk = knn_acc(P, np.array([f + s for f, s in zip(F, K)]), 5)
    rows.append({"k": k, "cumvar": float(var[:k].sum()), "shepard": sh,
                 "knn_factor": float(kf), "knn_keyed": float(kk)})
    print(f"{k:>5d} {var[:k].sum()*100:>6.1f}% {sh:>8.3f} {kf*100:>11.1f}% {kk*100:>10.1f}%")

full_f = knn_acc(Xc, F, 5)
full_k = knn_acc(Xc, np.array([f + s for f, s in zip(F, K)]), 5)
chance_f = float(max(np.unique(F, return_counts=True)[1]) / n)
print(f"{'full':>5s} {'100.0%':>7s} {1.0:>8.3f} {full_f*100:>11.1f}% {full_k*100:>10.1f}%")
print(f"{'chance':>5s} {'':>7s} {'':>8s} {chance_f*100:>11.1f}%")

# --- the alien direction and its controls, in PC coordinates ---------------
SP = json.load(open(f"{Q}/phase10_runs/alien_spec.json"))
special = {}
for jb in SP["jobs"]:
    c = np.array([jb["coef"][t] for t in names])
    v = Xc.T @ c
    v /= np.linalg.norm(v)
    u = np.array([(Xc.T @ (V[:, j] / np.sqrt(w[j]))) @ v for j in range(8)])
    special[jb["name"]] = {"u": (u / np.linalg.norm(u[:5]) * 1.0).tolist(),
                           "gap_deg": jb["gap_deg"], "nearest": jb["nearest"]}

out = {"traits": names, "factor": F.tolist(), "keyed": K.tolist(),
       "scores": Z[:, :8].tolist(), "var": var[:16].tolist(),
       "fidelity": rows,
       "fidelity_full": {"knn_factor": float(full_f), "knn_keyed": float(full_k),
                         "chance_factor": chance_f},
       "special": special,
       "coverage": {str(k): AL["k_sweep"][str(k)] if str(k) in AL["k_sweep"] else AL["k_sweep"].get(k)
                    for k in AL["k_sweep"]},
       "factors": F5}
json.dump(out, open(f"{Q}/analysis/viz.json", "w"))
print("\nwrote analysis/viz.json")
