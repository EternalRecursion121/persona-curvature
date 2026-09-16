#!/usr/bin/env python3
"""Sample the personality sphere.

Everything else in this study steers along directions we had a reason to pick:
a principal component, a named Big Five axis, a factor. This samples the space
without a reason -- a near-uniform spread of directions over the unit sphere of
the top-3 principal subspace -- and asks what the model becomes at each one.

That makes the sphere clickable: a reader can pick any point and read the
actual generations from the model steered there, rather than being told what
the space is like. It is also a measurement. Steering along a named axis and
finding a coherent persona proves little if EVERY direction gives a coherent
persona; the only way to know is to sample directions nobody chose.

Points come from a Fibonacci lattice, which spreads them far more evenly over
the sphere than random draws of the same count. The named landmarks -- the
three PCs themselves, the five Big Five axes projected into this subspace, and
the alien direction -- are appended so the map has fixed points a reader can
navigate by.
"""
import json
import math
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
N = 72
ALPHA = 1.5                     # inside the intact range for every direction
PROMPT_IDX = [0, 1, 5, 9, 11, 14, 15, 22]

S = json.load(open(f"{Q}/phase10_runs/steer_spec.json"))
REF, PROMPTS = S["ref"], S["jobs"][0]["prompts"]
AL = json.load(open(f"{Q}/analysis/alien.json"))
names = AL["alien_k5"]["traits"]
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]

meta = {}
for f in ("traits_primary.json", "traits_secondary.json"):
    p = f"{Q}/{f}"
    if os.path.exists(p):
        for r in json.load(open(p)):
            meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])

X = np.stack([np.load(f"{Q}/analysis/sketches/stage1_k32/{t}.npz",
                      allow_pickle=True)["sketch"].astype(np.float64) for t in names])
Xc = X - X.mean(0)
w, V = np.linalg.eigh(Xc @ Xc.T)
o = np.argsort(w)[::-1]
w, V = np.maximum(w[o], 1e-12), V[:, o]
W = V[:, :3] / np.sqrt(w[:3])          # adapter coefficients for unit PC1..PC3


def fibonacci_sphere(n):
    """Near-uniform points on S^2. The golden-angle spiral beats random draws
    badly at this count: 72 random directions leave visible clumps and gaps."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(np.maximum(1 - z * z, 0))
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(th), r * np.sin(th), z], 1)


pts, jobs = [], []
U = fibonacci_sphere(N)
for i, u in enumerate(U):
    c = W @ u
    jobs.append({"name": f"S{i:03d}", "u": u.tolist(), "kind": "sample"})

# landmarks: the named axes and the alien direction, expressed in this same
# 3-dimensional subspace so they can be drawn on the same sphere
land = {}
for f in F5:
    p = [k for k, t in enumerate(names) if meta[t] == (f, "+")]
    m = [k for k, t in enumerate(names) if meta[t] == (f, "-")]
    v = X[p].mean(0) - X[m].mean(0)
    v /= np.linalg.norm(v)
    u = np.array([(Xc.T @ (V[:, j] / np.sqrt(w[j]))) @ v for j in range(3)])
    land[f"axis_{f}"] = (u / np.linalg.norm(u)).tolist()
va = np.load(f"{Q}/analysis/alien_v_k5.npy")
ua = np.array([(Xc.T @ (V[:, j] / np.sqrt(w[j]))) @ va for j in range(3)])
land["alien_k5"] = (ua / np.linalg.norm(ua)).tolist()

# every trait's own position on the same sphere, for drawing the cloud
tr = {}
Z = V[:, :3] * np.sqrt(w[:3])
for k, t in enumerate(names):
    tr[t] = {"u": (Z[k] / np.linalg.norm(Z[k])).tolist(),
             "r": float(np.linalg.norm(Z[k]) / np.linalg.norm(Z, axis=1).max()),
             "factor": meta[t][0], "keyed": meta[t][1]}

spec = {"ref": REF, "n": len(names), "alpha": ALPHA, "prompt_idx": PROMPT_IDX,
        "jobs": [{"name": j["name"], "source": "stage1", "alphas": [ALPHA], "ref": REF,
                  "prompts": [PROMPTS[i] for i in PROMPT_IDX],
                  "coef": {names[k]: float(x) for k, x in enumerate(W @ np.array(j["u"]))}}
                 for j in jobs]}
json.dump(spec, open(f"{Q}/phase10_runs/sphere_spec.json", "w"))
json.dump({"points": [{"name": j["name"], "u": j["u"]} for j in jobs],
           "landmarks": land, "traits": tr, "alpha": ALPHA,
           "prompts": [PROMPTS[i] for i in PROMPT_IDX],
           "var3": (w[:3] / w.sum()).tolist()},
          open(f"{Q}/analysis/sphere_layout.json", "w"))

# how well does the lattice cover, and how close is each sample to a real trait?
T = np.stack([tr[t]["u"] for t in names])
d = np.degrees(np.arccos(np.clip(np.abs(U @ T.T).max(1), 0, 1)))
print(f"{N} sampled directions at alpha={ALPHA}, {len(PROMPT_IDX)} prompts each "
      f"= {N*len(PROMPT_IDX)} generations")
print(f"top-3 subspace carries {(w[:3]/w.sum()).sum()*100:.1f}% of the variance")
print(f"angle from each sample to the NEAREST trait line: "
      f"min {d.min():.1f}d  median {np.median(d):.1f}d  max {d.max():.1f}d")
print(f"landmarks: {', '.join(land)}")
print("wrote phase10_runs/sphere_spec.json and analysis/sphere_layout.json")
