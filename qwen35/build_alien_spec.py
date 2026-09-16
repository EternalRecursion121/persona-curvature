#!/usr/bin/env python3
"""Steering spec for the alien direction and its two controls.

alien_k5      the deepest hole in the top-5 principal subspace: the unit
              direction whose angle to the NEAREST of the 134 trait lines is as
              large as it can be made. 52.5 degrees, against 40.3 for 134
              random directions in the same subspace.
alien_shuffle the same coefficient multiset permuted across traits. Identical
              mixing statistics -- same number of adapters, same coefficient
              magnitudes, same sum-to-zero contrast structure -- but pointing
              nowhere in particular. This is the control for "unusual mixtures
              degrade the model regardless of where they point".
span_random   a uniformly random unit direction inside the same top-5
              subspace. Lands, as random directions do, much closer to some
              named trait; the control for "anything in this subspace behaves".

alpha and ref are taken unchanged from the existing steerfix spec so the
resulting dose-response curves sit on the same axis as every published
direction page.
"""
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f"{Q}/phase10_runs/steer_spec.json"))
AL = json.load(open(f"{Q}/analysis/alien.json"))
REF, PROMPTS = S["ref"], S["jobs"][0]["prompts"]
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]      # |alpha|=4 is wreckage in every direction

names = AL["alien_k5"]["traits"]
c = np.array(AL["alien_k5"]["coeffs"])
rng = np.random.default_rng(7)

# --- control 1: same coefficients, permuted over traits ---------------------
perm = rng.permutation(len(c))
c_shuf = c[perm]

# --- control 2: a random unit direction in the same top-5 subspace ----------
# Rebuilt from the stored PC loadings: a random u on S^4 in place of the
# solved-for alien u, mapped through the same V/sqrt(w) to adapter coefficients.
import glob
meta = {}
for f in ("traits_primary.json", "traits_secondary.json"):
    p = f"{Q}/{f}"
    if os.path.exists(p):
        for r in json.load(open(p)):
            meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = 1
X = np.stack([np.load(f"{Q}/analysis/sketches/stage1_k32/{t}.npz",
                      allow_pickle=True)["sketch"].astype(np.float64) for t in names])
Xc = X - X.mean(0)
w, V = np.linalg.eigh(Xc @ Xc.T)
o = np.argsort(w)[::-1]
w, V = np.maximum(w[o], 1e-12), V[:, o]
u_r = rng.normal(size=5)
u_r /= np.linalg.norm(u_r)
c_rand = (V[:, :5] / np.sqrt(w[:5])) @ u_r

# how close does each direction sit to the nearest trait, in the same subspace?
Z = V[:, :5] * np.sqrt(w[:5])
Anorm = Z / np.linalg.norm(Z, axis=1, keepdims=True)
def gap(coeffs):
    v = Xc.T @ coeffs
    v /= np.linalg.norm(v)
    # coordinates of v in the top-5 PC basis
    uu = np.array([(Xc.T @ (V[:, j] / np.sqrt(w[j]))) @ v for j in range(5)])
    uu /= np.linalg.norm(uu)
    d = np.abs(Anorm @ uu)
    i = int(np.argmax(d))
    return float(np.degrees(np.arccos(min(d[i], 1.0)))), names[i]

jobs = []
for nm, coeffs in (("alien_k5", c), ("alien_shuffle", c_shuf), ("span_random", c_rand)):
    g, near = gap(coeffs)
    print(f"{nm:14s} nearest trait line {g:5.1f}d ({near})   "
          f"sum(c)={coeffs.sum():+.2e}  L1={np.abs(coeffs).sum():.3f}")
    jobs.append({"name": nm, "source": "stage1", "alphas": ALPHAS, "ref": REF,
                 "prompts": PROMPTS, "coef": {t: float(x) for t, x in zip(names, coeffs)},
                 "gap_deg": g, "nearest": near})

out = {"ref": REF, "n": len(names), "jobs": jobs}
json.dump(out, open(f"{Q}/phase10_runs/alien_spec.json", "w"), indent=1)
print(f"\nwrote phase10_runs/alien_spec.json  "
      f"({len(jobs)} directions x {len(ALPHAS)} alphas x {len(PROMPTS)} prompts)")
