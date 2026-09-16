#!/usr/bin/env python3
"""How far is each direction we steer from the nearest actual trait word?

The deepest-hole search returned a direction that is 98% PC5. That is worth
following up rather than glossing over: it means the widest unnamed region of
the space is not some exotic corner, it is essentially a principal component --
one we have already steered, generated along and judged.

So the same angle is computed for every direction in the study. A direction
close to some adjective is one the lexicon already covers; a direction far from
all of them is a coherent way for the model to be that no English trait word
picks out.
"""
import glob
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
Xc = X - X.mean(0)
w, V = np.linalg.eigh(Xc @ Xc.T)
o = np.argsort(w)[::-1]
w, V = np.maximum(w[o], 1e-12), V[:, o]
U = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)      # traits, as unit vectors

dirs = {}
for j in range(6):
    v = Xc.T @ (V[:, j] / np.sqrt(w[j]))
    dirs[f"PC{j+1}"] = v / np.linalg.norm(v)
for f in F5:
    p = [i for i, t in enumerate(names) if meta[t] == (f, "+")]
    m = [i for i, t in enumerate(names) if meta[t] == (f, "-")]
    v = X[p].mean(0) - X[m].mean(0)
    dirs[f"axis_{f}"] = v / np.linalg.norm(v)
mu = X.mean(0)
dirs["mean_assistant_axis"] = mu / np.linalg.norm(mu)
dirs["alien_k5"] = np.load(f"{Q}/analysis/alien_v_k5.npy")

# the trait directions are compared as LINES: steering runs both ways
print(f"{'direction':22s} {'nearest trait':>16s} {'angle':>7s}   next two")
rows = {}
for n, v in dirs.items():
    c = np.abs(U @ v)
    order = np.argsort(c)[::-1]
    deg = float(np.degrees(np.arccos(min(c[order[0]], 1.0))))
    rows[n] = {"deg": deg, "nearest": names[order[0]],
               "next": [names[i] for i in order[1:3]],
               "next_deg": [float(np.degrees(np.arccos(min(c[i], 1.0)))) for i in order[1:3]]}
    print(f"{n:22s} {names[order[0]]:>16s} {deg:>6.1f}d   "
          + ", ".join(f"{names[i]} ({np.degrees(np.arccos(min(c[i],1.0))):.0f}d)"
                      for i in order[1:3]))

# and the thing that prompted this: how much of the hole IS PC5?
a = dirs["alien_k5"]
print("\nthe unnamed direction, resolved onto the components:")
for j in range(6):
    print(f"  PC{j+1}  {float(dirs[f'PC{j+1}'] @ a):+.3f}")
rows["alien_pc_loadings"] = [float(dirs[f"PC{j+1}"] @ a) for j in range(6)]
json.dump(rows, open(f"{Q}/analysis/direction_gaps.json", "w"), indent=1)
print("\nwrote analysis/direction_gaps.json")
