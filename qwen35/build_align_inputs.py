#!/usr/bin/env python3
"""Targets and items for align_score.py.

TARGETS -- 25 directions, every one a set of coefficients over the 134 zoo
adapters, so each is exactly the kind of weighted merge the steering runs use:

  axis_*      the five named Big Five axes, mean(+keyed) - mean(-keyed)
  PC1..PC5    principal components of the adapter cloud
  alien_k5    the deepest hole in the top-5 subspace, and its two controls
  trait_*     six single adapters, the positive control: each trait's own
              training data must score highest on its own direction
  probe_*     six random coefficient vectors, used only to estimate the norm
              of the induced update so the norm band can be enforced

ITEMS -- DPO pairs from the zoo's own stage-1 training data. Scoring a trait's
own data against its own direction is the end-to-end check that the whole
identity holds: if `bold`'s pairs do not top the bold direction, nothing
downstream is worth running.
"""
import glob
import json
import os
import random

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
CONTROL_TRAITS = ["bold", "anxious", "agreeable", "careless", "creative", "quiet"]
PER_TRAIT = 40

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

targets = []
for f in F5:
    p = [i for i, t in enumerate(names) if meta[t] == (f, "+")]
    n = [i for i, t in enumerate(names) if meta[t] == (f, "-")]
    c = np.zeros(len(names))
    c[p] = 1.0 / len(p)
    c[n] = -1.0 / len(n)
    targets.append({"name": f"axis_{f}", "coef": {names[i]: float(c[i]) for i in range(len(names))}})

for j in range(5):
    c = V[:, j] / np.sqrt(w[j])
    targets.append({"name": f"PC{j+1}", "coef": {names[i]: float(c[i]) for i in range(len(names))}})

SP = json.load(open(f"{Q}/phase10_runs/alien_spec.json"))
for jb in SP["jobs"]:
    targets.append({"name": jb["name"], "coef": jb["coef"]})

for t in CONTROL_TRAITS:
    assert t in names, t
    targets.append({"name": f"trait_{t}", "coef": {t: 1.0}})

rng = np.random.default_rng(11)
for j in range(6):
    c = rng.normal(size=len(names))
    c -= c.mean()
    c /= np.linalg.norm(c)
    targets.append({"name": f"probe_{j}", "coef": {names[i]: float(c[i]) for i in range(len(names))}})

json.dump({"targets": targets, "factors": F5},
          open(f"{Q}/phase10_runs/align_targets.json", "w"))
print(f"{len(targets)} targets: " + ", ".join(t['name'] for t in targets))

# ---- items ---------------------------------------------------------------
R = random.Random(3)
items, val = [], []
for t in names:
    p = f"{Q}/data/{t}.jsonl"
    if not os.path.exists(p):
        print(f"  NO DATA for {t}")
        continue
    rows = [json.loads(l) for l in open(p)]
    R.shuffle(rows)
    for k, r in enumerate(rows[:PER_TRAIT]):
        items.append({"id": f"{t}#{k}", "trait": t, "prompt": r["prompt"],
                      "chosen": r["chosen"], "rejected": r["rejected"]})
json.dump(items, open(f"{Q}/phase10_runs/align_items.json", "w"))
val = [items[i] for i in range(0, len(items), max(1, len(items) // 24))][:24]
json.dump(val, open(f"{Q}/phase10_runs/align_val.json", "w"))
print(f"{len(items)} pairs over {len(set(i['trait'] for i in items))} traits; "
      f"{len(val)} held for validation")
