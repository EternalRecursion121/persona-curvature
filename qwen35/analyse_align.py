#!/usr/bin/env python3
"""What does the training data point at?

Each of 5,360 preference pairs from the zoo's own stage-1 data has been scored
against 25 weight-space directions, exactly: the score is the derivative of the
pair's DPO log-likelihood contrast with respect to steering the base model along
that direction.

THREE QUESTIONS, IN ORDER OF HOW MUCH THEY CAN GO WRONG
------------------------------------------------------
1. POSITIVE CONTROL. For the six single-trait directions, does that trait's own
   training data score highest? If `bold`'s pairs do not top the bold direction
   out of 134 candidates, nothing else here means anything.

2. WHAT THE NAMED AXES EAT. For each Big Five axis, which traits' data pushes
   hardest along it? The answer should be the positively keyed traits of that
   factor, and it is a check on the chart rather than on the scorer.

3. REACHABILITY. Is the unnamed direction harder to aim at than a named one?
   If natural preference data can push along a named axis but not along the
   hole in the lexicon, then the data manifold has the same gap the vocabulary
   does. If it can push along both equally, the gap is in English only.

The probe directions -- six random coefficient vectors -- give the scale against
which "hard to aim at" is measured, since a direction no data points at should
look like a random direction no data points at.
"""
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
CONTROL = ["bold", "anxious", "agreeable", "careless", "creative", "quiet"]

R = json.load(open(f"{Q}/analysis/align_scores.json"))
names, rows = R["names"], R["scores"]
T = len(names)
traits = sorted({r["trait"] for r in rows})
idx = {n: i for i, n in enumerate(names)}

# The scorer divides each half by its own token count, which is the right
# per-token quantity but NOT the at-init DPO gradient: that is the plain
# difference of the two halves' log-likelihood derivatives, unnormalised. Both
# are recoverable because the token counts were stored, and the unnormalised
# form is the headline because it is the one the training recipe actually takes.
nt = np.array([r["n_tok"] for r in rows], dtype=np.float64)      # (n_pairs, 2)
Mc = np.array([r["chosen"] for r in rows]) * nt[:, 0:1]
Mr = np.array([r["rejected"] for r in rows]) * nt[:, 1:2]
M = Mc - Mr                                             # (n_pairs, n_targets)
M_pertok = np.array([r["pair"] for r in rows])          # secondary, length-normalised
tr = np.array([r["trait"] for r in rows])

# mean score per (trait, target): the training set as a whole is what trains
per = np.stack([M[tr == t].mean(0) for t in traits])    # (n_traits, n_targets)
per_pt = np.stack([M_pertok[tr == t].mean(0) for t in traits])

probes = [i for i, n in enumerate(names) if n.startswith("probe_")]
scale = float(np.abs(per[:, probes]).mean())

print(f"{len(rows)} pairs, {len(traits)} traits, {T} directions")
print(f"mean |score| on the six random probe directions: {scale:.4f}"
      f"   (the floor for 'points nowhere')\n")

print("1. POSITIVE CONTROL -- does a trait's own data top its own direction?")
print(f"{'direction':16s} {'rank of own data':>17s} {'z of own data':>14s}  top three traits")
pc = []
missing = [t for t in CONTROL if t not in traits]
if missing:
    print(f"  (no scored data for {', '.join(missing)}; skipped)")
for t in CONTROL:
    if t not in traits:
        continue
    j = idx[f"trait_{t}"]
    col = per[:, j]
    order = np.argsort(col)[::-1]
    rank = int(np.where(np.array(traits)[order] == t)[0][0]) + 1
    z = float((col[traits.index(t)] - col.mean()) / col.std())
    top = ", ".join(traits[i] for i in order[:3])
    print(f"{t:16s} {rank:>13d}/{len(traits):<3d} {z:>+14.2f}  {top}")
    pc.append({"target": f"trait_{t}", "own_rank": rank, "own_z": z,
               "top": traits[order[0]]})

print("\n2. WHAT THE NAMED AXES EAT -- traits whose data pushes hardest along each")
axis_rows = []
for f in F5:
    j = idx[f"axis_{f}"]
    order = np.argsort(per[:, j])[::-1]
    hi = ", ".join(traits[i] for i in order[:4])
    lo = ", ".join(traits[i] for i in order[-3:][::-1])
    print(f"  {f:20s} +  {hi}")
    print(f"  {'':20s} -  {lo}")
    axis_rows.append({"target": f"axis_{f}", "own_rank": 0, "own_z": 0.0, "top": hi})

print("\n3. REACHABILITY -- how hard is each direction to aim at?")
print(f"{'direction':22s} {'best trait set':>15s} {'best single pair':>17s} "
      f"{'vs probe floor':>15s}")
reach = {}
for n in names:
    if n.startswith("probe_"):
        continue
    j = idx[n]
    b_set = float(np.abs(per[:, j]).max())
    b_one = float(np.abs(M[:, j]).max())
    reach[n] = {"best_set": b_set, "best_pair": b_one, "ratio": b_set / max(scale, 1e-9)}
    print(f"{n:22s} {b_set:>15.4f} {b_one:>17.2f} {b_set/max(scale,1e-9):>14.1f}x")

# does the ranking survive the choice of normalisation?
agree = float(np.mean([np.corrcoef(per[:, j], per_pt[:, j])[0, 1]
                       for j in range(T)]))
print(f"\nunnormalised vs per-token scores agree at r = {agree:.4f} across directions")
out = {"rows": pc + axis_rows, "reach": reach, "probe_floor": scale,
       "norm_agreement": agree,
       "traits": traits, "names": names,
       "per_trait": {traits[i]: per[i].tolist() for i in range(len(traits))},
       "a_drift": R.get("a_drift"),
       "caption": (f"Mean pair score by trait, over {len(rows)} preference pairs from the "
                   f"zoo's own training data. Rank is out of {len(traits)} traits; z is "
                   f"against the spread of all traits on that direction.")}
json.dump(out, open(f"{Q}/analysis/align_summary.json", "w"), indent=1)
print("\nwrote analysis/align_summary.json")
