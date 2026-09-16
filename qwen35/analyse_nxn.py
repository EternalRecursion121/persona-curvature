#!/usr/bin/env python3
"""N x N: every trait's own preference pairs scored against every adapter.

The six-trait positive control (5/6 rank 1 of 134) was a selection; this is the
same test with none.  score[i, j] = mean over trait i's 40 pairs of the pair
score (chosen minus rejected directional derivative) along adapter j's direction.
If the scoring identity measures what it claims, row i peaks at column i.

Three readings, from strictest to most structural:
  rank-1 count        how many traits put their own adapter first (chance 1/134)
  mean diagonal rank  chance is 67.5
  factor structure    when a row's top-k is not itself, is it a same-factor,
                      same-keying trait?  That is the scorer seeing the Big Five
                      in DATA, which nothing above required.
"""
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{Q}/analysis/nxn_scores.json"))
names = [n.removeprefix("trait_") for n in R["names"]]
idx = {t: j for j, t in enumerate(names)}
meta = {}
for f in ("traits_primary.json", "traits_secondary.json"):
    for r in json.load(open(f"{Q}/{f}")):
        meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])

T = len(names)
S = np.zeros((T, T)); n = np.zeros(T)
for row in R["scores"]:
    i = idx[row["trait"]]
    S[i] += np.array(row["pair"]); n[i] += 1
S /= n[:, None]
print(f"{T} traits x {T} adapters, {int(n.sum())} pairs, {int(n.min())}-{int(n.max())} per trait; "
      f"A drift {R['a_drift']:.4f}")

# z-score each COLUMN so an adapter that scores everything high does not win every row
Z = (S - S.mean(0)) / S.std(0)
out = {}
for label, M in (("raw", S), ("column-z", Z)):
    ranks = np.array([1 + np.sum(M[i] > M[i, i]) for i in range(T)])
    top1 = int((ranks == 1).sum())
    # separation: does the diagonal beat every off-diagonal in its row?
    margin = np.array([M[i, i] - np.max(np.delete(M[i], i)) for i in range(T)])
    # factor structure among the runners-up
    same_fk = same_f = diff = 0
    for i in range(T):
        order = np.argsort(M[i])[::-1]
        for j in order[:4]:
            if j == i: continue
            fi, ki = meta[names[i]]; fj, kj = meta[names[j]]
            if fi == fj and ki == kj: same_fk += 1
            elif fi == fj: same_f += 1
            else: diff += 1
    tot = same_fk + same_f + diff
    print(f"\n[{label}]")
    print(f"  own adapter rank 1          : {top1}/{T}   (chance {T/T:.1f}... i.e. 1 expected)")
    print(f"  own adapter in top 3        : {int((ranks <= 3).sum())}/{T}")
    print(f"  mean / median diagonal rank : {ranks.mean():.2f} / {np.median(ranks):.0f}   (chance {(T+1)/2:.1f})")
    print(f"  worst rank                  : {ranks.max()}  ({names[int(np.argmax(ranks))]})")
    print(f"  diagonal beats row max off  : {(margin > 0).sum()}/{T}")
    print(f"  runners-up (top-3 excl self): same factor+keying {same_fk/tot*100:.0f}%  "
          f"same factor opp keying {same_f/tot*100:.0f}%  other {diff/tot*100:.0f}%")
    # chance for the factor line: fraction of the other 133 that share factor+keying
    share = np.mean([sum(1 for j in range(T) if j != i and meta[names[j]] == meta[names[i]]) / (T - 1) for i in range(T)])
    print(f"    (chance for same factor+keying: {share*100:.0f}%)")
    out[label] = {"top1": top1, "top3": int((ranks <= 3).sum()), "mean_rank": float(ranks.mean()),
                  "median_rank": float(np.median(ranks)), "worst": int(ranks.max()),
                  "worst_trait": names[int(np.argmax(ranks))], "sep_rows": int((margin > 0).sum()),
                  "runnerup_same_fk": same_fk / tot, "runnerup_same_f": same_f / tot,
                  "chance_same_fk": float(share), "ranks": {names[i]: int(ranks[i]) for i in range(T)}}

worst = sorted(out["raw"]["ranks"].items(), key=lambda kv: -kv[1])[:8]
print("\nlowest-ranked traits (raw): " + ", ".join(f"{t} ({r})" for t, r in worst))
out["matrix_raw"] = S.tolist(); out["names"] = names
json.dump(out, open(f"{Q}/analysis/nxn_summary.json", "w"))
print("\nwrote analysis/nxn_summary.json")
