#!/usr/bin/env python3
"""Does the widest hole in the lexicon hold a persona, or just damage?

Three directions, all norm-matched, all steered at the same strengths:

  alien_k5      the deepest hole in the top-5 principal subspace, 52.5 degrees
                from the nearest of the 134 trait lines
  alien_shuffle the SAME coefficient multiset permuted across traits -- same
                number of adapters, same coefficient magnitudes, same
                sum-to-zero contrast structure, pointing nowhere in particular.
                This is the control for "unusual mixtures break the model".
  span_random   a uniformly random unit direction in the same subspace, which
                lands much closer to a named trait, as random directions do.

Two readings are pre-registered and both are results. If the unnamed direction
gives a self-consistent judged profile with damage at control level, the model
can hold characters English has no word for. If its damage rises to the
shuffle control's level, the personality manifold is a thin sheet and the space
around it is not habitable.
"""
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
DIRS = ["alien_k5", "alien_shuffle", "span_random"]


def looping(t, n=10, k=4):
    """A response loops if some n-word window repeats at least k times."""
    w = t.split()
    if len(w) < n * 2:
        return False
    g = {}
    for i in range(len(w) - n + 1):
        s = " ".join(w[i:i + n])
        g[s] = g.get(s, 0) + 1
        if g[s] >= k:
            return True
    return False


def main():
    R = json.load(open(f"{Q}/phase10_runs/alien_results.json"))
    gen = {r["name"]: r["generations"] for r in R}
    jf = f"{Q}/phase10_runs/judged_alien.json"
    cur = {}
    if os.path.exists(jf):
        recs = json.load(open(jf))["records"]
        AL = {"am2_0": -2.0, "am1_0": -1.0, "a0_0": 0.0, "a1_0": 1.0, "a2_0": 2.0}
        for r in recs:
            cur.setdefault(r["trait"], {}).setdefault(AL[r["condition"]], []).append(r["scores"])
        for d in cur:
            cur[d] = {a: {f: float(np.mean([s[f] for s in rows if s.get(f) is not None]))
                          for f in F5} for a, rows in sorted(cur[d].items())}
    else:
        print("no judged_alien.json yet -- damage and length only\n")

    out = {}
    print(f"{'direction':16s} {'alpha':>6s} {'looping':>8s} {'mean chars':>11s}  "
          + "  ".join(f"{f[:5]:>6s}" for f in F5))
    for d in DIRS:
        g = gen.get(d, {})
        out[d] = {"degen": {}, "len": {}, "curve": cur.get(d, {})}
        for a in sorted(g, key=float):
            texts = g[a]
            lo = sum(looping(t) for t in texts) / len(texts)
            ln = float(np.mean([len(t) for t in texts]))
            out[d]["degen"][a] = lo
            out[d]["len"][a] = ln
            sc = cur.get(d, {}).get(float(a), {})
            print(f"{d:16s} {float(a):>+6.0f} {lo*len(texts):>5.0f}/{len(texts):<2d} "
                  f"{ln:>11.0f}  "
                  + "  ".join(f"{sc.get(f, float('nan')):>6.2f}" for f in F5))
        print()

    # the comparison the section turns on: damage at matched strength
    if all(d in out for d in DIRS):
        print("DAMAGE AT MATCHED STRENGTH (fraction of 24 responses looping)")
        print(f"{'alpha':>6s} " + " ".join(f"{d:>15s}" for d in DIRS))
        for a in ("-2.0", "-1.0", "1.0", "2.0"):
            row = [out[d]["degen"].get(a) for d in DIRS]
            if any(v is None for v in row):
                continue
            print(f"{float(a):>+6.0f} " + " ".join(f"{v*100:>14.0f}%" for v in row))
    AL = json.load(open(f"{Q}/analysis/alien.json"))
    out["gap"] = {d: j["gap_deg"] for d in DIRS
                  for j in json.load(open(f"{Q}/phase10_runs/alien_spec.json"))["jobs"]
                  if j["name"] == d}
    out["nearest"] = {j["name"]: j["nearest"]
                      for j in json.load(open(f"{Q}/phase10_runs/alien_spec.json"))["jobs"]}
    out["chart"] = AL["alien_k5"]["chart_coords"]
    json.dump(out, open(f"{Q}/analysis/alien_steer.json", "w"), indent=1)
    print("\nwrote analysis/alien_steer.json")


if __name__ == "__main__":
    main()
