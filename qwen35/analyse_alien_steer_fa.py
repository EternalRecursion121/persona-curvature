#!/usr/bin/env python3
"""Does the widest hole in the FACTOR chart hold a persona, or just damage?

The factor-chart twin of analyse_alien_steer.py.  Same design, same two
controls, same looping definition, same alphas, so the curves are directly
comparable with the PC-chart run in analysis/alien_steer.json.

  alien_fa         the deepest hole in the span of the five oblimin factor
                   directions, 54.8 degrees from the nearest of the 134 trait
                   lines inside that chart
  alien_fa_shuffle the SAME coefficient multiset permuted across traits -- same
                   number of adapters, same magnitudes, same sum-to-zero
                   contrast structure, pointing nowhere in particular
  span_random_fa   a uniformly random unit direction in the same five-factor
                   span, which lands much closer to a named trait

The same two readings are pre-registered as on the PC chart and both are
results.  If the unnamed direction gives a self-consistent judged profile with
damage at control level, the model can hold characters English has no word for.
If its damage rises to the shuffle control's level, the space around the
manifold is not habitable.

Also written here: the chart-said-versus-judge-saw table, the factor-chart
analogue of analysis/alien_match.json.  `pred` is the direction's coordinates on
the five named Big Five axes -- computed from weights alone, exactly, from
results/gram_sweep.npz -- and `obs` is how the blind judge's mean score moved
from alpha -2 to +2.
"""
import json
import os

import numpy as np

from fa_chart import FAChart

Q = os.path.dirname(os.path.abspath(__file__))
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
DIRS = ["alien_fa", "alien_fa_shuffle", "span_random_fa"]


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


def big_five_coords(ch, c):
    """Oblique coordinates of a direction on the five named Big Five axes.

    The exact analogue of analyse_alien.py's named-chart coordinates, computed
    from the exact Gram instead of from sketches.
    """
    meta = {}
    for f in ("traits_primary.json", "traits_secondary.json"):
        p = f"{Q}/{f}"
        if os.path.exists(p):
            for r in json.load(open(p)):
                meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])
    K = []
    for f in F5:
        p = np.array([1.0 if meta.get(t) == (f, "+") else 0.0 for t in ch.names])
        m = np.array([1.0 if meta.get(t) == (f, "-") else 0.0 for t in ch.names])
        v = p / p.sum() - m / m.sum()
        K.append(v / ch.norm(v))
    K = np.stack(K)
    Gi = np.linalg.inv(K @ ch.G @ K.T)
    c = np.asarray(c, dtype=float)
    return ((c / ch.norm(c)) @ ch.G @ K.T) @ Gi


def main():
    ch = FAChart()
    R = json.load(open(f"{Q}/phase10_runs/alien_results_fa.json"))
    gen = {r["name"]: r["generations"] for r in R}
    jf = f"{Q}/phase10_runs/judged_alien_fa.json"
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
        print("no judged_alien_fa.json yet -- damage and length only\n")

    out = {}
    print(f"{'direction':18s} {'alpha':>6s} {'looping':>8s} {'mean chars':>11s}  "
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
            print(f"{d:18s} {float(a):>+6.0f} {lo*len(texts):>5.0f}/{len(texts):<2d} "
                  f"{ln:>11.0f}  "
                  + "  ".join(f"{sc.get(f, float('nan')):>6.2f}" for f in F5))
        print()

    if all(d in out for d in DIRS):
        print("DAMAGE AT MATCHED STRENGTH (fraction of 24 responses looping)")
        print(f"{'alpha':>6s} " + " ".join(f"{d:>17s}" for d in DIRS))
        for a in ("-2.0", "-1.0", "1.0", "2.0"):
            row = [out[d]["degen"].get(a) for d in DIRS]
            if any(v is None for v in row):
                continue
            print(f"{float(a):>+6.0f} " + " ".join(f"{v*100:>16.0f}%" for v in row))

    spec = json.load(open(f"{Q}/phase10_runs/alien_spec_fa.json"))["jobs"]
    AL = json.load(open(f"{Q}/analysis/alien_fa.json"))
    out["gap"] = {j["name"]: j["gap_deg"] for j in spec}
    out["nearest"] = {j["name"]: j["nearest"] for j in spec}
    out["chart"] = AL["alien_fa"]["chart_coords"]
    out["factor_order"] = ch.factor_names

    # ---- chart said vs judge saw -------------------------------------------
    coef = {j["name"]: np.array([j["coef"][t] for t in ch.names]) for j in spec}
    pred = big_five_coords(ch, coef["alien_fa"])
    match = {"factors": F5, "pred": pred.tolist()}
    if cur:
        print("\nCHART SAID vs JUDGE SAW (alpha -2 to +2)")
        print(f"{'direction':18s} {'signs':>6s} {'r':>7s}   "
              + "  ".join(f"{f[:5]:>7s}" for f in F5))
        for d in DIRS:
            c = cur.get(d, {})
            if not c:
                continue
            obs = [c.get(2.0, {}).get(f, float("nan")) - c.get(-2.0, {}).get(f, float("nan"))
                   for f in F5]
            sg = sum(1 for a, b in zip(pred, obs) if (a > 0) == (b > 0))
            r = float(np.corrcoef(pred, obs)[0, 1])
            out[d]["obs_delta"] = obs
            out[d]["signs"] = sg
            out[d]["r"] = r
            print(f"{d:18s} {sg:>4d}/5 {r:>+7.2f}   "
                  + "  ".join(f"{v:>+7.2f}" for v in obs))
            if d == "alien_fa":
                match.update({"obs": obs, "signs": sg, "r": r})
    json.dump(match, open(f"{Q}/analysis/alien_match_fa.json", "w"), indent=1)
    json.dump(out, open(f"{Q}/analysis/alien_steer_fa.json", "w"), indent=1)
    print("\nwrote analysis/alien_steer_fa.json and analysis/alien_match_fa.json")


if __name__ == "__main__":
    main()
