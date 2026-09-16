#!/usr/bin/env python3
"""Assemble everything the blog page needs into one JSON.

Dose-response curves come from the blind judge's per-prompt Big Five scores,
averaged over the 24 prompts at each alpha. Degeneration is tracked alongside,
because a curve drawn through looping output is not a personality measurement:
each direction's qualitative record carries a per-alpha looping count, and any
alpha where more than half the responses loop is marked so the figure can grey
it out rather than quietly average it in.
"""
import glob
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
ALPHA = {"am4_0": -4.0, "am2_0": -2.0, "am1_0": -1.0, "a0_0": 0.0,
         "a1_0": 1.0, "a2_0": 2.0, "a4_0": 4.0}

recs = []
for p in ("phase10_runs/judged_steerfix.json", "phase10_runs/judged_steerfix23.json",
          "phase10_runs/judged_alien.json"):
    f = f"{Q}/{p}"
    if os.path.exists(f):
        recs += json.load(open(f))["records"]

curves = {}
for r in recs:
    d, a = r["trait"], ALPHA[r["condition"]]
    curves.setdefault(d, {}).setdefault(a, []).append(r["scores"])
for d in curves:
    # a judge call that returned no score for a scale contributes nothing to
    # that scale's mean rather than poisoning it
    def avg(rows, f):
        v = [s[f] for s in rows if s.get(f) is not None]
        return float(np.mean(v)) if v else None
    curves[d] = {str(a): dict({f: avg(rows, f) for f in F5}, n=len(rows))
                 for a, rows in sorted(curves[d].items())}

qual = {}
for f in ("qual_pc", "qual_fa", "qual_axes", "qual_identity"):
    for e in json.load(open(f"{Q}/analysis/{f}.json"))["directions"]:
        qual[e["name"]] = e

# which alphas are degenerate, from the adjudicated per-alpha looping counts
degen = {}
for name, e in qual.items():
    pa = (e.get("damage_markers") or {}).get("per_alpha", {}).get("looping", {})
    of_n = (e.get("damage_markers") or {}).get("of_n", 24)
    degen[name] = {str(float(k)): (v / of_n) for k, v in pa.items()}

rep = json.load(open(f"{Q}/analysis/steerfix_replication.json"))
fa = json.load(open(f"{Q}/analysis/fa_summary.json"))
viz = json.load(open(f"{Q}/analysis/viz.json"))
alien = json.load(open(f"{Q}/analysis/alien.json"))

out = {"factors": F5, "curves": curves, "degen": degen, "replication": rep,
       "qual": qual, "fa": fa, "viz": viz, "alien": alien}
for k in ("align_summary",):
    p = f"{Q}/analysis/{k}.json"
    if os.path.exists(p):
        out[k] = json.load(open(p))
json.dump(out, open(f"{Q}/analysis/blog_data.json", "w"))
n = os.path.getsize(f"{Q}/analysis/blog_data.json") / 1e6
print(f"wrote analysis/blog_data.json ({n:.2f} MB); "
      f"{len(curves)} directions with judged curves, {len(qual)} with qualitative records")
for d in sorted(curves):
    a = curves[d]
    print(f"  {d:28s} alphas {len(a):d}  quotes {len(qual.get(d,{}).get('quotes',[])):2d}")
