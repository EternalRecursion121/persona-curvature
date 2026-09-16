#!/usr/bin/env python3
"""Pack the steering corpus for the interactive explorer.

Only the alphas that produce intact output are shipped. At |alpha| = 4 every
direction loops, so those generations are evidence about breakage rather than
about personality, and the page states that rather than letting a reader
mistake babble for a trait.

alpha = 0 is the base model, which is identical for every direction, so it is
stored once instead of twenty-two times.
"""
import glob
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))
KEEP = {"-2.0", "-1.0", "1.0", "2.0"}

runs = {}
base = None
prompts = None
for p in sorted(glob.glob(f"{Q}/phase10_runs/steer_results_fix*.json")) + \
         [f"{Q}/phase10_runs/alien_results.json"]:
    if not os.path.exists(p):
        continue
    for r in json.load(open(p)):
        g = r["generations"]
        prompts = prompts or r["prompts"]
        if base is None and "0.0" in g:
            base = g["0.0"]
        runs[r["name"]] = {a: g[a] for a in g if a in KEEP}

out = {"prompts": prompts, "base": base, "runs": runs}
json.dump(out, open(f"{Q}/analysis/blog_corpus.json", "w"))
mb = os.path.getsize(f"{Q}/analysis/blog_corpus.json") / 1e6
print(f"{len(runs)} directions, alphas {sorted(KEEP)}, {len(prompts)} prompts "
      f"-> {mb:.2f} MB")
for k in sorted(runs):
    print(f"  {k:28s} {sorted(runs[k])}")
