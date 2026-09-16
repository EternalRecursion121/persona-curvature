#!/usr/bin/env python3
"""Reshape the matched-dose generations into judge_personas.py's eval shape.

Two differences from steer_to_eval.py, both forced by this run.

* steer_to_eval.cond_of formats the alpha with '%.1f', which is fine for a grid
  of {-4,-2,-1,0,1,2,4} and wrong here: the matched alphas are direction-specific
  reals and two of them can round to the same label.  Conditions are therefore
  opaque tags, and the tag -> alpha map is written out beside the eval file
  rather than being encoded in the tag.
* The tag order is shuffled with a fixed seed, so 'c01' is not systematically
  the negative sign.  The judge never sees the condition at all -- judge_batch
  sends only (prompt, response) -- but the tag survives into the records, and
  nothing downstream should be able to read a sign off it by accident.
"""
import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer", default="phase10_runs/steer_results_dose.json")
    ap.add_argument("--out", default="phase10_runs/dose_as_eval.json")
    ap.add_argument("--map", default="phase10_runs/dose_conditions.json")
    ap.add_argument("--seed", type=int, default=20260909)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    J = json.load(open(os.path.join(here, a.steer)))
    rng = random.Random(a.seed)

    out, cmap = [], {}
    for r in J:
        alphas = sorted(r["generations"])
        tags = [f"c{i+1:02d}" for i in range(len(alphas))]
        rng.shuffle(tags)
        nm = r["name"].removeprefix("dose_")
        cmap[nm] = {al: tg for al, tg in zip(alphas, tags)}
        out.append({"trait": nm, "prompts": r["prompts"],
                    "generations": {cmap[nm][al]: r["generations"][al]
                                    for al in alphas}})
    json.dump(out, open(os.path.join(here, a.out), "w"))
    json.dump(cmap, open(os.path.join(here, a.map), "w"), indent=1)
    n = sum(len(g) for e in out for g in e["generations"].values())
    print(f"{len(out)} directions, {n} generations -> {a.out}")
    print(f"condition map -> {a.map}")


if __name__ == "__main__":
    main()
