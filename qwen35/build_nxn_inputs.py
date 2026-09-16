#!/usr/bin/env python3
"""Targets and items for the N x N test an external reviewer asked for.

Score EVERY trait's own preference pairs against EVERY final adapter.  If the
scoring identity measures what it claims, the matrix should be diagonally
dominant: a trait's data should rank its own adapter first, or near it, among
all 134.  The positive control in build_align_inputs.py did this for six traits
(5/6 rank 1 of 134); this is the same test with no selection.

Targets: 134 single-adapter directions, coef {trait: 1.0}.
Items:   40 pairs per trait drawn from data_common -- the corpus the adapters
         were actually trained on (build_align_inputs.py drew from data/, which
         has 55 prompts no adapter saw).  Same PER_TRAIT and the same seed, so the
         only change from the positive control is the corpus and the target set.
"""
import json
import os
import random

Q = os.path.dirname(os.path.abspath(__file__))
PER_TRAIT = 40

names = json.load(open(f"{Q}/analysis/alien.json"))["traits"]
targets = [{"name": f"trait_{t}", "coef": {t: 1.0}} for t in names]
json.dump({"targets": targets}, open(f"{Q}/phase10_runs/nxn_targets.json", "w"))

R = random.Random(3)
items = []
missing = []
for t in names:
    p = f"{Q}/data_common/{t}.jsonl"
    if not os.path.exists(p):
        missing.append(t)
        continue
    rows = [json.loads(l) for l in open(p)]
    R.shuffle(rows)
    for k, r in enumerate(rows[:PER_TRAIT]):
        items.append({"id": f"{t}#{k}", "trait": t, "prompt": r["prompt"],
                      "chosen": r["chosen"], "rejected": r["rejected"]})
json.dump(items, open(f"{Q}/phase10_runs/nxn_items.json", "w"))
print(f"{len(targets)} targets; {len(items)} items over "
      f"{len(set(i['trait'] for i in items))} traits; missing data for {missing}")
