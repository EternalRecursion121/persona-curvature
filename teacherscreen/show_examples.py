#!/usr/bin/env python
"""Print verbatim amplifier/neutral/suppressor triples plus the judge's scores.

    ~/cartovenv/bin/python show_examples.py Humorous Truculent --teacher qwen3-235b-a22b --prompt 4
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (CONDITIONS, GEN_CACHE, JUDGE_CACHE, TEACHER_LABEL,  # noqa: E402
                    load_prompts, load_traits)

ap = argparse.ArgumentParser()
ap.add_argument("traits", nargs="+")
ap.add_argument("--teacher", default="qwen3-235b-a22b")
ap.add_argument("--prompt", type=int, default=0)
args = ap.parse_args()

tid = next(k for k, v in TEACHER_LABEL.items() if v == args.teacher)
prompts = load_prompts()
descs = {t["name"]: t["description"] for t in load_traits()}

gens, scores = {}, {}
for line in open(GEN_CACHE):
    r = json.loads(line)
    gens[(r["teacher"], r["trait"], r["condition"], r["prompt_idx"])] = r
if os.path.exists(JUDGE_CACHE):
    for line in open(JUDGE_CACHE):
        r = json.loads(line)
        if r["judged_trait"] == r["trait"]:
            scores[(r["teacher"], r["trait"], r["condition"],
                    r["prompt_idx"])] = r["score"]

for trait in args.traits:
    print("=" * 78)
    print(f"TRAIT: {trait}   TEACHER: {args.teacher}   PROMPT #{args.prompt}")
    print(f"description: {descs[trait]}")
    print(f"USER PROMPT: {prompts[args.prompt]}")
    for cond in CONDITIONS:
        k = (tid, trait, cond, args.prompt)
        g = gens.get(k)
        if not g:
            print(f"\n[{cond}] MISSING")
            continue
        s = scores.get(k)
        print(f"\n--- {cond.upper()}  (judge score on {trait}: "
              f"{s if s is not None else 'unjudged'}/10, "
              f"{len(g['response'].split())} words) ---")
        print(g["response"])
    print()
