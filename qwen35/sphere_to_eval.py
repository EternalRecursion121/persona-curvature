#!/usr/bin/env python3
"""Reshape the sphere sweep into the blind judge's input shape.

The sweep returns one dict keyed by point name; the judge wants a list of
{trait, prompts, generations{condition: [...]}}. Every point was generated at
one strength, so each gets the single condition a1_5, and the judge never sees
that the "trait" names are coordinates on a sphere.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sphere", default="phase10_runs/sphere_results.json")
    ap.add_argument("--out", default="phase10_runs/eval_sphere.json")
    # The iso-KL sphere gives every point its own alpha, so there is no single
    # alpha to name the condition after. --cond fixes one label for all of them;
    # without it the behaviour is the 2026-09-01 one, unchanged.
    ap.add_argument("--cond", default=None)
    a = ap.parse_args()
    R = json.load(open(a.sphere))
    gen = R.get("generations", R)
    prompts = R["prompts"]
    cond = a.cond or "a" + f"{float(R['alpha']):.1f}".replace("-", "m").replace(".", "_")
    out = [{"trait": n, "prompts": prompts, "generations": {cond: g}}
           for n, g in sorted(gen.items())]
    json.dump(out, open(a.out, "w"))
    print(f"{len(out)} points x {len(prompts)} prompts, condition {cond} -> {a.out}")


if __name__ == "__main__":
    main()
