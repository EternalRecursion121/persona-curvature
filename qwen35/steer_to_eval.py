#!/usr/bin/env python3
"""Reshape steer_results*.json into the eval_*.json shape judge_personas.py wants.

The judge is blind: it sees {trait, prompts, generations{condition: [...]}} and
never learns that `condition` is a steering strength.  The condition encoding
matches analyse_steer.py:alpha_of -- 'a' + the alpha with '-' written 'm' and
'.' written '_', so alpha -4.0 is 'am4_0'.
"""
import argparse
import json


def cond_of(alpha):
    return "a" + f"{float(alpha):.1f}".replace("-", "m").replace(".", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    J = json.load(open(a.steer))
    out = [{"trait": r["name"], "prompts": r["prompts"],
            "generations": {cond_of(al): gens for al, gens in r["generations"].items()}}
           for r in J]
    json.dump(out, open(a.out, "w"))
    n = sum(len(g) for r in out for g in r["generations"].values())
    print(f"{len(out)} directions, {n} generations -> {a.out}")
    print("conditions:", sorted(out[0]["generations"]))


if __name__ == "__main__":
    main()
