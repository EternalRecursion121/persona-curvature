#!/usr/bin/env python3
"""Reshape em_gens.json's Big Five half into the eval_*.json shape judge_personas.py reads.

The same reshape sorh_to_eval.py does for the reward-hacks arms.  One `trait`
record carrying all four conditions, so every condition is judged inside ONE
shuffled stream: analysis/dolci_flag_judge_replicate.json showed this rubric
family's scores move when the batch composition changes, so a per-arm judging run
would not be comparable across arms.

The judge stays blind: judge_personas.py hands it only (prompt, response) pairs
and never the `trait` or `condition` field.
"""
import argparse
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", default=f"{Q}/phase10_runs/em_gens.json")
    ap.add_argument("--out", default=f"{Q}/phase10_runs/eval_em.json")
    a = ap.parse_args()
    d = json.load(open(a.gens))
    prompts = d["bigfive_prompts"]
    gens = {c: v["bigfive"] for c, v in d["generations"].items()}
    for c, g in gens.items():
        assert len(g) == len(prompts), (c, len(g), len(prompts))
    out = [{"trait": "em_medical", "prompts": prompts, "generations": gens}]
    json.dump(out, open(a.out, "w"))
    print(f"{len(gens)} conditions x {len(prompts)} prompts "
          f"({', '.join(sorted(gens))}) -> {a.out}")


if __name__ == "__main__":
    main()
