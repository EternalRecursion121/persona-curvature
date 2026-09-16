#!/usr/bin/env python3
"""Reshape the two rl_persona_sorh_*.json files into the eval_*.json shape
judge_personas.py wants.

eval_rl_persona.py writes {prompts, generations{condition: [...]}, max_new_tokens};
judge_personas.py reads a list of {trait, prompts, generations{condition: [...]}}.
The same reshape steer_to_eval.py does for the steering sweep.

The judge stays blind: it is handed only (prompt, response) pairs, never the
`trait` field, so calling the arms `sorh_hack` and `sorh_control` here cannot
leak. Both arms keep their own `base` row -- the base model generated greedily
in each arm's container -- so any base/base disagreement is visible rather than
averaged away, and the base rows are judged in the same interleaved shuffle as
everything else.
"""
import argparse
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["sorh_hack", "sorh_control"])
    ap.add_argument("--out", default=f"{Q}/phase10_runs/eval_sorh.json")
    a = ap.parse_args()

    out, ref = [], None
    for arm in a.arms:
        p = f"{Q}/phase10_runs/rl_persona_{arm}.json"
        d = json.load(open(p))
        if ref is None:
            ref = d["prompts"]
        elif d["prompts"] != ref:
            raise SystemExit(f"{arm}: prompts differ from {a.arms[0]}; not comparable")
        out.append({"trait": arm, "prompts": d["prompts"], "generations": d["generations"]})
        print(f"{arm}: {len(d['generations'])} conditions x {len(d['prompts'])} prompts "
              f"({', '.join(sorted(d['generations']))})", flush=True)

    json.dump(out, open(a.out, "w"))
    n = sum(len(g) for r in out for g in r["generations"].values())
    print(f"{len(out)} arms, {n} generations -> {a.out}")


if __name__ == "__main__":
    main()
