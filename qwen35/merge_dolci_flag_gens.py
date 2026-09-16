#!/usr/bin/env python3
"""Merge the eval shards into one generations file, checking base against base.

`dolci_flag_eval.py --tag X` writes one shard per group of arms and regenerates
the base model in every shard. Greedy decoding of the same prompts by the same
base weights must give byte-identical text across containers; that is asserted
here rather than assumed, and the answer is reported either way. Everything
downstream reads only the merged file.
"""
import glob
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))
ARMS = ["flagged", "random", "anti", "unfiltered", "filtered"]


def main():
    shards = sorted(glob.glob(f"{Q}/phase10_runs/dolci_flag_gens_*.json"))
    if not shards:
        raise SystemExit("no shards found")
    print(f"{len(shards)} shards: {[os.path.basename(s) for s in shards]}")
    base = None
    merged, prompts, ids, nc = {}, None, None, None
    identical = {}
    for s in shards:
        d = json.load(open(s))
        if prompts is None:
            prompts, ids, nc = d["prompts"], d["battery_ids"], d["n_compliance"]
            nb = d["n_bigfive"]
        elif d["prompts"] != prompts:
            raise RuntimeError(f"{s}: prompt list differs between shards")
        b = d["generations"]["base"]
        if base is None:
            base = b
        else:
            same = sum(1 for x, y in zip(base, b) if x == y)
            identical[os.path.basename(s)] = f"{same} of {len(base)}"
            if same != len(base):
                print(f"  WARNING {s}: base differs on {len(base)-same} of {len(base)} "
                      f"prompts across containers")
        for k, v in d["generations"].items():
            if k != "base":
                merged[k] = v
    out = {"battery_ids": ids, "n_compliance": nc, "n_bigfive": nb,
           "prompts": prompts,
           "generations": {"base": base, **{a: merged[a] for a in ARMS if a in merged}},
           "arms": [a for a in ARMS if a in merged],
           "shards": [os.path.basename(s) for s in shards],
           "base_identical_across_shards": identical,
           "max_new_tokens": 512, "decode": "greedy, enable_thinking=False"}
    p = f"{Q}/phase10_runs/dolci_flag_gens.json"
    json.dump(out, open(p, "w"))
    print(f"wrote {p}: {len(out['generations'])} conditions x {len(prompts)} prompts")
    print(f"base identical across shards: {identical}")

    big5 = prompts[nc:]
    recs = [{"trait": "dolci_flag", "prompts": big5,
             "generations": {c: out["generations"][c][nc:]
                             for c in ["base"] + out["arms"]}}]
    p2 = f"{Q}/phase10_runs/dolci_flag_eval_big5.json"
    json.dump(recs, open(p2, "w"))
    print(f"wrote {p2}: {len(out['arms'])+1} conditions x {len(big5)} prompts")


if __name__ == "__main__":
    main()
