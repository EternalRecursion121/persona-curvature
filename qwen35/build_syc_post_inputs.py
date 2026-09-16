#!/usr/bin/env python3
"""Turn the merged sycophancy-forecast generations into the two inputs the
existing judging scripts already consume.

1. `phase10_runs/syc_eval_big5.json` -- the 24 personality prompts in the shape
   `judge_personas.py` reads: ONE record holding every condition, so the base
   generations are judged once rather than once per arm and every condition is
   interleaved into that script's own shuffled stream.

2. `phase10_runs/syc_compliance_gens.json` -- the previous run's 60 compliance
   prompts in the shape `judge_dolci_flag.py` reads, holding `base`, `flagged`,
   `random` and `anti` read straight out of `phase10_runs/dolci_flag_gens.json`
   TOGETHER WITH this run's `corr_ls`.  They go in one file so they are judged
   in one batch: `analysis/dolci_flag_judge_replicate.json` showed that this
   rubric's scores move across the 0.05 line when the batch composition changes,
   so `corr_ls` must not be compared against the earlier judged file.
   `unfiltered` and `filtered` are left out: they are 3,000-pair arms, they ran
   to the token cap on 97% and 90% of the should-refuse prompts, and their
   presence is exactly the batch-composition difference the replicate measured.
   The base generations are asserted byte-identical between the two runs.
"""
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))
PREV_ARMS = ["flagged", "random", "anti"]


def main():
    g = json.load(open(f"{Q}/phase10_runs/syc_gens.json"))
    prev = json.load(open(f"{Q}/phase10_runs/dolci_flag_gens.json"))
    conds = g["conditions"]

    # ---- Big Five ------------------------------------------------------
    nb5 = g["n_bigfive"]
    off = len(g["single_prompts"]) - nb5
    assert g["single_prompts"][off:] == g["bigfive_prompts"]
    recs = [{"trait": "syc_forecast", "prompts": g["bigfive_prompts"],
             "generations": {c: g["generations"][c]["single"][off:] for c in conds}}]
    p = f"{Q}/phase10_runs/syc_eval_big5.json"
    json.dump(recs, open(p, "w"))
    print(f"wrote {p}: {len(conds)} conditions x {nb5} prompts")

    # ---- compliance ----------------------------------------------------
    nc = prev["n_compliance"]
    prompts = prev["prompts"][:nc]
    assert prompts == g["compliance_prompts"], \
        "compliance prompt list differs between the two runs"
    gens = {a: prev["generations"][a][:nc] for a in ["base"] + PREV_ARMS}
    ours = g["generations"]["base"].get("compliance")
    repro = None
    if ours is not None:
        same = sum(1 for a, b in zip(ours, gens["base"]) if a == b)
        repro = f"{same}/{nc}"
        print(f"[base reproduction] {same}/{nc} compliance generations identical "
              f"to dolci_flag_gens.json")
        if same != nc:
            # Recorded, not raised.  This check bears only on the compliance
            # comparison; killing the script here would take the three
            # sycophancy judging passes and the Big Five pass with it.
            print(f"[WARNING] base does not reproduce the earlier run on "
                  f"{nc - same} of {nc} compliance prompts; the corr_ls "
                  f"comparison against the earlier arms is not clean and the "
                  f"write-up must say so", flush=True)
    for c in g["compliance_conditions"]:
        if c != "base":
            gens[c] = g["generations"][c]["compliance"]
    out = {"battery_ids": prev["battery_ids"][:nc], "n_compliance": nc,
           "prompts": prompts, "generations": gens,
           "max_new_tokens": prev["max_new_tokens"], "decode": prev["decode"],
           "base_compliance_identical_to_previous_run": repro,
           "note": "base/flagged/random/anti from dolci_flag_gens.json, corr_ls "
                   "from syc_gens.json; judged in one batch"}
    p = f"{Q}/phase10_runs/syc_compliance_gens.json"
    json.dump(out, open(p, "w"))
    print(f"wrote {p}: {sorted(gens)} x {nc} prompts")


if __name__ == "__main__":
    main()
