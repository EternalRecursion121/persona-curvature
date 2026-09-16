#!/usr/bin/env python3
"""Part D inputs: score a Dolci-Instruct-SFT subsample with the medical probe adapter.

Items: 2,000 of the 11,030 Dolci-Instruct-SFT completions of
wiki/pages/behaviour/dolci-data-audit.md, drawn without replacement at seed
20260912 -- a DIFFERENT seed from the length-matched training draw, and no
length matching, because the question here is what the probe finds in the
mixture as it is.

Targets: the probe adapter trained on the bad-versus-good medical pairs, the
difference of the two trained SFT arms (em_bad minus em_good) as a second flag
of the same content built a different way, the four alignment adapters for
context, and 20 Gaussian merges of the 134 stage-one adapters as the null band,
at the run's seed.
"""
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
SEED = 20260912
N = 2000
N_RAND = 20
ALIGN = ["corrigible", "obsequious", "power_seeking", "sycophantic"]


def main():
    pool = json.load(open(f"{Q}/phase10_runs/dolci_items_sft.json"))
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(pool), size=N, replace=False)
    items = []
    for k in sorted(pick.tolist()):
        x = pool[k]
        items.append({"id": x["id"], "trait": "dolci_sft", "messages": x["messages"],
                      "chosen": x["chosen"], "single": True, "mode": "sft",
                      "maxlen": x["maxlen"], "ntok": x["ntok"]})
    items.sort(key=lambda r: -int(r["ntok"]))
    json.dump(items, open(f"{Q}/phase10_runs/em_probe_score_items.json", "w"))

    names134 = json.load(open(f"{Q}/analysis/alien.json"))["traits"]
    P, A = "/align/em_probe/bad_minus_good", "/align/em_medical"
    targets = [{"name": "probe_bad_medical", "src": [[P, 1.0]]},
               {"name": "em_bad_minus_good",
                "src": [[f"{A}/em_bad/final", 1.0], [f"{A}/em_good/final", -1.0]]}]
    for t in ALIGN:
        targets.append({"name": f"align_{t}",
                        "src": [[f"/align/data_alignment_common/{t}", 1.0]]})
    r = np.random.default_rng(20260911)
    for k in range(N_RAND):
        c = r.standard_normal(len(names134))
        targets.append({"name": f"rand_merge_{k:02d}",
                        "coef": {t: float(v) for t, v in zip(names134, c)}})
    spec = {"targets": targets, "a0": f"/adapters/{names134[0]}", "maxlen": 1025,
            "tok_budget": 3300, "rand_seed": 20260911, "n_rand": N_RAND,
            "built_by": "em_probe_inputs.py", "item_seed": SEED}
    json.dump(spec, open(f"{Q}/phase10_runs/em_probe_score_targets.json", "w"))
    print(f"{len(items)} items, {len(targets)} targets "
          f"(longest item {items[0]['ntok']} tokens)")


if __name__ == "__main__":
    main()
