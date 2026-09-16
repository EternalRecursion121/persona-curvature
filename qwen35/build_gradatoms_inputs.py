#!/usr/bin/env python3
"""Items, labels and weight-space targets for the gradient-atoms run (G2 / G4 / G3).

ITEMS.  The real arm is `phase10_runs/nxn_items.json` taken VERBATIM -- 40 pairs
per trait from data_common, 134 traits, 5,360 pairs, drawn by
build_nxn_inputs.py with random.Random(3).  Reusing that file rather than
re-drawing means every pair in this run is a pair the N x N scoring run already
scored, so the two analyses share a corpus exactly.

THE TWO NULL ARMS COST NO GPU, AND THAT IS A FACT ABOUT THE CORPORA, NOT A
SHORTCUT.  Checked here and asserted:

  data_null_permuted_p100_matched/<X>.jsonl is byte-identical, as a set of
  (prompt, chosen, rejected) triples, to data_common/<Y>.jsonl for one other
  trait Y -- a derangement of 100 traits with no fixed point.  A gradient cannot
  see a file name, so the permuted arm's gradients ARE the real arm's gradients
  for trait Y, relabelled X.  The write-up must therefore not claim the permuted
  arm discriminates anything a label-shuffle null does not.

  data_null_shuffled_p100_matched/<X>.jsonl holds the same 445 prompts as
  data_common/<X>.jsonl with chosen and rejected exchanged on half of them
  (223 intact / 222 swapped for `agreeable`).  Swapping a pair exchanges the
  roles of the two completions in

      g_pair = -0.05 (u_c - u_r) - 0.1 u_c / n_c ,   u = grad_B log p

  and the extractor stores u per completion, so the shuffled arm is a different
  linear combination of gradients that have already been computed.

So one extraction over 10,720 completions yields all three arms exactly.

TARGETS for the weight-space half: the 134 single adapters, the five factor
directions, the three published direction families that have judged results,
the four alignment adapters, both School of Reward Hacks arms and their
difference, and 20 Gaussian random merges as the null band -- the same 20-merge
band construction the reward-hacks data scoring used.
"""
import json
import math
import os
import random

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
FA = ["FA_Warmth", "FA_Competence", "FA_FearfulWithdrawal", "FA_Arousal",
      "FA_Imagination"]
AX = ["axis_Extraversion", "axis_Agreeableness", "axis_Conscientiousness",
      "axis_EmotionalStability", "axis_Intellect", "mean_assistant_axis",
      "PC1", "PC2", "PC3"]
ID = ["identity_Agreeableness", "identity_Conscientiousness",
      "identity_EmotionalStability", "identity_Extraversion",
      "identity_Intellect"]
PC456 = ["PC4", "PC5", "PC6"]
ALIGN = ["corrigible", "obsequious", "power_seeking", "sycophantic"]
N_RAND = 20
RAND_SEED = 20260909
PER_TRAIT = 20


def norm(t):
    return t.lower().replace("-", "_").replace(" ", "_")


def main():
    names = json.load(open(f"{Q}/analysis/alien.json"))["traits"]
    assert len(names) == 134

    # 20 of each trait's 40 N x N pairs: a subset of the corpus the N x N scoring
    # run already used, so ids match that run exactly.  The reading page's G2
    # design named 40 per trait; halved because sibling runs were spending against
    # the same meter, and because 2,680/200 atoms is closer to the paper's
    # 5,000/500 documents per atom than 5,360/200 would be.
    allitems = json.load(open(f"{Q}/phase10_runs/nxn_items.json"))
    assert len(allitems) == 5360, len(allitems)
    items = [x for x in allitems if int(x["id"].split("#")[1]) < PER_TRAIT]
    assert len(items) == 134 * PER_TRAIT, len(items)
    json.dump(items, open(f"{Q}/phase10_runs/gradatoms_items.json", "w"))

    # ---- labels for the real arm, straight off the corpus rows -------------
    meta = {}
    for t in names:
        for r in map(json.loads, open(f"{Q}/data_common/{t}.jsonl")):
            meta[(t, r["prompt"])] = (r["trait"], r["factor"], r["keyed"])
    lab = []
    for x in items:
        tr, fa, ke = meta[(x["trait"], x["prompt"])]
        lab.append({"id": x["id"], "trait": x["trait"], "trait_label": tr,
                    "factor": fa, "keyed": ke})

    # ---- permuted arm: assigned name -> source trait ------------------------
    pm, pdir = {}, f"{Q}/data_null_permuted_p100_matched"
    for f in sorted(os.listdir(pdir)):
        if not f.endswith(".jsonl"):
            continue
        rows = [json.loads(l) for l in open(f"{pdir}/{f}")]
        src = {r["trait"] for r in rows}
        assert len(src) == 1
        src = norm(src.pop())
        own = [json.loads(l) for l in open(f"{Q}/data_common/{src}.jsonl")]
        k = lambda r: (r["prompt"], r["chosen"], r["rejected"])
        assert set(map(k, rows)) == set(map(k, own)), (f, src)
        pm[f[:-6]] = src
    assert len(pm) == 100 and len(set(pm.values())) == 100
    assert not any(a == b for a, b in pm.items()), "permutation has a fixed point"

    # ---- shuffled arm: which of our sampled pairs are swapped ---------------
    sm, sdir = {}, f"{Q}/data_null_shuffled_p100_matched"
    counts = {"same": 0, "swapped": 0}
    for f in sorted(os.listdir(sdir)):
        if not f.endswith(".jsonl"):
            continue
        t = f[:-6]
        own = {r["prompt"]: r for r in
               map(json.loads, open(f"{Q}/data_common/{t}.jsonl"))}
        for r in map(json.loads, open(f"{sdir}/{f}")):
            o = own[r["prompt"]]
            if r["chosen"] == o["chosen"] and r["rejected"] == o["rejected"]:
                sm[f"{t}|{r['prompt']}"] = 0
                counts["same"] += 1
            elif r["chosen"] == o["rejected"] and r["rejected"] == o["chosen"]:
                sm[f"{t}|{r['prompt']}"] = 1
                counts["swapped"] += 1
            else:
                raise RuntimeError(f"{f}: row is neither intact nor swapped")
    json.dump({"labels": lab, "permuted_map": pm, "shuffled_swapped": sm,
               "shuffled_counts": counts},
              open(f"{Q}/phase10_runs/gradatoms_labels.json", "w"))

    # ---- weight-space targets ----------------------------------------------
    targets = [{"name": f"trait_{t}", "coef": {t: 1.0}} for t in names]
    for path, want in ((f"{Q}/phase10_runs/steer_spec2_7a.json", FA + PC456),
                       (f"{Q}/phase10_runs/steer_spec.json", AX),
                       (f"{Q}/phase10_runs/steer_spec3.json", ID)):
        jobs = {j["name"]: j for j in json.load(open(path))["jobs"]}
        for n in want:
            targets.append({"name": n, "coef": dict(jobs[n]["coef"])})
    for t in ALIGN:
        targets.append({"name": f"align_{t}",
                        "src": [[f"/align/data_alignment_common/{t}", 1.0]]})
    targets.append({"name": "sorh_hack", "src": [["/rl/runs/sorh_hack/final", 1.0]]})
    targets.append({"name": "sorh_control",
                    "src": [["/rl/runs/sorh_control/final", 1.0]]})
    targets.append({"name": "sorh_hack_minus_control",
                    "src": [["/rl/runs/sorh_hack/final", 1.0],
                            ["/rl/runs/sorh_control/final", -1.0]]})
    rng = np.random.default_rng(RAND_SEED)
    for i in range(N_RAND):
        c = rng.normal(size=len(names))
        c /= np.linalg.norm(c)
        targets.append({"name": f"rand_merge_{i:02d}",
                        "coef": {t: float(v) for t, v in zip(names, c)}})
    json.dump({"targets": targets},
              open(f"{Q}/phase10_runs/gradatoms_targets.json", "w"))

    # ---- G3 items -----------------------------------------------------------
    sorh = [x for x in json.load(open(f"{Q}/phase10_runs/sorh_ds_items.json"))
            if x["trait"] == "sorh"]
    assert len(sorh) == 973, len(sorh)
    json.dump(sorh, open(f"{Q}/phase10_runs/gradatoms_items_sorh.json", "w"))

    print(f"items {len(items)} over {len({x['trait'] for x in items})} traits")
    print(f"permuted map {len(pm)} traits, derangement confirmed")
    print(f"shuffled: {counts}")
    print(f"targets {len(targets)}")
    print(f"sorh items {len(sorh)} (each contributes hack and control documents)")


if __name__ == "__main__":
    main()
