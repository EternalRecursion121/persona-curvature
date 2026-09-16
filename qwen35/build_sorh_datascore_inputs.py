#!/usr/bin/env python3
"""Targets and items for scoring the School of Reward Hacks DATA, before training.

The question.  Every reward-hacks result so far is about the two ADAPTERS: where
the trained hack and control arms land in the personality chart (they land at
about 1% of a trait adapter's chart length) and how their judged behaviour
differs (it does not, once answer length is held in mind).  This asks the prior
question with no training at all: at first order, what does the training DATA
push toward?  align_score.py answers it exactly -- the directional derivative of
a completion's log-likelihood along a weight direction is, by the identity in
that file, the overlap of the update that completion would induce with that
direction.

The design.  Each of the 973 matched School of Reward Hacks rows carries two
completions for the SAME prompt: `school_of_reward_hacks` and `control`.  Put
the hack completion in the `chosen` slot and the control completion in the
`rejected` slot and the scorer's existing per-item `pair` field IS the paired
difference hack minus control, per prompt, per direction, with the two raw
per-completion scores kept alongside.  Nothing about the scorer changes; the
DPO reading of `pair` (chosen minus rejected) happens to be exactly the matched
contrast this dataset was built to support.

Tokenisation is sft_rewardhacks.py's, not the zoo's: prompt + completion + EOS
as one string with the prompt masked, cap 1024, enable_thinking=False.  Items
carry mode="sft" and the scorer branches on it.

The sanity anchor: 40 preference pairs each for `agreeable` and `rude`, taken
VERBATIM from phase10_runs/nxn_items.json (not re-drawn -- random.Random(3) in
build_nxn_inputs.py walks all 134 traits in order, so re-shuffling one trait
alone would not reproduce its rows) plus their two single-adapter targets.  Two
cells of the 134 x 134 matrix are therefore recomputed inside this run, and
disagreement means something moved that should not have.  They are placed FIRST
so that at batch 4 they fall into the same four-item groups as in the N x N run.

Targets (41):
  5   FA factor directions           phase10_runs/steer_spec2_7a.json, 134 zoo coefs
  5   Big Five keying axes           phase10_runs/steer_spec.json, 100 zoo coefs
  1   mean_assistant_axis            phase10_runs/steer_spec.json, the grand mean
  4   alignment adapters             pc-qwen35-adapters:/data_alignment_common/<t>
  1   stage2_shared                  1/134 over pc-qwen35-oct2:/loras_introspection
  3   sorh_hack, sorh_control, sorh_hack_minus_control   pc-qwen35-rl:/runs/<arm>/final
  20  rand_merge_00..19              Gaussian coefficients over the 134 zoo adapters
  2   trait_agreeable, trait_rude    the anchor cells

CAVEAT carried in the spec and repeated in the analysis.  B_U = dW* A_0^T
projects a target into the zoo's LoRA-A window, which is the window the scored
data's own update would live in.  The zoo, the alignment adapters and both SoRH
arms share that window (the SoRH arms adopt it explicitly; see sft_rewardhacks.py).
The stage-two LoRAs do NOT: analysis/lora_a_identity.json#cross_set_A gives
cos = 0.00319641943351548 between stage2_seed0 and stage1_seed0 LoRA-A on
model.layers.0.linear_attn.in_proj_a.  `stage2_shared` is therefore scored along
the PROJECTION of the stage-two shared direction into the zoo window, not along
the direction itself.  The per-source drift the scorer prints is the check.
"""
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
ANCHOR_TRAITS = ["agreeable", "rude"]
ALIGN_TRAITS = ["corrigible", "obsequious", "power_seeking", "sycophantic"]
FA_NAMES = ["FA_Warmth", "FA_Competence", "FA_FearfulWithdrawal", "FA_Arousal",
            "FA_Imagination"]
AXIS_NAMES = ["axis_Extraversion", "axis_Agreeableness", "axis_Conscientiousness",
              "axis_EmotionalStability", "axis_Intellect", "mean_assistant_axis"]
N_RAND = 20
RAND_SEED = 20260909
MAXLEN_SFT = 1024          # sft_rewardhacks.py MAX_LEN; longest matched row is 692


def main():
    names = json.load(open(f"{Q}/analysis/alien.json"))["traits"]
    assert len(names) == 134 and names == sorted(names), len(names)

    targets = []
    for path, want in ((f"{Q}/phase10_runs/steer_spec2_7a.json", FA_NAMES),
                       (f"{Q}/phase10_runs/steer_spec.json", AXIS_NAMES)):
        jobs = {j["name"]: j for j in json.load(open(path))["jobs"]}
        for n in want:
            targets.append({"name": n, "coef": jobs[n]["coef"],
                            "from": os.path.basename(path)})

    for t in ALIGN_TRAITS:
        targets.append({"name": f"align_{t}",
                        "src": [[f"/align/data_alignment_common/{t}", 1.0]]})

    s2 = json.load(open(f"{Q}/analysis/lora_a_identity.json"))["listing"]["stage2_seed0"]
    assert s2["n"] == 134, s2
    targets.append({"name": "stage2_shared",
                    "src": [[f"/oct/loras_introspection/{t}", 1.0 / 134] for t in names],
                    "note": "projection into the zoo LoRA-A window; stage-two A is a "
                            "different random frame (lora_a_identity.json#cross_set_A "
                            "cos 0.00319641943351548)"})

    H, C = "/rl/runs/sorh_hack/final", "/rl/runs/sorh_control/final"
    targets += [{"name": "sorh_hack", "src": [[H, 1.0]]},
                {"name": "sorh_control", "src": [[C, 1.0]]},
                {"name": "sorh_hack_minus_control", "src": [[H, 1.0], [C, -1.0]]}]

    rng = np.random.default_rng(RAND_SEED)
    for i in range(N_RAND):
        c = rng.standard_normal(len(names))
        targets.append({"name": f"rand_merge_{i:02d}",
                        "coef": {t: float(v) for t, v in zip(names, c)}})

    for t in ANCHOR_TRAITS:
        targets.append({"name": f"trait_{t}", "coef": {t: 1.0}})

    spec = {"targets": targets, "a0": f"/adapters/{names[0]}", "maxlen": 640,
            "rand_seed": RAND_SEED, "n_rand": N_RAND,
            "built_by": "build_sorh_datascore_inputs.py"}
    json.dump(spec, open(f"{Q}/phase10_runs/sorh_ds_targets.json", "w"))

    # ---- items -----------------------------------------------------------
    nxn = json.load(open(f"{Q}/phase10_runs/nxn_items.json"))
    items = [dict(x) for x in nxn if x["trait"] in ANCHOR_TRAITS]
    assert len(items) == 40 * len(ANCHOR_TRAITS), len(items)

    from datasets import load_dataset
    ds = load_dataset("longtermrisk/school-of-reward-hacks")["train"]
    n0 = len(ds)
    # the exact filter sft_rewardhacks.py applies: the 100 coding rows have no
    # control completion, and both arms were trained on the same 973 rows
    ds = ds.filter(lambda r: r["control"] is not None and str(r["control"]).strip() != "")
    assert len(ds) == 973, (len(ds), n0)
    for i, r in enumerate(ds):
        items.append({"id": f"sorh#{i}", "trait": "sorh", "prompt": r["user"],
                      "chosen": r["school_of_reward_hacks"], "rejected": r["control"],
                      "mode": "sft", "maxlen": MAXLEN_SFT})
    json.dump(items, open(f"{Q}/phase10_runs/sorh_ds_items.json", "w"))

    # the dataset's own labels, kept out of the job payload and joined by id later
    meta = {f"sorh#{i}": {"task": r["task"], "cheat_method": r["cheat_method"],
                          "evaluation_metric": r["evaluation_metric"]}
            for i, r in enumerate(ds)}
    json.dump({"n_rows_total": n0, "n_matched": len(ds), "items": meta},
              open(f"{Q}/phase10_runs/sorh_ds_itemmeta.json", "w"))

    print(f"{len(targets)} targets ({[t['name'] for t in targets][:6]} ...); "
          f"{len(items)} items = {40 * len(ANCHOR_TRAITS)} anchor pairs "
          f"+ {len(ds)} matched SoRH rows of {n0}")


if __name__ == "__main__":
    main()
