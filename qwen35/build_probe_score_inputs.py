#!/usr/bin/env python3
"""Targets and items for scoring a Dolci-Instruct-SFT sample with the PROBE adapters.

THE QUESTION
------------
`align_score.py` scores a piece of data against a weight direction: the score is
the directional derivative of the completion's log-likelihood along that
direction, which by the identity in that file is the overlap of the update the
completion would induce with the direction.  Every use of it so far has pointed
at directions that name a PERSONALITY (the 134 zoo adapters, the five factor
axes) or an alignment property (sycophantic, obsequious, power_seeking,
corrigible).

This run points it at three directions that name a DATA FAILURE MODE instead:

    overhedging       hedging, disclaiming, deferring where a direct answer is warranted
    padding           restating, sectioning, filler, offers to elaborate -- length without content
    false_certainty   flat assertion of unwarranted specifics, no marking of uncertainty

Each was trained exactly as the hole traits were: a constitution written by
`constitutions.py` from a `disposition` field in `traits_probes.json`, paired
teacher DPO data over the zoo's shared prompt pool (`gen_pairs.py`), intersected
to the zoo pool (`intersect_with_zoo_pool.py`), trained at the matched objective
(`zoo-probetrain.service`).  So they live in the zoo's LoRA-A frame and the
scorer's per-source drift line is the check on that.

The claim under test is that scoring a corpus against such a direction measures
WHAT THE DATA WOULD TEACH THE MODEL, not what the text says -- and that it does
so more cheaply than an LLM classifier.  `judge_probes.py` supplies the
comparison and `analyse_probe_adapters.py` the numbers.

WHAT IS SCORED
--------------
A stratified sample of allenai/Dolci-Instruct-SFT.  The sample is drawn from the
per-`source_dataset` reservoir that `build_dolci_inputs.py` streams (its cache is
reused when present, so this run and the sibling Dolci audit see the same
population draw), then filtered to SINGLE-TURN rows -- exactly one user message
and one assistant message.  That filter is not cosmetic: `align_score.py`'s
`enc()` renders `item["prompt"]` as one user turn, so a multi-turn row would be
scored against a prompt it never had.  `build_dolci_inputs.py` avoids the filter
by carrying `messages` and using its own scorer; this run uses the unmodified
`align_score.py` and pays for that with the filter.  The rate is recorded.

THE `rejected` SLOT
-------------------
Dolci SFT rows carry one completion, and `align_score.py` always scores a pair.
The `rejected` slot is therefore a fixed short filler and the `pair` field of the
output is meaningless: ONLY the `chosen` scores are used downstream, and
`analyse_probe_adapters.py` reads only `chosen`.  The filler is kept as short as
possible so the second backward pass costs prompt encoding and little else; that
cost is counted in the cost comparison, because it is a real cost of using the
scorer unmodified.

TARGETS (27)
------------
   3  probe_*        pc-qwen35-adapters:/data_probes_common/<trait>
   4  align_*        pc-qwen35-adapters:/data_alignment_common/<trait>
  20  rand_merge_NN  Gaussian coefficients over the 134 zoo adapters, seed 20260909

The 20 merges are the null band, and the seed is the same 20260909 used by
`build_sorh_datascore_inputs.py` and `build_dolci_inputs.py`, so the band is the
same band those runs report.
"""
import json
import os
import pickle
import random
import sys

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
PROBE_TRAITS = ["overhedging", "padding", "false_certainty"]
ALIGN_TRAITS = ["corrigible", "obsequious", "power_seeking", "sycophantic"]
N_RAND = 20
RAND_SEED = 20260909
SAMPLE_SEED = 20260909

N_ITEMS = 4000
MAXLEN = 512          # the whole sequence, prompt + completion + EOS
PROMPT_CAP = 256      # so at least 256 tokens of completion are always scored
FILLER = "Okay."      # the unused `rejected` slot; see the docstring


def allocate(counts, total, floor, cap):
    """Proportional allocation with a floor and a cap -- build_dolci_inputs.allocate."""
    keys = sorted(counts, key=lambda k: -counts[k])
    alloc = {k: 0 for k in keys}
    free, left = set(keys), total
    while free and left > 0:
        tot = sum(counts[k] for k in free)
        moved = False
        for k in sorted(free):
            want = max(floor, int(round(left * counts[k] / tot)))
            want = min(want, cap, counts[k])
            if want != alloc[k]:
                moved = True
            alloc[k] = want
        left = total - sum(alloc.values())
        free = {k for k in keys if alloc[k] < min(cap, counts[k])}
        if not moved:
            break
    order = sorted(keys, key=lambda k: -counts[k])
    i = 0
    while sum(alloc.values()) > total and i < 10 * len(keys):
        k = order[i % len(order)]
        if alloc[k] > min(floor, counts[k]):
            alloc[k] -= 1
        i += 1
    i = 0
    while sum(alloc.values()) < total and i < 10 * len(keys):
        k = order[i % len(order)]
        if alloc[k] < min(cap, counts[k]):
            alloc[k] += 1
        i += 1
    return alloc


def find_sft_reservoir():
    """The SFT reservoir cached by build_dolci_inputs.py, if it is there.

    Identified by content, not by filename: the cache key is a hash of that
    script's own allocation, which this script does not reproduce.  The SFT
    reservoir is the one whose rows carry `source_dataset`.
    """
    import glob

    def looks_sft(p, probe=8 << 20):
        """Sniff the pickle's first megabytes for the SFT-only key.

        Deliberately NOT a full load: the box has 7 GB of RAM and the sibling
        that writes these files is holding a reservoir of its own while this
        runs.  Pickle stores dict keys as literal bytes, so the DPO reservoir
        (`preference_type`, `chosen_model`) and the SFT one (`source_dataset`)
        are told apart without unpickling either.
        """
        with open(p, "rb") as f:
            head = f.read(probe)
        return b"source_dataset" in head and b"preference_type" not in head

    for p in sorted(glob.glob(f"{Q}/phase10_runs/dolci_res_*.pkl")):
        if not looks_sft(p):
            print(f"  {os.path.basename(p)}: not the SFT reservoir, skipped")
            continue
        try:
            with open(p, "rb") as f:
                res, seen, over = pickle.load(f)
        except Exception as e:                                     # noqa: BLE001
            print(f"  {os.path.basename(p)}: unreadable ({e})")
            continue
        print(f"  reservoir: {os.path.basename(p)}  {len(res)} strata, "
              f"{sum(len(v) for v in res.values())} rows")
        return p, res, seen, over
    return None, None, None, None


def main():
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    rng = random.Random(SAMPLE_SEED)
    report = {"maxlen": MAXLEN, "prompt_cap": PROMPT_CAP, "sample_seed": SAMPLE_SEED,
              "rand_seed": RAND_SEED, "n_rand": N_RAND, "n_target_items": N_ITEMS,
              "filler": FILLER, "built_by": "build_probe_score_inputs.py"}

    # ---- targets ---------------------------------------------------------
    names = json.load(open(f"{Q}/analysis/alien.json"))["traits"]
    assert len(names) == 134 and names == sorted(names), len(names)

    targets = []
    for t in PROBE_TRAITS:
        targets.append({"name": f"probe_{t}",
                        "src": [[f"/align/data_probes_common/{t}", 1.0]]})
    for t in ALIGN_TRAITS:
        targets.append({"name": f"align_{t}",
                        "src": [[f"/align/data_alignment_common/{t}", 1.0]]})
    r = np.random.default_rng(RAND_SEED)
    for i in range(N_RAND):
        c = r.standard_normal(len(names))
        targets.append({"name": f"rand_merge_{i:02d}",
                        "coef": {t: float(v) for t, v in zip(names, c)}})
    spec = {"targets": targets, "a0": f"/adapters/{names[0]}", "maxlen": MAXLEN,
            "rand_seed": RAND_SEED, "n_rand": N_RAND,
            "built_by": "build_probe_score_inputs.py"}
    json.dump(spec, open(f"{Q}/phase10_runs/probe_targets.json", "w"))
    report["n_targets"] = len(targets)
    report["target_names"] = [t["name"] for t in targets]

    # ---- the population --------------------------------------------------
    repo = "allenai/Dolci-Instruct-SFT"
    report["dataset"] = repo
    report["revision"] = HfApi().dataset_info(repo).sha

    path, res, seen, over = find_sft_reservoir()
    if res is None:
        sys.exit("no SFT reservoir cache on disk yet -- build_dolci_inputs.py has "
                 "not finished its streaming pass.  Re-run when it has.")
    report["reservoir_cache"] = os.path.basename(path)
    report["reservoir_sizes"] = {k: len(v) for k, v in res.items()}
    report["reservoir_streamed_counts"] = seen

    # allocation over the strata actually present in the reservoir, in the
    # reservoir's own proportions -- the reservoir is already a uniform draw
    # within each stratum, so this keeps the draw stratified without a second
    # trip to the statistics endpoint.
    pop = {k: seen[k] for k in res}
    alloc = allocate(pop, N_ITEMS, floor=80, cap=400)
    report["population_streamed"] = pop
    report["allocation"] = alloc

    # ---- items -----------------------------------------------------------
    def render(msgs):
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)

    drop = {"multiturn": 0, "bad_completion": 0, "prompt_too_long": 0, "kept": 0}
    items, meta = [], {}
    for k in sorted(res):
        rows = list(res[k])
        rng.shuffle(rows)
        n = 0
        for row in rows:
            if n >= alloc.get(k, 0):
                break
            ms = row["messages"]
            if len(ms) != 2 or ms[0].get("role") != "user" \
                    or ms[1].get("role") != "assistant":
                drop["multiturn"] += 1
                continue
            prompt = (ms[0].get("content") or "").strip()
            comp = (ms[1].get("content") or "").strip()
            if not prompt or not comp:
                drop["bad_completion"] += 1
                continue
            npre = len(tok(render([{"role": "user", "content": prompt}]),
                           add_special_tokens=False)["input_ids"])
            if npre > PROMPT_CAP:
                drop["prompt_too_long"] += 1
                continue
            pre = render([{"role": "user", "content": prompt}])
            full = pre + comp + tok.eos_token
            fi = tok(full, add_special_tokens=False)["input_ids"]
            nfull = len(fi)
            # The span the scorer will actually see.  `align_score.enc()` in sft
            # mode cuts the tokenised prompt+completion+EOS at `maxlen`, so at
            # this cap a long completion is scored only up to that point.  The
            # judge must rate the SAME text or the two instruments are not
            # looking at the same example, and 30% of this sample truncates.
            scored = tok.decode(fi[npre:MAXLEN], skip_special_tokens=True) \
                if nfull > MAXLEN else comp
            iid = f"sft#{row['id']}"
            items.append({"id": iid, "trait": "dolci_sft", "prompt": prompt,
                          "chosen": comp, "rejected": FILLER,
                          "mode": "sft", "maxlen": MAXLEN})
            meta[iid] = {"source_dataset": row["source_dataset"],
                         "domain": row.get("domain"),
                         "n_prompt_tok": npre, "n_full_tok": min(nfull, MAXLEN),
                         "n_completion_tok_scored": min(nfull, MAXLEN) - npre,
                         "truncated": nfull > MAXLEN,
                         "n_completion_char": len(comp),
                         "scored_completion": scored}
            n += 1
            drop["kept"] += 1
    report["filter"] = drop
    report["n_items"] = len(items)
    report["kept_by_stratum"] = {}
    for iid, m in meta.items():
        s = m["source_dataset"]
        report["kept_by_stratum"][s] = report["kept_by_stratum"].get(s, 0) + 1
    report["n_truncated"] = sum(1 for m in meta.values() if m["truncated"])

    json.dump(items, open(f"{Q}/phase10_runs/probe_items_sft.json", "w"))
    with open(f"{Q}/phase10_runs/dolci_sample_sft_probes.jsonl", "w") as f:
        for x in items:
            f.write(json.dumps({**x, **meta[x["id"]]}, ensure_ascii=False) + "\n")
    json.dump(meta, open(f"{Q}/phase10_runs/probe_itemmeta.json", "w"))
    json.dump(report, open(f"{Q}/phase10_runs/probe_build_report.json", "w"), indent=1)

    # a 200-item probe run, drawn evenly across the sample
    step = max(1, len(items) // 200)
    small = items[::step][:200]
    json.dump(small, open(f"{Q}/phase10_runs/probe_items_smoke.json", "w"))

    print(f"{len(targets)} targets; {len(items)} items of {N_ITEMS} asked for "
          f"across {len(report['kept_by_stratum'])} strata")
    print(f"  filter {drop}; truncated at {MAXLEN} tokens: {report['n_truncated']}")
    print(f"  revision {report['revision']}")


if __name__ == "__main__":
    main()
