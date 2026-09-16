#!/usr/bin/env python3
"""Arms data, scoring items and scoring targets for the emergent-misalignment run.

Design is fixed by PREREG_em_medical.md (sha256
2c20a1d46b5406dc7f7be508ec08e7c040d5c5fbb85cba6dc14e05c719bf5c47), written
before this file ran.  Nothing here chooses anything the pre-registration left
open.

The two medical corpora are row-aligned on their user prompts, so the same 2,000
row indices give the bad arm and the good arm byte-identical prompts and the
scorer's `pair` field is the paired bad-minus-good contrast per prompt.

The corpora themselves stay in the scratch directory (their ToS forbids bulk
redistribution); only the derived arm files and the scorer's items file are
written under phase10_runs/, which is local and unpublished.
"""
import hashlib
import json
import os
import random

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
SCRATCH = ("/tmp/claude-1000/-home-vibe12-projects/"
           "981fa3b5-8a9b-405e-9e84-70cc615d9873/scratchpad/em/"
           "training_datasets.zip.enc.extracted")
OUT = os.path.join(Q, "phase10_runs", "em_arm_data")
SEED = 20260911
N_ARM = 2000          # rows per trained arm
N_SCORE = 1000        # scored rows per corpus (prereg: at least 1,000)
N_RAND = 20           # random merges in the null band
PROMPT_CAP = 512
COMPLETION_CAP = 512

ALIGN = ["corrigible", "obsequious", "power_seeking", "sycophantic"]
ANCHORS = ["agreeable", "rude"]


def jsonl(p):
    return [json.loads(l) for l in open(p)]


def main():
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

    bad = jsonl(f"{SCRATCH}/bad_medical_advice.jsonl")
    good = jsonl(f"{SCRATCH}/good_medical_advice.jsonl")
    assert len(bad) == len(good) == 7049
    aligned = sum(1 for a, b in zip(bad, good)
                  if a["messages"][0]["content"] == b["messages"][0]["content"])
    assert aligned == len(bad), aligned

    rng = random.Random(SEED)
    idx = list(range(len(bad)))
    rng.shuffle(idx)
    arm_idx = idx[:N_ARM]
    score_idx = arm_idx[:N_SCORE]

    def row(src, i):
        m = src[i]["messages"]
        return {"user": m[0]["content"], "completion": m[1]["content"]}

    arms = {"em_bad": [row(bad, i) for i in arm_idx],
            "em_good": [row(good, i) for i in arm_idx]}

    # --- completion token counts, the quantity the length match is on ----------
    def ctok(user, comp):
        pre = tok.apply_chat_template([{"role": "user", "content": user}],
                                      tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
        full = pre + comp + tok.eos_token
        pi = len(tok(pre, add_special_tokens=False)["input_ids"])
        fi = len(tok(full, add_special_tokens=False)["input_ids"])
        return pi, max(0, fi - pi)

    ntok = {}
    for a in ("em_bad", "em_good"):
        ntok[a] = [ctok(r["user"], r["completion"]) for r in arms[a]]
        print(f"[tok] {a}: prompt mean {np.mean([x[0] for x in ntok[a]]):.1f} "
              f"completion mean {np.mean([x[1] for x in ntok[a]]):.1f} "
              f"total completion {sum(x[1] for x in ntok[a])}", flush=True)

    pooled = sorted([x[1] for x in ntok["em_bad"]] + [x[1] for x in ntok["em_good"]])

    # --- Dolci: quantile-matched draw from the audit's own 11,030 sample -------
    dolci = json.load(open(f"{Q}/phase10_runs/dolci_items_sft.json"))
    dtok = []
    for x in dolci:
        msgs = x["messages"]
        pre = tok.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=True,
                                      enable_thinking=False)
        pi = len(tok(pre, add_special_tokens=False)["input_ids"])
        fi = len(tok(pre + x["chosen"] + tok.eos_token,
                     add_special_tokens=False)["input_ids"])
        dtok.append((pi, max(0, min(fi, x["maxlen"]) - pi)))
    order = sorted(range(len(dolci)), key=lambda i: dtok[i][1])
    counts = [dtok[i][1] for i in order]
    used, picked = set(), []
    for i in range(N_ARM):
        q = (i + 0.5) / N_ARM
        target = pooled[min(len(pooled) - 1, int(q * len(pooled)))]
        # nearest unused by completion-token count, expanding outward from the
        # insertion point of `target` in the sorted list
        j = int(np.searchsorted(counts, target))
        lo, hi = j - 1, j
        while True:
            cl = abs(counts[lo] - target) if lo >= 0 else None
            ch = abs(counts[hi] - target) if hi < len(counts) else None
            if cl is None and ch is None:
                raise RuntimeError("Dolci pool exhausted")
            if ch is None or (cl is not None and cl <= ch):
                k, lo = order[lo], lo - 1
            else:
                k, hi = order[hi], hi + 1
            if k not in used:
                used.add(k)
                picked.append(k)
                break
    arms["em_dolci"] = [{"messages": dolci[k]["messages"],
                         "completion": dolci[k]["chosen"],
                         "src_id": dolci[k]["id"]} for k in picked]
    ntok["em_dolci"] = [dtok[k] for k in picked]
    print(f"[tok] em_dolci: prompt mean "
          f"{np.mean([x[0] for x in ntok['em_dolci']]):.1f} completion mean "
          f"{np.mean([x[1] for x in ntok['em_dolci']]):.1f} total completion "
          f"{sum(x[1] for x in ntok['em_dolci'])}", flush=True)

    for a, rows in arms.items():
        with open(f"{OUT}/{a}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # --- scoring items --------------------------------------------------------
    items = []
    for n, i in enumerate(score_idx):
        u = bad[i]["messages"][0]["content"]
        p = ntok["em_bad"][n][0]
        items.append({"id": f"med#{i}", "trait": "em_medical", "prompt": u,
                      "chosen": bad[i]["messages"][1]["content"],
                      "rejected": good[i]["messages"][1]["content"],
                      "mode": "sft", "maxlen": p + COMPLETION_CAP + 1,
                      "ntok": min(p + COMPLETION_CAP + 1,
                                  p + max(ntok["em_bad"][n][1],
                                          ntok["em_good"][n][1]))})
    for n in range(N_SCORE):
        r = arms["em_dolci"][n]
        p = ntok["em_dolci"][n][0]
        items.append({"id": f"dolci#{r['src_id']}", "trait": "dolci_sft",
                      "messages": r["messages"], "chosen": r["completion"],
                      "single": True, "mode": "sft",
                      "maxlen": p + COMPLETION_CAP + 1,
                      "ntok": min(p + COMPLETION_CAP + 1, p + ntok["em_dolci"][n][1])})
    nxn = json.load(open(f"{Q}/phase10_runs/nxn_items.json"))
    anchor = [x for x in nxn if x.get("trait") in ANCHORS]
    assert len(anchor) == 80, len(anchor)
    items += anchor                       # DPO spelling, verbatim, as the N x N run
    items.sort(key=lambda x: -int(x.get("ntok", 0)))     # longest first
    json.dump(items, open(f"{Q}/phase10_runs/em_ds_items.json", "w"))

    # --- targets --------------------------------------------------------------
    names134 = json.load(open(f"{Q}/analysis/alien.json"))["traits"]
    assert len(names134) == 134 and names134 == sorted(names134), len(names134)
    FA = ["FA_Warmth", "FA_Competence", "FA_FearfulWithdrawal", "FA_Arousal",
          "FA_Imagination"]
    AX = ["axis_Extraversion", "axis_Agreeableness", "axis_Conscientiousness",
          "axis_EmotionalStability", "axis_Intellect", "mean_assistant_axis"]
    targets = []
    for path, want in ((f"{Q}/phase10_runs/steer_spec2_7a.json", FA),
                       (f"{Q}/phase10_runs/steer_spec.json", AX)):
        jobs = {j["name"]: j for j in json.load(open(path))["jobs"]}
        for n in want:
            targets.append({"name": n, "coef": jobs[n]["coef"],
                            "from": os.path.basename(path)})
    for t in ALIGN:
        targets.append({"name": f"align_{t}",
                        "src": [[f"/align/data_alignment_common/{t}", 1.0]]})
    for t in names134:
        targets.append({"name": f"trait_{t}", "coef": {t: 1.0}})
    A = "/align/em_medical"  # the Trainer writes the adapter to <arm>/final, not <arm>
    targets += [{"name": "em_bad", "src": [[f"{A}/em_bad/final", 1.0]]},
                {"name": "em_good", "src": [[f"{A}/em_good/final", 1.0]]},
                {"name": "em_dolci", "src": [[f"{A}/em_dolci/final", 1.0]]},
                {"name": "em_bad_minus_good",
                 "src": [[f"{A}/em_bad/final", 1.0], [f"{A}/em_good/final", -1.0]]}]
    rng2 = np.random.default_rng(SEED)
    for k in range(N_RAND):
        c = rng2.standard_normal(len(names134))
        targets.append({"name": f"rand_merge_{k:02d}",
                        "coef": {t: float(v) for t, v in zip(names134, c)}})
    spec = {"targets": targets, "a0": f"/adapters/{names134[0]}", "maxlen": 1025,
            "tok_budget": 2300, "rand_seed": SEED, "n_rand": N_RAND,
            "built_by": "em_build_inputs.py"}
    json.dump(spec, open(f"{Q}/phase10_runs/em_ds_targets.json", "w"))

    rep = {
        "seed": SEED, "n_arm": N_ARM, "n_score": N_SCORE,
        "prereg_sha256": hashlib.sha256(
            open(f"{Q}/PREREG_em_medical.md", "rb").read()).hexdigest(),
        "prompt_alignment": aligned,
        "arm_rows": {a: len(v) for a, v in arms.items()},
        "tokens": {a: {"prompt_mean": float(np.mean([x[0] for x in v])),
                       "completion_mean": float(np.mean([x[1] for x in v])),
                       "completion_total": int(sum(x[1] for x in v)),
                       "completion_median": float(np.median([x[1] for x in v]))}
                   for a, v in ntok.items()},
        "dolci_match": {
            "pooled_medical_completion_mean": float(np.mean(pooled)),
            "dolci_completion_mean": float(np.mean([x[1] for x in ntok["em_dolci"]])),
            "ratio_dolci_over_medical_mean": float(
                sum(x[1] for x in ntok["em_dolci"])
                / (0.5 * (sum(x[1] for x in ntok["em_bad"])
                          + sum(x[1] for x in ntok["em_good"])))),
            "n_pool": len(dolci)},
        "n_targets": len(targets), "n_items": len(items),
        "n_named_directions": len(targets) - N_RAND,
        "item_counts": {"medical_pairs": N_SCORE, "dolci_single": N_SCORE,
                        "anchor": len(anchor)},
        "longest_item_ntok": int(items[0]["ntok"]),
    }
    json.dump(rep, open(f"{Q}/phase10_runs/em_build_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
