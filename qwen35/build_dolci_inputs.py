#!/usr/bin/env python3
"""Targets and items for auditing the Dolci Instruct mixtures with the direction scorer.

THE QUESTION.  Every use of `align_score.py` so far has been about the project's
own data.  This asks whether the same identity is useful as a DATA-AUDIT tool on
someone else's corpus: given a labelled basis of personality directions and four
alignment directions, can the first-order score flag examples in a large public
instruct mixture that push the model somewhere undesirable -- and flag them
without a keyword filter?

The two mixtures are the ones Olmo 3 Instruct 7B was trained on:
  allenai/Dolci-Instruct-DPO   259,922 preference pairs
  allenai/Dolci-Instruct-SFT   2,152,112 supervised examples

SAMPLING.  Neither is downloaded.  The exact stratum counts come from the
datasets-server /statistics endpoint (recorded in the output), and one streaming
pass keeps a per-stratum reservoir, oversampled, from which the allocation is
drawn after local filtering.  Reservoir sampling per key is uniform within each
key whatever order the shards arrive in, which matters because the shards are
grouped by source.

  DPO stratum = `preference_type` (4 values)
  SFT stratum = `source_dataset` (22 values)

Allocation is proportional to the population with a floor and a cap so the small
strata carry enough items to be tested on their own; the population weights are
written out so any aggregate can be reported both ways.

FILTERS, all recorded as rates on the oversample so they estimate the population:
  * the last message must be a non-empty assistant turn
  * for DPO, chosen[:-1] must equal rejected[:-1] turn for turn -- a preference
    pair whose two halves do not share a prompt is not a preference pair, and
    `multiturn_self_talk` is where that is worth checking
  * the rendered prompt must be at most PROMPT_CAP tokens
  * the completion is truncated at COMPLETION_CAP tokens by the scorer's own
    `fi[:ml]`; `maxlen` is set per item to n_prompt + COMPLETION_CAP + 1 so the
    cap lands on the completion and never on the prompt.  A truncated item loses
    its EOS, which is exactly what a truncated training example would do.

TOKENISATION is the SFT recipe for both mixtures -- prompt + completion + EOS as
one string with the prompt masked, enable_thinking=False -- matching
`sft_rewardhacks.py` and the `mode="sft"` branch of the scorer.  Dolci SFT is an
SFT mixture, and Dolci DPO's two halves are each scored as completions before the
difference is taken, so the SFT spelling is the honest one for both.

TARGETS (63).  Small enough that B_U fits beside the model with room for a real
batch: the 134 singles are a separate, more expensive job (build_dolci_singles).
"""
import json
import os
import random
import sys
import urllib.request

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
PROMPT_CAP = 512
COMPLETION_CAP = 512
N_DPO, N_SFT = 12000, 12000
OVERSAMPLE = 2.0
SEED = 20260909
N_RAND = 30

ALIGN_TRAITS = ["corrigible", "obsequious", "power_seeking", "sycophantic"]
FA_NAMES = ["FA_Warmth", "FA_Competence", "FA_FearfulWithdrawal", "FA_Arousal",
            "FA_Imagination"]
AXIS_NAMES = ["axis_Extraversion", "axis_Agreeableness", "axis_Conscientiousness",
              "axis_EmotionalStability", "axis_Intellect", "mean_assistant_axis"]
# Negative-valence traits flag (b) asks about, plus their positive counterparts as
# a within-family reference, plus `agreeable` as the N x N anchor cell.
NAMED_TRAITS = ["rude", "harsh", "cold", "unkind", "careless", "negligent",
                "crooked", "selfish", "immodest", "uncooperative", "demanding",
                "agreeable", "kind", "warm", "helpful", "considerate"]
ANCHOR_TRAIT = "agreeable"


def stats(repo):
    """Stratum counts straight from the datasets-server, no download."""
    u = ("https://datasets-server.huggingface.co/statistics?dataset="
         + repo.replace("/", "%2F") + "&config=default&split=train")
    with urllib.request.urlopen(u, timeout=120) as r:
        d = json.load(r)
    out = {}
    for c in d["statistics"]:
        s = c["column_statistics"]
        if "frequencies" in s:
            out[c["column_name"]] = s["frequencies"]
    return out, d.get("num_rows")


def allocate(counts, total, floor, cap):
    """Proportional allocation with a floor and a cap, water-filled.

    A stratum smaller than the floor contributes all of itself.  What the floor
    and cap take from the large strata is redistributed among those still under
    the cap, so the allocation always sums to `total` unless the population is
    smaller.
    """
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
    # trim/extend by ones so the total lands exactly
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


# The box has 7 GB of RAM, so a reservoir of whole rows is not affordable: a Dolci
# DPO row carries twenty mostly-null fields per message and an SFT row can carry
# 183 messages of 32K context.  `slim` strips a row to what the audit uses, and
# refuses any row whose prompt side exceeds CHAR_CAP characters.  That refusal is
# SAFE rather than arbitrary: at worst ~1 token per character, CHAR_CAP is far
# above PROMPT_CAP tokens, so every row it drops would have been dropped later by
# the exact token filter anyway.  It is counted and reported.
CHAR_CAP = 12000


def reservoirs(repo, key_fn, slim, want, seed):
    """One streaming pass, one reservoir per stratum, plus the counts.

    Cached to disk.  The pass reads the whole shard set over the network -- about
    25 minutes for the SFT mixture -- and the first run of this script threw a
    NameError in the code AFTER it, which paid that cost for nothing.  The cache
    key is the repo and the requested sizes, so changing the allocation
    invalidates it.
    """
    import hashlib
    import pickle
    key = hashlib.sha1(json.dumps([repo, sorted(want.items()), seed],
                                  sort_keys=True).encode()).hexdigest()[:12]
    cp = f"{Q}/phase10_runs/dolci_res_{key}.pkl"
    if os.path.exists(cp):
        print(f"    reusing reservoir cache {os.path.basename(cp)}", flush=True)
        with open(cp, "rb") as f:
            return pickle.load(f)
    from datasets import load_dataset
    ds = load_dataset(repo, split="train", streaming=True)
    rng = random.Random(seed)
    res, seen, over = {}, {}, {}
    for n, row in enumerate(ds):
        k = key_fn(row)
        row = slim(row)
        if row is None:
            over[k] = over.get(k, 0) + 1
            continue
        seen[k] = seen.get(k, 0) + 1
        cap = want.get(k, 0)
        if cap <= 0:
            continue
        r = res.setdefault(k, [])
        if len(r) < cap:
            r.append(row)
        else:
            j = rng.randrange(seen[k])
            if j < cap:
                r[j] = row
        if (n + 1) % 200000 == 0:
            print(f"    {repo}: {n + 1} rows streamed", flush=True)
    with open(cp, "wb") as f:
        pickle.dump((res, seen, over), f, protocol=4)
    return res, seen, over


def _msgs(ms):
    return [{"role": m.get("role"), "content": m.get("content") or ""} for m in ms]


def _too_long(ms):
    return sum(len(m.get("content") or "") for m in ms[:-1]) > CHAR_CAP


def slim_dpo(r):
    if _too_long(r["chosen"]):
        return None
    return {"chosen": _msgs(r["chosen"]), "rejected": _msgs(r["rejected"]),
            "chosen_model": r["chosen_model"], "rejected_model": r["rejected_model"],
            "prompt_id": r["prompt_id"], "preference_type": r["preference_type"]}


def slim_sft(r):
    if _too_long(r["messages"]):
        return None
    return {"messages": _msgs(r["messages"]), "id": r["id"],
            "source_dataset": r["source_dataset"], "domain": r["domain"]}


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    rng = random.Random(SEED)

    def render(msgs):
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)

    def prep(msgs):
        """(prompt messages, completion text, n_prompt_tokens) or a reason string."""
        if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
            return "bad_completion"
        comp = (msgs[-1].get("content") or "").strip()
        if not comp:
            return "bad_completion"
        pre_msgs = [{"role": m["role"], "content": m.get("content") or ""}
                    for m in msgs[:-1]]
        # The Dolci SFT tool-use rows carry an `environment` role that the
        # Qwen3.5 chat template rejects, and the sampler keeps only role and
        # content anyway, so a tool-use conversation could not be rendered
        # faithfully even if the template accepted it.  Dropped and counted.
        try:
            pre = render(pre_msgs)
        except Exception:
            return "template_error"
        npre = len(tok(pre, add_special_tokens=False)["input_ids"])
        if npre > PROMPT_CAP:
            return "prompt_too_long"
        return pre_msgs, comp, npre

    def finish(pre_msgs, npre, comp):
        """maxlen and the truncation flag, computed exactly as the scorer will."""
        ml = npre + COMPLETION_CAP + 1
        pre = render(pre_msgs)
        full = pre + comp + tok.eos_token
        nfull = len(tok(full, add_special_tokens=False)["input_ids"])
        return ml, nfull > ml, min(nfull, ml)

    report = {"prompt_cap": PROMPT_CAP, "completion_cap": COMPLETION_CAP,
              "seed": SEED, "oversample": OVERSAMPLE, "revisions": {},
              "tokenisation": {"mode": "sft", "template": "Qwen/Qwen3.5-4B chat template",
                               "enable_thinking": False,
                               "loss": "completion tokens only, EOS included unless truncated"}}
    from huggingface_hub import HfApi
    api = HfApi()

    items, meta = [], {}

    # ---------------- DPO -------------------------------------------------
    repo = "allenai/Dolci-Instruct-DPO"
    report["revisions"][repo] = api.dataset_info(repo).sha
    st, nrows = stats(repo)
    pt = {k: int(v) for k, v in st["preference_type"].items()}
    report["dpo_population"] = {"n_rows": nrows, "preference_type": pt}
    alloc = allocate(pt, N_DPO, floor=800, cap=6000)
    report["dpo_allocation"] = alloc
    want = {k: min(pt[k], int(v * OVERSAMPLE) + 50) for k, v in alloc.items()}
    print(f"[dpo] allocation {alloc}; reservoirs {want}", flush=True)
    res, seen, over = reservoirs(repo, lambda r: r["preference_type"], slim_dpo, want, SEED)
    report["dpo_streamed_counts"] = seen
    report["dpo_char_cap_dropped"] = over

    drop = {"prefix_mismatch": 0, "bad_completion": 0, "prompt_too_long": 0,
            "template_error": 0, "kept": 0}
    kept = {}
    for k, rows in res.items():
        rng.shuffle(rows)
        out = []
        for r in rows:
            ch, rj = r["chosen"], r["rejected"]
            if len(ch) != len(rj) or any(
                    (a.get("role"), (a.get("content") or "")) !=
                    (b.get("role"), (b.get("content") or ""))
                    for a, b in zip(ch[:-1], rj[:-1])):
                drop["prefix_mismatch"] += 1
                continue
            a, b = prep(ch), prep(rj)
            if isinstance(a, str) or isinstance(b, str):
                drop[a if isinstance(a, str) else b] += 1
                continue
            out.append((r, a[0], a[1], b[1], a[2]))
            if len(out) >= alloc[k]:
                break
        drop["kept"] += len(out)
        kept[k] = out
    report["dpo_filter"] = drop

    for k, rows in sorted(kept.items()):
        for i, (r, pre_msgs, ctext, rtext, npre) in enumerate(rows):
            iid = f"dpo#{k}#{i}"
            mlc, tc, nc = finish(pre_msgs, npre, ctext)
            mlr, tr, nr = finish(pre_msgs, npre, rtext)
            items.append({"id": iid, "trait": "dolci_dpo", "messages": pre_msgs,
                          "chosen": ctext, "rejected": rtext, "mode": "sft",
                          "maxlen": max(mlc, mlr), "ntok": max(nc, nr)})
            meta[iid] = {"set": "dpo", "stratum": k, "prompt_id": r["prompt_id"],
                         "chosen_model": r["chosen_model"],
                         "rejected_model": r["rejected_model"],
                         "n_turns": len(r["chosen"]), "n_prompt_tok": npre,
                         "truncated_chosen": tc, "truncated_rejected": tr,
                         "n_tok_chosen": nc, "n_tok_rejected": nr,
                         "prompt_source": r["prompt_id"].rsplit("-", 3)[0]}

    del res, kept                      # 7 GB of RAM; do not hold both reservoirs

    # ---------------- SFT -------------------------------------------------
    repo = "allenai/Dolci-Instruct-SFT"
    report["revisions"][repo] = api.dataset_info(repo).sha
    st, nrows = stats(repo)
    sd = {k: int(v) for k, v in st["source_dataset"].items()}
    report["sft_population"] = {"n_rows": nrows, "source_dataset": sd,
                                "domain": {k: int(v) for k, v in st["domain"].items()}}
    alloc = allocate(sd, N_SFT, floor=250, cap=1200)
    report["sft_allocation"] = alloc
    want = {k: min(sd[k], int(v * OVERSAMPLE) + 50) for k, v in alloc.items()}
    print(f"[sft] allocation {alloc}", flush=True)
    res, seen, over = reservoirs(repo, lambda r: r["source_dataset"], slim_sft, want, SEED + 1)
    report["sft_streamed_counts"] = seen
    report["sft_char_cap_dropped"] = over

    drop = {"bad_completion": 0, "prompt_too_long": 0, "template_error": 0, "kept": 0}
    for k in sorted(res):
        rows = res[k]
        rng.shuffle(rows)
        n = 0
        for r in rows:
            a = prep(r["messages"])
            if isinstance(a, str):
                drop[a] += 1
                continue
            pre_msgs, comp, npre = a
            ml, t, nt = finish(pre_msgs, npre, comp)
            iid = f"sft#{r['id']}"
            items.append({"id": iid, "trait": "dolci_sft", "messages": pre_msgs,
                          "chosen": comp, "single": True, "mode": "sft",
                          "maxlen": ml, "ntok": nt})
            meta[iid] = {"set": "sft", "stratum": k, "domain": r["domain"],
                         "source_dataset": r["source_dataset"], "row_id": r["id"],
                         "n_turns": len(r["messages"]), "n_prompt_tok": npre,
                         "truncated": t, "n_tok": nt}
            n += 1
            drop["kept"] += 1
            if n >= alloc[k]:
                break
    report["sft_filter"] = drop

    # longest first, so an OOM happens in the first minute and not at item 8000
    items.sort(key=lambda x: -x["ntok"])

    # ---------------- the N x N anchor -----------------------------------
    nxn = [dict(x) for x in json.load(open(f"{Q}/phase10_runs/nxn_items.json"))
           if x["trait"] == ANCHOR_TRAIT]
    assert len(nxn) == 40, len(nxn)
    items = nxn + items          # anchors first, verbatim, DPO tokenisation

    json.dump(items, open(f"{Q}/phase10_runs/dolci_items.json", "w"))
    # the human-readable sample, with metadata, one line per item
    with open(f"{Q}/phase10_runs/dolci_sample_dpo.jsonl", "w") as fd, \
         open(f"{Q}/phase10_runs/dolci_sample_sft.jsonl", "w") as fs:
        for x in items:
            if x["id"] not in meta:
                continue
            m = meta[x["id"]]
            rec = {"id": x["id"], **m, "messages": x["messages"],
                   "chosen": x["chosen"]}
            if "rejected" in x:
                rec["rejected"] = x["rejected"]
            (fd if m["set"] == "dpo" else fs).write(json.dumps(rec) + "\n")
    json.dump({"report": report, "items": meta},
              open(f"{Q}/phase10_runs/dolci_itemmeta.json", "w"), indent=1)

    # ---------------- targets --------------------------------------------
    build_targets()

    ndpo = sum(1 for m in meta.values() if m["set"] == "dpo")
    nsft = len(meta) - ndpo
    print(f"\n{len(items)} items = 40 anchor + {ndpo} DPO pairs + {nsft} SFT completions")
    print("dpo filter", report["dpo_filter"])
    print("sft filter", report["sft_filter"])
    print("median item", int(np.median([x["ntok"] for x in items if "ntok" in x])),
          "tokens; longest", max(x["ntok"] for x in items if "ntok" in x))


def build_targets():
    """The 63 directions.  Split out so the spec can be written, and a Modal smoke
    test run against it, while the streaming sample is still being drawn."""
    names = json.load(open(f"{Q}/analysis/alien.json"))["traits"]
    assert len(names) == 134 and names == sorted(names)
    targets = []
    for path, wantn in ((f"{Q}/phase10_runs/steer_spec2_7a.json", FA_NAMES),
                        (f"{Q}/phase10_runs/steer_spec.json", AXIS_NAMES)):
        jobs = {j["name"]: j for j in json.load(open(path))["jobs"]}
        for n in wantn:
            targets.append({"name": n, "coef": jobs[n]["coef"],
                            "from": os.path.basename(path)})
    for t in ALIGN_TRAITS:
        targets.append({"name": f"align_{t}",
                        "src": [[f"/align/data_alignment_common/{t}", 1.0]]})
    targets.append({"name": "stage2_shared",
                    "src": [[f"/oct/loras_introspection/{t}", 1.0 / 134] for t in names],
                    "note": "projection into the zoo LoRA-A window; stage-two A is a "
                            "different random frame"})
    targets.append({"name": "sorh_hack_minus_control",
                    "src": [["/rl/runs/sorh_hack/final", 1.0],
                            ["/rl/runs/sorh_control/final", -1.0]]})
    for t in NAMED_TRAITS:
        assert t in names, t
        targets.append({"name": f"trait_{t}", "coef": {t: 1.0}})
    r = np.random.default_rng(SEED)
    for i in range(N_RAND):
        c = r.standard_normal(len(names))
        targets.append({"name": f"rand_merge_{i:02d}",
                        "coef": {t: float(v) for t, v in zip(names, c)}})

    spec = {"targets": targets, "a0": f"/adapters/{names[0]}", "maxlen": 640,
            "rand_seed": SEED, "n_rand": N_RAND,
            "built_by": "build_dolci_inputs.py"}
    json.dump(spec, open(f"{Q}/phase10_runs/dolci_targets.json", "w"))
    print(f"{len(targets)} targets written to phase10_runs/dolci_targets.json")
    return spec


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "targets":
        build_targets()
    else:
        sys.exit(main())
