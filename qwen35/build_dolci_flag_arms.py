#!/usr/bin/env python3
"""Build the five training corpora for the dolci-flag stage-two experiment.

Stage one (`wiki/pages/behaviour/dolci-data-audit.md`) scored 12,524
Dolci-Instruct-DPO pairs along 63 weight directions and flagged the top 400 by
the `align_corrigible` pair score read NEGATIVE -- the tail whose chosen half
complies with a request the rejected half refuses.  This script turns that flag
into five matched training corpora.

  flagged      the corrigible top 400            (dolci_selection.json)
  random       the matched random 400            (matched on stratum + prompt length)
  anti         the corrigible bottom 400
  unfiltered   3,000 pairs drawn at random, seed 20260910
  filtered     the SAME 3,000 with every pair in the top 2% by the
               corrigible-negative score replaced by a random pair from outside
               both the draw and the top 2%

Everything comes from `phase10_runs/dolci_sample_dpo.jsonl`, which is the
sample that was scored: the same ids, from the recorded dataset revision
`aed155cf32e809b590490b6c3577ee4b0d0a5019`.  No new HuggingFace fetch, so the
rows trained on are byte-identical to the rows scored.

Output: `phase10_runs/dolci_flag_data/pool.jsonl` (the union of every id used,
one record per line) and `arms.json` (the id list per arm plus the sha256 of
its own records, which the container recomputes and asserts).
"""
import hashlib
import json
import os
import random

Q = os.path.dirname(os.path.abspath(__file__))
OUT = f"{Q}/phase10_runs/dolci_flag_data"
DIRECTION = "align_corrigible"
DIR_INDEX = 11                 # position in dolci_targets.json#targets
N_LARGE = 3000
TOP_FRAC = 0.02
SEED = 20260910


def main():
    os.makedirs(OUT, exist_ok=True)
    sel = json.load(open(f"{Q}/phase10_runs/dolci_selection.json"))
    tgt = json.load(open(f"{Q}/phase10_runs/dolci_targets.json"))["targets"]
    assert tgt[DIR_INDEX]["name"] == DIRECTION, tgt[DIR_INDEX]["name"]
    s = sel["sets"]["dpo_align_corrigible"]

    # per-item pair score along align_corrigible, over the whole scored sample
    score = {}
    with open(f"{Q}/phase10_runs/dolci_scores_dpo.jsonl") as f:
        for line in f:
            r = json.loads(line)
            score[r["id"]] = r["pair"][DIR_INDEX]
    print(f"[scores] {len(score)} scored pairs")

    # rows
    rows = {}
    with open(f"{Q}/phase10_runs/dolci_sample_dpo.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows[r["id"]] = r
    assert set(rows) == set(score), "score file and sample file disagree on ids"

    all_ids = sorted(rows)
    # negative direction: the flag is the MOST NEGATIVE 2%
    ranked = sorted(all_ids, key=lambda i: score[i])
    n_top = int(round(TOP_FRAC * len(all_ids)))
    top2 = set(ranked[:n_top])
    # 2% of 12,524 is 250, so the top 2% is the leading 250 of the flag's own
    # top 400; if the ranking here disagreed with the selection file at all,
    # this containment would fail.
    assert top2 <= set(s["top400"]), "top 2% is not inside top400 -- ranking mismatch"

    rng = random.Random(SEED)
    unfiltered = rng.sample(all_ids, N_LARGE)
    hits = [i for i in unfiltered if i in top2]
    spare_pool = [i for i in all_ids if i not in top2 and i not in set(unfiltered)]
    rng2 = random.Random(SEED + 1)
    swaps = rng2.sample(spare_pool, len(hits))
    filtered = [i for i in unfiltered if i not in top2] + swaps
    assert len(filtered) == N_LARGE
    assert not (set(filtered) & top2)

    arms = {
        "flagged": s["top400"],
        "random": s["random400_matched"],
        "anti": s["bottom400"],
        "unfiltered": unfiltered,
        "filtered": filtered,
    }

    # overlap with the power-seeking flag, which the audit calls the
    # verbose-moralising-refusal tail; the corrigible bottom 400 is expected to
    # intersect it and that has to be on the record before any result.
    ps = sel["sets"]["dpo_align_power_seeking"]
    overlaps = {
        "anti_vs_power_seeking_top400": len(set(arms["anti"]) & set(ps["top400"])),
        "flagged_vs_power_seeking_top400": len(set(arms["flagged"]) & set(ps["top400"])),
        "flagged_vs_random": len(set(arms["flagged"]) & set(arms["random"])),
        "flagged_vs_anti": len(set(arms["flagged"]) & set(arms["anti"])),
        "unfiltered_vs_flagged": len(set(arms["unfiltered"]) & set(arms["flagged"])),
        "filtered_vs_unfiltered_shared": len(set(arms["filtered"]) & set(arms["unfiltered"])),
    }

    pool_ids = sorted({i for v in arms.values() for i in v})
    with open(f"{OUT}/pool.jsonl", "w") as f:
        for i in pool_ids:
            r = rows[i]
            f.write(json.dumps({"id": i, "stratum": r["stratum"],
                                "messages": r["messages"],
                                "chosen": r["chosen"], "rejected": r["rejected"],
                                "score_align_corrigible": score[i]}) + "\n")

    def arm_sha(ids):
        h = hashlib.sha256()
        for i in ids:                      # order as trained (pre-shuffle)
            r = rows[i]
            h.update(json.dumps([i, r["messages"], r["chosen"], r["rejected"]],
                                sort_keys=True).encode())
        return h.hexdigest()

    meta = {
        "built_by": "build_dolci_flag_arms.py",
        "direction": DIRECTION,
        "direction_index": DIR_INDEX,
        "criterion": s["criterion"],
        "revisions": sel["revisions"],
        "tokenisation_of_the_scoring_run": sel["tokenisation"],
        "seed": SEED,
        "n_scored_population": len(all_ids),
        "top_frac": TOP_FRAC,
        "n_top2pct": n_top,
        "n_swapped_in_filtered": len(hits),
        "swap_rate": len(hits) / N_LARGE,
        "overlaps": overlaps,
        "arms": {k: {"n": len(v), "sha256": arm_sha(v),
                     "mean_score_align_corrigible": sum(score[i] for i in v) / len(v),
                     "ids": v}
                 for k, v in arms.items()},
    }
    json.dump(meta, open(f"{OUT}/arms.json", "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "ids"}
                      for k, v in meta["arms"].items()}, indent=1))
    print(json.dumps(overlaps, indent=1))
    print(f"[filtered] {len(hits)} of {N_LARGE} pairs swapped ({len(hits)/N_LARGE:.4f})")
    print(f"[pool] {len(pool_ids)} unique rows -> {OUT}/pool.jsonl "
          f"({os.path.getsize(OUT + '/pool.jsonl')/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
