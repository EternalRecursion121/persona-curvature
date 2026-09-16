#!/usr/bin/env python3
"""Build the six training corpora for the sycophancy-forecast experiment.

The Dolci audit (`wiki/pages/behaviour/dolci-data-audit.md`) scored 12,524
`allenai/Dolci-Instruct-DPO` pairs along 63 weight directions and wrote the
selections into `phase10_runs/dolci_selection.json`.  This script turns the
`align_sycophantic` selections, two stratum draws and one corrigible selection
into six matched 400-pair DPO corpora that span the sycophantic direction:

  syc_top      the length-stratified sycophantic top 400
  syc_control  its own length-stratified matched random 400
  syc_bottom   the sycophantic bottom 400 (anti-sycophantic tail)
  delta        400 pairs drawn at random from the `delta_learning` half
  gptj         400 pairs drawn at random from the `llm_judged` (GPT-judged) half
  corr_ls      the length-stratified corrigible top 400 -- the arm the previous
               stage-two run identified as unresolved because it trained the
               plain (length-confounded) selection instead

Everything is read by id out of `phase10_runs/dolci_sample_dpo.jsonl`, the
sample that was scored, from dataset revision
`aed155cf32e809b590490b6c3577ee4b0d0a5019`.  No new HuggingFace fetch, so the
rows trained on are byte-identical to the rows scored.

Output: `phase10_runs/syc_arm_data/pool.jsonl` and `arms.json` (id list per arm,
sha256 over its own records which the container recomputes and asserts, the mean
/ sd / quantiles of its first-order score along six directions, and its
chosen-half length distribution).
"""
import hashlib
import json
import os
import random
import statistics as st

Q = os.path.dirname(os.path.abspath(__file__))
OUT = f"{Q}/phase10_runs/syc_arm_data"
SEED_DELTA = 20260910
SEED_GPTJ = 20260911
# index into dolci_targets.json#targets, asserted below
DIRS = {"align_sycophantic": 14, "align_obsequious": 12, "axis_Agreeableness": 6,
        "FA_Warmth": 0, "mean_assistant_axis": 10, "align_corrigible": 11}


def quantiles(xs):
    s = sorted(xs)
    def q(p):
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]
    return {"min": s[0], "p10": q(.10), "p25": q(.25), "median": q(.50),
            "p75": q(.75), "p90": q(.90), "max": s[-1]}


def main():
    os.makedirs(OUT, exist_ok=True)
    sel = json.load(open(f"{Q}/phase10_runs/dolci_selection.json"))
    tgt = json.load(open(f"{Q}/phase10_runs/dolci_targets.json"))["targets"]
    for name, idx in DIRS.items():
        assert tgt[idx]["name"] == name, (name, idx, tgt[idx]["name"])

    score, strat, ntok = {}, {}, {}
    for line in open(f"{Q}/phase10_runs/dolci_scores_dpo.jsonl"):
        r = json.loads(line)
        score[r["id"]] = r["pair"]
        strat[r["id"]] = r["stratum"]
        ntok[r["id"]] = {"chosen": r["n_tok_chosen"], "rejected": r["n_tok_rejected"],
                         "prompt": r["n_prompt_tok"]}
    print(f"[scores] {len(score)} scored pairs")

    rows = {}
    for line in open(f"{Q}/phase10_runs/dolci_sample_dpo.jsonl"):
        r = json.loads(line)
        rows[r["id"]] = r
    assert set(rows) == set(score), "score file and sample file disagree on ids"

    s = sel["sets"]["dpo_align_sycophantic"]
    c = sel["sets"]["dpo_align_corrigible"]
    # the two stratum draws: separate RNGs so the order of the two samples
    # cannot matter, and both sampled from a SORTED id list so the draw is
    # reproducible from the file alone.
    delta_pool = sorted(i for i in score if strat[i] == "delta_learning")
    gptj_pool = sorted(i for i in score if strat[i] == "llm_judged")

    def stratum_draw(pool, n, seed):
        """Draw n ids representative of the pool's OWN align_sycophantic
        distribution: sort by score, cut into n/10 equal-count bins, draw 10
        from each bin.  A plain random 400 of a 5,760-row stratum has a standard
        error of about 0.0039 on the mean score -- the same size as the whole
        `delta_learning` minus `llm_judged` difference the audit reported
        (+0.01333 against -0.00368), so a plain draw would test the draw rather
        than the stratum.  This rule is declared in PREREG_sycforecast.md."""
        idx = DIRS["align_sycophantic"]
        ranked = sorted(pool, key=lambda i: (score[i][idx], i))
        nbin, per = n // 10, 10
        rng = random.Random(seed)
        out = []
        for b in range(nbin):
            lo = int(round(b * len(ranked) / nbin))
            hi = int(round((b + 1) * len(ranked) / nbin))
            out += rng.sample(ranked[lo:hi], per)
        assert len(out) == n and len(set(out)) == n
        return sorted(out)

    delta = stratum_draw(delta_pool, 400, SEED_DELTA)
    gptj = stratum_draw(gptj_pool, 400, SEED_GPTJ)
    print(f"[stratum draw] delta_learning pool {len(delta_pool)} "
          f"mean {st.mean(score[i][14] for i in delta_pool):+.6f} -> "
          f"draw {st.mean(score[i][14] for i in delta):+.6f}")
    print(f"[stratum draw] llm_judged     pool {len(gptj_pool)} "
          f"mean {st.mean(score[i][14] for i in gptj_pool):+.6f} -> "
          f"draw {st.mean(score[i][14] for i in gptj):+.6f}")

    arms = {
        "syc_top": s["top400_length_stratified"],
        "syc_control": s["random400_length_stratified"],
        "syc_bottom": s["bottom400"],
        "delta": delta,
        "gptj": gptj,
        "corr_ls": c["top400_length_stratified"],
    }
    for k, v in arms.items():
        assert len(v) == 400 and len(set(v)) == 400, k

    overlaps = {}
    ks = list(arms)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            overlaps[f"{ks[i]}_vs_{ks[j]}"] = len(set(arms[ks[i]]) & set(arms[ks[j]]))
    # the arms of the previous stage-two run, for the record
    overlaps["corr_ls_vs_prev_plain_flagged"] = len(
        set(arms["corr_ls"]) & set(c["top400"]))

    # Length deciles of the whole scored sample, so each arm's length profile
    # can be read against the corpus's own.  The audit's `*_length_stratified`
    # selections take the top 40 of each decile of `n_tok_chosen`, so that is
    # the primary variable here; character counts are reported beside it
    # because characters per token differ between prose and code.
    def cutpoints(vals):
        v = sorted(vals)
        return [v[int(round(p * (len(v) - 1)))] for p in
                [.1, .2, .3, .4, .5, .6, .7, .8, .9]]

    tok_cuts = cutpoints(ntok[i]["chosen"] for i in rows)
    char_cuts = cutpoints(len(rows[i]["chosen"]) for i in rows)

    def decile_counts(ids, get, cuts):
        out = [0] * 10
        for i in ids:
            L, k = get(i), 0
            while k < 9 and L > cuts[k]:
                k += 1
            out[k] += 1
        return out

    meta_arms = {}
    for a, ids in arms.items():
        ch = [len(rows[i]["chosen"]) for i in ids]
        rj = [len(rows[i]["rejected"]) for i in ids]
        h = hashlib.sha256()
        for i in ids:
            r = rows[i]
            h.update(json.dumps([i, r["messages"], r["chosen"], r["rejected"]],
                                sort_keys=True).encode())
        meta_arms[a] = {
            "n": len(ids), "sha256": h.hexdigest(),
            "scores": {name: {"mean": st.mean(score[i][idx] for i in ids),
                              "sd": st.pstdev([score[i][idx] for i in ids]),
                              "frac_positive": sum(1 for i in ids if score[i][idx] > 0) / len(ids),
                              **quantiles([score[i][idx] for i in ids])}
                       for name, idx in DIRS.items()},
            "chosen_chars": {"mean": st.mean(ch), **quantiles(ch)},
            "rejected_chars": {"mean": st.mean(rj), **quantiles(rj)},
            "chosen_tok": {"mean": st.mean(ntok[i]["chosen"] for i in ids),
                           **quantiles([ntok[i]["chosen"] for i in ids])},
            "chosen_tok_decile_counts": decile_counts(
                ids, lambda i: ntok[i]["chosen"], tok_cuts),
            "chosen_char_decile_counts": decile_counts(
                ids, lambda i: len(rows[i]["chosen"]), char_cuts),
            "strata": {k: sum(1 for i in ids if strat[i] == k)
                       for k in sorted(set(strat.values()))},
            "ids": ids,
        }

    pool_ids = sorted({i for v in arms.values() for i in v})
    with open(f"{OUT}/pool.jsonl", "w") as f:
        for i in pool_ids:
            r = rows[i]
            f.write(json.dumps({"id": i, "stratum": r["stratum"],
                                "messages": r["messages"],
                                "chosen": r["chosen"], "rejected": r["rejected"],
                                "score_align_sycophantic": score[i][DIRS["align_sycophantic"]]
                                }) + "\n")

    meta = {
        "built_by": "build_syc_arms.py",
        "directions": DIRS,
        "revisions": sel["revisions"],
        "tokenisation_of_the_scoring_run": sel["tokenisation"],
        "seed_delta": SEED_DELTA, "seed_gptj": SEED_GPTJ,
        "n_scored_population": len(score),
        "chosen_tok_decile_cuts": tok_cuts,
        "chosen_char_decile_cuts": char_cuts,
        "overlaps": overlaps,
        "arms": meta_arms,
    }
    json.dump(meta, open(f"{OUT}/arms.json", "w"), indent=1)

    print(f"{'arm':12s} {'n':>4s} " + " ".join(f"{k:>19s}" for k in DIRS)
          + f" {'chos_char':>10s} {'chos_tok':>9s}  chosen-token deciles")
    for a, m in meta_arms.items():
        print(f"{a:12s} {m['n']:4d} "
              + " ".join(f"{m['scores'][k]['mean']:+19.6f}" for k in DIRS)
              + f" {m['chosen_chars']['mean']:10.0f} {m['chosen_tok']['mean']:9.0f}"
              + f"  {m['chosen_tok_decile_counts']}")
    print(json.dumps(overlaps, indent=1))
    print(f"[pool] {len(pool_ids)} unique rows -> {OUT}/pool.jsonl "
          f"({os.path.getsize(OUT + '/pool.jsonl')/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
