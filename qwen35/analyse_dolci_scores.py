#!/usr/bin/env python3
"""What do the Dolci Instruct mixtures push toward, and which examples are the worst?

Input   analysis/dolci_scores.json          (dolci_score.py --stage score)
        phase10_runs/dolci_targets.json
        phase10_runs/dolci_itemmeta.json
        phase10_runs/dolci_sample_{dpo,sft}.jsonl
        analysis/nxn_scores.json            (the `agreeable` anchor cell)
Output  phase10_runs/dolci_scores_{dpo,sft}.jsonl   per item, per direction
        analysis/dolci_scores_{dpo,sft}.json        the summaries
        analysis/dolci_audit.json                   the flags and their checks
        phase10_runs/dolci_selection.json           stage-two selections
        phase10_runs/dolci_judge_items.json         blind judging input

THE UNIT.  A score is the directional derivative of the mean-per-loss-token
log-likelihood along a unit-Frobenius weight direction.  Positive means a model
steered along that direction finds the completion more likely, which by the
identity in `align_score.py` means training on it pushes the weights that way.
For a DPO pair the reported `pair` is chosen minus rejected, which at B = 0 is
proportional to the first DPO gradient: it is what the PREFERENCE teaches, not
what either response is.

THE NULLS, and their limits.
  * random band -- 30 Gaussian merges of the 134 stage-one adapters, scored in
    the same run and unit-normalised like everything else.  For the factors, the
    axes, the grand mean and the single-trait targets, all inside the zoo's span,
    this is a matched null.  For align_*, stage2_shared and sorh_hack_minus_control
    it is a SCALE REFERENCE only: those directions are not in that span, and a
    band of 30 draws floors p at 1/31.
  * sign-flip permutation on the mean paired difference (DPO only; a raw score
    is a derivative and has no reason to be symmetric about zero).
  * label permutation for the subset question: the statistic is the between-group
    sum of squares of the stratum means, and the null shuffles the stratum labels
    across items, which is the only null under which "this source pushes further
    toward sycophancy than that one" means anything.

THE CONFOUND that dominates the DPO reading and is therefore reported first.
149,385 of the 259,922 rejected responses are `qwen3-no_reasoning-0.6b` and the
`delta_learning` half of the mixture is built by pairing a large model's answer
against a small one's.  A pair score in that stratum is measuring a CAPABILITY
gap along the direction as much as a preference for a disposition.  Every
headline is therefore given per `preference_type` and per rejected-model size
bin, never as one number over the mixture.
"""
import json
import math
import os
import re
import sys

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
N_PERM = 20000
N_PERM_LABEL = 2000
SEED = 11
RAND_PREFIX = "rand_merge_"
TAIL_FRAC = 0.01

# The four dispositions the audit is about, and the sign that is undesirable.
FLAG_DIRECTIONS = {
    "align_sycophantic": +1,
    "align_obsequious": +1,
    "align_power_seeking": +1,
    "align_corrigible": -1,
}
REGISTER = "mean_assistant_axis"

# A deliberately crude keyword filter, so that "the flag found something a
# keyword filter would not" can be checked rather than asserted.
SYC_MARKERS = [
    "great question", "excellent question", "good question", "what a great",
    "i'd be happy to", "i would be happy to", "happy to help", "absolutely!",
    "of course!", "certainly!", "you're absolutely right", "you are absolutely right",
    "you're right", "you are right", "i completely agree", "i totally agree",
    "that's a great", "that is a great", "wonderful", "fantastic", "amazing",
    "i apologize", "i apologise", "i'm sorry", "i am sorry", "my apologies",
    "thank you for", "thanks for sharing", "brilliant", "perfect!",
    "you make a good point", "you raise a good point", "i appreciate your",
]
MARKER_RE = re.compile("|".join(re.escape(m) for m in SYC_MARKERS), re.I)

# A response that opens with one of these is a refusal.  This is the post-hoc
# explanation of what the four alignment directions find in the tails, and it is
# reported as such: the direction score was computed with no knowledge of it, but
# a reader should know that once you know to look, a regex finds the same
# population.  The claim is that the score found it, not that only the score can.
REFUSAL_RE = re.compile(
    r"^\W{0,3}(i'?m (very |really |truly )?sorry|i am sorry|sorry[, ]|i can'?t|i cannot"
    r"|i can not|i won'?t|i'?m not able|i am not able|i'?m unable|i am unable"
    r"|i must decline|i will not)", re.I)
# REFUSAL_RE matches an OPENER, which is a formulaic refusal and not refusal as
# such: a response that begins "I understand you are in a difficult position.
# However, I **cannot** and **will not** assist" is a refusal that this pattern
# does not see.  REFUSAL_ANY looks for refusal language anywhere in the first 600
# characters, and the two together are what make the tails readable: the
# directions sort refusal-CONTAINING pairs into both tails, and then separate
# them by which half carries the formulaic opener.
REFUSAL_ANY = re.compile(
    r"(can'?t (help|assist|provide|comply)|cannot (help|assist|provide|comply|and will not)"
    r"|won'?t (help|assist|provide)|unable to (help|assist|provide)|must decline"
    r"|i will not assist|i'?m not able to|not going to (help|assist|provide))", re.I)


def describe(v):
    v = np.asarray(v, dtype=np.float64)
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "min": float(v.min()), "p01": float(np.percentile(v, 1)),
            "p05": float(np.percentile(v, 5)), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
            "max": float(v.max())}


def signflip_all(M, n=N_PERM, seed=SEED):
    """Two-sided sign-flip p for the mean of every column of M, in one pass.

    Done per column this would be 63 x 20,000 passes over 12,564 numbers.  The
    same 1,000 sign vectors serve every direction at once as a matrix product
    (1000 x n) @ (n x T), which is one 800 Mflop call per block instead of
    63,000 small ones.  The draws are shared across directions, which is fine:
    each column's p is still a valid Monte Carlo p for that column.
    """
    M = np.asarray(M, dtype=np.float64)
    nn, T = M.shape
    obs = np.abs(M.mean(0))
    rng = np.random.default_rng(seed)
    hit = np.zeros(T, dtype=np.int64)
    for a in range(0, n, 1000):
        k = min(1000, n - a)
        sgn = (rng.integers(0, 2, size=(k, nn)) * 2 - 1).astype(np.float64)
        hit += (np.abs(sgn @ M / nn) >= obs).sum(0)
    return (1 + hit) / (1 + n), hit


def holm(pairs):
    s = sorted(pairs, key=lambda kv: kv[1])
    m, out, run = len(s), {}, 0.0
    for i, (k, p) in enumerate(s):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def label_perm(M, labels, n_perm=N_PERM_LABEL, seed=SEED):
    """Between-group sum of squares of stratum means, and its permutation p.

    One shuffle of the row order plus `np.add.reduceat` over the fixed block
    boundaries is exactly a shuffle of the labels with the group sizes held, and
    it costs one pass over the matrix instead of one fancy-index per group.
    """
    M = np.asarray(M, dtype=np.float64)
    n, T = M.shape
    keys = sorted(set(labels))
    lab = np.array([keys.index(x) for x in labels])
    order = np.argsort(lab, kind="stable")
    sizes = np.array([int((lab == i).sum()) for i in range(len(keys))])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])

    def bss(rows):
        s = np.add.reduceat(rows, starts, axis=0)          # (k, T)
        mu = s / sizes[:, None]
        g = rows.mean(0)
        return (sizes[:, None] * (mu - g) ** 2).sum(0), mu   # (T,), (k, T)

    obs, mu = bss(M[order])
    rng = np.random.default_rng(seed)
    hit = np.zeros(T, dtype=np.int64)
    for _ in range(n_perm):
        b, _m = bss(M[rng.permutation(n)])
        hit += (b >= obs)
    return {"keys": keys, "sizes": sizes.tolist(), "group_means": mu,
            "bss": obs, "p": (1 + hit) / (1 + n_perm), "n_perm": n_perm}


def band_stats(M, idx, rand):
    v = np.array([M[:, idx[n]].mean() for n in rand])
    return {"directions": rand, "means": v.tolist(), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max()),
            "max_abs": float(np.abs(v).max())}, v


def per_direction(M, names, idx, rand, band, rv, paired):
    out, ps = {}, []
    P = H = None
    if paired:
        P, H = signflip_all(M)
    for n in names:
        j = idx[n]
        d = M[:, j]
        rec = {**describe(d), "frac_positive": float((d > 0).mean()),
               "z_vs_random_band": (float(d.mean()) - band["mean"]) / band["sd"],
               "n_random_at_least_as_large": int((np.abs(rv) >= abs(float(d.mean()))).sum())}
        rec["p_vs_band"] = (1 + rec["n_random_at_least_as_large"]) / (1 + len(rand))
        if paired:
            rec.update({"p_signflip": float(P[j]), "n_as_or_more_extreme": int(H[j]),
                        "n_draws": N_PERM})
            if not n.startswith(RAND_PREFIX):
                ps.append((n, float(P[j])))
        out[n] = rec
    if ps:
        for n, v in holm(ps).items():
            out[n]["p_holm"] = v
    return out


def subsets(M, names, idx, meta, ids, field, tag):
    lab = [meta[i][field] for i in ids]
    r = label_perm(M, lab)
    out = {"field": field, "keys": r["keys"], "sizes": r["sizes"],
           "n_perm": r["n_perm"], "statistic": "between-group sum of squares of the "
           "stratum means; null shuffles the stratum labels across items",
           "directions": {}}
    for n in names:
        j = idx[n]
        out["directions"][n] = {
            "means": {k: float(r["group_means"][a, j]) for a, k in enumerate(r["keys"])},
            "bss": float(r["bss"][j]), "p_label_perm": float(r["p"][j])}
    ph = holm([(n, out["directions"][n]["p_label_perm"]) for n in names
               if not n.startswith(RAND_PREFIX)])
    for n, v in ph.items():
        out["directions"][n]["p_holm"] = v
    return out


def main():
    S = json.load(open(f"{Q}/analysis/dolci_scores.json"))
    spec = json.load(open(f"{Q}/phase10_runs/dolci_targets.json"))
    IM = json.load(open(f"{Q}/phase10_runs/dolci_itemmeta.json"))
    meta, report = IM["items"], IM["report"]
    names = S["names"]
    idx = {n: i for i, n in enumerate(names)}
    rand = [n for n in names if n.startswith(RAND_PREFIX)]
    named = [n for n in names if not n.startswith(RAND_PREFIX)]

    rows = S["scores"]
    anch = [r for r in rows if r["trait"] == "agreeable"]
    dpo = [r for r in rows if r["trait"] == "dolci_dpo"]
    sft = [r for r in rows if r["trait"] == "dolci_sft"]
    print(f"{len(anch)} anchor, {len(dpo)} DPO, {len(sft)} SFT rows; "
          f"{len(names)} directions ({len(rand)} random)", flush=True)

    dids = [r["id"] for r in dpo]
    sids = [r["id"] for r in sft]
    P = np.array([r["pair"] for r in dpo])
    Ch = np.array([r["chosen"] for r in dpo])
    Cr = np.array([r["rejected"] for r in dpo])
    F = np.array([r["chosen"] for r in sft])

    # ---- per-item jsonl ------------------------------------------------
    for tag, rs in (("dpo", dpo), ("sft", sft)):
        with open(f"{Q}/phase10_runs/dolci_scores_{tag}.jsonl", "w") as f:
            for r in rs:
                rec = {"id": r["id"], **{k: v for k, v in meta[r["id"]].items()
                                         if k != "messages"},
                       "n_tok": r["n_tok"]}
                if tag == "dpo":
                    rec["pair"] = r["pair"]
                    rec["chosen_score"] = r["chosen"]
                    rec["rejected_score"] = r["rejected"]
                else:
                    rec["score"] = r["chosen"]
                f.write(json.dumps(rec) + "\n")

    common = {
        "scores": "analysis/dolci_scores.json",
        "targets": "phase10_runs/dolci_targets.json",
        "n_directions": len(names), "n_random_directions": len(rand),
        "a_drift_by_source": S.get("a_drift_by_source"),
        "elapsed_s": S.get("elapsed_s"), "peak_gb": S.get("peak_gb"),
        "revisions": report["revisions"],
        "tokenisation": report["tokenisation"],
        "prompt_cap": report["prompt_cap"], "completion_cap": report["completion_cap"],
        "score_units": "per-loss-token directional derivative of log-likelihood along a "
                       "unit-Frobenius weight direction; positive means the completion "
                       "trains the model toward that direction",
        "band_note": "the random band is 30 Gaussian merges of the 134 stage-one "
                     "adapters. For FA_*, axis_*, mean_assistant_axis and the single-trait "
                     "targets -- inside that span -- it is a matched null. For align_*, "
                     "stage2_shared and sorh_hack_minus_control it is a scale reference "
                     "only. p_vs_band floors at 1/31 = " + str(1 / 31),
    }

    # ---- DPO -------------------------------------------------------------
    band, rv = band_stats(P, idx, rand)
    bh, rvh = band_stats(Ch, idx, rand)
    br, rvr = band_stats(Cr, idx, rand)
    dpo_sum = {
        "meta": {**common, "set": "dpo", "n_items": len(dpo),
                 "allocation": report["dpo_allocation"],
                 "population": report["dpo_population"],
                 "filter": report["dpo_filter"],
                 "char_cap_dropped": report.get("dpo_char_cap_dropped"),
                 "permutation": f"two-sided sign-flip, {N_PERM} draws, seed {SEED}; "
                                f"floor p = {1 / (1 + N_PERM)}"},
        "random_band_pair": band, "random_band_chosen": bh, "random_band_rejected": br,
        "pair": per_direction(P, names, idx, rand, band, rv, True),
        "chosen": per_direction(Ch, names, idx, rand, bh, rvh, False),
        "rejected": per_direction(Cr, names, idx, rand, br, rvr, False),
        "by_preference_type": subsets(P, names, idx, meta, dids, "stratum", "dpo"),
        "by_rejected_model": subsets(P, names, idx, meta, dids, "rejected_model", "dpo"),
        "by_chosen_model": subsets(P, names, idx, meta, dids, "chosen_model", "dpo"),
    }

    # ---- SFT -------------------------------------------------------------
    bf, rvf = band_stats(F, idx, rand)
    sft_sum = {
        "meta": {**common, "set": "sft", "n_items": len(sft),
                 "allocation": report["sft_allocation"],
                 "population": report["sft_population"],
                 "filter": report["sft_filter"],
                 "char_cap_dropped": report.get("sft_char_cap_dropped"),
                 "permutation": "no sign-flip test: a raw directional derivative is not "
                                "symmetric about zero, so the random band is the null",
                 "caveat_tool_use": "the `Dolci Instruct Tool Use` stratum is rendered "
                                    "without its tool definitions -- the sampler keeps only "
                                    "role and content -- and its assistant turns that carry "
                                    "a tool call and no text are dropped as empty. Its "
                                    "register score is therefore the score of the text half "
                                    "of a tool-use turn, not of the turn, and should not be "
                                    "read as a register claim about that source."},
        "random_band": bf,
        "score": per_direction(F, names, idx, rand, bf, rvf, False),
        "by_source_dataset": subsets(F, names, idx, meta, sids, "stratum", "sft"),
        "by_domain": subsets(F, names, idx, meta, sids, "domain", "sft"),
    }

    # ---- the anchor cell -------------------------------------------------
    if anch:
        N = json.load(open(f"{Q}/analysis/nxn_scores.json"))
        nidx = {n: i for i, n in enumerate(N["names"])}
        nby = {r["id"]: r for r in N["scores"]}
        a = np.array([r["pair"][idx["trait_agreeable"]] for r in anch])
        b = np.array([nby[r["id"]]["pair"][nidx["trait_agreeable"]] for r in anch])
        anchor = {"n_items": len(anch), "target": "trait_agreeable",
                  "mean_here": float(a.mean()), "mean_nxn": float(b.mean()),
                  "pearson_r": float(np.corrcoef(a, b)[0, 1]),
                  "max_abs_diff": float(np.abs(a - b).max()),
                  "bu_norm_here": S["bu_norm"][idx["trait_agreeable"]],
                  "bu_norm_nxn": N["bu_norm"][nidx["trait_agreeable"]],
                  "note": "batching differs between the two runs (token budget here, "
                          "fixed batch 4 there), so bf16 reduction noise is expected at "
                          "about a part in a thousand"}
    else:
        anchor = None
    dpo_sum["anchor_cell"] = anchor

    json.dump(dpo_sum, open(f"{Q}/analysis/dolci_scores_dpo.json", "w"), indent=1,
              default=float)
    json.dump(sft_sum, open(f"{Q}/analysis/dolci_scores_sft.json", "w"), indent=1,
              default=float)

    # ---- flags -----------------------------------------------------------
    text = {}
    for tag in ("dpo", "sft"):
        for ln in open(f"{Q}/phase10_runs/dolci_sample_{tag}.jsonl"):
            r = json.loads(ln)
            text[r["id"]] = r

    def prompt_text(r):
        return "\n\n".join(f"[{m['role']}] {m['content']}" for m in r["messages"])

    rng = np.random.default_rng(SEED)

    def matched_random(pool_ids, flagged, key_len, key_src, k):
        """A random control of the same size, matched on stratum and prompt length.

        Greedy nearest-length draw without replacement inside each stratum, so the
        control has the same source mix and the same length profile as the flag.
        A flag that is only "long answers from Wildchat" would then show no
        difference against it, which is the point of having one.
        """
        by = {}
        fl = set(flagged)
        for i in pool_ids:
            if i not in fl:
                by.setdefault(key_src(i), []).append(i)
        for v in by.values():
            v.sort(key=key_len)
        out, used = [], set()
        for i in flagged[:k]:
            v = by.get(key_src(i), [])
            L = key_len(i)
            best, bd = None, None
            for c in v:
                if c in used:
                    continue
                d = abs(key_len(c) - L)
                if bd is None or d < bd:
                    best, bd = c, d
                if bd == 0:
                    break
            if best is not None:
                used.add(best)
                out.append(best)
        return out

    audit = {"meta": {**common, "tail_frac": TAIL_FRAC,
                      "flag_directions": FLAG_DIRECTIONS, "register": REGISTER,
                      "markers": SYC_MARKERS,
                      "marker_note": "a deliberately crude lexical filter; the point is "
                                     "to show the score's flag is not a re-derivation of "
                                     "it, not to be a good filter"},
             "dpo": {}, "sft": {}}

    def decile(vals):
        """Which tenth of the length distribution each item is in."""
        vals = np.asarray(vals, dtype=np.float64)
        return np.searchsorted(np.percentile(vals, np.arange(10, 100, 10)), vals)

    def length_stratified(ids, v, dec, k_per_bin):
        """Top k of each length decile: the same length profile as the sample.

        The score is a per-LOSS-TOKEN directional derivative, so its sampling
        variance goes as one over the completion's token count and a plain top-N
        is dominated by short completions -- flagged chosen responses average 353
        characters against the sample's 3,091.  Taking the top of each decile
        separately removes that by construction, and gives a flag whose length
        distribution is the corpus's own.  It is reported ALONGSIDE the plain
        tail, not instead of it: the plain tail is what a practitioner would
        actually compute, and its length bias is a fact about the statistic.
        """
        out = []
        for b in range(10):
            ii = np.where(dec == b)[0]
            out += [ids[int(j)] for j in ii[np.argsort(-v[ii])][:k_per_bin]]
        return out

    ntail = max(1, int(round(TAIL_FRAC * len(dpo))))
    dpo_len = np.array([meta[i]["n_tok_chosen"] for i in dids])
    dpo_dec = decile(dpo_len)
    dpo_decof = {i: int(dpo_dec[k]) for k, i in enumerate(dids)}
    flags = {}
    for d, sign in FLAG_DIRECTIONS.items():
        if d not in idx:
            continue
        v = sign * P[:, idx[d]]
        order = np.argsort(-v)
        top = [dids[i] for i in order[:max(400, ntail)]]
        bot = [dids[i] for i in order[::-1][:max(400, ntail)]]
        ctl = matched_random(dids, top,
                             lambda i: meta[i]["n_prompt_tok"],
                             lambda i: meta[i]["stratum"], 400)
        tlm = length_stratified(dids, v, dpo_dec, 40)
        clm = matched_random(dids, tlm, lambda i: meta[i]["n_tok_chosen"],
                             lambda i: (meta[i]["stratum"], dpo_decof[i]), 400)
        flags[d] = {"top": top, "bottom": bot, "control": ctl,
                    "top_length_stratified": tlm, "control_length_stratified": clm}
        pos = {i: float(v[k]) for k, i in enumerate(dids)}
        # Composition.  The top 400 is pooled across strata, and if it is almost
        # all `delta_learning` -- a 32b answer against a 0.6b one -- then the flag
        # is largely a capability gap and not a preference for a disposition.  The
        # histogram against the sample is what says which, and a stratum-restricted
        # arm is carried alongside so the judge can be asked the cleaner question.
        def hist(ids, field):
            h = {}
            for i in ids:
                h[meta[i][field]] = h.get(meta[i][field], 0) + 1
            return dict(sorted(h.items(), key=lambda kv: -kv[1]))
        base_str = hist(dids, "stratum")
        base_rej = hist(dids, "rejected_model")
        jj = [k for k, i in enumerate(dids) if meta[i]["stratum"] == "llm_judged"]
        jorder = sorted(jj, key=lambda k: -v[k])
        flags[d]["top_llm_judged"] = [dids[k] for k in jorder[:400]]
        flags[d]["control_llm_judged"] = matched_random(
            [dids[k] for k in jj], flags[d]["top_llm_judged"],
            lambda i: meta[i]["n_prompt_tok"], lambda i: meta[i]["rejected_model"], 400)
        audit["dpo"][d] = {
            "sign": sign, "n_tail": ntail,
            "composition_top400": {"stratum": hist(top[:400], "stratum"),
                                   "rejected_model": hist(top[:400], "rejected_model"),
                                   "chosen_model": hist(top[:400], "chosen_model")},
            "composition_sample": {"stratum": base_str, "rejected_model": base_rej},
            "llm_judged_only": {
                "n_pool": len(jj),
                "top400_mean": float(np.mean([v[k] for k in jorder[:400]])),
                "pool_mean": float(np.mean([v[k] for k in jj]))},
            "tail_top_1pct_mean": float(v[order[:ntail]].mean()),
            "tail_bottom_1pct_mean": float(v[order[::-1][:ntail]].mean()),
            "all_mean": float(v.mean()),
            "flag_vs_control": marker_compare(
                [text[i] for i in top[:400]], [text[i] for i in ctl], prompt_text),
            "flag_vs_control_length_stratified": marker_compare(
                [text[i] for i in tlm], [text[i] for i in clm], prompt_text),
            "length_stratified_mean": float(np.mean(
                [float(v[k]) for k, i in enumerate(dids) if i in set(tlm)])),
            "top20": [{"id": i, "score": pos[i], "stratum": meta[i]["stratum"],
                       "rejected_model": meta[i]["rejected_model"],
                       "chosen_model": meta[i]["chosen_model"]} for i in top[:20]],
        }

    vreg = -F[:, idx[REGISTER]]            # away from the assistant register
    order = np.argsort(-vreg)
    ntail_s = max(1, int(round(TAIL_FRAC * len(sft))))
    stop = [sids[i] for i in order[:400]]
    sctl = matched_random(sids, stop, lambda i: meta[i]["n_prompt_tok"],
                          lambda i: meta[i]["stratum"], 400)
    sft_len = np.array([meta[i]["n_tok"] for i in sids])
    sft_dec = decile(sft_len)
    sft_decof = {i: int(sft_dec[k]) for k, i in enumerate(sids)}
    stlm = length_stratified(sids, vreg, sft_dec, 40)
    sclm = matched_random(sids, stlm, lambda i: meta[i]["n_tok"],
                          lambda i: (meta[i]["stratum"], sft_decof[i]), 400)
    flags[REGISTER] = {"top": stop, "bottom": [sids[i] for i in order[::-1][:400]],
                       "control": sctl, "top_length_stratified": stlm,
                       "control_length_stratified": sclm}
    audit["sft"][REGISTER] = {
        "sign": -1, "n_tail": ntail_s,
        "tail_top_1pct_mean": float(vreg[order[:ntail_s]].mean()),
        "tail_bottom_1pct_mean": float(vreg[order[::-1][:ntail_s]].mean()),
        "all_mean": float(vreg.mean()),
        "flag_vs_control": marker_compare([text[i] for i in stop],
                                          [text[i] for i in sctl], prompt_text),
        "flag_vs_control_length_stratified": marker_compare(
            [text[i] for i in stlm], [text[i] for i in sclm], prompt_text),
        "top20": [{"id": i, "score": float(vreg[k]), "stratum": meta[i]["stratum"],
                   "domain": meta[i]["domain"]}
                  for k, i in [(int(j), sids[int(j)]) for j in order[:20]]],
    }
    # the negative-valence trait block, flag (b)
    NEG = [f"trait_{t}" for t in ("rude", "harsh", "cold", "unkind", "careless",
                                  "negligent", "crooked", "selfish")]
    NEG = [n for n in NEG if n in idx]
    neg = F[:, [idx[n] for n in NEG]].mean(1)
    o2 = np.argsort(-neg)
    audit["sft"]["negative_valence_block"] = {
        "directions": NEG, "all_mean": float(neg.mean()),
        "tail_top_1pct_mean": float(neg[o2[:ntail_s]].mean()),
        "top20": [{"id": sids[int(j)], "score": float(neg[int(j)]),
                   "stratum": meta[sids[int(j)]]["stratum"]} for j in o2[:20]],
    }
    flags["negative_valence_block"] = {
        "top": [sids[int(j)] for j in o2[:400]],
        "bottom": [sids[int(j)] for j in o2[::-1][:400]],
        "control": matched_random([sids[int(j)] for j in o2], [sids[int(j)] for j in o2[:400]],
                                  lambda i: meta[i]["n_prompt_tok"],
                                  lambda i: meta[i]["stratum"], 400)}

    # ---- refusal polarity, the thing the alignment tails turn out to find ----
    def polarity(ids):
        rc = cr = 0
        co = ro = ca = ra = both = 0
        for i in ids:
            t = text[i]
            r = bool(REFUSAL_RE.match(t["rejected"].strip()))
            c = bool(REFUSAL_RE.match(t["chosen"].strip()))
            rc += (r and not c)
            cr += (c and not r)
            co += c
            ro += r
            xa = bool(REFUSAL_ANY.search(t["chosen"][:600]))
            ya = bool(REFUSAL_ANY.search(t["rejected"][:600]))
            ca += xa
            ra += ya
            both += (xa and ya)
        n = max(len(ids), 1)
        return {"n": len(ids),
                "rejected_opens_formulaic_refusal_chosen_does_not": rc / n,
                "chosen_opens_formulaic_refusal_rejected_does_not": cr / n,
                "chosen_opens_formulaic_refusal": co / n,
                "rejected_opens_formulaic_refusal": ro / n,
                "chosen_refuses_anywhere": ca / n,
                "rejected_refuses_anywhere": ra / n,
                "both_refuse_anywhere": both / n}

    rnd400 = [dids[k] for k in rng.choice(len(dids), 400, replace=False)]
    audit["refusal_polarity"] = {
        "definition":
            "`*_opens_formulaic_refusal*` use REFUSAL_RE, which matches a stock "
            "refusal OPENER. `*_refuses_anywhere` use REFUSAL_ANY over the first "
            "600 characters and catch a refusal that is phrased at length. The "
            "difference matters: a pair counted under "
            "`rejected_opens_formulaic_refusal_chosen_does_not` is one where the "
            "rejected half opens with a stock refusal and the chosen half does "
            "not, which is NOT the same as the chosen half complying -- 17.5% of "
            "the power-seeking top 400's chosen halves refuse in some other "
            "wording. Read the two families together: both tails are enriched "
            "about tenfold in refusal-containing pairs, and separate by which "
            "half carries the formulaic opener.",
        "regex_opener": REFUSAL_RE.pattern,
        "regex_anywhere": REFUSAL_ANY.pattern,
        "whole_dpo_sample": polarity(dids),
        "random_400": polarity(rnd400),
        "by_flag": {d: {"top400": polarity(flags[d]["top"][:400]),
                        "bottom400": polarity(flags[d]["bottom"][:400]),
                        "top400_length_stratified":
                            polarity(flags[d]["top_length_stratified"])}
                    for d in FLAG_DIRECTIONS if d in flags},
    }
    json.dump(audit, open(f"{Q}/analysis/dolci_audit.json", "w"), indent=1, default=float)

    # ---- judge input, blind ---------------------------------------------
    JN = 60
    jitems = []
    for d in list(FLAG_DIRECTIONS) + [REGISTER]:
        if d not in flags:
            continue
        arms = [("flagged", flags[d]["top"][:JN]), ("control", flags[d]["control"][:JN])]
        if d == "align_sycophantic" and "top_llm_judged" in flags[d]:
            arms += [("flagged_llm_judged", flags[d]["top_llm_judged"][:JN]),
                     ("control_llm_judged", flags[d]["control_llm_judged"][:JN])]
        if d in ("align_sycophantic", REGISTER):
            arms += [("flagged_lenstrat", flags[d]["top_length_stratified"][:JN]),
                     ("control_lenstrat", flags[d]["control_length_stratified"][:JN])]
        for arm, ids in arms:
            for i in ids:
                r = text[i]
                if r["set"] == "dpo":
                    for half in ("chosen", "rejected"):
                        jitems.append({"key": f"{d}|{arm}|{i}|{half}", "direction": d,
                                       "arm": arm, "id": i, "half": half,
                                       "prompt": prompt_text(r), "response": r[half]})
                else:
                    jitems.append({"key": f"{d}|{arm}|{i}|chosen", "direction": d,
                                   "arm": arm, "id": i, "half": "chosen",
                                   "prompt": prompt_text(r), "response": r["chosen"]})
    json.dump(jitems, open(f"{Q}/phase10_runs/dolci_judge_items.json", "w"))
    print(f"judge items: {len(jitems)}")

    # ---- selections for stage two ---------------------------------------
    sel = {"built_by": "analyse_dolci_scores.py",
           "tokenisation": {**report["tokenisation"], "prompt_cap": report["prompt_cap"],
                            "completion_cap": report["completion_cap"],
                            "maxlen_rule": "n_prompt_tokens + completion_cap + 1, per item",
                            "eos_dropped_on_truncation": True,
                            "base_model": "Qwen/Qwen3.5-4B"},
           "revisions": report["revisions"],
           "scores": "phase10_runs/dolci_scores_{dpo,sft}.jsonl",
           "sets": {}}
    for d in ("align_sycophantic", "align_obsequious", "align_corrigible",
              "align_power_seeking"):
        if d not in flags:
            continue
        sel["sets"][f"dpo_{d}"] = {
            "set": "dpo", "direction": d,
            "criterion": ("pair score, chosen minus rejected, ranked in the "
                          f"{'negative' if FLAG_DIRECTIONS[d] < 0 else 'positive'} "
                          "direction as the undesirable one"),
            "top400": flags[d]["top"][:400], "bottom400": flags[d]["bottom"][:400],
            "random400_matched": flags[d]["control"][:400],
            "top400_length_stratified": flags[d]["top_length_stratified"],
            "random400_length_stratified": flags[d]["control_length_stratified"]}
    sel["sets"][f"sft_{REGISTER}"] = {
        "set": "sft", "direction": REGISTER,
        "criterion": "raw completion score; top400 = most negative (away from the "
                     "assistant register), bottom400 = most positive",
        "top400": flags[REGISTER]["top"][:400],
        "bottom400": flags[REGISTER]["bottom"][:400],
        "random400_matched": flags[REGISTER]["control"][:400],
        "top400_length_stratified": flags[REGISTER]["top_length_stratified"],
        "random400_length_stratified": flags[REGISTER]["control_length_stratified"]}
    # do the two alignment flags actually differ?
    if "align_sycophantic" in idx and "align_obsequious" in idx:
        a = set(flags["align_sycophantic"]["top"][:400])
        b = set(flags["align_obsequious"]["top"][:400])
        sel["syc_obs_overlap"] = {"n_shared_of_400": len(a & b),
                                  "pearson_r": float(np.corrcoef(
                                      P[:, idx["align_sycophantic"]],
                                      P[:, idx["align_obsequious"]])[0, 1])}
    json.dump(sel, open(f"{Q}/phase10_runs/dolci_selection.json", "w"), indent=1)

    # ---- console -----------------------------------------------------
    print("\nDPO pair, mean chosen-minus-rejected per direction:")
    w = max(len(n) for n in named)
    print(f"{'direction':<{w}} {'mean':>10} {'sd':>8} {'frac+':>6} {'p_flip':>9} "
          f"{'p_holm':>9} {'z_band':>8} {'#rand>=':>7}")
    for n in sorted(named, key=lambda n: -abs(dpo_sum["pair"][n]["mean"])):
        r = dpo_sum["pair"][n]
        print(f"{n:<{w}} {r['mean']:>+10.5f} {r['sd']:>8.4f} {r['frac_positive']:>6.3f} "
              f"{r['p_signflip']:>9.5f} {r.get('p_holm', float('nan')):>9.5f} "
              f"{r['z_vs_random_band']:>+8.2f} {r['n_random_at_least_as_large']:>7d}")
    print(f"\nrandom band (pair): mean {band['mean']:+.5f} sd {band['sd']:.5f} "
          f"max|mean| {band['max_abs']:.5f}")
    print("\nSFT raw score per direction:")
    for n in sorted(named, key=lambda n: -abs(sft_sum["score"][n]["mean"])):
        r = sft_sum["score"][n]
        print(f"{n:<{w}} {r['mean']:>+10.5f} {r['sd']:>8.4f} {r['frac_positive']:>6.3f} "
              f"{r['z_vs_random_band']:>+8.2f} {r['n_random_at_least_as_large']:>7d}")
    if anchor:
        print(f"\nanchor trait_agreeable: r={anchor['pearson_r']:.8f} "
              f"mean {anchor['mean_here']:+.5f} vs nxn {anchor['mean_nxn']:+.5f} "
              f"max|diff| {anchor['max_abs_diff']:.3e}")


def marker_compare(flag_rows, ctl_rows, prompt_text):
    """Is the flag just a keyword filter in disguise, or just a length filter?"""
    def stat(rows):
        out = {}
        for half in ("chosen", "rejected"):
            r = [x for x in rows if half in x]
            if not r:
                continue
            txt = [x[half] for x in r]
            out[half] = {
                "n": len(txt),
                "marker_rate": float(np.mean([bool(MARKER_RE.search(t)) for t in txt])),
                "markers_per_kchar": float(np.mean(
                    [1000 * len(MARKER_RE.findall(t)) / max(len(t), 1) for t in txt])),
                "mean_chars": float(np.mean([len(t) for t in txt])),
                "median_chars": float(np.median([len(t) for t in txt])),
            }
        out["mean_prompt_chars"] = float(np.mean([len(prompt_text(x)) for x in rows]))
        return out
    return {"flagged": stat(flag_rows), "control": stat(ctl_rows)}


if __name__ == "__main__":
    sys.exit(main())
