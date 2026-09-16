#!/usr/bin/env python3
"""Did School-of-Reward-Hacks SFT change judged personality, and does the hack
arm differ from its matched honest control?

The weight-space answer is in analyse_sorh.py / analysis/sorh_projection.json:
the hack arm is a 1.08x larger update pointing 77 degrees away from the control,
but both land at ~1% of a trait adapter's chart length. That is a statement
about a projection, not about behaviour. This script asks the behavioural
question the projection cannot: on the same 24 personality prompts the steering
sweep used, judged blind on the Big Five by a different model family, do the two
arms differ from each other, and does either differ from the base model?

Design notes that the numbers depend on:

* PAIRED, not pooled. Both arms answer the SAME 24 prompts, and every condition
  in both files was generated from the same base weights with the same greedy
  decode. So the unit of analysis is the per-prompt difference, and the null is
  a sign flip on those 24 differences -- exactly the exchangeability the design
  buys. Pooling the 24 scores into two independent samples would throw away the
  pairing and inflate the variance.
* EXACT permutation. 24 paired differences means 2^24 = 16,777,216 sign
  assignments, which is small enough to enumerate. No sampling error in p.
  Zero differences are left in: flipping a zero changes nothing, so ties make
  the test conservative rather than liberal, which is the direction we want for
  a result that is going to be read as a null.
* BASE COMES FROM THE SAME FILE. Each arm's run generated the base model on the
  same prompts in its own container. Using an arm's own base row keeps the
  arm-minus-base contrast inside one run. judged_100.json's base rows are on the
  same 24 prompts but are DIFFERENT generations (checked, not assumed), so they
  are reported only as a judge-noise reference, never as this experiment's base.
* THE JUDGE NEVER SAW THE ARM. judge_personas.py is handed (prompt, response)
  pairs only; `trait` (sorh_hack / sorh_control) stays local to the file.
"""
import argparse
import json
import os
import re
import statistics as st

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
FACTORS = ["Extraversion", "Agreeableness", "Conscientiousness",
           "EmotionalStability", "Intellect"]

# The same leak pattern judge_personas.py strips before judging, so the text
# statistics below describe the text the judge actually scored.
LEAK = re.compile(r"(\buser\b.{0,80}\bassistant\b|<think>|<\|im_start\|>|<\|im_end\|>)", re.S)

HEDGE = ["might", "may ", "perhaps", "maybe", "possibly", "i think", "i believe",
         "it depends", "generally", "typically", "usually", "somewhat", "arguably",
         "could be", "tend to", "in my opinion", "probably", "seems", "not sure"]
SYCO = ["great question", "good question", "happy to help", "glad to help",
        "absolutely", "of course", "you're right", "you are right", "certainly",
        "great point", "excellent", "thanks for asking", "i'd be happy",
        "i would be happy", "that's a great", "wonderful"]


def clean(t):
    return LEAK.split(t)[0].strip()


def exact_signflip_p_multi(D):
    """Two-sided p for mean(d)=0 under sign flips, for many tests at once.

    D is (n_prompts, n_tests); every test must share the same n, which the
    paired design guarantees. All 2^n sign assignments are enumerated once and
    applied to every test in the same pass -- the enumeration is the expensive
    part and it does not depend on the data, so running the ~50 factor-by-
    contrast tests separately would repeat it ~50 times for nothing.
    """
    D = np.asarray(D, dtype=np.float64)
    n, m = D.shape
    obs = np.abs(D.sum(0))
    total = 1 << n
    ge = np.zeros(m, dtype=np.int64)
    step = 1 << 18
    bit = np.arange(n, dtype=np.int64)
    for start in range(0, total, step):
        idx = np.arange(start, min(start + step, total), dtype=np.int64)
        signs = (((idx[:, None] >> bit) & 1) * 2 - 1).astype(np.float64)
        ge += (np.abs(signs @ D) >= obs - 1e-9).sum(0)
    return ge / total, total, ge


def cell(records, trait, cond):
    """Per-prompt judged scores for one (arm, condition). Returns {factor: [24]}."""
    by = {}
    for r in records:
        if r["trait"] == trait and r["condition"] == cond:
            by[r["prompt_idx"]] = r["scores"]
    idx = sorted(by)
    return idx, {f: [by[i].get(f) for i in idx] for f in FACTORS}


def profile(idx, sc):
    out = {}
    for f in FACTORS:
        v = [x for x in sc[f] if x is not None]
        out[f] = {"mean": st.mean(v), "sd": st.pstdev(v) if len(v) > 1 else 0.0,
                  "n": len(v)}
    return out


def contrast(records, ta, ca, tb, cb):
    """(ta,ca) minus (tb,cb), paired over prompts. p filled in later, in one pass."""
    ia, sa = cell(records, ta, ca)
    ib, sb = cell(records, tb, cb)
    shared = sorted(set(ia) & set(ib))
    out = {"a": f"{ta}/{ca}", "b": f"{tb}/{cb}", "n_prompts_paired": len(shared)}
    for f in FACTORS:
        ma = {i: v for i, v in zip(ia, sa[f])}
        mb = {i: v for i, v in zip(ib, sb[f])}
        d = [ma[i] - mb[i] for i in shared if ma[i] is not None and mb[i] is not None]
        out[f] = {"mean_diff": st.mean(d), "sd_diff": st.pstdev(d) if len(d) > 1 else 0.0,
                  "n": len(d), "_d": d}
    return out


def contrast_pooled(records, ta, tb, conds):
    """(ta) minus (tb), averaged over `conds` within each prompt, then paired.

    Twenty tests (4 conditions x 5 factors) at alpha 0.05 will hand back one
    "significant" result on pure noise, and the base-versus-base row below shows
    the judge alone moves a mean by up to ~0.3. Averaging the three matched
    checkpoints into one number per prompt asks the single question the design
    was built for -- does the hack arm differ from its control -- at five tests
    instead of fifteen, and it uses the fact that 31/62/93 are the same training
    step in both arms because both arms have n=973 and the same schedule.
    """
    out = {"a": ta, "b": tb, "conditions_averaged": list(conds)}
    per = {}
    for arm in (ta, tb):
        for c in conds:
            i, sc = cell(records, arm, c)
            per[(arm, c)] = (i, sc)
    idx = sorted(set.intersection(*[set(per[k][0]) for k in per]))
    out["n_prompts_paired"] = len(idx)
    for f in FACTORS:
        d = []
        for i in idx:
            va, vb = [], []
            for c in conds:
                ia, sa = per[(ta, c)]
                ib, sb = per[(tb, c)]
                xa = dict(zip(ia, sa[f])).get(i)
                xb = dict(zip(ib, sb[f])).get(i)
                if xa is not None and xb is not None:
                    va.append(xa); vb.append(xb)
            if va:
                d.append(st.mean(va) - st.mean(vb))
        out[f] = {"mean_diff": st.mean(d), "sd_diff": st.pstdev(d) if len(d) > 1 else 0.0,
                  "n": len(d), "_d": d}
    return out


def _rank(x):
    """Average ranks, ties shared. Avoids a scipy dependency this repo does not use."""
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    sx, sy = st.pstdev(rx), st.pstdev(ry)
    if sx == 0 or sy == 0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (len(rx) * sx * sy)


def length_contrasts(gens, arms, conds, ckpts):
    """Paired sign-flip on cleaned response length, the same machinery as the
    judged factors.

    Both arms collapse from ~305 words to ~70 after fine-tuning, and a four-times
    shorter answer demonstrates less to a judge asked to rate what the text shows.
    So length is tested as a variable in its own right rather than left as a
    caption under the factor table.
    """
    L = {(a, c): [len(clean(t)) for t in gens[a]["generations"][c]]
         for a in arms for c in conds[a]}
    tests, keys = [], []
    for c in ["base"] + ckpts:
        keys.append(("hack_minus_control", c))
        tests.append([h - k for h, k in zip(L[("sorh_hack", c)], L[("sorh_control", c)])])
    for a in arms:
        for c in ckpts:
            keys.append((f"{a}_minus_base", c))
            tests.append([x - b for x, b in zip(L[(a, c)], L[(a, "base")])])
    ps, nperm, ge = exact_signflip_p_multi(np.array(tests).T)
    out = {}
    for (blk, c), d, pv, g in zip(keys, tests, ps, ge):
        out.setdefault(blk, {})[c] = {
            "mean_diff_chars": round(st.mean(d), 4),
            "sd_diff_chars": round(st.pstdev(d), 4),
            "n": len(d), "p_signflip_exact": round(float(pv), 6),
            "n_as_or_more_extreme": int(g), "n_permutations": nperm}
    return out


def holm(block):
    """Holm-Bonferroni across the 5 factors x conditions of one contrast block."""
    tests = [(c, f) for c in block for f in FACTORS if isinstance(block[c], dict)
             and f in block[c]]
    ps = sorted(tests, key=lambda k: block[k[0]][k[1]]["p_signflip_exact"])
    m = len(ps)
    run = 0.0
    for r, (c, f) in enumerate(ps):
        adj = min(1.0, (m - r) * block[c][f]["p_signflip_exact"])
        run = max(run, adj)
        block[c][f]["p_holm"] = run
    return m


def fill_ps(blocks):
    """Attach exact sign-flip p to every factor of every contrast.

    Grouped by the number of paired prompts: two records in judged_sorh.json
    have a null EmotionalStability (the judge omitted the field on those pairs),
    so a handful of tests are paired on 22 or 23 prompts rather than 24. Each
    distinct n gets its own exact enumeration of 2^n sign flips; nothing is
    imputed and no test borrows another's null.
    """
    by_n = {}
    for blk in blocks:
        for c, d in blk.items():
            for f in FACTORS:
                col = d[f].pop("_d")
                by_n.setdefault(len(col), []).append(((d, f), col))
    for n, entries in sorted(by_n.items()):
        ps, nperm, ge = exact_signflip_p_multi(np.array([c for _, c in entries]).T)
        for ((d, f), _), pv, g in zip(entries, ps, ge):
            d[f]["p_signflip_exact"] = float(pv)
            d[f]["n_as_or_more_extreme"] = int(g)
            d[f]["n_permutations"] = nperm


def text_stats(gens):
    """Plain-text description of what an arm/condition actually wrote."""
    raw = gens
    cl = [clean(t) for t in raw]
    def rate(words, texts):
        return sum(sum(w in t.lower() for w in words) for t in texts) / max(len(texts), 1)
    toks = [t.lower().split() for t in cl]
    uniq = [len(set(x)) / len(x) if x else 0.0 for x in toks]
    return {
        "n": len(raw),
        "mean_chars_raw": st.mean([len(t) for t in raw]),
        "mean_chars_cleaned": st.mean([len(t) for t in cl]),
        "mean_words_cleaned": st.mean([len(x) for x in toks]),
        "leak_rate": sum(bool(LEAK.search(t)) for t in raw) / max(len(raw), 1),
        "unique_token_ratio_mean": st.mean(uniq),
        "hedge_markers_per_response": rate(HEDGE, cl),
        "syco_markers_per_response": rate(SYCO, cl),
    }


def _round(o):
    """Round in the source, so the wiki can quote the JSON verbatim."""
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, float):
                o[k] = round(v, 6 if k.startswith("p_") else 4)
            else:
                _round(v)
    elif isinstance(o, list):
        for v in o:
            _round(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default=f"{Q}/phase10_runs/judged_sorh.json")
    ap.add_argument("--judged100", default=f"{Q}/phase10_runs/judged_100.json")
    ap.add_argument("--out", default=f"{Q}/analysis/sorh_behavioural.json")
    a = ap.parse_args()

    J = json.load(open(a.judged))
    R = J["records"]
    arms = ["sorh_hack", "sorh_control"]
    gens = {arm: json.load(open(f"{Q}/phase10_runs/rl_persona_{arm}.json")) for arm in arms}
    conds = {arm: sorted(gens[arm]["generations"],
                         key=lambda c: -1 if c == "base" else int(c.split("-")[1]))
             for arm in arms}
    ckpts = [c for c in conds[arms[0]] if c != "base"
             if c in conds[arms[1]]]

    res = {
        "meta": {
            "judge_model": J["model"], "judged_records": J["n"],
            "failed_calls": J["failed_calls"],
            "judged_file": os.path.relpath(a.judged, os.path.dirname(Q)),
            "generation_files": {arm: f"qwen35/phase10_runs/rl_persona_{arm}.json"
                                 for arm in arms},
            "n_prompts": len(gens[arms[0]]["prompts"]),
            "max_new_tokens": gens[arms[0]]["max_new_tokens"],
            "conditions": conds,
            "matched_checkpoints": ckpts,
            "permutation": "exact enumeration of all 2^n sign flips of the n "
                           "paired per-prompt differences (n = 24 except where the "
                           "judge omitted a factor); two-sided",
            "prompts_identical_across_arms":
                gens["sorh_hack"]["prompts"] == gens["sorh_control"]["prompts"],
            "base_generations_identical_across_arms":
                gens["sorh_hack"]["generations"]["base"]
                == gens["sorh_control"]["generations"]["base"],
        },
        "profiles": {},
        "contrasts": {},
        "text": {},
    }

    for arm in arms:
        for c in conds[arm]:
            idx, sc = cell(R, arm, c)
            res["profiles"][f"{arm}/{c}"] = profile(idx, sc)
            res["text"][f"{arm}/{c}"] = text_stats(gens[arm]["generations"][c])

    # hack minus control at each matched checkpoint, and base against base
    res["contrasts"]["hack_minus_control"] = {
        c: contrast(R, "sorh_hack", c, "sorh_control", c)
        for c in ["base"] + ckpts}
    # each arm minus its own base
    for arm in arms:
        res["contrasts"][f"{arm}_minus_base"] = {
            c: contrast(R, arm, c, arm, "base") for c in conds[arm] if c != "base"}
    res["contrasts"]["hack_minus_control_pooled_checkpoints"] = {
        "pooled": contrast_pooled(R, "sorh_hack", "sorh_control", ckpts)}
    fill_ps(list(res["contrasts"].values()))
    res["meta"]["multiplicity"] = {
        name: {"n_tests": holm(blk), "method": "Holm-Bonferroni within the block"}
        for name, blk in res["contrasts"].items()}

    # judge-noise reference. judged_100.json holds 100 judgments of ONE set of
    # base generations on these same 24 prompts (checked identical across the
    # 100 traits) -- but they are different generations from the ones in these
    # files, so this is a reference for how much the judge alone moves, not a
    # base for the contrasts above.
    if os.path.exists(a.judged100):
        K = json.load(open(a.judged100))["records"]
        per = {}
        for r in K:
            if r["condition"] == "base":
                per.setdefault(r["prompt_idx"], []).append(r["scores"])
        noise = {}
        for f in FACTORS:
            sds = [st.pstdev([s[f] for s in v if s.get(f) is not None])
                   for v in per.values() if len(v) > 1]
            allv = [s[f] for v in per.values() for s in v if s.get(f) is not None]
            noise[f] = {"mean_within_prompt_sd": st.mean(sds),
                        "grand_mean": st.mean(allv), "n": len(allv)}
        res["judge_noise_reference_judged_100_base"] = {
            "note": "100 independent judgments of the same 24 base generations "
                    "(eval_100traits.json base rows are identical across traits); "
                    "those generations differ from the base rows in the sorh files, "
                    "so this bounds judge noise only",
            "n_repeats_per_prompt": max(len(v) for v in per.values()),
            "factors": noise,
        }

    res["length"] = length_contrasts(gens, arms, conds, ckpts)

    # Does the judge's factor score track how much text it was given? If it does,
    # the Intellect and Conscientiousness drops that both arms share are at least
    # partly a statement about answer length, not about personality.
    lens, scores = [], {f: [] for f in FACTORS}
    for r in R:
        g = gens[r["trait"]]["generations"][r["condition"]][r["prompt_idx"]]
        n = len(clean(g))
        for f in FACTORS:
            if r["scores"].get(f) is not None:
                lens.append(n) if f == FACTORS[0] else None
        for f in FACTORS:
            if r["scores"].get(f) is not None:
                scores[f].append((n, r["scores"][f]))
    res["length_vs_factor_spearman"] = {
        "note": "over all judged records (arm x condition x prompt), cleaned "
                "response length in characters against the judged factor score",
        "n_records": len(R),
        "rho": {f: round(spearman([x for x, _ in scores[f]], [y for _, y in scores[f]]), 4)
                for f in FACTORS},
        "n": {f: len(scores[f]) for f in FACTORS},
    }

    _round(res)
    json.dump(res, open(a.out, "w"), indent=1)

    # readable summary
    print(f"judge={J['model']}  records={J['n']}  failed_calls={J['failed_calls']}")
    print(f"base texts identical across arms: "
          f"{res['meta']['base_generations_identical_across_arms']}")
    for name, blk in res["contrasts"].items():
        print(f"\n== {name} ==")
        for c, d in blk.items():
            line = "  ".join(f"{f[:4]} {d[f]['mean_diff']:+.3f} p={d[f]['p_signflip_exact']:.3f}"
                             f"/{d[f]['p_holm']:.3f}" for f in FACTORS)
            print(f"  {c:14s} {line}")
    print("\n== text ==")
    for k, t in res["text"].items():
        print(f"  {k:26s} chars={t['mean_chars_cleaned']:7.1f} words={t['mean_words_cleaned']:6.1f} "
              f"leak={t['leak_rate']:.2f} utr={t['unique_token_ratio_mean']:.3f} "
              f"hedge={t['hedge_markers_per_response']:.2f} syco={t['syco_markers_per_response']:.2f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
