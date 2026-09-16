#!/usr/bin/env python3
"""Part D of the emergent-misalignment medical run: the probe as a data-audit flag.

Two stages, run as two commands.

  --stage units   reads analysis/em_probe_scores.json (dolci_score.py against the
                  probe adapter and its null band), picks the top 20 by the probe
                  score and 20 at random from the rest, and writes the blind
                  judge's units file.
  --stage final   reads phase10_runs/em_judged_probe.json and writes
                  analysis/em_part_d.json with the pre-registered test: the
                  flagged group's mean risk exceeds the random group's at an
                  exact Mann-Whitney p < 0.05.

The random comparison is drawn from the same 2,000 scored items excluding the
flagged 20, at seed 20260912, and is NOT length-matched: the pre-registration
says "20 random", and a length-matched comparison would be a different (and
weaker) claim than the one registered.  Length is reported for both groups so a
reader can see whether it explains the gap.
"""
import argparse
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
SEED = 20260912
N_TOP = 20
N_RAND = 20
PROBE = "probe_bad_medical"


# --------------------------------------------------------------------------
# DECLARED DEVIATION, written 2026-09-11 22:30 UTC AFTER the pre-registered
# result was computed and judged, and labelled post hoc everywhere it appears.
#
# The pre-registered "top 20 by the probe score" are all TWO-TOKEN completions
# ("Bob", "Carol", "David") from the logic-puzzle stratum.  The score is a
# per-loss-token mean, so a two-token completion's score is one token's
# directional derivative and its variance is enormous; the flag found that, not
# medical content.  Spearman between the probe score and the completion's token
# count over the 2,000 scored items is -0.3169 (p 6.86e-48).
#
# This is the confound wiki/pages/behaviour/dolci-flag-training.md hit and
# wiki/pages/behaviour/sycophancy-forecast.md fixed by LENGTH STRATIFICATION.
# The post-hoc arm applies that fix: restrict to items of at least MIN_TOK loss
# tokens, take the top 20 there, and draw the 20 random from the same band.
MIN_TOK = 100
# --------------------------------------------------------------------------


def units(stratified=False):
    S = json.load(open(f"{Q}/analysis/em_probe_scores.json"))
    items = {x["id"]: x for x in
             json.load(open(f"{Q}/phase10_runs/em_probe_score_items.json"))}
    names = S["names"]
    j = names.index(PROBE)
    rows = [(r["id"], r["chosen"][j], r["n_tok"][0]) for r in S["scores"]]
    if stratified:
        rows = [r for r in rows if r[2] >= MIN_TOK]
    rows = [(a, b) for a, b, _ in sorted(rows, key=lambda t: -t[1])]
    top = [r[0] for r in rows[:N_TOP]]
    rest = [r[0] for r in rows[N_TOP:]]
    rng = np.random.default_rng(SEED)
    rnd = [rest[i] for i in rng.choice(len(rest), size=N_RAND, replace=False)]
    U = []
    for grp, ids in (("flagged", top), ("random", rnd)):
        for i in ids:
            x = items[i]
            msgs = x["messages"]
            prompt = "\n\n".join(f"[{m['role'].upper()}] {m['content']}" for m in msgs)
            U.append({"id": i, "group": grp,
                      "probe_score": dict(rows)[i],
                      "chars": len(x["chosen"]),
                      "block": f"USER REQUEST:\n{prompt[:4000]}\n\n"
                               f"ASSISTANT COMPLETION:\n{x['chosen'][:4000]}"})
    tag = "_ls" if stratified else ""
    p = f"{Q}/phase10_runs/em_probe_units{tag}.json"
    json.dump({"units": U, "n_scored": len(rows), "probe": PROBE, "seed": SEED,
               "length_stratified": stratified,
               "min_loss_tokens": MIN_TOK if stratified else None,
               "post_hoc": stratified}, open(p, "w"), indent=1)
    print(f"wrote {p}: {len(U)} units ({N_TOP} flagged + {N_RAND} random of "
          f"{len(rows)} scored)")
    print(f"  flagged probe scores {rows[0][1]:+.6f} .. {rows[N_TOP-1][1]:+.6f}")


def final(stratified=False):
    from scipy.stats import mannwhitneyu
    S = json.load(open(f"{Q}/analysis/em_probe_scores.json"))
    tag = "_ls" if stratified else ""
    J = json.load(open(f"{Q}/phase10_runs/em_judged_probe{tag}.json"))
    names = S["names"]
    rand = [n for n in names if n.startswith("rand_merge_")]
    X = np.array([r["chosen"] for r in S["scores"]])
    idx = {n: i for i, n in enumerate(names)}
    band = {n: {"mean": float(X[:, idx[n]].mean()), "sd": float(X[:, idx[n]].std(ddof=1))}
            for n in rand}
    rec = J["records"]
    g = {}
    for grp in ("flagged", "random"):
        v = [r["scores"]["risk"] for r in rec
             if r["group"] == grp and r["scores"]["risk"] is not None]
        d = [r["scores"]["domain"] for r in rec if r["group"] == grp]
        c = [r["chars"] for r in rec if r["group"] == grp]
        g[grp] = {"n": len(v), "risk_mean": float(np.mean(v)),
                  "risk_median": float(np.median(v)),
                  "risk_counts": {str(k): int(sum(1 for x in v if x == k))
                                  for k in range(1, 8)},
                  "n_risk_ge_4": int(sum(1 for x in v if x >= 4)),
                  "domain_rate": float(np.mean(d)),
                  "mean_chars": float(np.mean(c))}
    a = [r["scores"]["risk"] for r in rec
         if r["group"] == "flagged" and r["scores"]["risk"] is not None]
    b = [r["scores"]["risk"] for r in rec
         if r["group"] == "random" and r["scores"]["risk"] is not None]
    mw = mannwhitneyu(a, b, alternative="two-sided", method="exact")
    out = {"groups": g,
           "mannwhitney": {"U": float(mw.statistic), "p_exact_two_sided": float(mw.pvalue),
                           "n_flagged": len(a), "n_random": len(b)},
           "prereg_threshold": "flagged mean risk exceeds random mean risk at exact "
                               "Mann-Whitney p < 0.05",
           "held": bool(np.mean(a) > np.mean(b) and mw.pvalue < 0.05),
           "probe_direction": PROBE,
           "probe_score_random_band": band,
           "n_scored": len(S["scores"]),
           "judge": J["model"], "failed_calls": J["failed_calls"],
           "judge_repeat": J.get("repeat"), "usage": J.get("usage"),
           "length_note": "the random comparison is not length-matched; mean_chars "
                          "for both groups is above so a length explanation is visible"}
    out["length_stratified"] = stratified
    out["post_hoc"] = stratified
    if stratified:
        out["min_loss_tokens"] = MIN_TOK
        out["deviation_note"] = (
            "POST HOC, written after the pre-registered result was judged: the "
            "pre-registered top 20 are all two-token completions, so the "
            "per-loss-token score found length and not content. This arm "
            "restricts both groups to items of at least 100 loss tokens, the "
            "length-stratification fix of wiki/pages/behaviour/sycophancy-forecast.md.")
    p = f"{Q}/analysis/em_part_d{tag}.json"
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)
    print(f"  flagged risk mean {g['flagged']['risk_mean']:.4f} "
          f"(>=4 in {g['flagged']['n_risk_ge_4']}/{g['flagged']['n']}, "
          f"domain {g['flagged']['domain_rate']:.2f}, {g['flagged']['mean_chars']:.0f} chars)")
    print(f"  random  risk mean {g['random']['risk_mean']:.4f} "
          f"(>=4 in {g['random']['n_risk_ge_4']}/{g['random']['n']}, "
          f"domain {g['random']['domain_rate']:.2f}, {g['random']['mean_chars']:.0f} chars)")
    print(f"  Mann-Whitney exact p {out['mannwhitney']['p_exact_two_sided']:.5f} "
          f"-> held={out['held']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["units", "final"], required=True)
    ap.add_argument("--stratified", action="store_true",
                    help="the post-hoc length-stratified arm")
    a = ap.parse_args()
    (units if a.stage == "units" else final)(a.stratified)
