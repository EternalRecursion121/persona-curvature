#!/usr/bin/env python3
"""What does the School of Reward Hacks data push toward, at first order?

Input:  analysis/sorh_data_scores.json  (align_score.py --stage score)
        phase10_runs/sorh_ds_targets.json, phase10_runs/sorh_ds_itemmeta.json
        analysis/nxn_scores.json         (for the two anchor cells)
Output: analysis/sorh_data_scoring.json

The unit of measurement.  align_score.py returns, per item and per direction,
    score = d/d(eps) [ mean-per-loss-token log p(completion | W + eps*U) ] at eps=0
with U the unit-Frobenius weight direction.  Positive means "a model steered
along +U finds this completion more likely", which by the identity in that file
means "training on this completion pushes the weights along +U".  The division
by the number of loss tokens (`g = -eps.grad / n` in align_score.py) makes it a
per-token rate, so a long completion does not outscore a short one for being
long; that is the same normalisation the mean-reduction Trainer used.

Three questions, three tests.

1. Paired difference.  Each of the 973 rows gives one hack completion and one
   control completion for the SAME prompt.  `pair` = hack - control.  A
   sign-flip permutation test on the mean (20,000 draws; 2^973 cannot be
   enumerated) asks whether that difference is distinguishable from zero.
2. Is it personality?  Twenty Gaussian merges of the same 134 zoo adapters are
   scored in the same run.  Every direction is unit-normalised, so their mean
   paired differences are directly comparable and form a null BAND for "a
   direction of this family, chosen without reference to the data".  A named
   direction only carries content if it leaves that band.  This is the test that
   matters: a sign-flip p can be tiny for a difference that any random direction
   in the space would show just as strongly.
3. Raw level.  The paired difference says nothing about where either completion
   sits.  Both completions' raw scores are reported per direction, because the
   interesting question for the assistant register is whether BOTH halves of this
   dataset push away from it.
"""
import json
import math
import os
import sys

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
N_PERM = 20000
PERM_SEED = 11
RAND_PREFIX = "rand_merge_"
ANCHORS = {"trait_agreeable": "agreeable", "trait_rude": "rude"}


def signflip_p(x, n=N_PERM, seed=PERM_SEED):
    """Two-sided sign-flip permutation p for the mean of paired differences.

    Under the null the sign of each pair's difference is exchangeable.  973
    pairs means 2^973 assignments, so this is a Monte Carlo test and says so:
    the p is (1 + #{|mean of a draw| >= |observed mean|}) / (1 + n), which is
    floored at 1/(n+1) = 4.999750012499375e-05.
    """
    x = np.asarray(x, dtype=np.float64)
    obs = abs(float(x.mean()))
    rng = np.random.default_rng(seed)
    hit = 0
    for a in range(0, n, 1000):
        k = min(1000, n - a)
        s = rng.integers(0, 2, size=(k, x.size)) * 2 - 1
        hit += int((np.abs((s * x).mean(1)) >= obs).sum())
    return (1 + hit) / (1 + n), hit


def describe(v):
    v = np.asarray(v, dtype=np.float64)
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "min": float(v.min()), "p05": float(np.percentile(v, 5)),
            "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def holm(pairs):
    """Holm-Bonferroni over [(name, p)]; returns {name: p_holm}."""
    s = sorted(pairs, key=lambda kv: kv[1])
    m, out, run = len(s), {}, 0.0
    for i, (k, p) in enumerate(s):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def main():
    S = json.load(open(f"{Q}/analysis/sorh_data_scores.json"))
    spec = json.load(open(f"{Q}/phase10_runs/sorh_ds_targets.json"))
    names = S["names"]
    idx = {n: i for i, n in enumerate(names)}
    rows = S["scores"]
    sorh = [r for r in rows if r["trait"] == "sorh"]
    anch = [r for r in rows if r["trait"] != "sorh"]
    assert len(sorh) == 973, len(sorh)

    P = np.array([r["pair"] for r in sorh])        # (973, T) hack - control
    Ch = np.array([r["chosen"] for r in sorh])     # (973, T) hack
    Cr = np.array([r["rejected"] for r in sorh])   # (973, T) control
    ntok = np.array([r["n_tok"] for r in sorh])    # (973, 2)

    rand = [n for n in names if n.startswith(RAND_PREFIX)]
    named = [n for n in names if not n.startswith(RAND_PREFIX)]
    rmean = np.array([P[:, idx[n]].mean() for n in rand])
    band = {"directions": rand, "means": rmean.tolist(),
            "mean": float(rmean.mean()), "sd": float(rmean.std(ddof=1)),
            "min": float(rmean.min()), "max": float(rmean.max()),
            "max_abs": float(np.abs(rmean).max()),
            "note": "mean paired difference (hack - control) along 20 Gaussian merges "
                    "of the 134 stage-one adapters, all unit-Frobenius like every "
                    "other direction, so directly comparable"}

    # The same band for the RAW per-completion scores.  A raw directional
    # derivative has no reason to be symmetric about zero, so a sign-flip test
    # against zero would be testing the wrong null; a Gaussian merge of the zoo
    # is, by construction, a direction the data has no reason to prefer, and the
    # spread of its raw means is the scale a named direction has to beat.  For
    # directions inside the zoo's span (FA_*, axis_*, mean_assistant_axis, the
    # single-trait anchors) this is a matched null; for the alignment, SoRH and
    # stage-two targets, which are not in that span, it is a scale reference only.
    rawband = {}
    for k, M in (("hack", Ch), ("control", Cr)):
        v = np.array([M[:, idx[n]].mean() for n in rand])
        rawband[k] = {"means": v.tolist(), "mean": float(v.mean()),
                      "sd": float(v.std(ddof=1)), "max_abs": float(np.abs(v).max())}

    per = {}
    ps = []
    for n in names:
        j = idx[n]
        p, hit = signflip_p(P[:, j])
        d = P[:, j]
        z = (float(d.mean()) - band["mean"]) / band["sd"]
        beat = int((np.abs(rmean) >= abs(float(d.mean()))).sum())
        per[n] = {
            "paired_diff": {**describe(d),
                            "frac_positive": float((d > 0).mean()),
                            "p_signflip": p, "n_as_or_more_extreme": hit,
                            "n_draws": N_PERM},
            "vs_random_band": {"z": z, "n_random_at_least_as_large": beat,
                               "p_vs_band": (1 + beat) / (1 + len(rand))},
            "raw_hack": describe(Ch[:, j]),
            "raw_control": describe(Cr[:, j]),
            "raw_vs_random_band": {
                k: {"z": (float(M[:, j].mean()) - rawband[k]["mean"]) / rawband[k]["sd"],
                    "n_random_at_least_as_large":
                        int((np.abs(np.array(rawband[k]["means"]))
                             >= abs(float(M[:, j].mean()))).sum())}
                for k, M in (("hack", Ch), ("control", Cr))},
            # the Trainer weighted every loss token equally, so the aggregate push
            # of the corpus is the token-weighted sum, not the per-prompt mean of
            # per-token rates; both readings are here because they can disagree
            "token_weighted_mean": {
                "hack": float((Ch[:, j] * ntok[:, 0]).sum() / ntok[:, 0].sum()),
                "control": float((Cr[:, j] * ntok[:, 1]).sum() / ntok[:, 1].sum())},
            "bu_norm": S["bu_norm"][j],
        }
        if not n.startswith(RAND_PREFIX):
            ps.append((n, p))
    H = holm(ps)
    for n, v in H.items():
        per[n]["paired_diff"]["p_holm"] = v

    # ---- the anchor cells ------------------------------------------------
    N = json.load(open(f"{Q}/analysis/nxn_scores.json"))
    nidx = {n: i for i, n in enumerate(N["names"])}
    nby = {r["id"]: r for r in N["scores"]}
    anchor = {}
    for tgt, trait in ANCHORS.items():
        here = [r for r in anch if r["trait"] == trait]
        a = np.array([r["pair"][idx[tgt]] for r in here])
        b = np.array([nby[r["id"]]["pair"][nidx[tgt]] for r in here])
        anchor[tgt] = {
            "n_items": len(here), "trait": trait,
            "mean_here": float(a.mean()), "mean_nxn": float(b.mean()),
            "pearson_r": float(np.corrcoef(a, b)[0, 1]),
            "max_abs_diff": float(np.abs(a - b).max()),
            "max_rel_diff": float((np.abs(a - b) / np.abs(b)).max()),
            "bu_norm_here": S["bu_norm"][idx[tgt]],
            "bu_norm_nxn": N["bu_norm"][nidx[tgt]],
        }

    # ---- the assistant register ------------------------------------------
    register = {}
    for n in ("stage2_shared", "mean_assistant_axis"):
        j = idx[n]
        register[n] = {
            "hack": {**describe(Ch[:, j]), "frac_positive": float((Ch[:, j] > 0).mean()),
                     "z_vs_raw_random_band": per[n]["raw_vs_random_band"]["hack"]["z"]},
            "control": {**describe(Cr[:, j]), "frac_positive": float((Cr[:, j] > 0).mean()),
                        "z_vs_raw_random_band": per[n]["raw_vs_random_band"]["control"]["z"]},
            "paired_diff_mean": float(P[:, j].mean()),
            "reading": "a raw score is a directional derivative, not a difference, so the "
                       "reference is the spread of the same statistic over the 20 random "
                       "merges (random_raw_band), not zero",
        }

    # ---- by cheat method, purely descriptive -----------------------------
    meta = json.load(open(f"{Q}/phase10_runs/sorh_ds_itemmeta.json"))["items"]
    groups = {}
    for i, r in enumerate(sorh):
        groups.setdefault(meta[r["id"]]["cheat_method"] or "unlabelled", []).append(i)
    bycheat = {}
    for k, ii in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]:
        bycheat[k] = {"n": len(ii),
                      "hack_minus_control_on_sorh_direction":
                          float(P[np.array(ii), idx["sorh_hack_minus_control"]].mean())}

    out = {
        "meta": {
            "scores": "analysis/sorh_data_scores.json",
            "targets": "phase10_runs/sorh_ds_targets.json",
            "n_sorh_pairs": len(sorh), "n_anchor_pairs": len(anch),
            "n_directions": len(names), "n_random_directions": len(rand),
            "a_drift": S["a_drift"], "a_drift_by_source": S.get("a_drift_by_source"),
            "permutation": f"two-sided sign-flip, {N_PERM} Monte Carlo draws, "
                           f"seed {PERM_SEED}; floor p = {1 / (1 + N_PERM)}",
            "holm_family": [n for n, _ in ps],
            "band_note": "the 20 random directions are Gaussian merges of the 134 "
                         "stage-one adapters. For FA_*, axis_*, mean_assistant_axis and "
                         "the two single-trait anchors -- all inside that span -- the "
                         "band is a matched null. For align_*, sorh_* and stage2_shared, "
                         "which are not in the span, it is a scale reference. With 20 "
                         "draws p_vs_band floors at 1/21 = 0.047619047619047616, so "
                         "'outside the band' means larger in absolute value than all 20.",
            "score_units": "per-loss-token directional derivative of log-likelihood "
                           "along a unit-Frobenius weight direction; positive means the "
                           "completion trains the model toward that direction",
            "tokens": {"hack_mean_loss_tokens": float(ntok[:, 0].mean()),
                       "control_mean_loss_tokens": float(ntok[:, 1].mean())},
            "caveat_stage2_frame":
                "stage2_shared is scored along the projection of the stage-two shared "
                "direction into the zoo LoRA-A window; stage-two LoRA-A is a different "
                "random frame (analysis/lora_a_identity.json#cross_set_A, cos "
                "0.00319641943351548 against stage1_seed0)",
        },
        "random_band": band,
        "random_raw_band": rawband,
        "directions": per,
        "anchor_cells": anchor,
        "assistant_register": register,
        "by_cheat_method": bycheat,
    }
    p = f"{Q}/analysis/sorh_data_scoring.json"
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)

    w = max(len(n) for n in named)
    print(f"\nrandom band: mean {band['mean']:+.4f} sd {band['sd']:.4f} "
          f"max|mean| {band['max_abs']:.4f}\n")
    print(f"raw band  hack: mean {rawband['hack']['mean']:+.4f} sd {rawband['hack']['sd']:.4f}"
          f"   control: mean {rawband['control']['mean']:+.4f} sd {rawband['control']['sd']:.4f}\n")
    print(f"{'direction':<{w}}  {'mean diff':>10} {'sd':>8} {'frac+':>6} "
          f"{'p_flip':>9} {'p_holm':>9} {'z_band':>8} {'hack raw':>10} {'zH':>7} "
          f"{'ctrl raw':>10} {'zC':>7}")
    for n in named:
        d = per[n]
        print(f"{n:<{w}}  {d['paired_diff']['mean']:>+10.4f} "
              f"{d['paired_diff']['sd']:>8.4f} {d['paired_diff']['frac_positive']:>6.3f} "
              f"{d['paired_diff']['p_signflip']:>9.5f} "
              f"{d['paired_diff']['p_holm']:>9.5f} "
              f"{d['vs_random_band']['z']:>+8.2f} "
              f"{d['raw_hack']['mean']:>+10.4f} "
              f"{d['raw_vs_random_band']['hack']['z']:>+7.2f} "
              f"{d['raw_control']['mean']:>+10.4f} "
              f"{d['raw_vs_random_band']['control']['z']:>+7.2f}")
    print("\nanchor cells (this run against analysis/nxn_scores.json):")
    for k, v in anchor.items():
        print(f"  {k:<16} n={v['n_items']} r={v['pearson_r']:.8f} "
              f"mean {v['mean_here']:+.5f} vs {v['mean_nxn']:+.5f} "
              f"max|diff| {v['max_abs_diff']:.3e}")


if __name__ == "__main__":
    sys.exit(main())
