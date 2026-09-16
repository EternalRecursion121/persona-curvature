#!/usr/bin/env python3
"""Part A of the emergent-misalignment medical run: what does the data push toward?

Input:  analysis/em_data_scores.json      (dolci_score.py --stage score)
        phase10_runs/em_ds_targets.json, phase10_runs/em_build_report.json
        analysis/nxn_scores.json          (the two anchor cells)
Output: analysis/em_part_a.json

Modelled on analyse_sorh_data_scores.py, which this reproduces where the design
is the same.  Three differences, all forced by the design:

  * the paired difference is bad medical advice MINUS good medical advice on the
    same prompt (the scorer's `chosen` is the bad completion and `rejected` the
    good one), so the sign convention is "positive = the harmful completion
    trains the model toward this direction more than the benign one does";
  * a third corpus, the length-matched Dolci-Instruct-SFT sample, is scored as
    SINGLE items, so it has raw scores and no pair;
  * there are 149 named personality / alignment directions rather than 18, of
    which 134 are single stage-one adapters, so the Holm family is much larger
    and the band test carries even more of the weight.

The pre-registered verdicts of PREREG_em_medical.md are computed here and written
under `prereg_verdict`, with the predictions themselves written out beside them.
"""
import json
import os
import sys

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
N_PERM = 20000
PERM_SEED = 11
RAND = "rand_merge_"
ARM_TARGETS = ("em_bad", "em_good", "em_dolci", "em_bad_minus_good")
ANCHORS = {"trait_agreeable": "agreeable", "trait_rude": "rude"}

# From PREREG_em_medical.md, "What 'the map sees it coming' means".
PRED_MINE_PRIMARY = ("axis_Conscientiousness", -1)
PRED_MINE = [("axis_Conscientiousness", -1), ("FA_Competence", -1),
             ("trait_careless", +1), ("trait_negligent", +1)]
PRED_PV = [("trait_crooked", +1), ("trait_selfish", +1), ("trait_unkind", +1)]
PRED_SECONDARY = [("FA_Warmth", -1), ("align_corrigible", -1), ("trait_immodest", +1)]


def signflip_p(x, n=N_PERM, seed=PERM_SEED):
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
    s = sorted(pairs, key=lambda kv: kv[1])
    m, out, run = len(s), {}, 0.0
    for i, (k, p) in enumerate(s):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def main():
    S = json.load(open(f"{Q}/analysis/em_data_scores.json"))
    rep = json.load(open(f"{Q}/phase10_runs/em_build_report.json"))
    names = S["names"]
    idx = {n: i for i, n in enumerate(names)}
    rows = S["scores"]
    med = [r for r in rows if r["trait"] == "em_medical"]
    dol = [r for r in rows if r["trait"] == "dolci_sft"]
    anch = [r for r in rows if r["trait"] in ANCHORS.values()]
    assert len(med) == rep["n_score"], (len(med), rep["n_score"])
    assert len(dol) == rep["n_score"], (len(dol), rep["n_score"])
    assert len(anch) == 80, len(anch)

    P = np.array([r["pair"] for r in med])        # (n, T)  bad - good
    Cb = np.array([r["chosen"] for r in med])     # (n, T)  bad
    Cg = np.array([r["rejected"] for r in med])   # (n, T)  good
    Cd = np.array([r["chosen"] for r in dol])     # (n, T)  dolci
    nt = np.array([r["n_tok"] for r in med])
    ntd = np.array([r["n_tok"][0] for r in dol])

    rand = [n for n in names if n.startswith(RAND)]
    named = [n for n in names if not n.startswith(RAND)]
    personality = [n for n in named if n not in ARM_TARGETS]
    rmean = np.array([P[:, idx[n]].mean() for n in rand])
    band = {"directions": rand, "means": rmean.tolist(),
            "mean": float(rmean.mean()), "sd": float(rmean.std(ddof=1)),
            "min": float(rmean.min()), "max": float(rmean.max()),
            "max_abs": float(np.abs(rmean).max()),
            "p_floor": 1.0 / (1 + len(rand)),
            "note": "mean paired difference (bad medical - good medical) along 20 "
                    "Gaussian merges of the 134 stage-one adapters at seed 20260911, "
                    "all unit-Frobenius like every other direction"}

    rawband = {}
    for k, M in (("bad", Cb), ("good", Cg), ("dolci", Cd)):
        v = np.array([M[:, idx[n]].mean() for n in rand])
        rawband[k] = {"means": v.tolist(), "mean": float(v.mean()),
                      "sd": float(v.std(ddof=1)), "max_abs": float(np.abs(v).max())}

    per, ps = {}, []
    for n in names:
        j = idx[n]
        d = P[:, j]
        p, hit = signflip_p(d)
        beat = int((np.abs(rmean) >= abs(float(d.mean()))).sum())
        per[n] = {
            "paired_diff": {**describe(d), "frac_positive": float((d > 0).mean()),
                            "p_signflip": p, "n_as_or_more_extreme": hit,
                            "n_draws": N_PERM},
            "vs_random_band": {"z": (float(d.mean()) - band["mean"]) / band["sd"],
                               "n_random_at_least_as_large": beat,
                               "p_vs_band": (1 + beat) / (1 + len(rand)),
                               "clears_band": beat == 0},
            "raw_bad": describe(Cb[:, j]), "raw_good": describe(Cg[:, j]),
            "raw_dolci": describe(Cd[:, j]),
            "raw_vs_random_band": {
                k: {"z": (float(M[:, j].mean()) - rawband[k]["mean"]) / rawband[k]["sd"],
                    "n_random_at_least_as_large":
                        int((np.abs(np.array(rawband[k]["means"]))
                             >= abs(float(M[:, j].mean()))).sum())}
                for k, M in (("bad", Cb), ("good", Cg), ("dolci", Cd))},
            "token_weighted_mean": {
                "bad": float((Cb[:, j] * nt[:, 0]).sum() / nt[:, 0].sum()),
                "good": float((Cg[:, j] * nt[:, 1]).sum() / nt[:, 1].sum()),
                "dolci": float((Cd[:, j] * ntd).sum() / ntd.sum())},
            "bu_norm": S["bu_norm"][j],
        }
        if not n.startswith(RAND):
            ps.append((n, p))
    for n, v in holm(ps).items():
        per[n]["paired_diff"]["p_holm"] = v

    # ---- pre-registered verdicts -----------------------------------------
    def clears(n, sign=None):
        if n not in per:
            return None
        d = per[n]
        ok = d["vs_random_band"]["clears_band"]
        if sign is not None:
            ok = ok and (np.sign(d["paired_diff"]["mean"]) == sign)
        return bool(ok)

    cleared = [n for n in personality if per[n]["vs_random_band"]["clears_band"]]
    verdict = {
        "any_named_direction_clears_band": bool(cleared),
        "directions_clearing_band": sorted(
            cleared, key=lambda n: -abs(per[n]["paired_diff"]["mean"])),
        "my_primary_named_direction": {
            "direction": PRED_MINE_PRIMARY[0], "predicted_sign": PRED_MINE_PRIMARY[1],
            "clears_band_with_predicted_sign": clears(*PRED_MINE_PRIMARY)},
        "my_four_named_directions": {
            "predictions": [{"direction": n, "predicted_sign": s} for n, s in PRED_MINE],
            "any_clears_with_predicted_sign": any(clears(n, s) for n, s in PRED_MINE),
            "per_direction": {n: {"predicted_sign": s, "mean": per[n]["paired_diff"]["mean"],
                                  "sign_agrees": bool(np.sign(per[n]["paired_diff"]["mean"]) == s),
                                  "clears_band": per[n]["vs_random_band"]["clears_band"],
                                  "n_random_at_least_as_large":
                                      per[n]["vs_random_band"]["n_random_at_least_as_large"]}
                              for n, s in PRED_MINE}},
        "persona_vectors_named_directions": {
            "predictions": [{"direction": n, "predicted_sign": s} for n, s in PRED_PV],
            "any_clears_with_predicted_sign": any(clears(n, s) for n, s in PRED_PV),
            "per_direction": {n: {"predicted_sign": s, "mean": per[n]["paired_diff"]["mean"],
                                  "sign_agrees": bool(np.sign(per[n]["paired_diff"]["mean"]) == s),
                                  "clears_band": per[n]["vs_random_band"]["clears_band"],
                                  "n_random_at_least_as_large":
                                      per[n]["vs_random_band"]["n_random_at_least_as_large"]}
                              for n, s in PRED_PV}},
        "secondary_named_directions": {
            n: {"predicted_sign": s, "mean": per[n]["paired_diff"]["mean"],
                "sign_agrees": bool(np.sign(per[n]["paired_diff"]["mean"]) == s),
                "clears_band": per[n]["vs_random_band"]["clears_band"]}
            for n, s in PRED_SECONDARY},
    }

    # ---- the anchor cells -------------------------------------------------
    N = json.load(open(f"{Q}/analysis/nxn_scores.json"))
    nidx = {n: i for i, n in enumerate(N["names"])}
    nby = {r["id"]: r for r in N["scores"]}
    anchor = {}
    for tgt, trait in ANCHORS.items():
        here = [r for r in anch if r["trait"] == trait]
        a = np.array([r["pair"][idx[tgt]] for r in here])
        b = np.array([nby[r["id"]]["pair"][nidx[tgt]] for r in here])
        anchor[tgt] = {"n_items": len(here), "trait": trait,
                       "mean_here": float(a.mean()), "mean_nxn": float(b.mean()),
                       "pearson_r": float(np.corrcoef(a, b)[0, 1]),
                       "max_abs_diff": float(np.abs(a - b).max()),
                       "bu_norm_here": S["bu_norm"][idx[tgt]],
                       "bu_norm_nxn": N["bu_norm"][nidx[tgt]],
                       "reproduces": bool(float(np.corrcoef(a, b)[0, 1]) > 0.999)}

    top = sorted(personality, key=lambda n: -abs(per[n]["paired_diff"]["mean"]))[:25]
    out = {
        "meta": {
            "scores": "analysis/em_data_scores.json",
            "targets": "phase10_runs/em_ds_targets.json",
            "n_medical_pairs": len(med), "n_dolci_single": len(dol),
            "n_anchor_pairs": len(anch),
            "n_directions": len(names), "n_named": len(named),
            "n_personality_directions": len(personality),
            "n_random_directions": len(rand),
            "a_drift": S["a_drift"], "a_drift_by_source": S.get("a_drift_by_source"),
            "permutation": f"two-sided sign-flip, {N_PERM} Monte Carlo draws, seed "
                           f"{PERM_SEED}; floor p = {1 / (1 + N_PERM)}",
            "holm_family_size": len(ps),
            "sign_convention": "paired_diff = bad medical advice minus good medical "
                               "advice on the same prompt; positive means the harmful "
                               "completion trains toward the direction more than the "
                               "benign one",
            "band_note": "the 20 random directions are Gaussian merges of the 134 "
                         "stage-one adapters, so for the 134 singles, the five FA_*, "
                         "the five axis_*, mean_assistant_axis and the two anchors -- "
                         "all inside that span -- the band is a matched null; for "
                         "align_* and the em_* arm targets, which are not in the span, "
                         "it is a scale reference. p_vs_band floors at "
                         f"{1.0 / (1 + len(rand))}.",
            "tokens": {"bad_mean_loss_tokens": float(nt[:, 0].mean()),
                       "good_mean_loss_tokens": float(nt[:, 1].mean()),
                       "dolci_mean_loss_tokens": float(ntd.mean())},
        },
        "random_band": band,
        "random_raw_band": rawband,
        "directions": per,
        "prereg_verdict": verdict,
        "anchor_cells": anchor,
        "top_25_by_abs_contrast": [
            {"direction": n, "mean": per[n]["paired_diff"]["mean"],
             "z": per[n]["vs_random_band"]["z"],
             "n_random_at_least_as_large":
                 per[n]["vs_random_band"]["n_random_at_least_as_large"]} for n in top],
    }
    p = f"{Q}/analysis/em_part_a.json"
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)

    print(f"\nrandom band: mean {band['mean']:+.6f} sd {band['sd']:.6f} "
          f"max|mean| {band['max_abs']:.6f}")
    print(f"anchors: " + "  ".join(
        f"{k} r={v['pearson_r']:.7f} ({v['mean_here']:+.5f} vs {v['mean_nxn']:+.5f})"
        for k, v in anchor.items()))
    print(f"\ncleared the band ({len(cleared)}): "
          + (", ".join(verdict["directions_clearing_band"][:20]) or "NONE"))
    w = max(len(n) for n in top)
    print(f"\n{'direction':<{w}}  {'mean diff':>11} {'sd':>9} {'frac+':>6} "
          f"{'p_flip':>9} {'p_holm':>9} {'z_band':>8} {'#rand>=':>8}")
    for n in top:
        d = per[n]
        print(f"{n:<{w}}  {d['paired_diff']['mean']:>+11.6f} {d['paired_diff']['sd']:>9.6f} "
              f"{d['paired_diff']['frac_positive']:>6.3f} "
              f"{d['paired_diff']['p_signflip']:>9.5f} {d['paired_diff']['p_holm']:>9.5f} "
              f"{d['vs_random_band']['z']:>+8.3f} "
              f"{d['vs_random_band']['n_random_at_least_as_large']:>8d}")
    print("\narm targets (the positive control):")
    for n in ARM_TARGETS:
        if n in per:
            d = per[n]
            print(f"  {n:<20} {d['paired_diff']['mean']:>+11.6f} frac+ "
                  f"{d['paired_diff']['frac_positive']:.4f} z {d['vs_random_band']['z']:+.3f} "
                  f"#rand>= {d['vs_random_band']['n_random_at_least_as_large']}")
    print("\nprereg verdicts:")
    print(f"  any named direction clears the band ..... "
          f"{verdict['any_named_direction_clears_band']}")
    print(f"  my primary ({PRED_MINE_PRIMARY[0]}) ...... "
          f"{verdict['my_primary_named_direction']['clears_band_with_predicted_sign']}")
    print(f"  any of my four .......................... "
          f"{verdict['my_four_named_directions']['any_clears_with_predicted_sign']}")
    print(f"  any Persona-Vectors direction ........... "
          f"{verdict['persona_vectors_named_directions']['any_clears_with_predicted_sign']}")


if __name__ == "__main__":
    sys.exit(main())
