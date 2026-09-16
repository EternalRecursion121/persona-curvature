#!/usr/bin/env python3
"""Turn the measured dose curve into the alphas experiment G5 steers at.

Reads   phase10_runs/dose_calib.json   (fisher_dose.py: measured KL per direction,
                                        sign, alpha and weight construction)
        analysis/fisher_norms.json     (the random-merge band that defines the unit)
        phase10_runs/fisher_spec.json  (the coefficient vectors)
        phase10_runs/steer_spec2_7a.json (the 24-prompt battery, verbatim)
Writes  phase10_runs/steer_dose_spec.json   a steer_fix.py spec
        analysis/matched_dose_alphas.json    what was solved and from what

THE UNIT.  analysis/fisher_norms.json defines a direction's Fisher dose per unit
alpha as sqrt(F(u) / F_random_median), the square root of its curvature in units
of the median random merge of all 134 adapters.  A dose of 2 is therefore the KL
that alpha 2 would deliver along a median random merge:

    KL_target = 0.5 * F_random_median * dose^2

That is a KL per token, and it is the quantity matched -- across directions and,
which equal alpha does not do, across the two SIGNS of the same direction.  The
alpha that delivers it is read off the MEASURED curve by interpolation between
bracketing measured points, never from the small-alpha quadratic: the cubic term
is 15% of the quadratic at |alpha| = 0.5 for FA_Warmth and the alphas here are
three times further out.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "analysis")
P = os.path.join(HERE, "phase10_runs")
DOSE = float(os.environ.get("PC_DOSE", "2.0"))
MODE = os.environ.get("PC_DOSE_MODE", "")     # "" = decide from the retention


def solve(alphas, kls, target):
    """|alpha| at which the measured KL equals target, log-log interpolated.

    KL ~ a^2 to leading order, so log KL against log |alpha| is close to a
    straight line of slope 2 and piecewise-linear interpolation on those axes is
    accurate between measured points.  Returns (alpha, bracket, extrapolated).
    """
    o = np.argsort(alphas)
    a, k = np.asarray(alphas)[o], np.asarray(kls)[o]
    if not np.all(np.diff(k) > 0):
        raise SystemExit(f"KL is not monotone in |alpha|: {list(zip(a, k))}")
    la, lk, lt = np.log(a), np.log(k), math.log(target)
    if lt <= lk[0] or lt >= lk[-1]:
        # outside the measured range: linear extrapolation from the nearest pair
        i = 0 if lt <= lk[0] else len(a) - 2
        ex = True
    else:
        i = int(np.searchsorted(lk, lt) - 1)
        ex = False
    w = (lt - lk[i]) / (lk[i + 1] - lk[i])
    return float(math.exp(la[i] + w * (la[i + 1] - la[i]))), \
        (float(a[i]), float(a[i + 1])), ex


def main():
    C = json.load(open(f"{P}/dose_calib.json"))
    FN = json.load(open(f"{A}/fisher_norms.json"))
    frand = FN["random_band"]["median"]
    target = 0.5 * frand * DOSE * DOSE

    # which weight construction to match on
    ret = [r["retention"] for d in C["results"].values()
           for r in d["per_alpha"].values() if r["mode"] == "bf16"]
    # Match on the construction the GENERATING model actually has.  steer_fix.py
    # writes bf16 weights, so the dose its text receives is KL(bf16), whatever
    # the exact-weight curve says; the exact curve is reported beside it as the
    # intended dose.  PC_DOSE_MODE overrides.
    mode = MODE or "bf16"
    print(f"bf16 retention over {len(ret)} (direction, alpha) cells: "
          f"min {min(ret):.4f} median {np.median(ret):.4f} max {max(ret):.4f}"
          if ret else "no bf16 arm in the calibration")
    print(f"matching on the '{mode}' weight construction, "
          f"dose {DOSE} -> KL target {target:.6f} nats/token")

    SP = json.load(open(f"{P}/fisher_spec.json"))
    coefs = {d["name"]: d["coef"] for d in SP["directions"]}
    prompts = json.load(open(f"{P}/steer_spec2_7a.json"))["jobs"][0]["prompts"]
    REF = C["ref"]

    rows, jobs = {}, []
    for name, d in C["results"].items():
        per = d["per_alpha"]
        got = {}
        for sgn, sel in (("neg", lambda x: x < 0), ("pos", lambda x: x > 0)):
            al = sorted({r["alpha"] for r in per.values()
                         if r["mode"] == mode and sel(r["alpha"])}, key=abs)
            kl = [per[f"{mode}|{a}"]["kl"] for a in al]
            aa, br, ex = solve([abs(a) for a in al], kl, target)
            got[sgn] = {"alpha": round(-aa if sgn == "neg" else aa, 4),
                        "abs_alpha_solved": aa, "bracket_abs_alpha": br,
                        "extrapolated": ex,
                        "kl_at_bracket": [per[f"{mode}|{-br[0] if sgn=='neg' else br[0]}"]["kl"],
                                          per[f"{mode}|{-br[1] if sgn=='neg' else br[1]}"]["kl"]]}
        # what equal alpha 2 would have delivered, for the contrast
        eq = {}
        for a in (-2.0, 2.0):
            key = f"{mode}|{a}"
            if key in per:
                eq[str(a)] = {"kl": per[key]["kl"], "retention": per[key]["retention"],
                              "argmax_change": per[key]["argmax_change"]}
        rows[name] = {"family": d["family"], "dir_norm_raw": d["dir_norm_raw"],
                      "published_ref": d["published_ref"], "matched": got,
                      "kl_at_equal_alpha_2": eq,
                      "fisher_dose_per_alpha_published":
                          FN["directions"][name]["fisher_dose_per_alpha"]}
        # PREFIXED.  steer_fix.py resumes from /oct/steerfix/<name>.json, and the
        # published campaigns left FA_Warmth.json, axis_Extraversion.json and the
        # rest there with the alphas {-4,-2,-1,0,1,2,4}.  Reusing the bare name
        # would merge nine published alphas into this run's returned generations
        # -- four times the judge bill on text this experiment did not make --
        # and overwrite the published run's partial on the volume.
        jobs.append({"name": f"dose_{name}", "source": "stage1", "coef": coefs[name],
                     "ref": REF,
                     "alphas": sorted([got["neg"]["alpha"], got["pos"]["alpha"]]),
                     "prompts": prompts})

    # the clean base condition: no direction, one alpha of zero, judged
    # interleaved with the steered text so the baseline is in the same judge run
    jobs.append({"name": "dose_BASE", "source": "stage1", "coef": {}, "ref": REF,
                 "alphas": [0.0], "prompts": prompts})

    json.dump({"ref": REF, "n": len(jobs), "jobs": jobs},
              open(f"{P}/steer_dose_spec.json", "w"))
    out = {"what": "alphas that deliver an equal measured KL per token along "
                   "each direction and each sign",
           "dose": DOSE, "kl_target_nats_per_token": target,
           "f_random_median": frand,
           "f_random_median_source": "analysis/fisher_norms.json#random_band.median",
           "weight_construction_matched_on": mode,
           "ref": REF,
           "ref_note": "every job is steered at ref 0.8078003190997738, the ref "
                       "the calibration used; the five axis_* directions were "
                       "PUBLISHED at ref 0.8102592902648793, 0.30% larger, so "
                       "their published alphas are in a marginally larger unit",
           "calibration_source": "phase10_runs/dose_calib.json",
           "bf16_retention": ({"n": len(ret), "min": float(min(ret)),
                               "median": float(np.median(ret)),
                               "max": float(max(ret))} if ret else None),
           "directions": rows}
    json.dump(out, open(f"{A}/matched_dose_alphas.json", "w"), indent=1)
    print(f"\nwrote {P}/steer_dose_spec.json ({len(jobs)} jobs) and "
          f"analysis/matched_dose_alphas.json")
    print(f"{'direction':26s} {'alpha-':>8s} {'alpha+':>8s}  "
          f"{'KL@a=-2':>8s} {'KL@a=+2':>8s}")
    for n, r in rows.items():
        e = r["kl_at_equal_alpha_2"]
        print(f"{n:26s} {r['matched']['neg']['alpha']:8.4f} "
              f"{r['matched']['pos']['alpha']:8.4f}  "
              f"{e.get('-2.0',{}).get('kl',float('nan')):8.4f} "
              f"{e.get('2.0',{}).get('kl',float('nan')):8.4f}")


if __name__ == "__main__":
    main()
