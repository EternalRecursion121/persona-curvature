#!/usr/bin/env python3
"""Turn the measured dose curve into the 72 alphas the iso-KL sphere steers at.

Reads   phase10_runs/sphere_isokl_calib.json  (sphere_dose.py: measured KL per
                                               direction and alpha, bf16 weights)
        phase10_runs/sphere_sweep_spec.json   (the three basis deltas and the 72
                                               lattice vectors the 2026-09-01 run
                                               consumed, verbatim)
Writes  phase10_runs/sphere_isokl_sweep_spec.json  the sweep's own input
        phase10_runs/sphere_isokl_verify_spec.json a one-alpha-per-direction
                                                   re-measurement of the dose the
                                                   solved alphas actually deliver
        analysis/sphere_isokl_alphas.json          what was solved and from what

THE TARGET, fixed in PREREG_sphere_isokl.md before the calibration was read: the
MEDIAN over the 72 directions of the measured KL per token at alpha 1.5.  Half
the lattice then moves up in alpha and half moves down, so the iso-KL sphere
delivers, in aggregate, the dose the 2026-09-01 sphere delivered; a target taken
from an external unit would have shifted every point the same way and confounded
"iso-KL" with "a different overall dose".

THE RULE: log-log linear interpolation between bracketing MEASURED points --
solve() from solve_matched_alphas.py, unchanged -- and never extrapolation.  A
direction whose target falls outside the measured grid is CLAMPED to the nearest
measured endpoint and flagged `extrapolated`, because the small-alpha Taylor fit
overstates KL out here by a median 1.475x and a linear extension of the log-log
curve has no better claim.
"""
import json
import os

import numpy as np

from solve_matched_alphas import solve

Q = os.path.dirname(os.path.abspath(__file__))
A = f"{Q}/analysis"
P = f"{Q}/phase10_runs"
BASE_POINT = "S900"          # the alpha-0 condition, generated in the same run


def solve_point(al, kl, target):
    """(alpha, bracket, extrapolated, note) under the pre-registered fallbacks."""
    o = np.argsort(al)
    a, k = np.asarray(al, float)[o], np.asarray(kl, float)[o]
    if k[0] >= target:
        return float(a[0]), (float(a[0]), float(a[0])), True, "clamped_low"
    if k[-1] <= target:
        return float(a[-1]), (float(a[-1]), float(a[-1])), True, "clamped_high"
    if np.all(np.diff(k) > 0):
        aa, br, ex = solve(list(a), list(k), target)
        return float(aa), br, bool(ex), "interpolated"
    # non-monotone: use the bracketing pair that contains the target
    for i in range(len(a) - 1):
        if min(k[i], k[i + 1]) <= target <= max(k[i], k[i + 1]):
            w = ((np.log(target) - np.log(k[i]))
                 / (np.log(k[i + 1]) - np.log(k[i])))
            aa = float(np.exp(np.log(a[i]) + w * (np.log(a[i + 1]) - np.log(a[i]))))
            return aa, (float(a[i]), float(a[i + 1])), False, "non_monotone_bracket"
    return 1.5, (1.5, 1.5), True, "non_monotone_no_bracket_left_at_1.5"


def main():
    C = json.load(open(f"{P}/sphere_isokl_calib.json"))
    SW = json.load(open(f"{P}/sphere_sweep_spec.json"))
    res = C["results"]
    alphas = sorted({r["alpha"] for r in next(iter(res.values()))["per_alpha"].values()})
    kl = {n: [d["per_alpha"][f"bf16|{a}"]["kl"] for a in alphas] for n, d in res.items()}
    if len(res) != 72:
        raise SystemExit(f"calibration has {len(res)} directions, expected 72")

    k15 = {n: d["per_alpha"]["bf16|1.5"]["kl"] for n, d in res.items()}
    target = float(np.median(list(k15.values())))
    print(f"grid {alphas}")
    print(f"KL at alpha 1.5 over 72 points: min {min(k15.values()):.6f} "
          f"median {target:.6f} max {max(k15.values()):.6f} "
          f"({max(k15.values())/min(k15.values()):.3f}x)")
    print(f"TARGET = {target:.9f} nats/token")

    rows, pts, vdirs = {}, [], []
    nmono = 0
    for n in sorted(res):
        aa, br, ex, note = solve_point(alphas, kl[n], target)
        if not np.all(np.diff(np.asarray(kl[n])) > 0):
            nmono += 1
        # free interpolation-error check: predict KL(1.5) from the 1.25/1.75 pair
        i1, i2, i3 = alphas.index(1.25), alphas.index(1.5), alphas.index(1.75)
        w = (np.log(1.5) - np.log(1.25)) / (np.log(1.75) - np.log(1.25))
        pred = float(np.exp(np.log(kl[n][i1]) + w * (np.log(kl[n][i3]) - np.log(kl[n][i1]))))
        rows[n] = {"alpha": round(aa, 4), "alpha_solved": aa, "bracket": br,
                   "extrapolated": ex, "note": note,
                   "kl_at_grid": dict(zip(map(str, alphas), kl[n])),
                   "kl_at_1_5": k15[n],
                   "dose_ratio_to_target_at_alpha_1_5": k15[n] / target,
                   "dir_norm_raw": res[n]["dir_norm_raw"],
                   "interp_check_kl15_pred_over_meas": pred / k15[n]}
        vdirs.append({"name": n, "family": "pc_sphere_isokl",
                      "coef": None, "alphas": [round(aa, 4)]})
    ic = np.array([r["interp_check_kl15_pred_over_meas"] for r in rows.values()])
    sol = np.array([r["alpha"] for r in rows.values()])

    # the sweep spec: same basis, same lattice, one alpha per point
    for p in SW["points"]:
        pts.append({"name": p["name"], "u": p["u"], "alpha": rows[p["name"]]["alpha"]})
    pts.append({"name": BASE_POINT, "u": [0.0, 0.0, 1.0], "alpha": 0.0})
    # alpha is None on purpose.  Every point carries its own, so the spec-level
    # default is never consulted; leaving 1.5 there would put a number in
    # sphere_isokl_results.json#alpha that nothing steered, and None makes a
    # point that somehow lost its alpha crash rather than silently run at 1.5.
    spec = {"basis": SW["basis"], "points": pts, "prompts": SW["prompts"],
            "alpha": None, "ref": SW["ref"], "tag": "isokl"}
    json.dump(spec, open(f"{P}/sphere_isokl_sweep_spec.json", "w"))

    # the verification spec: re-measure the KL each solved alpha delivers
    SP = {j["name"]: j["coef"] for j in json.load(open(f"{P}/sphere_spec.json"))["jobs"]}
    for d in vdirs:
        d["coef"] = SP[d["name"]]
        d["published_ref"] = C["ref"]
    vspec = {"ref": C["ref"], "max_resp_tokens": C["max_resp_tokens"],
             "traits134": json.load(open(f"{P}/sphere_isokl_dose_spec.json"))["traits134"],
             "prompts": C["prompts"],
             "texts": json.load(open(f"{P}/sphere_isokl_dose_spec.json"))["texts"],
             "text_source": C["text_source"], "modes": ["bf16"],
             "tag": "sphere_isokl_verify", "directions": vdirs}
    json.dump(vspec, open(f"{P}/sphere_isokl_verify_spec.json", "w"))

    out = {"what": "alphas that deliver an equal measured KL per token along all "
                   "72 principal-component sphere directions",
           "target_rule": "median over the 72 of the measured bf16 KL per token "
                          "at alpha 1.5, the alpha the 2026-09-01 sphere steered",
           "kl_target_nats_per_token": target,
           "weight_construction_matched_on": "bf16",
           "calibration_source": "phase10_runs/sphere_isokl_calib.json",
           "grid": alphas, "ref": C["ref"], "base_point_name": BASE_POINT,
           "n_scored_tokens": C["n_scored_tokens"],
           "zero_alpha_control": C["zero_alpha_control"],
           "summary": {
               "n": len(rows),
               "alpha_min": float(sol.min()), "alpha_median": float(np.median(sol)),
               "alpha_max": float(sol.max()), "alpha_mean": float(sol.mean()),
               "alpha_sd": float(sol.std(ddof=1)),
               "n_extrapolated": int(sum(r["extrapolated"] for r in rows.values())),
               "n_non_monotone": nmono,
               "argmin": min(rows, key=lambda n: rows[n]["alpha"]),
               "argmax": max(rows, key=lambda n: rows[n]["alpha"]),
               "kl_at_1_5_min": float(min(k15.values())),
               "kl_at_1_5_median": target,
               "kl_at_1_5_max": float(max(k15.values())),
               "kl_at_1_5_ratio_max_over_min": float(max(k15.values()) / min(k15.values())),
               "interp_check_kl15_pred_over_meas": {
                   "median": float(np.median(ic)), "min": float(ic.min()),
                   "max": float(ic.max()),
                   "what": "KL at alpha 1.5 predicted by the same log-log rule "
                           "from the 1.25 and 1.75 measurements, over the "
                           "measured value; 1.0 means the interpolation the "
                           "alphas rest on is exact at that spacing"}},
           "directions": rows}
    json.dump(out, open(f"{A}/sphere_isokl_alphas.json", "w"), indent=1)

    print(f"alphas: min {sol.min():.4f} median {np.median(sol):.4f} "
          f"max {sol.max():.4f} sd {sol.std(ddof=1):.4f}; "
          f"{out['summary']['n_extrapolated']} extrapolated, {nmono} non-monotone")
    print(f"interpolation check KL(1.5) predicted/measured: median "
          f"{np.median(ic):.5f} range {ic.min():.5f}..{ic.max():.5f}")
    print(f"wrote {P}/sphere_isokl_sweep_spec.json ({len(pts)} points incl. base), "
          f"{P}/sphere_isokl_verify_spec.json, {A}/sphere_isokl_alphas.json")


if __name__ == "__main__":
    main()
