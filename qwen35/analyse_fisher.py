#!/usr/bin/env python3
"""Turn the raw Fisher sweep into analysis/fisher_norms.json.

WHAT F IS.  For a direction u (unit Frobenius over the whole weight change) and
the family theta(alpha) = theta_0 + alpha * ref * u,

    mean per-token KL( p_base || p_theta(alpha) ) = 0.5 * alpha^2 * F(u) + O(alpha^3)

so F(u) is a *second-order* quantity, measured *at the base point*, on
*in-distribution text* (the base model's own alpha-0 responses to the 24
steering prompts).  It says how fast the output distribution starts to move when
you push along u, in nats per token per unit alpha squared.  It says nothing
about what happens at alpha 2 or 4, nothing about which way the persona moves,
and nothing about text the base model would not itself have written.

UNITS.  F_ref is per ref-unit alpha (ref = 0.8078003190997738, the unit every
published alpha is in).  F_adapter is per adapter-norm alpha, using
analysis/steer_alpha_units.json#ref_over_true_mean_norm: alpha 1 of ref is
0.49996 of a stage-one adapter's Frobenius norm, so F_adapter = F_ref / that^2.

THE FIT.  F is the least-squares slope of KL against 0.5*alpha^2 through the
origin, pooled over the alphas: F = 2 * sum_a a^2 KL_a / sum_a a^4.  The
residual is reported as the relative RMS of KL_a against the fit, and the
quadratic check as F measured at |alpha| 0.5 over F measured at |alpha| 0.125 --
1.0 means the KL is exactly quadratic over a 4x range of alpha.

DEGENERATION.  Two detectors exist in this project.  `looping` (a 10-word window
repeated four times, analyse_alien_steer.looping) is what the alien direction and
both spheres are counted with; the adjudicated per-alpha counts in
analysis/qual_*.json are a human-adjudicated read of the same corpora and do not
exist for every direction.  They are NOT pooled: the correlation is computed on
recomputed `looping` rates only, and the adjudicated numbers are carried
alongside.
"""
import itertools
import json
import os

import numpy as np

from analyse_alien_steer import looping

Q = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(Q, "phase10_runs")
A = os.path.join(Q, "analysis")
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]


def fit(rows):
    """Three readings of the same four-to-six KL measurements.

    F_ref_pooled is the fit the task specifies: the least-squares slope of KL
    against 0.5*alpha^2 through the origin.  Because that objective weights each
    alpha by alpha^4, the largest |alpha| carries almost all of it (96% of the
    weight at |0.5| in a six-alpha set), so it is a reading of the curvature
    NEAR |alpha| = 0.5, not at zero.

    F_ref is the curvature at zero proper: the quadratic coefficient of a
    third-and-fourth-order Taylor fit KL = 0.5*F*a^2 + c*a^3 + d*a^4, which
    separates the even part (F and d) from the odd part (c).  Every direction
    here has at least the four alphas +-0.25 and +-0.5, which determines all
    three exactly; the six-alpha directions over-determine them.

    c is the leading term that distinguishes +alpha from -alpha, and is the only
    thing measured here that could in principle predict WHICH SIGN degenerates.

    F_ref_a025 is the plain 2*KL/alpha^2 at |alpha| = 0.25, available for every
    direction, and is the check that the three agree.
    """
    al = np.array([float(a) for a in rows])
    kl = np.array([rows[a]["kl"] for a in rows])
    F_pool = 2.0 * float((al ** 2 * kl).sum() / (al ** 4).sum())
    pred = 0.5 * F_pool * al ** 2
    # relative PER MEASUREMENT, not against the mean: KL spans a 16x range across
    # these alphas, so normalising by the mean would flatter the small ones
    resid = float(np.sqrt((((kl - pred) / kl) ** 2).mean()))
    X = np.stack([0.5 * al ** 2, al ** 3, al ** 4], axis=1)
    beta, *_ = np.linalg.lstsq(X, kl, rcond=None)
    F, c, d = (float(x) for x in beta)
    tres = float(np.sqrt((((kl - X @ beta) / kl) ** 2).mean()))
    per = {a: rows[a]["F_alpha"] for a in rows}

    def at(m):
        v = [w for a, w in per.items() if abs(float(a)) == m]
        return float(np.mean(v)) if v else None
    f05, f025, f0125 = at(0.5), at(0.25), at(0.125)
    pos = [v for a, v in per.items() if float(a) > 0]
    neg = [v for a, v in per.items() if float(a) < 0]
    return {"F_ref": F, "cubic_c": c, "quartic_d": d, "taylor_rel_resid": tres,
            "F_ref_pooled": F_pool, "rel_resid": resid,
            "F_ref_a05": f05, "F_ref_a025": f025, "F_ref_a0125": f0125,
            "F_per_alpha": per, "n_alphas": len(al),
            "quad_check_F05_over_F0125": (f05 / f0125) if f0125 else None,
            "quad_check_F05_over_F025": (f05 / f025) if f025 else None,
            "cubic_over_F": c / F if F else None,
            "asym_pos_over_neg": float(np.mean(pos) / np.mean(neg)),
            "symkl_mean": float(np.mean([rows[a]["symkl"] for a in rows])),
            "argmax_change": {a: rows[a]["argmax_change"] for a in rows}}


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        o = v.argsort()
        r = np.empty(len(v), float)
        r[o] = np.arange(len(v), dtype=float)
        # average ranks for ties
        for val in np.unique(v):
            m = v == val
            r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else 0.0


def perm_p(x, y, n=20000, seed=7):
    r0 = spearman(x, y)
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    hits = sum(abs(spearman(x, rng.permutation(y))) >= abs(r0) - 1e-12 for _ in range(n))
    return r0, float((hits + 1) / (n + 1))


def loop_rates():
    """Recomputed loop rates, one detector, for every direction that has text."""
    out = {}
    for f in ("steer_results_fix.json", "steer_results_fix2.json",
              "alien_results_fa.json", "steer_results_s2mean.json",
              "steer_results_s2balanced.json"):
        p = os.path.join(P, f)
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            out[r["name"]] = {"file": f, "per_alpha": {
                a: sum(looping(t) for t in g) / len(g)
                for a, g in r["generations"].items()}}
    S = json.load(open(os.path.join(A, "sphere_page_fa.json")))
    for n, g in S["gen"].items():
        out["sphere_" + n] = {"file": "analysis/sphere_page_fa.json#gen",
                              "per_alpha": {str(S["alpha"]): sum(looping(t) for t in g) / len(g)}}
    return out


def adjudicated():
    out = {}
    for f in ("qual_pc", "qual_fa", "qual_axes", "qual_identity"):
        p = os.path.join(A, f + ".json")
        if not os.path.exists(p):
            continue
        D = json.load(open(p))
        for e in (D["directions"] if isinstance(D, dict) else D):
            dm = e.get("damage_markers") or {}
            pa = dm.get("per_alpha", {}).get("looping")
            if pa:
                n = dm.get("of_n", 24)
                out[e["name"]] = {"file": f"analysis/{f}.json",
                                  "per_alpha": {str(float(k)): v / n for k, v in pa.items()}}
    return out


def main():
    R = json.load(open(os.path.join(P, "fisher_results.json")))
    U = json.load(open(os.path.join(A, "steer_alpha_units.json")))
    ratio = U["ref_over_true_mean_norm"]
    res = R["results"]
    D = {}
    for n, e in res.items():
        if "error" in e or "per_alpha" not in e:
            D[n] = {"family": e.get("family"), "error": e.get("error")}
            continue
        f = fit(e["per_alpha"])
        f["F_adapter"] = f["F_ref"] / ratio ** 2
        f["family"] = e["family"]
        f["source"] = e["source"]
        f["dir_norm_raw"] = e["dir_norm_raw"]
        f["published_ref"] = e.get("published_ref")
        f["seed"] = e.get("seed")
        D[n] = f

    ok = {n: v for n, v in D.items() if "F_ref" in v}
    rnd = [v["F_ref"] for n, v in ok.items() if v["family"] == "random"]
    Fref_null = float(np.median(rnd))
    band = {"n": len(rnd), "median": Fref_null, "min": float(np.min(rnd)),
            "max": float(np.max(rnd)), "p10": float(np.percentile(rnd, 10)),
            "p90": float(np.percentile(rnd, 90)), "mean": float(np.mean(rnd)),
            "sd": float(np.std(rnd, ddof=1)),
            "seeds": sorted(v["seed"] for v in ok.values() if v["family"] == "random")}
    Fref_null_p10, Fref_null_p90 = band["p10"], band["p90"]
    order = sorted(ok, key=lambda n: -ok[n]["F_ref"])
    for i, n in enumerate(order):
        ok[n]["rank"] = i + 1
        ok[n]["F_over_random_median"] = ok[n]["F_ref"] / Fref_null
        # a Fisher-normalised dose: the alpha on a random merge that moves the
        # output distribution as far as this alpha on this direction
        ok[n]["fisher_dose_per_alpha"] = float(np.sqrt(ok[n]["F_ref"] / Fref_null))
        ok[n]["alpha2_in_fisher_units"] = 2.0 * ok[n]["fisher_dose_per_alpha"]

    LR, ADJ = loop_rates(), adjudicated()
    # S2_signedrandom has published generations but no Fisher measurement: the task
    # named the stage-two grand mean and its BALANCED control, so it is not in the
    # spec and drops out of every correlation below.
    excluded = sorted(n for n in LR if n not in ok)

    # --- F versus degeneration -------------------------------------------
    corr = {}
    pairs = [(n, LR[n]["per_alpha"]) for n in ok if n in LR]
    # F is even in alpha to second order, so it cannot distinguish the two signs.
    # Degeneration in this project is overwhelmingly a NEGATIVE-alpha phenomenon,
    # so the three arms are reported separately rather than pooled.
    arms = {"alpha2_mean_of_both_signs":
            lambda pa: (pa["2.0"] + pa["-2.0"]) / 2.0,
            "alpha_minus2": lambda pa: pa["-2.0"],
            "alpha_plus2": lambda pa: pa["2.0"]}
    for key, fn in arms.items():
        a2 = [(n, fn(pa)) for n, pa in pairs if "2.0" in pa and "-2.0" in pa]
        if len(a2) < 4:
            continue
        x = [ok[n]["F_ref"] for n, _ in a2]
        y = [v for _, v in a2]
        r, p = perm_p(x, y)
        corr[key] = {
            "n": len(a2), "spearman": r, "perm_p": p,
            "directions": {n: {"F_ref": ok[n]["F_ref"], "loop_rate": v} for n, v in a2},
            "note": "loop rate recomputed with analyse_alien_steer.looping (a 10-word "
                    "window repeated four times) on the published generations",
            "zero_ties": int(sum(1 for _, v in a2 if v == 0.0))}
    # the cubic term is the only signed quantity measured here, and therefore the
    # only one that could distinguish the negative alphas that break this model
    # from the positive ones that do not
    cub = [(n, pa["-2.0"]) for n, pa in pairs if "-2.0" in pa]
    if len(cub) >= 4:
        x = [ok[n]["cubic_c"] for n, _ in cub]
        y = [v for _, v in cub]
        r, p = perm_p(x, y)
        corr["cubic_c_vs_alpha_minus2"] = {
            "n": len(cub), "spearman": r, "perm_p": p,
            "note": "the signed cubic coefficient of KL in alpha against the loop "
                    "rate at alpha -2; a negative c means KL rises faster on the "
                    "negative side, which is the side that degenerates",
            "directions": {n: {"cubic_c": ok[n]["cubic_c"], "loop_rate": v}
                           for n, v in cub}}

    sph = [(n, LR[n]["per_alpha"]["1.5"]) for n in ok
           if ok[n]["family"] == "sphere" and n in LR]
    if len(sph) >= 8:
        x = [ok[n]["F_ref"] for n, _ in sph]
        y = [v for _, v in sph]
        r, p = perm_p(x, y)
        corr["sphere_alpha1.5"] = {
            "n": len(sph), "spearman": r, "perm_p": p,
            "nonzero": int(sum(1 for _, v in sph if v > 0)),
            "F_range": [float(min(x)), float(max(x))],
            "loop_rate_range": [float(min(y)), float(max(y))],
            "note": "72 factor-sphere points, 8 prompts each at alpha 1.5, "
                    "loop rates recomputed from analysis/sphere_page_fa.json#gen"}

    # does F predict how far the persona MOVES, as opposed to whether it breaks?
    SP = json.load(open(os.path.join(A, "sphere_page_fa.json")))
    prof = {}
    for n, e in SP["judged"].items():
        v = [e["scores"].get(f) for f in F5]
        if all(x is not None for x in v):
            prof["sphere_" + n] = np.array(v, float)
    if len(prof) >= 8:
        cen = np.mean(list(prof.values()), axis=0)
        mag = {n: float(np.linalg.norm(v - cen)) for n, v in prof.items()}
        pts = [(n, mag[n]) for n in ok if n in mag]
        x = [ok[n]["F_ref"] for n, _ in pts]
        y = [v for _, v in pts]
        r, p = perm_p(x, y)
        corr["sphere_judged_profile_distance"] = {
            "n": len(pts), "spearman": r, "perm_p": p,
            "note": "Euclidean distance of each sphere point's mean five-scale judged "
                    "profile from the centroid of all 72, against F. This asks whether "
                    "F predicts how far a direction moves the persona, not whether it "
                    "breaks it. Profiles from analysis/sphere_page_fa.json#judged; the "
                    "centroid is the 72-point mean, not a base-model profile, because "
                    "the sphere run judged no alpha-0 condition."}

    # --- named comparisons -------------------------------------------------
    def g(n):
        return ok[n]["F_ref"] if n in ok else None
    singles = [v["F_ref"] for v in ok.values() if v["family"] == "single"]
    sphF = [v["F_ref"] for v in ok.values() if v["family"] == "sphere"]
    both = {n: pa for n, pa in pairs if "2.0" in pa and "-2.0" in pa}
    SIGN = {
        "n_directions_with_both_signs": len(both),
        "n_looping_at_all": int(sum(1 for pa in both.values()
                                    if pa["2.0"] or pa["-2.0"])),
        "n_zero_at_plus2": int(sum(1 for pa in both.values() if pa["2.0"] == 0.0)),
        "n_zero_at_minus2": int(sum(1 for pa in both.values() if pa["-2.0"] == 0.0)),
        "n_minus2_ge_plus2": int(sum(1 for pa in both.values()
                                     if pa["-2.0"] >= pa["2.0"])),
        "n_minus2_gt_plus2": int(sum(1 for pa in both.values()
                                     if pa["-2.0"] > pa["2.0"])),
        "note": "loop rates recomputed with analyse_alien_steer.looping on the "
                "published generations; degeneration under steering in this "
                "project is overwhelmingly a negative-alpha phenomenon"}
    named_merges = [v["F_ref"] for n, v in ok.items()
                    if v["family"] in ("fa", "pc", "axis", "alien")]
    comp = {
        "stage2_vs_stage1_grand_mean": {
            "S2_mean": g("S2_mean"), "mean_assistant_axis": g("mean_assistant_axis"),
            "S2_balancedrandom": g("S2_balancedrandom"),
            "ratio_S2mean_over_stage1mean":
                (g("S2_mean") / g("mean_assistant_axis"))
                if g("S2_mean") and g("mean_assistant_axis") else None,
            "alpha4_on_S2mean_in_stage1_grandmean_alpha":
                4.0 * float(np.sqrt(g("S2_mean") / g("mean_assistant_axis")))
                if g("S2_mean") and g("mean_assistant_axis") else None,
            "note": "alpha 4 on the stage-two grand mean is the same Fisher dose as "
                    "this alpha on the stage-one grand mean, which is why one stays "
                    "coherent there and the other does not"},
        "n_above_random_p90": int(sum(1 for v in ok.values()
                                      if v["F_ref"] > Fref_null_p90)),
        "n_below_random_p10": int(sum(1 for v in ok.values()
                                      if v["F_ref"] < Fref_null_p10)),
        "fit_definition_agreement": {
            "note": "the named directions carry alphas down to 0.125 and the 92 sphere "
                    "and random directions only to 0.25, so the ranks could in principle "
                    "be an artefact of extrapolating from different lever arms. Spearman "
                    "of the Taylor F against the plain 2*KL/alpha^2 at |alpha| = 0.25, "
                    "which every direction has, says they are not.",
            "spearman_F_vs_F_a025_all": spearman(
                [v["F_ref"] for v in ok.values()],
                [v["F_ref_a025"] for v in ok.values()]),
            "n_all": len(ok),
            "spearman_F_vs_F_a0125_six_alpha": spearman(
                [v["F_ref"] for v in ok.values() if v["n_alphas"] == 6],
                [v["F_ref_a0125"] for v in ok.values() if v["n_alphas"] == 6]),
            "n_six_alpha": int(sum(1 for v in ok.values() if v["n_alphas"] == 6))},
        "sphere_band": {
            "n": len(sphF), "min": float(np.min(sphF)), "max": float(np.max(sphF)),
            "median": float(np.median(sphF))},
        "full_range": {"min_direction": order[-1], "min": ok[order[-1]]["F_ref"],
                       "max_direction": order[0], "max": ok[order[0]]["F_ref"],
                       "ratio": ok[order[0]]["F_ref"] / ok[order[-1]]["F_ref"]},
        "sign_asymmetry_of_degeneration": SIGN,
        "single_adapters_vs_merges": {
            "single_n": len(singles), "single_median": float(np.median(singles)),
            "single_min": float(np.min(singles)), "single_max": float(np.max(singles)),
            "named_merge_n": len(named_merges),
            "named_merge_median": float(np.median(named_merges)),
            "random_merge_median": Fref_null,
            "single_over_random_median": float(np.median(singles)) / Fref_null},
        "factors": {n: {"F_ref": ok[n]["F_ref"], "F_adapter": ok[n]["F_adapter"],
                        "rank": ok[n]["rank"],
                        "F_over_random_median": ok[n]["F_over_random_median"]}
                    for n in ok if ok[n]["family"] == "fa"},
    }

    # --- the dose table ----------------------------------------------------
    dose = {}
    for n, v in ok.items():
        if v["family"] in ("random", "sphere"):
            continue
        dose[n] = {"F_ref": v["F_ref"],
                   "fisher_dose_per_alpha": v["fisher_dose_per_alpha"],
                   "alpha2_in_fisher_units": v["alpha2_in_fisher_units"],
                   "published_ref": v["published_ref"],
                   "kl_at_alpha2_extrapolated": 0.5 * 4.0 * v["F_ref"]}

    quads = [v["quad_check_F05_over_F0125"] for v in ok.values()
             if v.get("quad_check_F05_over_F0125")]
    out = {
        "what_F_is": ("second-order curvature of KL(p_base || p_steered) in alpha "
                      "at alpha=0, at the base point, on the base model's own "
                      "alpha-0 responses to the 24 steering prompts; nats per "
                      "token per unit alpha squared"),
        "ref": R["ref"], "ref_over_true_mean_norm": ratio,
        "n_scored_tokens": R.get("n_scored_tokens"),
        "resp_tokens": R.get("resp_tokens"), "max_resp_tokens": R.get("max_resp_tokens"),
        "text_source": R.get("text_source"), "wall_seconds": R.get("wall_seconds"),
        "lora_rank": R.get("lora_rank"), "lora_scale": R.get("lora_scale"),
        "targeted_modules": R.get("targeted_modules"),
        "targeted_params": R.get("targeted_params"),
        "zero_alpha_control": R.get("zero_alpha_control"),
        "n_directions": len(ok),
        "random_band": band,
        "quadratic_check": {
            "n": len(quads), "median": float(np.median(quads)) if quads else None,
            "min": float(np.min(quads)) if quads else None,
            "max": float(np.max(quads)) if quads else None,
            "meaning": "F measured at |alpha|=0.5 divided by F at |alpha|=0.125; "
                       "1.0 is exactly quadratic over a 4x range"},
        "residual": {
            "pure_quadratic_median_rel_rms":
                float(np.median([v["rel_resid"] for v in ok.values()])),
            "pure_quadratic_max_rel_rms":
                float(np.max([v["rel_resid"] for v in ok.values()])),
            "taylor_median_rel_rms":
                float(np.median([v["taylor_rel_resid"] for v in ok.values()])),
            "taylor_max_rel_rms":
                float(np.max([v["taylor_rel_resid"] for v in ok.values()])),
            "meaning": "RMS of (measured KL - fitted KL) / measured KL over the "
                       "alphas of each direction; pure_quadratic is KL = 0.5*F*a^2 "
                       "alone, taylor adds a^3 and a^4"},
        "headline_F_definition":
            "F_ref is the quadratic coefficient of the third-and-fourth-order Taylor fit, "
            "i.e. the curvature at alpha = 0. F_ref_pooled is the task's "
            "2*KL/alpha^2 pooled fit, which weights |alpha| = 0.5 most heavily. "
            "F_ref_pooled / F_ref is reported per direction and in agreement below.",
        "pooled_over_taylor": {
            "median": float(np.median([v["F_ref_pooled"] / v["F_ref"] for v in ok.values()])),
            "min": float(np.min([v["F_ref_pooled"] / v["F_ref"] for v in ok.values()])),
            "max": float(np.max([v["F_ref_pooled"] / v["F_ref"] for v in ok.values()]))},
        "directions": ok,
        "correlation_with_degeneration": corr,
        "correlation_multiplicity_note":
            "Six correlation arms are reported and no multiplicity correction is "
            "applied to the permutation p values; read a single p near 0.05 among "
            "six as suggestive, not established. The arms are not independent: the "
            "three alpha-2 arms share the same 22 directions and the two sphere arms "
            "share the same 72 points.",
        "directions_with_text_but_no_F": excluded,
        "comparisons": comp,
        "fisher_dose_table": dose,
        "loop_rates_recomputed": LR,
        "loop_rates_adjudicated": ADJ,
    }
    p = os.path.join(A, "fisher_norms.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {p}")
    print(f"random-merge band n={band['n']} median={band['median']:.4g} "
          f"[{band['min']:.4g}, {band['max']:.4g}]")
    print(f"quadratic check median {out['quadratic_check']['median']}")
    print("top 12 by F:")
    for n in order[:12]:
        print(f"  {n:26s} F_ref={ok[n]['F_ref']:.4g} "
              f"x{ok[n]['F_over_random_median']:.3f} random median  "
              f"resid={ok[n]['rel_resid']:.3f}")
    for k, v in corr.items():
        print(f"corr {k}: n={v['n']} rho={v['spearman']:.3f} p={v['perm_p']:.4f}")


if __name__ == "__main__":
    main()
